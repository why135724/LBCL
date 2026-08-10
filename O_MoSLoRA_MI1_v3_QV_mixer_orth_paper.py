# A_history = F.normalize(A_history, dim=1, eps=1e-8)
# A_current = F.normalize(A_current, dim=1, eps=1e-8)
import sys
import time
start_time = time.time()

NUM_MIXERS = int(sys.argv[1]) if len(sys.argv) > 1 else 2
GPU_ID = sys.argv[2] if len(sys.argv) > 2 else "2"
RUN_TAG = sys.argv[3] if len(sys.argv) > 3 else "default"

import os
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import copy, torch, torch.nn as nn, torch.nn.functional as F, json, os, random, numpy as np, pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq

# =======================
# Global subject context
# =======================
class SubjectContext:
    _current_subject_id = None

    @classmethod
    def set(cls, sid):
        cls._current_subject_id = sid

    @classmethod
    def get(cls):
        return cls._current_subject_id


def set_subject_context(subject_id):
    SubjectContext.set(subject_id)


# =======================
# Device & Seed
# =======================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.bfloat16 if device != 'cpu' and torch.cuda.is_bf16_supported() else torch.float32
seed = 1
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print(f'device: {device}\ndtype: {dtype}\nSeed:{seed} GPU:{os.environ["CUDA_VISIBLE_DEVICES"]}')


# =======================
# O-MoSLoRA Orthogonal Loss
# =======================
def orthogonal_loss(A_history: torch.Tensor, A_current: torch.Tensor) -> torch.Tensor:
    """
    Orthogonality loss consistent with the paper:
        U_j = orth(A_j), U_t = orth(A_t)
        L_orth = ||U_j^T U_t||_F^2

    In this implementation, LoRA A is stored as [r, d], whereas the paper
    denotes A as [d, r]. Therefore, A.T is used to obtain an orthonormal
    basis for the corresponding column space.

    A_history: [r, d]
    A_current: [r, d]
    """
    U_history, _ = torch.linalg.qr(A_history.T.float(), mode="reduced")  # [d, r]
    U_current, _ = torch.linalg.qr(A_current.T.float(), mode="reduced")  # [d, r]

    gram = U_history.T @ U_current  # [r, r]
    return torch.sum(gram ** 2)


# =======================
# LoraLinear (O-MoSLoRA)
# =======================
# PyTorch nn.Linear stores weight as [out_features, in_features] and computes
# x @ weight.T.  Therefore the code-space parameters are transposed relative
# to the paper's row-vector notation:
#   A_paper = lora_A.T, B_paper = lora_B.T, M_paper^(k) = lora_M^(k).T.
# Consequently, applying A -> M1 -> M2 -> B below is exactly
# x @ A_paper @ M1_paper @ M2_paper @ B_paper.
class LoraLinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, r=8, alpha=32, dropout_p=0.1, num_mixers=2):
        super().__init__()
        self.base_layer = copy.deepcopy(base_layer)
        self.r = r
        self.alpha = alpha
        self.num_mixers = num_mixers
        self.dropout = nn.Dropout(dropout_p)
        self.scaling = float(alpha) / float(r)

        self.lora_A_list = nn.ParameterList()
        self.lora_B_list = nn.ParameterList()
        self.lora_M_list = nn.ParameterList()  # flattened: subject_id * num_mixers + k

        for param in self.base_layer.parameters():
            param.requires_grad = False

    def add_adapter(self, init=True):
        device = self.base_layer.weight.device
        dtype = self.base_layer.weight.dtype

        lora_A = nn.Parameter(
            torch.randn((self.r, self.base_layer.in_features),
                       dtype=dtype, device=device) * 0.01
        )
        lora_B = nn.Parameter(
            torch.zeros((self.base_layer.out_features, self.r),
                       dtype=dtype, device=device)
        )

        if init:
            nn.init.normal_(lora_A, 0.0, 0.02)
            nn.init.zeros_(lora_B)

        self.lora_A_list.append(lora_A)
        self.lora_B_list.append(lora_B)
        for _ in range(self.num_mixers):
            lora_M = nn.Parameter(
                torch.randn((self.r, self.r),
                           dtype=dtype, device=device) * 0.01
            )
            if init:
                nn.init.normal_(lora_M, 0.0, 0.02)
            self.lora_M_list.append(lora_M)

    def subject_mixers(self, t):
        return [self.lora_M_list[t * self.num_mixers + k] for k in range(self.num_mixers)]

    def adapter_params(self, subject_id):
        params = [self.lora_A_list[subject_id], self.lora_B_list[subject_id]]
        params += self.subject_mixers(subject_id)
        return params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Read subject_id from global context
        """
        subject_id = SubjectContext.get()
        if subject_id is None:
            subject_id = len(self.lora_A_list) - 1

        output = self.base_layer(x)

        for t in range(subject_id + 1):
            if t < len(self.lora_A_list):
                A = self.lora_A_list[t]
                B = self.lora_B_list[t]
                mixers = self.subject_mixers(t)

                lora_out = F.linear(self.dropout(x), A)
                for M in mixers:
                    lora_out = F.linear(lora_out, M)
                lora_out = F.linear(lora_out, B)
                output = output + lora_out * self.scaling

        return output


# =======================
# Replace ONLY W_Q / W_V -> LoraLinear
# =======================
# Paper: LoRA adapters are introduced exclusively into W_Q and W_V.
# In Hugging Face Qwen2/Qwen2.5 these projections are named q_proj and v_proj.
LORA_TARGET_MODULES = {"q_proj", "v_proj"}


def replace_linear_with_lora(
    module, r=8, alpha=16, dropout_p=0.0,
    num_mixers=2,
    target_modules=LORA_TARGET_MODULES
):
    """
    Replace only the attention query/value projections with O-MoSLoRA layers.

    All non-target pretrained parameters must remain frozen.  The caller freezes
    the whole base model before invoking this function; each inserted
    LoraLinear then exposes only its subject-specific A/B/M parameters.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            if name in target_modules:
                setattr(
                    module,
                    name,
                    LoraLinear(
                        child,
                        r=r,
                        alpha=alpha,
                        dropout_p=dropout_p,
                        num_mixers=num_mixers,
                    ),
                )
            # k_proj, o_proj, MLP projections, lm_head, etc. remain unchanged/frozen.
        else:
            replace_linear_with_lora(
                child,
                r=r,
                alpha=alpha,
                dropout_p=dropout_p,
                num_mixers=num_mixers,
                target_modules=target_modules,
            )


def verify_lora_targets(model: nn.Module):
    """Fail fast if any LoRA layer is attached outside q_proj/v_proj."""
    lora_names = [
        name for name, module in model.named_modules()
        if isinstance(module, LoraLinear)
    ]
    invalid = [
        name for name in lora_names
        if name.rsplit('.', 1)[-1] not in LORA_TARGET_MODULES
    ]
    if invalid:
        raise RuntimeError(f"LoRA attached to non-Q/V modules: {invalid}")

    q_count = sum(name.endswith('.q_proj') for name in lora_names)
    v_count = sum(name.endswith('.v_proj') for name in lora_names)
    print(f"O-MoSLoRA targets verified: q_proj={q_count}, v_proj={v_count}, total={len(lora_names)}")


def print_trainable_parameters(model: nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {trainable_params:,} || "
          f"all params: {total_params:,} || "
          f"trainable%: {100 * trainable_params / total_params:.4f}")


# =======================
# Adapter Collection & Merge
# =======================
def collect_adapters(model, subject_id):
    adapters = {}
    for name, module in model.named_modules():
        if isinstance(module, LoraLinear):
            d = {
                'lora_A': module.lora_A_list[subject_id].detach().clone(),
                'lora_B': module.lora_B_list[subject_id].detach().clone(),
            }
            for k in range(module.num_mixers):
                d[f'lora_M{k}'] = module.lora_M_list[subject_id * module.num_mixers + k].detach().clone()
            adapters[name] = d
    return adapters


def merge_adapters(model):
    """
    Merge all historical adapters into base_layer.

    In PyTorch weight storage:
        delta_weight = B_code @ M2_code @ M1_code @ A_code
    which equals
        (A_paper @ M1_paper @ M2_paper @ B_paper).T
    and is therefore exactly equivalent to the paper update under nn.Linear.
    """
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, LoraLinear):
                delta = 0
                nm = module.num_mixers
                for t in range(len(module.lora_A_list)):
                    A = module.lora_A_list[t]      # [r, in]
                    B = module.lora_B_list[t]      # [out, r]
                    M_prod = None
                    for k in range(nm):
                        M = module.lora_M_list[t * nm + k]
                        M_prod = M if M_prod is None else (M @ M_prod)
                    delta = delta + (B @ M_prod @ A) * module.scaling

                module.base_layer.weight.data += delta


# =======================
# Dataset Processing
# =======================
def dataset_jsonl_transfer(origin_path, new_path, dataset_type):
    instr_map = {
        "MI1": "You are an expert in EEG signal classification. You will receive a segment of CSP-processed motor imagery EEG signal and several candidate categories (0 for left hand, 1 for right hand). This is the MI1 dataset. Please output the correct type of the EEG signal."
    }
    instruction = instr_map.get(dataset_type,
        f"You are an expert in EEG signal classification. Please classify the following EEG signal. This data is from the {dataset_type} dataset.")

    messages = []
    with open(origin_path, "r") as file:
        for line in file:
            data = json.loads(line)
            messages.append({
                "instruction": instruction,
                "input": f"Text: {data['text']}, Category options: {data['category']}",
                "output": data['output']
            })
    with open(new_path, "w", encoding="utf-8") as file:
        for m in messages:
            file.write(json.dumps(m, ensure_ascii=False) + "\n")


def process_func(example, tokenizer, MAX_LENGTH=384):
    prompt = (
        f"<|im_start|>system\n"
        f"You are an expert in EEG signal classification. You will receive a segment of CSP-processed motor imagery EEG signal and several candidate categories (0 for left hand, 1 for right hand). Please output the correct type of the EEG signal."
        f"<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n"
    )

    instruction = tokenizer(prompt, add_special_tokens=False)
    response = tokenizer(f"{example['output']}", add_special_tokens=False)

    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.eos_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.eos_token_id]

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def predict(messages, model, tokenizer, device="cuda:0"):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=5,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
    )
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


# =======================
# Main Program
# =======================
base_dir = os.path.dirname(os.path.abspath(__file__))

paths = {
    "MI1": (
        os.path.join(base_dir, "data", "MI1_train.jsonl"),
        os.path.join(base_dir, "data", "MI1_test.jsonl"),
        f"new_MI1_train_{RUN_TAG}.jsonl",
        f"new_MI1_test_{RUN_TAG}.jsonl"
    )
}

for ds_type, (train_path, test_path, train_new, test_new) in paths.items():
    train_new_path = os.path.join(base_dir, "data", train_new)
    test_new_path = os.path.join(base_dir, "data", test_new)
    if os.path.exists(train_new_path):
        os.remove(train_new_path)
    if os.path.exists(test_new_path):
        os.remove(test_new_path)
    dataset_jsonl_transfer(train_path, train_new_path, ds_type)
    dataset_jsonl_transfer(test_path, test_new_path, ds_type)

train_df = pd.read_json(os.path.join(base_dir, "data", f"new_MI1_train_{RUN_TAG}.jsonl"), lines=True)
print('train_df.len', len(train_df))

test_df = pd.read_json(os.path.join(base_dir, "data", f"new_MI1_test_{RUN_TAG}.jsonl"), lines=True)
print('test_df.len', len(test_df))


n_splits = 7
split_dfs_train_list = [df for df in np.array_split(train_df, n_splits)]
split_dfs_test_list = [df for df in np.array_split(test_df, n_splits)]

print(f"Original rows training: {len(train_df)}")
print(f"Total rows after splitting training: {sum(len(df) for df in split_dfs_train_list)}")
print(f"Original rows test: {len(test_df)}")
print(f"Total rows after splitting test: {sum(len(df) for df in split_dfs_test_list)}")

# =======================
# Initialize Model (once for all subjects)
# =======================
print("\nInitializing model (once for all subjects)...")
tokenizer = AutoTokenizer.from_pretrained(
    "/data2/hywu/LLaMA-Factory/Qwen/Qwen2.5-1.5B-Instruct/",
    use_fast=False, trust_remote_code=True, local_files_only=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    "/data2/hywu/LLaMA-Factory/Qwen/Qwen2.5-1.5B-Instruct/",
    torch_dtype=torch.bfloat16, local_files_only=True
)
model.enable_input_require_grads()

# Paper: W_init is frozen. Freeze the complete pretrained model first, then
# introduce trainable O-MoSLoRA parameters only on q_proj and v_proj.
for param in model.parameters():
    param.requires_grad = False

replace_linear_with_lora(
    model,
    r=8,
    alpha=32,
    dropout_p=0.1,
    num_mixers=NUM_MIXERS,
    target_modules=LORA_TARGET_MODULES,
)
verify_lora_targets(model)
model.to(device)
print_trainable_parameters(model)

history_adapters = {}
ACC_all = []

lambda_orth = 0.1
warmup_steps = 0
num_epochs = 5

for subject_id in range(7):
    print(f"\n{'='*40}")
    print(f"Training Subject {subject_id + 1}/7")
    print(f"{'='*40}")

    # ===== Add adapter =====
    for module in model.modules():
        if isinstance(module, LoraLinear):
            module.add_adapter(init=True)

    # ===== Optimize only current subject's parameters =====
    optimizer = torch.optim.AdamW(
        [
            p for module in model.modules()
            if isinstance(module, LoraLinear)
            for p in module.adapter_params(subject_id)
        ],
        lr=3e-4
    )

    # ===== Data =====
    train_df_sub = split_dfs_train_list[subject_id]
    train_ds = Dataset.from_pandas(train_df_sub)
    train_dataset = train_ds.map(
        process_func,
        fn_kwargs={'tokenizer': tokenizer},
        remove_columns=train_ds.column_names
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    )

    # ===== Training =====
    model.train()
    global_step = 0

    for epoch in range(num_epochs):
        total_loss, total_step = 0.0, 0

        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            set_subject_context(subject_id)

            outputs = model(**batch)
            ce_loss = outputs.loss

            # ===== Orthogonal loss (Q and V adapters only) =====
            # Every LoraLinear in the model is either q_proj or v_proj, so this
            # sums L_orth separately over W_Q and W_V (and over Transformer layers).
            orth_loss = 0.0
            if subject_id > 0 and global_step >= warmup_steps:
                for name, module in model.named_modules():
                    if isinstance(module, LoraLinear):
                        A_current = module.lora_A_list[subject_id]
                        for t in range(subject_id):
                            if name in history_adapters[t]:
                                A_hist = history_adapters[t][name]['lora_A'].to(A_current.device)
                                orth_loss += orthogonal_loss(A_hist, A_current)

            loss = ce_loss + lambda_orth * orth_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_step += 1
            global_step += 1

            if total_step % 5 == 0:
                print(f"Step {step + 1}, CE: {ce_loss.item():.4f}, "
                      f"Orth: {orth_loss:.4f}, Total: {loss.item():.4f}")

        print(f"Epoch {epoch + 1} avg loss: {total_loss / total_step:.4f}")

    # ===== Save adapter =====
    history_adapters[subject_id] = collect_adapters(model, subject_id)

    # ===== Testing =====
    print("\nTesting...")
    set_subject_context(subject_id)

    test_df_sub = split_dfs_test_list[subject_id]
    preds = []
    for _, row in test_df_sub.iterrows():
        messages = [
            {"role": "system", "content": row["instruction"]},
            {"role": "user", "content": row["input"]}
        ]
        preds.append(predict(messages, model, tokenizer, device))

    correct = sum(
        1 for i, p in enumerate(preds)
        if p == str(test_df_sub['output'].iloc[i]).strip()
    )
    acc = correct / len(preds)
    print(f"Subject {subject_id + 1} ACC: {acc:.4f}")
    ACC_all.append(acc)

# ===== Final merge =====
print("\nMerging all adapters into base model...")
merge_adapters(model)

print("\nAll subjects ACC:", ACC_all)
print(f"Average ACC: {np.mean(ACC_all):.4f}")

end_time = time.time()
print(f"Total running time: {end_time - start_time:.2f}s")