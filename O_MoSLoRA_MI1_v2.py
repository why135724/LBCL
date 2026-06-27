import time
start_time = time.time()

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
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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
    A_history: [r, d]
    A_current: [r, d]
    """
    gram = A_history @ A_current.T
    return torch.sum(gram ** 2)


# =======================
# LoraLinear (O-MoSLoRA)
# =======================
class LoraLinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, r=8, alpha=32, dropout_p=0.1):
        super().__init__()
        self.base_layer = copy.deepcopy(base_layer)
        self.r = r
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout_p)
        self.scaling = float(alpha) / float(r)

        self.lora_A_list = nn.ParameterList()
        self.lora_B_list = nn.ParameterList()
        self.lora_AB_list = nn.ParameterList()
        self.lora_AB1_list = nn.ParameterList()

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
        lora_AB = nn.Parameter(
            torch.randn((self.r, self.r),
                       dtype=dtype, device=device) * 0.01
        )
        lora_AB1 = nn.Parameter(
            torch.randn((self.r, self.r),
                       dtype=dtype, device=device) * 0.01
        )

        if init:
            nn.init.normal_(lora_A, 0.0, 0.02)
            nn.init.normal_(lora_AB, 0.0, 0.02)
            nn.init.normal_(lora_AB1, 0.0, 0.02)
            nn.init.zeros_(lora_B)

        self.lora_A_list.append(lora_A)
        self.lora_B_list.append(lora_B)
        self.lora_AB_list.append(lora_AB)
        self.lora_AB1_list.append(lora_AB1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        subject_id = SubjectContext.get()
        if subject_id is None:
            subject_id = len(self.lora_A_list) - 1

        output = self.base_layer(x)

        for t in range(subject_id + 1):
            if t < len(self.lora_A_list):
                A = self.lora_A_list[t]
                B = self.lora_B_list[t]
                M1 = self.lora_AB_list[t]
                M2 = self.lora_AB1_list[t]

                lora_out = F.linear(self.dropout(x), A)
                lora_out = F.linear(lora_out, M1)
                lora_out = F.linear(lora_out, M2)
                lora_out = F.linear(lora_out, B)
                output = output + lora_out * self.scaling

        return output


# =======================
# Replace Linear -> LoraLinear
# =======================
def replace_linear_with_lora(
    module, r=8, alpha=16, dropout_p=0.0,
    embed_requires_grad=False,
    norm_requires_grad=False,
    head_requires_grad=False
):
    for name, child in module.named_children():
        if any(s in name for s in ['embed', 'norm', 'lm_head']):
            requires_grad = (
                embed_requires_grad if 'embed' in name else
                norm_requires_grad if 'norm' in name else
                head_requires_grad
            )
            for param in child.parameters():
                param.requires_grad = requires_grad
        elif isinstance(child, nn.Linear):
            setattr(module, name, LoraLinear(child, r, alpha, dropout_p))
        else:
            replace_linear_with_lora(
                child, r, alpha, dropout_p,
                embed_requires_grad, norm_requires_grad, head_requires_grad
            )


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
            adapters[name] = {
                'lora_A': module.lora_A_list[subject_id].detach().clone(),
                'lora_B': module.lora_B_list[subject_id].detach().clone(),
                'lora_AB': module.lora_AB_list[subject_id].detach().clone(),
                'lora_AB1': module.lora_AB1_list[subject_id].detach().clone(),
            }
    return adapters


def merge_adapters(model):
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, LoraLinear):
                delta = 0
                for t in range(len(module.lora_A_list)):
                    A = module.lora_A_list[t]
                    B = module.lora_B_list[t]
                    M1 = module.lora_AB_list[t]
                    M2 = module.lora_AB1_list[t]
                    delta = delta + (B @ M2.T @ M1.T @ A) * module.scaling

                module.base_layer.weight.data += delta


# =======================
# Dataset Processing
# =======================
def dataset_jsonl_transfer(origin_path, new_path, dataset_type):
    instr_map = {
        "MI2": "You are an expert in EEG signal classification. You will receive a segment of CSP-processed motor imagery EEG signal and several candidate categories (0 for left hand, 1 for right hand). This is the MI2 dataset. Please output the correct type of the EEG signal.",
        "MI1": "You are an expert in EEG signal classification. You will receive a segment of CSP-processed motor imagery EEG signal and several candidate categories (0 for left hand, 1 for right hand). This is the MI1 dataset. Please output the correct type of the EEG signal.",
        "ERN": "You are an expert in EEG signal classification. You will receive a segment of XDAWN-processed event-related potential (ERN) EEG signal (output category 0 or 1). This is the ERN dataset. Please output the correct type of the EEG signal."
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
        "new_MI1_train.jsonl",
        "new_MI1_test.jsonl"
    ),
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

# =======================
# Load MI1 data
# =======================

train_df = pd.read_json(os.path.join(base_dir, "data", "new_MI1_train.jsonl"), lines=True)
print('train_df.len', len(train_df))

test_df = pd.read_json(os.path.join(base_dir, "data", "new_MI1_test.jsonl"), lines=True)
print('test_df.len', len(test_df))


n_splits = 7
split_dfs_train_list = [df for df in np.array_split(train_df, n_splits)]
split_dfs_test_list = [df for df in np.array_split(test_df, n_splits)]

print(f"Original rows training: {len(train_df)}")
print(f"Total rows after splitting training: {sum(len(df) for df in split_dfs_train_list)}")
print(f"Original rows test: {len(test_df)}")
print(f"Total rows after splitting test: {sum(len(df) for df in split_dfs_test_list)}")

# =======================
# Initialize Model
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
    device_map="auto", torch_dtype=torch.bfloat16, local_files_only=True
)
model.enable_input_require_grads()
replace_linear_with_lora(model, r=8, alpha=32, dropout_p=0.1)
model.to(device)
print_trainable_parameters(model)

history_adapters = {}
ACC_all = []

lambda_orth = 0.1
warmup_steps = 0
num_epochs = 15


for subject_id in range(7):
    print(f"\n{'='*40}")
    print(f"Training Subject {subject_id + 1}/7")
    print(f"{'='*40}")

    for module in model.modules():
        if isinstance(module, LoraLinear):
            module.add_adapter(init=True)

    optimizer = torch.optim.AdamW(
        [
            p for module in model.modules()
            if isinstance(module, LoraLinear)
            for p in [
                module.lora_A_list[subject_id],
                module.lora_B_list[subject_id],
                module.lora_AB_list[subject_id],
                module.lora_AB1_list[subject_id],
            ]
        ],
        lr=3e-4
    )

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

    history_adapters[subject_id] = collect_adapters(model, subject_id)

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

print("\nMerging all adapters into base model...")
merge_adapters(model)

print("\nAll subjects ACC:", ACC_all)
print(f"Average ACC: {np.mean(ACC_all):.4f}")

end_time = time.time()
print(f"Total running time: {end_time - start_time:.2f}s")