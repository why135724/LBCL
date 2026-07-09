import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from mne.decoding import CSP
from scipy.io import loadmat
from tqdm import tqdm


# =========================
# Set random seed
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# MLP model
# =========================
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# =========================
# Compute gradient embeddings
# =========================
def compute_gradient_embeddings(model, X, y, device):
    model.eval()
    grads = []
    criterion = nn.CrossEntropyLoss()

    for i in range(len(X)):
        x = X[i].unsqueeze(0).to(device)
        target = y[i].unsqueeze(0).to(device)

        model.zero_grad()
        logits = model(x)
        loss = criterion(logits, target)
        loss.backward()

        g = model.net[-1].weight.grad.detach().cpu().numpy().reshape(-1)
        g = g / (np.linalg.norm(g) + 1e-8)
        grads.append(g)

    return np.stack(grads)


# =========================
# Compute uncertainty
# =========================
def compute_uncertainty(model, X, device):
    model.eval()

    with torch.no_grad():
        X = X.to(device)
        logits = model(X)
        probs = F.softmax(logits, dim=1)
        uncertainty = 1.0 - probs.max(dim=1)[0]

    return uncertainty.cpu().numpy()


# =========================
# Compute representativeness
# =========================
def compute_representativeness(grads, n_clusters):
    n_samples = len(grads)

    if n_samples == 0:
        return np.array([])

    n_clusters = min(n_clusters, n_samples)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    kmeans.fit(grads)
    centers = kmeans.cluster_centers_

    sim = grads @ centers.T
    dist = 1.0 - sim.max(axis=1)

    representativeness = 1.0 / (1.0 + dist)

    return representativeness


# =========================
# URBSS: one-round selection version
# Final sample count = int(7/8 * len(X))
# Final samples = initial random samples + one-time selected samples
# =========================
def URBSS(
    X_train_csp,
    y_train,
    input_dim,
    num_classes,
    device,
    N_init=100,
    alpha=0.1,
    n_clusters=6,
    target_ratio=7 / 8,
    seed=42
):
    set_seed(seed)

    X = torch.tensor(X_train_csp, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.long)

    total_size = len(X)
    final_size = int(target_ratio * total_size)

    if final_size <= 0:
        raise ValueError("final_size <= 0, please check the data size or target_ratio.")

    # The initial random sample count cannot exceed the final target sample count
    initial_size = min(N_init, final_size)

    # =========================
    # 1. Initial random selection
    # =========================
    init_idx = np.random.choice(
        total_size,
        size=initial_size,
        replace=False
    )

    selected_idx = init_idx.tolist()
    selected_set = set(selected_idx)

    # Number of additional samples needed in one round
    need_more = final_size - initial_size

    print(f"Total samples: {total_size}")
    print(f"Target selected samples: {final_size}")
    print(f"Initial random samples: {initial_size}")
    print(f"Need more samples in one round: {need_more}")

    # =========================
    # 2. Train the MLP once using only the initial random samples
    # =========================
    model = MLP(input_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()

    for _ in range(10):
        for i in selected_idx:
            x = X[i].unsqueeze(0).to(device)
            t = y[i].unsqueeze(0).to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, t)
            loss.backward()
            optimizer.step()

    # =========================
    # 3. Select need_more samples from the remaining samples in one round
    # =========================
    if need_more > 0:
        candidate_idx = np.array(
            [i for i in range(total_size) if i not in selected_set]
        )

        X_candidate = X[candidate_idx]
        y_candidate = y[candidate_idx]

        # 3.1 Compute gradient embeddings
        grads = compute_gradient_embeddings(
            model,
            X_candidate,
            y_candidate,
            device
        )

        # 3.2 Compute uncertainty
        uncertainty = compute_uncertainty(
            model,
            X_candidate,
            device
        )

        # 3.3 Compute representativeness
        representativeness = compute_representativeness(
            grads,
            n_clusters=n_clusters
        )

        # 3.4 softmax normalization
        uncertainty_score = F.softmax(
            torch.tensor(uncertainty, dtype=torch.float32),
            dim=0
        ).numpy()

        representativeness_score = F.softmax(
            torch.tensor(representativeness, dtype=torch.float32),
            dim=0
        ).numpy()

        # 3.5 Combined score
        score = (
            alpha * uncertainty_score
            + (1.0 - alpha) * representativeness_score
        )

        # 3.6 Select the top need_more samples in one round
        top_idx_in_candidate = np.argsort(score)[-need_more:][::-1]
        selected_from_candidate = candidate_idx[top_idx_in_candidate].tolist()

        selected_idx.extend(selected_from_candidate)

    # =========================
    # 4. Final check
    # =========================
    selected_idx = np.array(selected_idx)

    assert len(selected_idx) == final_size, (
        f"Selection count error: current {len(selected_idx)}, target {final_size}"
    )

    assert len(set(selected_idx.tolist())) == len(selected_idx), (
        "Duplicate sample indices exist in the selection result."
    )

    print(f"Final selected samples: {len(selected_idx)} / {total_size}")

    return X[selected_idx], y[selected_idx]


# =========================
# Load MI dataset + CSP
# =========================
def load_MI_dataset(root, subject_files, csp_components=10):
    X_all, y_all = [], []

    csp = CSP(n_components=csp_components, log=True)

    for f in subject_files:
        mat_path = os.path.join(root, f)
        mat = loadmat(mat_path)

        x = mat["x"]          # [C, T, N]
        y = mat["y"].flatten()

        X = np.transpose(x, (2, 0, 1))  # [N, C, T]
        y = y.copy()

        split = int(0.8 * len(X))

        X_train = X[:split]
        y_train = y[:split]

        X_csp = csp.fit_transform(X_train, y_train)

        X_all.append(X_csp)
        y_all.append(y_train)

    X_all = np.concatenate(X_all)
    y_all = np.concatenate(y_all)

    return X_all, y_all


# =========================
# Save JSONL
# =========================
def save_to_jsonl(
    X_selected,
    y_selected,
    output_path,
    category_list=["0", "1"]
):
    """
    Save the data selected by URBSS in JSONL format
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for i in tqdm(range(len(X_selected)), desc="Writing JSONL"):
            sample = {
                "text": np.array2string(
                    X_selected[i].cpu().numpy(),
                    separator=", ",
                    formatter={"float_kind": lambda x: f"{x:.4f}"},
                    threshold=np.inf,
                    edgeitems=np.inf,
                ),
                "category": category_list,
                "output": str(y_selected[i].item())
            }

            json.dump(sample, f, ensure_ascii=False)
            f.write("\n")

    print(f"Saved JSONL to: {output_path}")


# =========================
# Main program
# =========================
if __name__ == "__main__":
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for dataset in ["MI1", "MI2"]:
        root = f"/data2/hywu/LLaMA-Factory/{dataset}"

        files = sorted([
            f for f in os.listdir(root)
            if f.endswith(".mat")
        ])

        print(f"\n=== Processing {dataset} ===")

        # =========================
        # Load data + CSP
        # =========================
        X_train, y_train = load_MI_dataset(root, files)

        # =========================
        # Standardization
        # =========================
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        # =========================
        # URBSS one-round selection
        # Final sample count = 7/8 of all samples
        # Final samples = initial random samples + one-time selected samples
        # =========================
        X_sel, y_sel = URBSS(
            X_train_csp=X_train,
            y_train=y_train,
            input_dim=X_train.shape[1],
            num_classes=len(np.unique(y_train)),
            device=device,
            N_init=100,
            alpha=0.1,
            n_clusters=6,
            target_ratio=7 / 8,
            seed=42
        )

        # =========================
        # Evaluation
        # =========================
        _, X_test, _, y_test = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            stratify=y_train,
            random_state=42
        )

        lda = LinearDiscriminantAnalysis()
        lda.fit(
            X_sel.cpu().numpy(),
            y_sel.cpu().numpy()
        )

        y_pred = lda.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"{dataset} Test Accuracy: {acc * 100:.2f}%")
        print(f"Selected samples: {len(X_sel)} / {len(X_train)}")
        print(f"Expected selected samples: {int(7 / 8 * len(X_train))}")

        # =========================
        # Save JSONL
        # =========================
        jsonl_path = (
            f"/data2/hywu/Lora_from_scratch/data/"
            f"{dataset}_train_urbss.jsonl"
        )

        save_to_jsonl(
            X_sel,
            y_sel,
            jsonl_path,
            category_list=["0", "1"]
        )