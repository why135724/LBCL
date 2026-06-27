import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from mne.decoding import CSP
from scipy.io import loadmat
from tqdm import tqdm

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
        g /= (np.linalg.norm(g) + 1e-8)
        grads.append(g)

    return np.stack(grads)

def compute_uncertainty(model, X, device):
    model.eval()
    with torch.no_grad():
        X = X.to(device)
        logits = model(X)
        probs = F.softmax(logits, dim=1)
        return 1 - probs.max(dim=1)[0].cpu().numpy()


def compute_representativeness(grads, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(grads)

    centers = kmeans.cluster_centers_
    sim = grads @ centers.T
    dist = 1 - sim.max(axis=1)
    return 1 / (1 + dist)

def URBSS(
    X_train_csp,
    y_train,
    input_dim,
    num_classes,
    device,
    N_init=100,
    I_max=10,
    B=50,
    alpha=0.1,
    n_clusters=6
):
    X = torch.tensor(X_train_csp, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.long)

    idx = np.random.choice(len(X), N_init, replace=False)
    S_idx = set(idx.tolist())

    model = MLP(input_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 初始训练
    for _ in range(10):
        model.train()
        for i in idx:
            x = X[i].unsqueeze(0).to(device)
            t = y[i].unsqueeze(0).to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), t)
            loss.backward()
            optimizer.step()

    # 迭代选择
    for _ in range(I_max):
        cand = list(set(range(len(X))) - S_idx)
        Xc = X[cand]
        yc = y[cand]

        grads = compute_gradient_embeddings(model, Xc, yc, device)
        unc = compute_uncertainty(model, Xc, device)
        rep = compute_representativeness(grads, n_clusters)

        unc_s = F.softmax(torch.tensor(unc), dim=0).numpy()
        rep_s = F.softmax(torch.tensor(rep), dim=0).numpy()

        score = alpha * unc_s + (1 - alpha) * rep_s
        topB = np.argsort(score)[-B:]

        S_idx.update([cand[i] for i in topB])

        # 更新模型
        model.train()
        for _ in range(5):
            for i in S_idx:
                x = X[i].unsqueeze(0).to(device)
                t = y[i].unsqueeze(0).to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), t)
                loss.backward()
                optimizer.step()

    final_size = int(7 / 8 * len(X))
    final_idx = np.array(list(S_idx))[:final_size]

    return X[final_idx], y[final_idx]

def load_MI_dataset(root, subject_files, csp_components=10):
    X_all, y_all = [], []

    csp = CSP(n_components=csp_components, log=True)

    for f in subject_files:
        mat = loadmat(os.path.join(root, f))
        x = mat["x"]          # [C, T, N]
        y = mat["y"].flatten()

        X = np.transpose(x, (2, 0, 1))
        y = y.copy()

        split = int(0.8 * len(X))
        X_train = X[:split]
        y_train = y[:split]

        X_csp = csp.fit_transform(X_train, y_train)
        X_all.append(X_csp)
        y_all.append(y_train)

    return np.concatenate(X_all), np.concatenate(y_all)

import json
import os
from tqdm import tqdm


def save_to_jsonl(
    X_selected,
    y_selected,
    output_path,
    category_list=["0", "1"]
):
    """
    保存 URBSS 筛选后的数据为 JSONL 格式
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    data = []
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
        data.append(sample)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

    print(f"✅ Saved JSONL to: {output_path}")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for dataset in ["MI1", "MI2"]:
        root = f"/data2/hywu/LLaMA-Factory/{dataset}"
        files = sorted([f for f in os.listdir(root) if f.endswith(".mat")])

        print(f"\n=== Processing {dataset} ===")

        # ===== 加载数据 + CSP =====
        X_train, y_train = load_MI_dataset(root, files)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        # ===== URBSS 样本选择 =====
        X_sel, y_sel = URBSS(
            X_train,
            y_train,
            input_dim=X_train.shape[1],
            num_classes=len(np.unique(y_train)),
            device=device
        )

        # ===== 评估 =====
        _, X_test, _, y_test = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train
        )

        lda = LinearDiscriminantAnalysis()
        lda.fit(X_sel.cpu().numpy(), y_sel.cpu().numpy())
        acc = accuracy_score(y_test, lda.predict(X_test))

        print(f"{dataset} Test Accuracy: {acc * 100:.2f}%")
        print(f"Selected samples: {len(X_sel)} / {len(X_train)}")

        # ===== 保存 JSONL =====
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
