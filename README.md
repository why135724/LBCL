# LCBE
**LCBE: LLM-Based Continual Learning Approach for Brain-Computer Interfaces with Efficient Tuning**

> **Authors:**  
> Huanyu Wu<sup>1</sup>, Jiayu An<sup>1</sup>, Siyuan Kan<sup>1</sup>, Dongrui Wu<sup>1</sup> ✉️  
> <sup>1</sup> School of Artificial Intelligence and Automation, Huazhong University of Science and Technology  

## 🧬 Overview

**LCBE** is a dual-branch convolutional Transformer network tailored for EEG decoding.  
It serves as a **benchmark codebase** for EEG decoding models, where we implement and fairly evaluate **13 state-of-the-art models**, including:

- CNN-based models  
- CNN–Transformer hybrid models  
- CNN–Mamba hybrid models  

### Key Components

- **T-Conformer**: Captures temporal dependencies  
- **S-Conformer**: Models spatial patterns  
- **Channel Attention Module**: Refines spatial representations via data-driven channel weighting  

<p align="center">
  <img src="assets/dbconformer_overview.png" width="720"/>
</p>

---

## 🔬 Features

- 🔀 **Dual-branch parallel design** for symmetric spatio-temporal modeling  
- 🧩 **Plug-and-play channel attention** for adaptive channel weighting  
- 📈 **Strong generalization** across CO, CV, and LOSO settings  
- 💡 **High interpretability**, aligned with sensorimotor priors in MI  
- 🧮 **Lightweight**: ~8× fewer parameters than large CNN–Transformer baselines (e.g., EEG Conformer)  

---

## 🏗️ Network Architecture Comparison

Comparison among CNNs, traditional serial Conformers, and the proposed **DBConformer**.

<p align="center">
  <img src="assets/architecture_comparison.png" width="720"/>
</p>

---

## 📂 Code Structure
<pre>
DBConformer/
│
├── DBConformer_CO.py # Chronological Order (CO) scenario
├── DBConformer_CV.py # Cross-Validation (CV) scenario
├── DBConformer_LOSO.py # Leave-One-Subject-Out (LOSO) scenario
│
├── models/ # Model architectures
│ ├── DBConformer.py # Dual-branch Convolutional Transformer (Ours)
│ ├── EEGNet.py
│ ├── SCNN.py
│ ├── DCNN.py
│ ├── FBCNet.py
│ ├── ADFCNN.py
│ ├── IFNet.py
│ ├── EEGWaveNet.py
│ ├── SlimSeiz.py
│ ├── CTNet.py
│ ├── MSVTNet.py
│ ├── MSCFormer.py
│ ├── TMSA-Net.py
│ └── EEGConformer.py
│
├── data/ # Datasets
│ ├── BNCI2014001/
│ └── ...
│
├── utils/ # Utilities
│ ├── data_utils.py # EEG preprocessing
│ ├── alg_utils.py # Euclidean Alignment, etc.
│ ├── network.py # Backbone definitions
│ └── ...
│
└── README.md
</pre>
---

## 🧪 Baselines

We reproduce and compare **10 representative EEG decoding models**:

| Type | Models |
|------|--------|
| CNNs | EEGNet, SCNN, DCNN, FBCNet, ADFCNN, IFNet, EEGWaveNet |
| Serial Conformers | CTNet, EEG Conformer |
| CNN–Mamba | SlimSeiz |

<p align="center">
  <img src="assets/baseline_comparison.png" width="720"/>
</p>

---

## 📊 Datasets

### Motor Imagery (MI)
- BNCI2014001  
- BNCI2014004  
- Zhou2016  
- Blankertz2007  
- BNCI2014002  

### Seizure Detection
- CHSZ (publicly available on Zenodo)  
- NICU  

> MI datasets can be obtained from [MOABB](https://moabb.github.io/).  
> Processed BNCI2014001 is also available in [MVCNet](https://github.com/...).

---

## ⚙️ Experimental Scenarios

| Scenario | Description |
|---------|-------------|
| **CO** | Within-subject; first 80% trials for training, last 20% for testing |
| **CV** | Within-subject; stratified 5-fold cross-validation |
| **LOSO** | Cross-subject; leave one subject out for testing |
| **CD** | Cross-dataset generalization (see Table S1 in Supplementary Material) |

---


## 🤝 Contributing

Feel free to open **issues** or submit **pull requests**.  
We welcome improvements, bug fixes, new baselines, and additional datasets.

---

<p align="center">
  ⭐ <b>Star this repo if you find DBConformer useful!</b> ⭐
</p>
