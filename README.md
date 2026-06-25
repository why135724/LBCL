# LCBE
**LCBE: LLM-Based Continual Learning Approach for Brain-Computer Interfaces with Efficient Tuning**

> **Authors:**  
> Huanyu Wu<sup>1</sup>, Jiayu An<sup>1</sup>, Siyuan Kan<sup>1</sup>, Dongrui Wu<sup>1</sup> ✉️  
> <sup>1</sup> School of Artificial Intelligence and Automation, Huazhong University of Science and Technology  

---

## 🧬 Overview

**LCBE** is an LLM-based continual learning framework tailored for EEG-based brain-computer interfaces.  
It addresses three critical limitations of existing EEG foundation models: **high pre-training costs**, **lack of effective sample selection**, and **absence of robust continual learning mechanisms**.

### Core Innovations

- **URBSS (Uncertainty-Representativeness Balanced Sample Selection)**  
  Combines cluster analysis with model prediction uncertainty to identify high-quality training samples
- **EPC (EEG-Specific Prompt Construction)**  
  Integrates EEG features with task instructions into structured prompts for LLM consumption
- **O-MoSLoRA (Orthogonal Mixture of Subspaces in Low-Rank Adaptation)**  
  Learns subject-specific LoRA parameters with mixer layers and orthogonal regularization to mitigate catastrophic forgetting

<p align="center">
  <img src="assets/flowchart.png" width="720"/>
</p>

---

## 🔬 Features

- 🎯 **High-quality sample mining** via hybrid uncertainty-representativeness selection  
- 🧩 **Structured prompting** for seamless EEG-LLM integration without architectural modification  
- 🔄 **Continual learning ready** with orthogonal subspace constraints for cross-subject adaptation  
- 💾 **Parameter-efficient tuning** via LoRA-based adaptation  
- 📈 **Superior performance**: +3.2% average accuracy gain over prevailing methods across 5 public datasets  
- 🔀 **Mergeable adapters**: All subject-specific adaptations can be merged back into the base model  

---

## 🏗️ Framework Pipeline

LCBE operates in four stages:

| Stage | Description |
|-------|-------------|
| **Preprocessing & Feature Extraction** | Band-pass filtering, trial alignment, and paradigm-specific feature vectorization |
| **URBSS** | Iterative sample selection balancing prediction uncertainty and gradient-space diversity |
| **EPC** | Conversion of selected samples into structured prompts combining task instructions |
| **O-MoSLoRA** | Subject-specific adapter learning with mixer layers and orthogonal constraints |

<p align="center">
  <img src="assets/architecture_overview.png" width="720"/>
</p>

---

## 📂 Code Structure

<pre>
LCBE/
│
├── train_CO.py              # Chronological Order (CO) scenario
├── train_CV.py              # Cross-Validation (CV) scenario
├── train_LOSO.py            # Leave-One-Subject-Out (LOSO) scenario
│
├── models/                  # Model architectures
│   ├── LCBE.py              # Main LCBE framework (Ours)
│   ├── feature_extractor.py # Paradigm-aware feature extractors
│   ├── urbss.py             # Uncertainty-Representativeness Balanced Sample Selection
│   ├── prompt_constructor.py# EEG-Specific Prompt Construction (EPC)
│   ├── omoslora.py          # Orthogonal Mixture of Subspaces in LoRA (O-MoSLoRA)
│   └── backbone/            # LLM backbones
│       ├── llama_adapter.py
│       └── ...
│
├── data/                    # Datasets
│   ├── BNCI2014001/
│   ├── BNCI2014004/
│   ├── Zhou2016/
│   ├── Blankertz2007/
│   └── BNCI2014002/
│
├── utils/                   # Utilities
│   ├── data_utils.py        # EEG preprocessing & loading
│   ├── metrics.py           # Evaluation metrics
│   ├── clustering.py        # Cluster analysis for URBSS
│   └── ...
│
└── README.md
</pre>

---

## 🧪 Baselines

LCBE is compared against **state-of-the-art EEG foundation models and continual learning approaches**:

| Category | Methods |
|----------|---------|
| EEG Foundation Models | EEG Conformer, Large-Scale EEG Foundation Models |
| CNN-based | EEGNet, SCNN, DCNN, FBCNet |
| Transformer-based | CTNet, MSCFormer, MSVTNet |
| CL Approaches | EWC, LwF, Replay-based methods |

---

## 📊 Datasets

### Motor Imagery (MI)
- BNCI2014001  
- BNCI2014004  
- Zhou2016  
- Blankertz2007  
- BNCI2014002  

> All MI datasets are accessible via [MOABB](https://moabb.github.io/).  
> Preprocessed versions are also provided in the repository.

---

## ⚙️ Experimental Scenarios

| Scenario | Description |
|---------|-------------|
| **CO** | Within-subject; first 80% trials for training, last 20% for testing |
| **CV** | Within-subject; stratified 5-fold cross-validation |
| **LOSO** | Cross-subject; leave one subject out for testing |
| **Continual Learning** | Sequential subject adaptation with evaluation on all seen subjects |

---

## 📈 Results Summary

- **+3.2% average accuracy gain** over prevailing approaches across 5 public datasets
- **Effective catastrophic forgetting mitigation** via orthogonal subspace constraints
- **Reduced computational overhead** compared to full fine-tuning of LLM backbones

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Open **issues** for bug reports or feature requests  
- Submit **pull requests** for improvements, new baselines, or additional datasets  
- Extend LCBE to other EEG paradigms beyond motor imagery  

---

<p align="center">
  ⭐ <b>Star this repo if you find LCBE useful!</b> ⭐
</p>
