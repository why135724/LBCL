# LBCL
**LBCL: LLM-Based Continual Learning with Parameter-Efficient Fine-Tuning for Brain-Computer Interfaces**

> **Authors:**  
> Huanyu Wu<sup>1</sup>, Jiayu An<sup>1</sup>, Siyuan Kan<sup>1</sup>, Dongrui Wu<sup>1</sup> ✉️  
> <sup>1</sup> School of Artificial Intelligence and Automation, Huazhong University of Science and Technology  

---

## 🧬 Overview

**LBCL** is an LLM-based continual learning framework tailored for EEG-based brain-computer interfaces.  
It alleviates three limitations of existing EEG foundation models: **unstable performance on downstream tasks**, **lack of effective sample selection**, and **absence of robust continual learning mechanisms**.

### Core Innovations

- **URBSS (Uncertainty-Representativeness Balanced Sample Selection)**  
  Combines clustering analysis with model prediction uncertainty to identify high-quality training samples
- **EPC (EEG-Specific Prompt Construction)**  
  Integrates EEG features with task instructions into structured prompts for LLM consumption
- **O-MoSLoRA (Orthogonal Mixture of Subspaces in Low-Rank Adaptation)**  
  Learns subject-specific LoRA parameters with mixer layers and orthogonal regularization to mitigate catastrophic forgetting

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

LBCL operates in four stages:

| Stage | Description |
|-------|-------------|
| **Preprocessing & Feature Extraction** | Band-pass filtering, trial alignment, and paradigm-specific feature vectorization |
| **URBSS** | Sample selection balancing prediction uncertainty and gradient-space diversity |
| **EPC** | Conversion of selected samples into structured prompts combining task instructions |
| **O-MoSLoRA** | Subject-specific adapter learning with mixer layers and orthogonal constraints |

---

## 📂 Code Structure

<pre>
LBCL/
│
├── MI1_EEGNET_WITHIN_CL.py  #  Performance of other representative EEG models (EEGNet, ShallowNet, DeepConvNet, EEGConformer, DBConformer) on MI1 dataset
├── MI2_EEGNET_WITHIN_CL.py  #  Performance of other representative EEG models (EEGNet, ShallowNet, DeepConvNet, EEGConformer, DBConformer) on MI1 dataset
├── ERP1_EEGNET_WITHIN_CL.py  #  Performance of other representative EEG models (EEGNet, ShallowNet, DeepConvNet, EEGConformer, DBConformer) on ERP11 dataset
├── ERP2_EEGNET_WITHIN_CL.py  #  Performance of other representative EEG models (EEGNet, ShallowNet, DeepConvNet, EEGConformer, DBConformer) on ERP2 dataset
├── Sleep_EEGNET_WITHIN_CL.py  #  Performance of other representative EEG models (EEGNet, ShallowNet, DeepConvNet, EEGConformer, DBConformer) on Sleep dataset
├── MI1_CBraMod_WITHIN_CL.py  #  Performance of CBraMod on the MI1 dataset
├── MI2_CBraMod_WITHIN_CL.py  #  Performance of CBraMod on the MI2 dataset
├── O_MoSLoRA_MI1_v2_mixer_orth_paper.py #  Performance of O-MoSLoRA on the MI1 dataset
├── O_MoSLoRA_MI2_v2_mixer_orth_paper.py #  Performance of O-MoSLoRA on the MI2 dataset
├── URBSS_MI_v3.py #  Data selection approach (URBSS) of MI datasets
├── requirements.txt
└── README.md
</pre>

---

## 🚀 Quick start

### 📦 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 🏋️ 2. Comparison with Different EEG-specialized Models in the  continual learning scenario

```bash
python MI1_EEGNET_WITHIN_CL.py  #  Performance of other representative EEG models (EEGNet, ShallowNet, DeepConvNet, EEGConformer, DBConformer) on MI1 dataset
python MI2_EEGNET_WITHIN_CL.py  #  Performance of other representative EEG models (EEGNet, ShallowNet, DeepConvNet, EEGConformer, DBConformer) on MI1 dataset
python ERP1_EEGNET_WITHIN_CL.py  #  Performance of other representative EEG models (EEGNet, ShallowNet, DeepConvNet, EEGConformer, DBConformer) on ERP11 dataset
python ERP2_EEGNET_WITHIN_CL.py  #  Performance of other representative EEG models (EEGNet, ShallowNet, DeepConvNet, EEGConformer, DBConformer) on ERP2 dataset
python Sleep_EEGNET_WITHIN_CL.py  #  Performance of other representative EEG models (EEGNet, ShallowNet, DeepConvNet, EEGConformer, DBConformer) on Sleep dataset
```

### 🏋️ 3. PEFT O-MoSLoRA

```bash
python O_MoSLoRA_MI1_v2_mixer_orth_paper.py #  Performance of O-MoSLoRA on the MI1 dataset
python O_MoSLoRA_MI2_v2_mixer_orth_paper.py #  Performance of O-MoSLoRA on the MI2 dataset
```

### 🏋️ 4. Data selection approach - URBSS (MI dataset)

```bash
python URBSS_MI_v3.py
```

---

## 🧪 Baselines

LBCL is compared against **state-of-the-art EEG foundation models and continual learning approaches**:

| Category | Methods |
|----------|---------|
| EEG Foundation Models | CBraMod, LaBraM, BENDR |
| CNN-based | EEGNet, ShallowNet, DeepConvNet |
| Transformer-based | Conformer, DBConformer  |
| CL Approaches | EWC, MAS, LwF, O-LoRA |

---

## 📊 Datasets

### Motor Imagery (MI)
- MI1 https://www.bbci.de/competition/iv/desc_1.html
- MI2 https://www.bbci.de/competition/iv/desc_2a.pdf

### Event-Related Potentials (ERPs)
- ERP1 https://physionet.org/physiobank/database/ltrsvp
- ERP2 https://www.kaggle.com/c/inria-bci-challenge

### Sleep Staging (SS)
- Sleep https://physionet.org/content/sleep-edfx/1.0.0/

---

## ⚙️ Experimental Scenarios

| Scenario | Description |
|---------|-------------|
| **Continual Learning + CO** | Sequential subject adaptation with evaluation on all seen subjects （Within-subject; first 70% trials for training, last 20% for testing） |

---

## 📈 Results Summary

- **+5.3% average BCA gain** over prevailing approaches across 5 public datasets
- **Effective catastrophic forgetting mitigation** via orthogonal subspace constraints
- **Acceptable computational overhead** compared to other EEG Foundation Models, CNNs

---
