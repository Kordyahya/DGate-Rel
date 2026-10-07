# DGate-Rel
Code for DGate-Rel: Reliability-Gated Reciprocal Distillation for Data-Scarce Medical Image Classification
# DGate-Rel

Code for the manuscript:

**DGate-Rel: Reliability-Gated Reciprocal Distillation from Complementary Weight-Selected Students for Data-Scarce Medical Image Classification**
Yahya Kord Tamandani (University of Sistan and Baluchestan) 
*Manuscript under review. This repository is provided for anonymous/inspection review.*

---

## What this code does

Trains two compact ViT-T students initialized from complementary importance-ranked subsets of a pretrained ViT-S, exchanges EMA-smoothed peer predictions between them, gates the peer supervision with a Jensen–Shannon disagreement signal, and adds a relational consistency loss. Only Student A is used at inference.

Evaluated on chest radiography, lung CT, and brain MRI at 1%, 5%, and 10% training data across 5 seeds.

## Requirements

- Python 3.10+
- PyTorch 2.x with CUDA (a GPU is strongly recommended)
- `timm >= 1.0.9`

Install:

```bash
pip install -r requirements.txt
```

## Datasets

The three public datasets used in the paper must be downloaded separately (they are not included here):

- **COVID-19 Radiography** — https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
- **COVID-CT slices** — https://www.kaggle.com/datasets/maitrisharma/covid-ct-md
- **Brain Tumor MRI** — https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
