# MultiGAI: Global Attention-Based Integration of Single-Cell Multi-Omics with Batch Effects Correction

MultiGAI is a **variational autoencoder (VAE) framework that incorporates a global attention mechanism** for integrating single-cell multi-omics data. It incorporates global information during encoding and uses specially designed components to restrict the propagation of batch-specific signals. This enables **effective batch effects correction while preserving essential biological signals**, resulting in high-quality latent representations of cells. MultiGAI provides a new strategy that jointly preserves biological information and corrects batch effects, offering fresh insights for single-cell multi-omics integration.

---

<img width="5029" height="6598" alt="MultiGAI" src="https://github.com/user-attachments/assets/568993a7-38ea-4a09-9ce3-10f15455e09e" />

---

## Features

- Integration of single-cell multi-omics datasets (scRNA-seq, scATAC-seq, ADT)
- Mapping and imputation of missing modalities
- Batch-effect correction with global attention

---

## File Structure

| File/Folder | Description |
|-------------|-------------|
| `MultiGAI.ipynb` | Source code for multi-omics integration. |
| `MultiGAI_mappingandimputing.ipynb` | Source code for mapping and imputation. |
| `MultiGAI.py` | Complete source code (all modules in one Python file). |
| `tutorial.ipynb` | Step-by-step tutorial reproducing experiments in the paper. |
| `data_process.ipynb` | Data preprocessing workflow. |
| `scib.ipynb` | Calculation of evaluation metrics (e.g., scIB metrics). |
| `umap.ipynb` | Generation of UMAP plots for visualization. |
| `environment.yml` | Conda environment configuration for experiments. |

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/Recklesszjy/Multigai.git
cd Multigai
```

### 2. Environment Setup

```bash
conda env create -f environment.yml
conda activate multigai
```

---

## Usage
Please refer to the source code `MultiGAI.py` and the tutorial `tutorial.ipynb`.


