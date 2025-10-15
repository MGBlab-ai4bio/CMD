# CMD: Genotype-Phenotype Association Analysis

A deep learning framework for genotype-phenotype association analysis using CMD architecture with Mamba2.

## Quick Start

1. **Run the main notebook**:
   ```bash
   jupyter notebook test.ipynb
   ```

2. **Prepare your data**:
   - Genotype data: `test_geno.csv`
   - Phenotype data: `test_pheno.txt`

   **If using your own data**, convert VCF format to CSV:
   ```bash
   python data/vcf_to_csv.py
   ```
   This script converts VCF files to the required CSV format with numeric encoding (0, 1, 2, -1 for missing).

## Data Format

### Input Requirements
- **Genotype Data**: CSV format with samples as rows, SNPs as columns
- **Encoding**: 0 (homozygous reference), 1 (heterozygous), 2 (homozygous alternative), -1 (missing)
- **Phenotype Data**: CSV format with sample IDs as index and continuous values

### VCF to CSV Conversion
The `data/vcf_to_csv.py` script handles:
- VCF genotype format conversion (0/0, 0/1, 1/1 → 0, 1, 2)
- Missing data handling (./. → -1)
- Sample numbering and variant ID generation

3. **Configure parameters**:
   ```python
   class Config:
       USE_PRETRAINED = True/False  #  Use pre-trained model
       MISSING_RATIO = 0.0   # Missing data ratio
       EPOCHS = 100          # Training epochs
       BATCH_SIZE = 32       # Batch size
       MAX_ROWS = 1000        # Maximum number of samples to load (adjust based on your data size)
   ```

## Features

- **CMD Autoencoder**: Pre-training for feature extraction
- **CMD Phenotype Predictor**: Final phenotype prediction
- **10-fold Cross-validation**: Robust model evaluation
- **Early Stopping**: Prevents overfitting
- **Pre-trained Model Support**: Transfer learning capabilities

## Usage

### Basic Configuration
```python
config = Config()
config.USE_PRETRAINED = True
config.MISSING_RATIO = 0.1
```

### Advanced Configuration
```python
config.OUT_CHANNELS = 512
config.D_STATE = 256
config.LEARNING_RATE = 0.0001
config.EARLY_STOPPING_PATIENCE = 30
```

## Output

The system generates:
- Correlation coefficients for each fold
- Mean correlation ± standard deviation
- Training logs and model checkpoints


## File Structure

```
CMD/
├── test.ipynb                    # Main analysis notebook
├── model/                        # Model checkpoints
├── train/                        # Training utilities
├── data/                         # Data processing
├── confidence/                   # Confidence analysis
└── README.md                     # This file
```

## Requirements

- Python 3.10+
- PyTorch with CUDA support
- Mamba-SSM
- NumPy, Pandas, Scikit-learn

## Citation

If you use this code, please cite the original paper.

