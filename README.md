# CMD Genotype-Phenotype Association Analysis based on Mamba2

## Project Overview

CMD is a deep learning project based on the Mamba2 architecture for genotype-phenotype association analysis. This project combines convolutional neural networks with Mamba state space models to process large-scale genomic data and predict phenotypic traits.

## Project Structure

```
CMD/
├── config.py              # Configuration file
├── requirements.txt        # Dependencies
├── data/                   # Data processing module
│   ├── data_utils.py       # Data loading and preprocessing utilities
│   ├── data_loder.py       # Data loader
│   └── vcf_to_csv.py       # VCF format conversion tool
├── model/                  # Model definitions
│   ├── model_utils.py      # Model architecture and utility functions
│   ├── pretrain_model.py   # Pre-training model
│   └── get_bv.py           # Breeding value prediction model
├── train/                  # Training module
│   ├── train.py            # Main training script
│   ├── cross_train.py      # Cross-validation training
│   └── evaluation_utils.py # Evaluation utilities
├── confidence/             # Confidence analysis
│   └── confidence_analysis.py
├── tutorial/               # Tutorials and examples
│   ├── test.ipynb          # Basic testing
│   ├── predict_bv.ipynb    # Breeding value prediction example
│   ├── interpretive_analysis.ipynb  # Interpretive analysis
│   └── confidence_analysis.ipynb     # Confidence analysis example
└── dataset/                # Example datasets
    ├── simulate_geno.csv   # Simulated genotype data
    ├── simulate_pheno.txt  # Simulated phenotype data
    ├── test_geno.csv       # Test genotype data
    └── test_pheno.csv      # Test phenotype data
```

## Core Features

### 1. Data Processing
- Support for multiple genotype data formats (CSV, VCF, etc.)
- Data preprocessing and standardization
- Missing value handling
- Data augmentation and simulation

### 2. Model Architecture
- **CMD-AutoEncoder**: Convolutional and Mamba-based autoencoder
- **CMD-Predictor**: Genotype to phenotype prediction model
- **Pre-training mechanism**: Support for unsupervised pre-training

### 3. Training and Evaluation
- Cross-validation training
- Early stopping mechanism
- Model evaluation metrics
- Confidence analysis

### 4. Interpretive Analysis
- Integrated gradients analysis
- Feature importance assessment
- Visualization tools

## Quick Start

### Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

1. **Data Preparation**
   ```python
   from data.data_utils import load_genotype_data
   from config import Config
   
   # Load data
   genotype_data = load_genotype_data('dataset/test_geno.csv')
   ```

2. **Model Training**
   ```python
   from train.train import train
   from model.model_utils import CMDModel
   
   # Create model
   model = CMDModel(Config.IN_CHANNELS, Config.OUT_CHANNELS)
   
   # Start training
   train(epoch, model, device, optimizer, criterion, train_loader, test_loader)
   ```

3. **Prediction and Evaluation**
   ```python
   from model.get_bv import predict_bv
   
   # Predict breeding values
   predictions = predict_bv(model, test_data)
   ```

## Configuration

Main configuration parameters in `config.py`:

- **Data Configuration**: Input file paths, maximum rows, missing ratios, etc.
- **Model Configuration**: Input/output channels, kernel sizes, Mamba parameters, etc.
- **Training Configuration**: Batch size, learning rate, training epochs, etc.
- **Analysis Configuration**: Cross-validation folds, feature importance analysis parameters, etc.

## Tutorials and Examples

Check the Jupyter notebook files in the `tutorial/` directory:

- `test.ipynb`: Basic functionality testing
- `predict_bv.ipynb`: Complete breeding value prediction workflow
- `interpretive_analysis.ipynb`: Model interpretability analysis
- `confidence_analysis.ipynb`: Confidence analysis

## Requirements

- Python >= 3.7
- PyTorch >= 1.7.0
- pandas >= 1.0.0
- numpy >= 1.18.0
- scikit-learn >= 0.24.0
- mamba-ssm >= 1.0.0
- causal-conv1d >= 1.0.0

## Notes

1. Ensure sufficient GPU memory for training large-scale models
2. Data preprocessing may take considerable time; SSD storage is recommended
3. Model checkpoints are automatically saved during training
4. Supports robustness analysis under various missing rates

## License

This project follows the corresponding open source license. Please check the LICENSE file in the project root directory for specific information.
