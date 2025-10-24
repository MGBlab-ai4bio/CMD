import torch
import numpy as np
import os
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

from model.model_utils import (
    CMDPhenotypePredictor, train_phenotype_predictor, set_random_seed, load_pretrained_weights
)
from data.data_utils import PhenotypeDataset


def cross_validation_phenotype_prediction(genotype_data, phenotype_data, scaler, device, config=None):
    """
    Execute 10-fold cross-validation for phenotype prediction
    
    Args:
        genotype_data: Genotype data
        phenotype_data: Phenotype data
        scaler: Scaler
        device: Device
        config: Configuration object
    
    Returns:
        list: Best correlation coefficients for each fold
    """
    set_random_seed(config.RANDOM_SEED if config else 42)
    
    kf = KFold(n_splits=config.N_SPLITS if config else 10, shuffle=True, random_state=config.RANDOM_SEED if config else 42)
    all_best_correlations = []
    fold_losses = [[] for _ in range(config.N_SPLITS if config else 10)]
    
    for fold, (train_index, test_index) in enumerate(kf.split(genotype_data), start=1):
        print(f"\n========== Cross-validation Fold {fold}/{config.N_SPLITS if config else 10} ==========")
        
        # Set random seed for each fold
        set_random_seed((config.RANDOM_SEED if config else 42) + fold)
        
        # Split data
        X_train, X_test = genotype_data[train_index], genotype_data[test_index]
        y_train, y_test = phenotype_data[train_index], phenotype_data[test_index]
        
        # Initialize model
        model = CMDPhenotypePredictor(
            in_channels=X_train.shape[2],
            out_channels=config.OUT_CHANNELS if config else 256,
            kernel_size=config.KERNEL_SIZE if config else 5,
            stride=config.STRIDE if config else 5,
            d_state=config.D_STATE if config else 128
        ).to(device)
        
        # Load pre-trained weights
        if config.USE_PRETRAINED:
            if os.path.exists("model/model_state.pth"):
                print(f"🔄 Fold {fold}: Loading pre-trained model...")
                if load_pretrained_weights(model, "model/model_state.pth", device):
                    print(f"✅ Fold {fold}: Pre-trained model loaded successfully")
                else:
                    print(f"⚠️ Fold {fold}: Pre-trained model loading failed, using random initialization")
            else:
                print(f"⚠️ Fold {fold}: Pre-trained model file not found, using random initialization")
        else:
            print(f"🔄 Fold {fold}: Using random initialization (pre-training disabled)")
        
        # Set optimizer and loss function
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0004)
        
        # Create data loaders
        train_dataset = PhenotypeDataset(X_train, y_train)
        test_dataset = PhenotypeDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
        
        # Train model
        best_corr = -1.0
        early_stopping_counter = 0
        
        for epoch in range(1, 101):  
            corr, predictions, epoch_loss_values = train_phenotype_predictor(
                epoch, model, device, optimizer, criterion, train_loader, test_loader, scaler
            )
            
            fold_losses[fold - 1].extend(epoch_loss_values)
            
            # Early stopping mechanism
            if corr > best_corr:
                best_corr = corr
                print(f"✅ Epoch {epoch}: New best correlation = {best_corr:.4f}")
                torch.save(model.state_dict(), f"model/fold_{fold}_best_model.pth")
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            
            if early_stopping_counter >= (config.EARLY_STOPPING_PATIENCE if config else 20):
                print(f"⏹️  Epoch {epoch} early stopping (no improvement for {config.EARLY_STOPPING_PATIENCE if config else 20} epochs)")
                break
        
        print(f"🏁 Fold {fold} best correlation: {best_corr:.4f}")
        all_best_correlations.append(best_corr)
    
    return all_best_correlations
