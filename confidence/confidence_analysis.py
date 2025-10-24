# =============================================================================
# Confidence Analysis Module
# =============================================================================
"""
Confidence analysis functions for genotype-phenotype association studies.
This module provides functions for training models and calculating confidence scores
across different missing data ratios.
"""

import torch
import numpy as np
import os
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from data.data_utils import GenotypeDataset
from model.model_utils import CMDAutoEncoder, train_autoencoder, evaluate_autoencoder
from train.evaluation_utils import get_confidence


def train_and_get_confidence_scores(train_loader, valid_loader, train_loaders_dict, valid_loaders_dict, 
                                  config, times=1, k=30):
    """
    Train models and calculate confidence scores for different missing ratios
    
    Args:
        train_loader: Main training data loader
        valid_loader: Main validation data loader
        train_loaders_dict: Dictionary containing training loaders for each missing ratio
        valid_loaders_dict: Dictionary containing validation loaders for each missing ratio
        config: Configuration object
        times (int): Number of training repetitions
        k (int): Number of nearest neighbors for confidence calculation
    
    Returns:
        tuple: Arrays of confidence scores and distances for each missing ratio
    """
    # Initialize result containers
    confidence_scores = {ratio: [] for ratio in [0, 25, 50, 75, 90]}
    distance_scores = {ratio: [] for ratio in [0, 25, 50, 75, 90]}
    
    for ii in range(times):
        print(f"\n========== Training iteration {ii+1}/{times} ==========")
        
        # Initialize model and training components
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = CMDAutoEncoder(
            in_channels=config.IN_CHANNELS,
            out_channels=config.OUT_CHANNELS,
            kernel_size=config.KERNEL_SIZE,
            stride=config.STRIDE,
            d_state=config.D_STATE
        ).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        
        # Create model save directory
        model_dir = "./model"
        os.makedirs(model_dir, exist_ok=True)
        
        # Training loop
        best_valid_acc = 0
        for epoch in tqdm(range(config.EPOCHS), desc="Training"):
            avg_loss, acc_train = train_autoencoder(model, train_loader, device, criterion, optimizer)
            acc_valid = evaluate_autoencoder(model, valid_loader, device)
            
            # Convert tensors to numpy
            if isinstance(acc_train, torch.Tensor):
                acc_train = acc_train.cpu().numpy()
            if isinstance(acc_valid, torch.Tensor):
                acc_valid = acc_valid.cpu().numpy()
            
            # Save best model
            if acc_valid > best_valid_acc:
                best_valid_acc = acc_valid
                model_file_path = os.path.join(model_dir, "model_state.pth")
                torch.save(model.state_dict(), model_file_path)
        
        print(f"Best validation accuracy: {best_valid_acc:.4f}")
        
        # Calculate confidence scores for each missing ratio
        for ratio in [0, 25, 50, 75, 90]:
            print(f"Calculating confidence scores for {ratio}% missing data...")
            
            # Get corresponding data loaders
            train_loader_ratio = train_loaders_dict[ratio]
            valid_loader_ratio = valid_loaders_dict[ratio]
            
            # Calculate confidence scores using imported function
            temp_confidence_scores, temp_distance_scores = get_confidence(
                train_loader_ratio, valid_loader_ratio, model, device, k=k
            )
            
            confidence_scores[ratio].append(temp_confidence_scores)
            distance_scores[ratio].append(temp_distance_scores)
    
    # Convert to numpy arrays
    result_confidence = {}
    result_distance = {}
    for ratio in [0, 25, 50, 75, 90]:
        result_confidence[ratio] = np.array(confidence_scores[ratio])
        result_distance[ratio] = np.array(distance_scores[ratio])
    
    return (result_confidence[0], result_confidence[25], result_confidence[50], 
            result_confidence[75], result_confidence[90],
            result_distance[0], result_distance[25], result_distance[50], 
            result_distance[75], result_distance[90])


def create_data_loaders_dict(train_X_dict, valid_X_dict, train_X_un_miss, valid_X_un_miss, 
                           batch_size=32, num_workers=4):
    """
    Create data loaders dictionary for different missing ratios
    
    Args:
        train_X_dict: Dictionary containing training data for each missing ratio
        valid_X_dict: Dictionary containing validation data for each missing ratio
        train_X_un_miss: Training data without missing values
        valid_X_un_miss: Validation data without missing values
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading
    
    Returns:
        tuple: (train_loaders_dict, valid_loaders_dict)
    """
    from data_utils import GenotypeDataset
    
    train_loaders_dict = {}
    valid_loaders_dict = {}
    
    for ratio in [0, 25, 50, 75, 90]:
        # Create datasets
        train_dataset = GenotypeDataset(train_X_dict[ratio], train_X_un_miss)
        valid_dataset = GenotypeDataset(valid_X_dict[ratio], valid_X_un_miss)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                 num_workers=num_workers, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset), 
                                 num_workers=num_workers, shuffle=False)
        
        train_loaders_dict[ratio] = train_loader
        valid_loaders_dict[ratio] = valid_loader
    
    return train_loaders_dict, valid_loaders_dict


def run_confidence_analysis(df_onehot_dict, df_onehot_no_miss, config, 
                           times=1, k=30, test_size=0.1, random_seed=42):
    """
    Run complete confidence analysis pipeline
    
    Args:
        df_onehot_dict: Dictionary containing one-hot encoded data for each missing ratio
        df_onehot_no_miss: One-hot encoded data without missing values
        config: Configuration object
        times: Number of training repetitions
        k: Number of nearest neighbors for confidence calculation
        test_size: Test set ratio
        random_seed: Random seed for reproducibility
    
    Returns:
        tuple: Results from train_and_get_confidence_scores
    """

    
    # Split data for each missing ratio
    train_X_dict = {}
    valid_X_dict = {}
    
    for ratio in [0, 25, 50, 75, 90]:
        train_X, valid_X = train_test_split(
            df_onehot_dict[ratio], test_size=test_size, random_state=random_seed
        )
        train_X_dict[ratio] = train_X
        valid_X_dict[ratio] = valid_X
    
    # Split data without missing values
    train_X_un_miss, valid_X_un_miss = train_test_split(
        df_onehot_no_miss, test_size=test_size, random_state=random_seed
    )
    
    # Create main data loaders
    train_dataset = GenotypeDataset(
        np.concatenate([df_onehot_dict[ratio] for ratio in [0, 25, 50, 75, 90]], axis=0),
        np.concatenate([df_onehot_no_miss] * 5, axis=0)
    )
    valid_dataset = GenotypeDataset(valid_X_dict[0], valid_X_un_miss)
    
    train_loader = DataLoader(train_dataset, batch_size=32, num_workers=4, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset), num_workers=4, shuffle=False)
    
    # Create data loaders dictionary
    train_loaders_dict, valid_loaders_dict = create_data_loaders_dict(
        train_X_dict, valid_X_dict, train_X_un_miss, valid_X_un_miss
    )
    
    # Run confidence analysis
    return train_and_get_confidence_scores(
        train_loader, valid_loader, train_loaders_dict, valid_loaders_dict, 
        config, times=times, k=k
    )


if __name__ == "__main__":
    # Example usage
    print("Confidence Analysis Module")
    print("This module provides functions for confidence analysis in genotype-phenotype studies.")
    print("Import and use train_and_get_confidence_scores() or run_confidence_analysis() functions.")
