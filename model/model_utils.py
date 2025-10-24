# =============================================================================
# Model architecture and training utilities
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from tqdm import tqdm
from mamba_ssm import Mamba2
from config import Config


# =============================================================================
# Model architecture definitions
# =============================================================================
class CMDAutoEncoder(nn.Module):
    """
    CMD-based autoencoder model - for pre-training
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, d_state=128):
        super(CMDAutoEncoder, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride)
        self.relu = nn.ReLU()
        
        # CMD layer
        self.mamba = Mamba2(
            d_model=out_channels,
            d_state=d_state,
            d_conv=4,
            expand=2,
        ).to("cuda")
        
        # Decoder layers
        self.conv1 = nn.Conv1d(out_channels, stride, kernel_size, stride=1, padding='same')
        self.conv2 = nn.ConvTranspose1d(stride, in_channels, kernel_size, stride=stride)

    def forward(self, x):
        # Encoder
        x = self.conv(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)  # (B, C, L) -> (B, L, C)
        x = self.mamba(x)
        x = x.permute(0, 2, 1)  # (B, L, C) -> (B, C, L)
        
        # Decoder
        x = self.conv1(x)
        x = self.conv2(x)
        return x

    def get_encoder_features(self, x):
        """Get encoder features"""
        x = self.conv(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)
        x = self.mamba(x)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        return x


class CMDPhenotypePredictor(nn.Module):
    """
    CMD-based phenotype prediction model
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, d_state=128, input_length=None):
        super(CMDPhenotypePredictor, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        # CMD layer
        self.mamba = Mamba2(
            d_model=out_channels,
            d_state=d_state,
            d_conv=4,
            expand=2,
        ).to("cuda")

        self.conv1 = nn.Conv1d(out_channels, stride, kernel_size, stride=1, padding="same")
        if input_length is None:
            input_length = self._calculate_input_length(20000, kernel_size, stride)
        
        self.predictor = nn.Sequential(
            nn.Linear(Config.INPUT_LENGTH, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def _calculate_input_length(self, input_length, kernel_size, stride):
        """Calculate linear layer input dimension"""
        conv1_output = (input_length - kernel_size) // stride + 1
        conv2_output = conv1_output  # padding='same'
        return stride * conv2_output

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = x.permute(0, 2, 1)
        x = self.mamba(x)
        x = x.permute(0, 2, 1)

        x = self.conv1(x)
        x = x.flatten(-2, -1)
        x = self.predictor(x)
        return x




def train_autoencoder(model, train_loader, device, criterion, optimizer):
    """
    Train autoencoder model
    
    Args:
        model: Autoencoder model
        train_loader: Training data loader
        device: Device
        criterion: Loss function
        optimizer: Optimizer
    
    Returns:
        tuple: (average loss, training accuracy)
    """
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        # Create mask
        row_sums = labels.sum(axis=1)
        bools = (row_sums != 0).unsqueeze(1).expand(labels.shape).to(device)
        inputs, labels = inputs.to(device), labels.to(device)
        true_labels = torch.argmax(labels, dim=1)

        outputs = model(inputs)
        
        # Adjust output dimensions
        if outputs.shape[2] < bools.shape[2]:
            outputs = F.pad(outputs, (0, bools.shape[2] - outputs.shape[2]))
        elif outputs.shape[2] > bools.shape[2]:
            outputs = outputs[:, :, :bools.shape[2]]

        outputs_ = outputs * bools
        loss = criterion(outputs_, true_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pred = torch.argmax(outputs, dim=1)
        correct += (pred == true_labels).sum().item()
        total += true_labels.numel()

    avg_loss = running_loss / len(train_loader)
    train_accuracy = correct / total
    
    return avg_loss, train_accuracy


def evaluate_autoencoder(model, valid_loader, device):
    """
    Evaluate autoencoder model
    
    Args:
        model: Autoencoder model
        valid_loader: Validation data loader
        device: Device
    
    Returns:
        float: Validation accuracy
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            pred = torch.argmax(outputs, dim=1)
            true_labels = torch.argmax(labels, dim=1)
            
            # Adjust prediction dimensions
            if pred.shape[1] < true_labels.shape[1]:
                pred = F.pad(pred, (0, true_labels.shape[1] - pred.shape[1]))
            elif pred.shape[1] > true_labels.shape[1]:
                pred = pred[:, :true_labels.shape[1]]
            
            correct += (pred == true_labels).sum().item()
            total += true_labels.numel()

    valid_accuracy = correct / total
    return valid_accuracy


def train_phenotype_predictor(epoch, model, device, optimizer, criterion, train_loader, test_loader, scaler):
    """
    Train phenotype prediction model
    
    Args:
        epoch: Current epoch
        model: Prediction model
        device: Device
        optimizer: Optimizer
        criterion: Loss function
        train_loader: Training data loader
        test_loader: Test data loader
        scaler: Scaler
    
    Returns:
        tuple: (correlation coefficient, prediction output, loss values list)
    """
    model.train()
    
    # Set all parameters trainable
    for param in model.parameters():
        param.requires_grad = True
    
    all_loss = 0
    fold_loss_values = []
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.float(), target.float()
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1), target.view(-1))
        loss.backward()
        optimizer.step()
        all_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f'Training epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    avg_loss = all_loss / len(train_loader)
    fold_loss_values.append(avg_loss)
    print(f'=========> Epoch: {epoch} Average loss: {avg_loss:.4f}')

    # Evaluate model
    corr, predictions = evaluate_phenotype_predictor(model, device, test_loader, scaler)
    return corr, predictions, fold_loss_values


def evaluate_phenotype_predictor(model, device, test_loader, scaler):
    """
    Evaluate phenotype prediction model
    
    Args:
        model: Prediction model
        device: Device
        test_loader: Test data loader
        scaler: Scaler (can be None if no normalization was applied)
    
    Returns:
        tuple: (correlation coefficient, prediction output)
    """
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.float(), target.float()
            data, target = data.to(device), target
            
            output = model(data)
            
            # Handle inverse normalization based on whether scaler is available
            if scaler is not None:
                # Inverse normalize prediction results and true values
                output_original = scaler.inverse_transform(
                    output.view(-1).detach().cpu().numpy().reshape(-1, 1)
                ).flatten()
                target_original = scaler.inverse_transform(
                    target.reshape(-1).reshape(-1, 1)
                ).flatten()
            else:
                # Use original values without inverse transformation
                output_original = output.view(-1).detach().cpu().numpy()
                target_original = target.reshape(-1).cpu().numpy()
            
            # Calculate correlation coefficient
            corr = np.corrcoef(output_original, target_original)[0, 1]
            print(f"Correlation coefficient: {corr:.4f}")
            return corr, output_original



def set_random_seed(seed=42):
    """Set random seed to ensure reproducible results"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False


def load_pretrained_weights(model, pretrained_path, device):
    """
    Load pre-trained weights
    
    Args:
        model: Target model
        pretrained_path: Pre-trained model path
        device: Device
    
    Returns:
        bool: Whether loading was successful
    """
    if not os.path.exists(pretrained_path):
        return False
    
    try:
        pretrained_state = torch.load(pretrained_path, map_location=device)
        model_state = model.state_dict()
        
        # Filter out shape-mismatched parameters
        filtered_params = {k: v for k, v in pretrained_state.items() 
                          if k in model_state and v.size() == model_state[k].size()}
        
        # Update model parameters
        model_state.update(filtered_params)
        model.load_state_dict(model_state, strict=False)
        
        # Print unmatched parameters
        unmatched_params = [name for name in model_state.keys() if name not in filtered_params]
        if unmatched_params:
            print(f"⚠️  The following parameters did not match pre-trained weights: {unmatched_params}")
        
        return True
    except Exception as e:
        print(f"❌ Pre-trained model loading failed: {e}")
        return False


def pretrain_autoencoder(train_loader, valid_loader, device, epochs=100, config=None):
    """
    Pre-train autoencoder model
    
    Args:
        train_loader: Training data loader
        valid_loader: Validation data loader
        device: Device
        epochs: Number of training epochs
        config: Configuration object
    
    Returns:
        str: Best model save path
    """
    # Initialize model
    model = CMDAutoEncoder(
        in_channels=config.IN_CHANNELS if config else 3,
        out_channels=config.OUT_CHANNELS if config else 256,
        kernel_size=config.KERNEL_SIZE if config else 5,
        stride=config.STRIDE if config else 5,
        d_state=config.D_STATE if config else 128
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE if config else 1e-3)
    
    # Create model save directory
    model_dir = "./model"
    os.makedirs(model_dir, exist_ok=True)
    
    # Training records
    train_accuracies = []
    valid_accuracies = []
    losses = []
    best_valid_acc = 0
    model_file_path = None
    
    print("Starting autoencoder pre-training...")
    for epoch in range(epochs):
        avg_loss, acc_train = train_autoencoder(model, train_loader, device, criterion, optimizer)
        acc_valid = evaluate_autoencoder(model, valid_loader, device)

        # Convert tensor to numpy
        if isinstance(acc_train, torch.Tensor):
            acc_train = acc_train.cpu().numpy()
        if isinstance(acc_valid, torch.Tensor):
            acc_valid = acc_valid.cpu().numpy()

        # Save best model
        if acc_valid > best_valid_acc:
            best_valid_acc = acc_valid
            model_file_path = os.path.join(model_dir, "model_state.pth")
            torch.save(model.state_dict(), model_file_path)

        train_accuracies.append(acc_train)
        valid_accuracies.append(acc_valid)
        losses.append(avg_loss)
        
        print(f'Epoch {epoch+1}/{epochs} - Training accuracy: {acc_train:.5f}, Validation accuracy: {acc_valid:.5f}, Loss: {avg_loss:.8f}')

    print(f"Pre-training completed! Best model saved at: {model_file_path}")
    return model_file_path


