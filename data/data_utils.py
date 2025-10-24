# =============================================================================
# Data loading and preprocessing utilities
# =============================================================================
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_genotype_data(file_path, max_rows=None):
    """
    Load genotype data from file
    
    Args:
        file_path (str): Path to genotype data file
        max_rows (int): Maximum number of rows to load
    
    Returns:
        pd.DataFrame: Loaded genotype data
    """
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, index_col=0)
        elif file_path.endswith('.raw'):
            df = pd.read_csv(file_path, sep=' ', index_col=0)
        else:
            # Try to read as CSV by default
            df = pd.read_csv(file_path, index_col=0)
        
        if max_rows is not None:
            df = df.iloc[:max_rows]
        
        return df
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None


def apply_missing_mask(data, missing_ratio):
    """
    Apply missing mask to genotype data
    
    Args:
        data (pd.DataFrame): Original genotype data
        missing_ratio (float): Missing ratio (0.0 to 1.0)
    
    Returns:
        np.ndarray: Data after applying mask
    """
    data_array = data.to_numpy().copy()
    if missing_ratio > 0:
        mask = np.random.uniform(0, 1, size=data_array.shape)
        data_array[mask < missing_ratio] = -1
    return data_array


def encode_genotype_to_categorical(genotype_data):
    """
    Encode genotype data to categorical format
    
    Args:
        genotype_data (np.ndarray): Genotype data array
    
    Returns:
        np.ndarray: Encoded categorical data
    """
    codebook = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
    return codebook[genotype_data.astype(int)]


def load_phenotype_data(file_path):
    """
    Load phenotype data
    
    Args:
        file_path (str): Phenotype data file path
    
    Returns:
        pd.DataFrame: Phenotype data frame
    """
    return pd.read_csv(file_path, sep=',', index_col=0)



def preprocess_phenotype_data(phenotype_df, genotype_df, phenotype_column=1, normalize_phenotype=True):
    """
    Preprocess phenotype data and align with genotype data
    
    Args:
        phenotype_df: Phenotype data frame
        genotype_df: Genotype data frame
        phenotype_column: Phenotype column index
        normalize_phenotype: Whether to normalize phenotype data (default: True)
    
    Returns:
        tuple: (genotype data, phenotype data, scaler)
    """
    # Find common samples
    common_samples = set(phenotype_df.index) & set(genotype_df.index)
    common_samples = list(common_samples)
    
    # Extract corresponding data
    phenotype_values = phenotype_df.loc[common_samples].iloc[:, phenotype_column].values
    genotype_values = genotype_df.loc[common_samples].values
    
    # Encode genotype data to categorical format
    genotype_encoded = encode_genotype_to_categorical(genotype_values)
    
    if normalize_phenotype:
        # Normalize phenotype data
        scaler = StandardScaler()
        phenotype_processed = scaler.fit_transform(phenotype_values.reshape(-1, 1)).flatten()
        print(f"Phenotype column name: {phenotype_df.columns[phenotype_column]}")
        print(f"After normalization - Mean: {np.mean(phenotype_processed):.4f}, Std: {np.std(phenotype_processed):.4f}")
    else:
        # Use original phenotype data without normalization
        phenotype_processed = phenotype_values
        scaler = None
        print(f"Phenotype column name: {phenotype_df.columns[phenotype_column]}")
        print(f"Original phenotype - Mean: {np.mean(phenotype_processed):.4f}, Std: {np.std(phenotype_processed):.4f}")
    
    return genotype_encoded, phenotype_processed, scaler



def simulate_phenotype(genotype_data, num_local=3, 
                      local_ranges=[(20, 300), (10000, 10300), (17000, 17300)],
                      num_effects=[50, 60, 50],
                      effect_sizes=[0.3, 0.3, 0.3],
                      heritability=1.8):
    """
    Simulate phenotype data with multiple local effects
    
    Args:
        genotype_data: Genotype data
        num_local: Number of local effect regions
        local_ranges: List of (start, end) tuples for effect regions
        num_effects: List of number of effects in each region
        effect_sizes: List of effect sizes for each region
        heritability: Heritability parameter
    
    Returns:
        np.ndarray: Simulated phenotype values
    """
    n_samples = genotype_data.shape[0]
    phenotype = np.zeros(n_samples)
    
    for i in range(num_local):
        start, end = local_ranges[i]
        n_effects = num_effects[i]
        effect_size = effect_sizes[i]
        
        # Randomly select SNP positions in the region
        snp_positions = np.random.choice(range(start, end), size=n_effects, replace=False)
        
        # Calculate effects
        for pos in snp_positions:
            if pos < genotype_data.shape[1]:
                snp_data = genotype_data[:, pos]
                # Convert to additive coding (0, 1, 2)
                snp_additive = np.sum(snp_data, axis=1) if len(snp_data.shape) > 1 else snp_data
                phenotype += effect_size * snp_additive
    
    # Add noise
    noise = np.random.normal(0, np.std(phenotype) * (1 - heritability), n_samples)
    phenotype += noise
    
    return phenotype


class GenotypeDataset(Dataset):
    """
    Genotype dataset class for pre-training
    """
    def __init__(self, masked_data, missing_perc=0):
        """
        Initialize dataset
        
        Args:
            masked_data (np.ndarray): Masked data
            original_data (np.ndarray): Original data
            missing_perc (float): Additional missing percentage to apply
        """
        self.masked_data = masked_data
        # self.original_data = original_data
        self.missing_perc = missing_perc

    def __len__(self):
        return len(self.masked_data)

    def __getitem__(self, idx):
        x = self.masked_data[idx].copy()
        
        # Apply additional missing data if specified
        if isinstance(self.missing_perc, (int, float)) and self.missing_perc > 0:
            missing_size = int(self.missing_perc * len(x))
            missing_index = np.random.choice(len(x), missing_size, replace=False)
            x[missing_index] = [0, 0, 0]
        
        y = self.masked_data[idx]
        # z = self.original_data[idx]

        # Adjust data dimensions
        if len(x.shape) == 2:
            x = x.transpose(1, 0)
            y = y.transpose(1, 0)
            # z = z.transpose(1, 0)
        elif len(x.shape) == 3:
            x = x.transpose(0, 2, 1)
            y = y.transpose(0, 2, 1)
            z = z.transpose(0, 2, 1)
            
        return (torch.tensor(x, dtype=torch.float32), 
                torch.tensor(y, dtype=torch.float32)
                # torch.tensor(z, dtype=torch.float32)
                )


class PhenotypeDataset(Dataset):
    """
    Phenotype dataset class for phenotype prediction
    """
    def __init__(self, genotype_data, phenotype_data):
        """
        Initialize phenotype dataset
        
        Args:
            genotype_data (np.ndarray): Genotype data
            phenotype_data (np.ndarray): Phenotype data
        """
        self.genotype_data = np.asarray(genotype_data, dtype=np.float32)
        self.phenotype_data = np.asarray(phenotype_data, dtype=np.float32)

    def __len__(self):
        return len(self.genotype_data)

    def __getitem__(self, idx):
        x = self.genotype_data[idx]
        y = self.phenotype_data[idx]

        # Adjust data dimensions
        if len(x.shape) == 2:
            x = x.transpose(1, 0)
        elif len(x.shape) == 3:
            x = x.transpose(0, 2, 1)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def prepare_pretraining_data(genotype_data, original_data, test_size=0.1, random_seed=42):
    """
    Prepare pre-training data
    
    Args:
        genotype_data: Genotype data
        original_data: Original data
        test_size: Test set ratio
        random_seed: Random seed
    
    Returns:
        tuple: (training data loader, validation data loader)
    """
    train_X, valid_X = train_test_split(genotype_data, test_size=test_size, random_state=random_seed)
    train_X_original, valid_X_original = train_test_split(original_data, test_size=test_size, random_state=random_seed)
    
    train_dataset = GenotypeDataset(train_X, train_X_original)
    valid_dataset = GenotypeDataset(valid_X, valid_X_original)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=False)
    
    return train_loader, valid_loader



#######################################################################################################################################################

def mask_data(data, ratio):
    data = data.to_numpy().copy()
    bools = np.random.uniform(0, 1, size=data.shape)
    data[bools < ratio] = -1
    return data


def to_categorical(y):
    codebook = np.array([[1,0,0], [0,1,0], [0,0,1], [0,0,0]])
    y = np.asarray(y, dtype=int)
    return codebook[y]


class GenotypeDataset_pretrain(Dataset):
    def __init__(self, data, missing_perc=0.1):
        self.data = data
        self.missing_perc = missing_perc

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        x = self.data[idx].copy()
        missing_size = int(self.missing_perc * len(x))
        missing_index = np.random.randint(len(x), size=missing_size)
        x[missing_index, :] = [0, 0, 0]
        y = self.data[idx]

        if len(x.shape)==2:
            x=x.transpose(1,0)
            y=y.transpose(1,0)

        if len(x.shape)==3:
            x=x.transpose(0,2,1)
            y=y.transpose(0,2,1)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class GenotypeDataset_bv(Dataset):
    def __init__(self, data, y):
        self.data = np.asarray(data, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.y[idx]
        if len(x.shape) == 2:
            x = x.transpose(1, 0)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def process_data(genotype: np.array,
                 phenotype:np.array, 
                 test_split_ratio_per=0.1 ,
                 test_split_ratio_bv=0.1,
                 use_seed=False,
                 random_seed=42,
                 batch_size_pre=16,
                 batch_size_bv=16,
                 missing_perc=0.5,
                 to_split_in_pre=False
                 ):
    if use_seed:
        if isinstance(random_seed,int):
            pass
        else:
            raise ValueError("random_seed must be int")
    genotype_one_hot = to_categorical(genotype)
    
    if to_split_in_pre:
        if use_seed:
            train_dataset, valid_dataset = train_test_split(genotype_one_hot, test_size=test_split_ratio_per, random_state=random_seed)
        else:
            train_dataset, valid_dataset = train_test_split(genotype_one_hot, test_size=test_split_ratio_per)
        train_dataset = GenotypeDataset_pretrain(train_dataset,missing_perc=missing_perc)
        valid_dataset = GenotypeDataset_pretrain(valid_dataset,missing_perc=missing_perc)
        train_loader_pre = DataLoader(train_dataset ,batch_size=batch_size_pre, shuffle=True)
        valid_loader_pre = DataLoader(train_dataset ,batch_size=len(train_dataset), shuffle=True)
    else:
        train_dataset = GenotypeDataset_pretrain(genotype_one_hot,missing_perc=missing_perc)
        train_loader_pre = DataLoader(train_dataset ,batch_size=batch_size_pre, shuffle=True)
        
    # train_dataset = GenotypeDataset(genotype_one_hot, missing_perc=missing_perc)
    # train_loader_pre = DataLoader(train_dataset ,batch_size=batch_size_pre, shuffle=True)
    # valid_loader_pre = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
    if use_seed:
        X_train,X_test,y_train,y_test = train_test_split(genotype_one_hot, phenotype, test_size=test_split_ratio_bv, random_state=random_seed)
    else:
        X_train,X_test,y_train,y_test = train_test_split(genotype_one_hot, phenotype, test_size=test_split_ratio_bv)
    train_dataset= GenotypeDataset_bv(X_train,y_train)
    test_dataset= GenotypeDataset_bv(X_test,y_test)
    train_loader_bv = DataLoader(train_dataset, batch_size=batch_size_bv, shuffle=True)
    test_loader_bv = DataLoader(test_dataset,batch_size=len(test_dataset),shuffle=True)
    if to_split_in_pre:
        return train_loader_pre, valid_loader_pre, train_loader_bv, test_loader_bv
    else:
        return train_loader_pre, None, train_loader_bv, test_loader_bv
