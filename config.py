# =============================================================================
# Configuration parameters
# =============================================================================

class Config:
    """Experimental configuration class for Mamba2 genotype-phenotype analysis"""
    # Pre-training model configuration
    USE_PRETRAINED = False  # True: use pre-trained model, False: don't use pre-trained model
    
    # Data configuration
    INPUT_FILE = 'test.csv'
    MAX_ROWS = 1000
    MISSING_RATIO = 0.0
    MISSING_RATIOS = [0.0, 0.25, 0.5, 0.75, 0.9]
    INPUT_LENGTH = 20000
    # Model configuration
    IN_CHANNELS = 3
    OUT_CHANNELS = 256
    KERNEL_SIZE = 5
    STRIDE = 5
    D_STATE = 128
    D_CONV = 4
    EXPAND = 2
    DROPOUT = 0.0
    
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0004
    EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 20
    
    # Cross-validation configuration
    N_SPLITS = 10
    RANDOM_SEED = 42
    
    # Phenotype simulation configuration
    N_REGIONS = 5
    EFFECT_SIZE = 0.1
    NOISE_LEVEL = 0.1
    
    # Feature importance configuration
    INTEGRATED_GRADIENTS_STEPS = 50
    FILTER_METHODS = ['top_k', 'threshold', 'percentile']
    TOP_K_VALUES = [10, 50, 100, 500]
    THRESHOLD_VALUES = [0.01, 0.05, 0.1]
    PERCENTILE_VALUES = [90, 95, 99]
    
    # Confidence analysis configuration
    CONFIDENCE_K = 30
    CONFIDENCE_EPSILON = 1e-6
    REPEAT_TIMES = 1
    
    # Visualization configuration
    FIGURE_SIZE = (10, 6)
    DPI = 300


class ConfidenceConfig(Config):
    """Configuration for confidence analysis"""
    # Confidence analysis specific parameters
    CONFIDENCE_K = 30
    CONFIDENCE_EPSILON = 1e-6
    REPEAT_TIMES = 1
    MISSING_RATIOS = [0.0, 0.25, 0.5, 0.75, 0.9]
    
    # Model configuration for confidence analysis
    KERNEL_SIZE = 20
    STRIDE = 20
    EPOCHS = 15



class ExplainConfig(Config):
    TEST_SIZE = 0.15  
    
    LEARNING_RATE_PRETRAIN = 1e-3  
    LEARNING_RATE_CV = 0.0004
    

    BATCH_SIZE_PRETRAIN = 32  
    BATCH_SIZE_CV = 16        
    IG_BATCH_SIZE = 16        
    

    IG_STEPS = 50

    KERNEL_SIZE = 20  
    STRIDE = 20       
    D_STATE = 128    
    

    EARLY_STOPPING_PATIENCE = 20
    
    N_SPLITS = 10
    RANDOM_SEED = 42
    
    INTEGRATED_GRADIENTS_STEPS = 50
    FILTER_METHODS = ['top_k', 'threshold', 'percentile']
    TOP_K_VALUES = [10, 50, 100, 500]
    THRESHOLD_VALUES = [0.01, 0.05, 0.1]
    PERCENTILE_VALUES = [90, 95, 99]
    
    N_REGIONS = 5
    EFFECT_SIZE = 0.1
    NOISE_LEVEL = 0.1