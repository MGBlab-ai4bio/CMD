# =============================================================================
# Evaluation and visualization utilities
# =============================================================================
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.fft import rfft, rfftfreq, irfft
from tqdm import tqdm
from torch.utils.data import DataLoader
from pygam import LinearGAM, s



def calculate_encoder_feature_correlations(train_loader, model, device):
    """
    Calculate correlations between encoder features
    
    Args:
        train_loader: Training data loader
        model: Trained model
        device: Device
    
    Returns:
        np.ndarray: Correlation matrix
    """
    model.eval()
    features = []
    
    with torch.no_grad():
        for inputs, _, _ in train_loader:
            inputs = inputs.to(device)
            encoder_features = model.get_encoder_features(inputs)
            features.append(encoder_features.cpu().numpy())
    
    features = np.concatenate(features, axis=0)
    features_flat = features.reshape(features.shape[0], -1)
    
    # Calculate correlation matrix
    correlation_matrix = np.corrcoef(features_flat.T)
    
    return correlation_matrix


def get_confidence(train_loader, valid_loader, model, device, epsilon=1e-6, k=10):
    """
    Calculate confidence scores based on encoder features
    
    Args:
        train_loader: Training data loader
        valid_loader: Validation data loader
        model: Trained model
        device: Device
        epsilon: Small constant to prevent division by zero
        k: Number of nearest neighbors
    
    Returns:
        tuple: (confidence_scores, average_distances)
    """
    model.eval()
    with torch.no_grad():
        # Process training set
        encoder_outputs_train = []
        for labels, _ in train_loader:
            labels = labels.to(device)
            encoder_outputs_train.append(model.get_encoder_features(labels).cpu())
        encoder_output_label_train = torch.cat(encoder_outputs_train, dim=0).flatten(-2, -1).numpy()
        
        # Process validation set
        encoder_outputs_valid = []
        for labels, _  in valid_loader:
            labels = labels.to(device)
            encoder_outputs_valid.append(model.get_encoder_features(labels).cpu())
        
        encoder_output_label_valid = torch.cat(encoder_outputs_valid, dim=0).flatten(-2, -1).numpy()
        print(f"Validation encoder output shape: {encoder_output_label_valid.shape}")
        
        # Calculate distance matrix between validation and training sets
        distance_matrix = distance.cdist(encoder_output_label_valid, encoder_output_label_train, metric='euclidean')
        
        # Get k nearest neighbors for each validation sample
        top_k_nearest_indices = np.argsort(distance_matrix, axis=1)[:, :k]
        top_k_nearest_distances = np.sort(distance_matrix, axis=1)[:, :k]
        
        # Calculate confidence scores
        d_min, d_max = np.min(top_k_nearest_distances.mean(axis=1)), np.max(top_k_nearest_distances.mean(axis=1))
        confidence_scores = 1 - (top_k_nearest_distances.mean(axis=1) - d_min) / (d_max - d_min + epsilon)
        
    return confidence_scores, top_k_nearest_distances.mean(axis=1)


def integrated_gradients_batch(inputs, model, baseline=None, steps=20, device='cuda'):
    """Batch calculate integrated gradients"""
    inputs = inputs.to(device)

    if baseline is None:
        baseline = torch.zeros_like(inputs, device=device)
    else:
        baseline = baseline.to(device)

    assert baseline.shape == inputs.shape, "baseline and inputs must have same shape"

    diff = inputs - baseline
    step_sizes = diff / steps

    integrated_grads = torch.zeros_like(inputs, device=device)

    for step in range(steps):
        x = baseline + step_sizes * step
        x.requires_grad_(True)
        output = model(x)
        grads = torch.autograd.grad(outputs=output.sum(), inputs=x)[0]
        integrated_grads += grads

    integrated_grads *= diff
    return integrated_grads


def calculate_integrated_gradients(model, dataset, batch_size=16, steps=10, device='cuda'):
    """Calculate integrated gradients for dataset"""
    model = model.to(device)
    model.eval()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    first_input = dataset[0][0].unsqueeze(0).to(device)
    baseline = torch.zeros_like(first_input, device=device)

    all_grads = []

    try:
        for batch in tqdm(dataloader, desc="Calculating integrated gradients"):
            inputs = batch[0].to(device)
            B = inputs.shape[0]
            baseline_batch = baseline.expand(B, *baseline.shape[1:])

            grads = integrated_gradients_batch(inputs, model, baseline=baseline_batch, 
                                              steps=steps, device=device)
            all_grads.append(grads.detach().cpu())

    except Exception as e:
        print(f"Integrated gradients error: {str(e)}")
        return None

    return torch.cat(all_grads, dim=0)


def calculate_mean_importance(gradients):
    """Calculate mean-based importance"""
    importance = gradients.abs().max(dim=1)[0].mean(dim=0)
    return importance.detach().numpy() if torch.is_tensor(importance) else importance


def calculate_std_importance(gradients):
    """Calculate std-based importance"""
    importance = gradients.abs().max(dim=1)[0].std(dim=0)
    return importance.detach().numpy() if torch.is_tensor(importance) else importance


def calculate_combined_importance(mean_scores, std_scores):
    """Calculate combined importance"""
    return mean_scores * std_scores


def calculate_fourier_filtered_importance(gradients, threshold=0.7):
    """Calculate Fourier-filtered importance"""
    mean_grad = gradients.abs().max(dim=1)[0].mean(dim=0)
    mean_grad_np = mean_grad.detach().numpy() if torch.is_tensor(mean_grad) else mean_grad
    
    yf = rfft(mean_grad_np)
    xf = rfftfreq(20000, 1)
    
    yf_clean = (np.abs(yf) > threshold) * yf
    filtered_importance = irfft(yf_clean)
    
    filter_condition = (np.abs(yf) > threshold)
    print(f"Original frequencies: {len(yf)}")
    print(f"Retained frequencies: {np.sum(filter_condition)}")
    print(f"Filter ratio: {1 - np.sum(filter_condition) / len(yf):.2%}")
    
    return filtered_importance, yf


def apply_kalman_filter(signal, dt=50.0, process_noise=1e-2, measurement_noise=150):
    """Apply Kalman filter for smoothing"""
    u = np.zeros((2, 1))
    x = np.zeros((2, 1))
    P = np.eye(2)
    
    F = np.array([[1, dt], [0, 1]])
    H = np.array([[1, 0]])
    Q = np.array([[process_noise, 0], [0, process_noise]])
    R = np.array([[measurement_noise]])
    
    filtered_signal = []
    for z in signal:
        x = F @ x + u
        P = F @ P @ F.T + Q
        
        y = z - (H @ x)
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (np.eye(2) - K @ H) @ P
        
        filtered_signal.append(x[0, 0])
    
    return np.array(filtered_signal)


def create_chromosome_colors(n_chromosomes=10, n_snps_per_chr=2000):
    """Create chromosome color list"""
    colors = []
    for i in range(n_chromosomes):
        color = '#406994' if i % 2 == 0 else '#95b3ca'
        colors.extend([color] * n_snps_per_chr)
    return colors



def plot_feature_importance(importance_scores, title='Feature Importance', 
                            save_path=None, show_plot=True):
    """Plot feature importance with alternating chromosome colors"""
    plt.figure(figsize=(8, 5))

    colors = []
    for i in range(10):  # 10个染色体
        color = '#406994' if i % 2 == 0 else '#95b3ca'  # 深浅蓝交替
        colors.extend([color] * 2000)  # 每个染色体对应2000个点

    scatter = plt.scatter(range(20000), importance_scores, s=0.5, alpha=0.8, c=colors)

    plt.xticks(range(1000, 20000, 2000), [f"chr{i}" for i in range(1, 11)])
    plt.title(title)
    plt.xlabel('Chromosome')
    plt.ylabel('Importance Score')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()

def calculate_roc_curves(true_effects, importance_scores_dict):
    """
    Calculate ROC curves for multiple methods
    
    Args:
        true_effects: True effect vector
        importance_scores_dict: Importance scores dict {Method name: Score array}
    
    Returns:
        dict: {Method name: (fpr, tpr, auc)}
    """
    true_labels = (true_effects > 0).astype(int)
    
    roc_results = {}
    for method_name, scores in importance_scores_dict.items():
        fpr, tpr, _ = roc_curve(true_labels, scores)
        auc = roc_auc_score(true_labels, scores)
        roc_results[method_name] = {'fpr': fpr, 'tpr': tpr, 'auc': auc}
    
    return roc_results


def plot_roc_comparison(roc_results, save_path=None):
    """Plot ROC curve comparison"""
    plt.figure(figsize=(8, 6))
    
    plot_config = {
        'combined': ('b-', 'Mean×Std'),
        'kalman': ('black', 'Kalman'),
        'std': ('g-', 'Std'),
        'fourier': ('r-', 'Fourier'),
        'mean': ('orange', 'Mean')
    }
    
    for method, (color, label_cn) in plot_config.items():
        if method in roc_results:
            result = roc_results[method]
            plt.plot(result['fpr'], result['tpr'], color, linewidth=2, 
                    label=f'{label_cn} (AUC = {result["auc"]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc='lower right')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()


def process_confidence_data(scores_dict, distance_dict):
    """
    Process confidence scores and distances for visualization
    
    Args:
        scores_dict: Dictionary containing scores for each missing ratio
        distance_dict: Dictionary containing distances for each missing ratio
    
    Returns:
        tuple: Processed scores and distances
    """
    # Sort by average scores (descending order)
    sort_index = np.argsort(scores_dict[0].mean(axis=1))[::-1]
    
    processed_scores = {}
    processed_distances = {}
    avg_scores = {}
    avg_distances = {}
    
    for ratio in [0, 25, 50, 75, 90]:
        # Sort data by confidence scores
        processed_scores[ratio] = scores_dict[ratio][sort_index, :]
        processed_distances[ratio] = distance_dict[ratio][sort_index, :]
        
        # Calculate averages
        avg_scores[ratio] = processed_scores[ratio].mean(axis=1)
        avg_distances[ratio] = processed_distances[ratio].mean(axis=1)
    
    return processed_scores, processed_distances, avg_scores, avg_distances


def fit_gam_models(avg_scores_dict):
    """
    Fit GAM models for confidence score relationships
    
    Args:
        avg_scores_dict: Dictionary containing average scores for each missing ratio
    
    Returns:
        tuple: (GAM models, x_plot for visualization)
    """
    # Sort by 0% missing scores (descending order)
    sort_index = np.argsort(avg_scores_dict[0])[::-1]
    x = np.array(avg_scores_dict[0])[sort_index].reshape(-1, 1)
    
    # Prepare data for other missing ratios
    y_data = {}
    for ratio in [25, 50, 75, 90]:
        y_data[ratio] = np.array(avg_scores_dict[ratio])[sort_index]
    
    # Fit GAM models
    gams = {}
    for ratio in [25, 50, 75, 90]:
        gams[ratio] = LinearGAM(s(0)).fit(x, y_data[ratio])
    
    # Create dense points for smooth plotting
    x_plot = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
    
    return gams, x_plot, x, y_data


def plot_gam_results(gams, x_plot, x, y_data):
    """
    Plot GAM fitting results with confidence intervals
    
    Args:
        gams: Dictionary of fitted GAM models
        x_plot: Dense x points for plotting
        x: Original x data
        y_data: Original y data for each missing ratio
    """
    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    labels = ['25%', '50%', '75%', '90%']
    ratios = [25, 50, 75, 90]
    
    for gam, color, label, ratio in zip([gams[r] for r in ratios], colors, labels, ratios):
        y_pred = gam.predict(x_plot)
        ci = gam.confidence_intervals(x_plot)
        
        # Plot GAM fit
        plt.plot(x_plot, y_pred, color=color, label=f'{label} missing', linewidth=2)
        plt.fill_between(x_plot.flatten(), ci[:, 0], ci[:, 1], color=color, alpha=0.2)
    
    plt.xlabel("Confidence Score (0% missing)", fontsize=12)
    plt.ylabel("Confidence Score (other missing ratios)", fontsize=12)
    plt.title("GAM Fit: Confidence Score Relationships", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("score_shrimp_GAM.pdf", dpi=300, bbox_inches='tight')
    plt.show()


def create_correlation_heatmap(avg_scores_dict):
    """
    Create correlation heatmap for confidence scores across different missing ratios
    
    Args:
        avg_scores_dict: Dictionary containing average scores for each missing ratio
    """
    # Construct DataFrame with confidence scores under different missing rates
    avg_df = pd.DataFrame({
        '0%': avg_scores_dict[0],
        '25%': avg_scores_dict[25],
        '50%': avg_scores_dict[50],
        '75%': avg_scores_dict[75],
        '90%': avg_scores_dict[90],
    })
    
    # Compute correlation matrix
    corr_matrix = avg_df.corr()
    
    # Create heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        annot_kws={"size": 14},
        cmap='coolwarm',
        fmt=".2f",
        square=True,
        vmin=0, vmax=1,
        linewidths=0.5,
        cbar_kws={"label": "Correlation", "ticks": [0, 0.2, 0.4, 0.6, 0.8, 1], "shrink": 0.9}
    )
    
    # Customize colorbar
    cbar = plt.gcf().axes[-1]
    cbar.tick_params(labelsize=14)
    
    plt.title("Correlation of Confidence Scores Across Different Missing Rates", fontsize=15)
    plt.xlabel("Missing Rate", fontsize=15)
    plt.ylabel("Missing Rate", fontsize=15)
    plt.xticks(rotation=0, fontsize=15)
    plt.yticks(rotation=0, fontsize=15)
    plt.gca().invert_yaxis()  # Invert y-axis for better visualization
    
    plt.tight_layout()
    plt.savefig("confidence_correlation_heatmap.pdf", dpi=300, bbox_inches='tight')
    plt.show()
    
    return corr_matrix


def calculate_runtime_summary(start_time):
    """Calculate runtime statistics"""
    import time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total runtime: {elapsed_time:.2f} seconds")
    return elapsed_time


def print_final_results(missing_ratio, correlations):
    """Print final results"""
    print(f"Missing ratio: {missing_ratio}")
    mean_correlation = np.mean(correlations)
    std_correlation = np.std(correlations)
    
    print(f"Mean correlation: {mean_correlation:.4f}")
    print(f"Standard deviation: {std_correlation:.4f}")
    print(f"Best correlation: {np.max(correlations):.4f}")
    print(f"Worst correlation: {np.min(correlations):.4f}")
    
    return mean_correlation
