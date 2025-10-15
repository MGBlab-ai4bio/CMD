import numpy as np
from scipy.spatial import distance
import torch

def get_confidence(train_loader: torch.utils.data.DataLoader, 
                         valid_loader: torch.utils.data.DataLoader,
                         model: torch.nn.Module,
                         device: torch.device, epsilon=1e-6,
                            k=10):
    model.eval()
    with torch.no_grad():
        encoder_outputs_train = []
        for inputs, _  in train_loader:
            inputs = inputs.to(device)
            encoder_outputs_train.append(model.get_encoder(inputs).cpu())
        encoder_output_label_train = torch.cat(encoder_outputs_train, dim=0).flatten(-2, -1).numpy()
        encoder_outputs_valid = []
        for inputs, _  in valid_loader:
            inputs = inputs.to(device)
            encoder_outputs_valid.append(model.get_encoder(inputs).cpu())
        encoder_output_label_valid = torch.cat(encoder_outputs_valid, dim=0).flatten(-2, -1).numpy()
        print(encoder_output_label_valid.shape)
        distance_matrix = distance.cdist(encoder_output_label_valid, encoder_output_label_train, metric='euclidean')
        top_10_nearest_indices = np.argsort(distance_matrix, axis=1)[:, :k]
        top_10_nearest_distances = np.sort(distance_matrix, axis=1)[:, :k]
        d_min, d_max = np.min(top_10_nearest_distances.mean(axis=1)), np.max(top_10_nearest_distances.mean(axis=1))
        confidence_scores = 1 - (top_10_nearest_distances.mean(axis=1) - d_min) / (d_max - d_min + epsilon) 
    return confidence_scores, top_10_nearest_distances.mean(axis=1)