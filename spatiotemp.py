import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import models
from torch.nn import SyncBatchNorm
from torchvision.models import ResNet50_Weights
from torchvision import transforms

class SpearmanCorrelationLoss(nn.Module):
    def __init__(self):
        super(SpearmanCorrelationLoss, self).__init__()

    def forward(self, predictions, targets):
        pred_ranks = torch.argsort(torch.argsort(predictions))
        target_ranks = torch.argsort(torch.argsort(targets))

        pred_ranks = pred_ranks.float()
        pred_ranks_centered = pred_ranks - pred_ranks.mean()
        target_ranks = target_ranks.float()
        target_ranks_centered = target_ranks - target_ranks.mean()

        numerator = torch.sum(pred_ranks_centered * target_ranks_centered)
        denominator = torch.sqrt(torch.sum(pred_ranks_centered**2) * torch.sum(target_ranks_centered**2))

        spearman_corr = numerator / (denominator + 1e-8)
        return 1 - spearman_corr

class SpatialQualityAnalyzer(nn.Module):
    def __init__(self, pretrained=True):
        super(SpatialQualityAnalyzer, self).__init__()
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        children = list(resnet.children())
        self.features = nn.Sequential(
            *[SyncBatchNorm.convert_sync_batchnorm(child) for child in children[:-2]]
        )
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x):
        batch_size, channels, frames, height, width = x.size()
        x = x.reshape(batch_size * frames, channels, height, width)
        if torch.all(x == 0):
            raise ValueError("All-zero frames detected before normalization.")
        #x = self.normalize(x)
        features = self.features(x)
        features = features.view(batch_size, frames, features.size(1), features.size(2), features.size(3))
        return features

class TemporalQualityAnalyzer(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_heads=4, num_layers=2, dropout=0.3):
        super(TemporalQualityAnalyzer, self).__init__()
        self.input_dim = input_dim
        self.positional_encoding = self._get_positional_encoding(length=8, d_model=input_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def _get_positional_encoding(self, length, d_model):
        position = torch.arange(0, length).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe = torch.zeros(length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # Shape: [1, length, d_model]

    def forward(self, x):
        batch_size, frames, _ = x.size()  # [batch_size, num_frames, input_dim]
        device = x.device

        positional_encoding = self.positional_encoding[:, :frames, :].to(device)

        x = x + positional_encoding
r
        x = self.transformer_encoder(x)

        x = x.mean(dim=1)
        return x

class QualityRegressor(nn.Module):
    def __init__(self, input_dim=1024, hidden_dims=[256, 128], output_dim=1):
        super(QualityRegressor, self).__init__()
        layers = []
        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dims[i - 1], hidden_dims[i]))
            layers.append(nn.ReLU(inplace=False))
            layers.append(nn.Dropout(0.3))
        layers.append(nn.Linear(hidden_dims[-1], output_dim))  # Single scalar output
        self.regressor = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.regressor:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        """
        Input: x of shape [batch_size, feature_dim]
        Output: single scalar per video [batch_size, 1]
        """
        return self.regressor(x)


class VideoQualityModelSimpleFusion(nn.Module):
    def __init__(self, device, spatial_feature_dim=2048, temporal_feature_dim=2048, combined_dim=2048):
        super(VideoQualityModelSimpleFusion, self).__init__()
        self.device = device

        # Feature analyzers
        self.spatial_analyzer = SpatialQualityAnalyzer().to(device)
        self.temporal_analyzer = TemporalQualityAnalyzer(input_dim=spatial_feature_dim).to(device)

        # Feature fusion and regression
        self.combined_projector = nn.Linear(spatial_feature_dim + temporal_feature_dim, combined_dim).to(device)
        self.regressor = QualityRegressor(input_dim=combined_dim).to(device)

    def forward(self, x):
        # Extract spatial features: [B, T, C, H, W]
        spatial_features = self.spatial_analyzer(x)
        batch_size, num_frames, spatial_feature_dim, height, width = spatial_features.size()

        # Pool spatial dimensions
        spatial_features_pooled = F.adaptive_avg_pool2d(
            spatial_features.view(-1, spatial_feature_dim, height, width), (1, 1))
        spatial_features_pooled = spatial_features_pooled.view(batch_size, num_frames, spatial_feature_dim)

        # Extract temporal features
        temporal_features = self.temporal_analyzer(spatial_features_pooled)

        # Average spatial features across frames
        spatial_features_avg = spatial_features_pooled.mean(dim=1)

        # Concatenate spatial and temporal features
        combined_features = torch.cat([spatial_features_avg, temporal_features], dim=-1)

        # Predict final quality score
        combined_features_proj = self.combined_projector(combined_features)
        quality_score = self.regressor(combined_features_proj)
        return quality_score.squeeze(-1)

