import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import numpy as np
from torchvision import transforms
from dataloader import LiveVQADataset, KonvidVQADataset
from spatiotemp import SpearmanCorrelationLoss, VideoQualityModelSimpleFusion
import random
import matplotlib.pyplot as plt

# Configuration
class Config:
    FRAME_DIR = './KoNViD_1k_extracted_frames'
    MOS_CSV = './KoNViD_1k/KoNViD_1k_mos.csv'
    SALIENCY_DIR = './KoNViD_1k_saliency_maps'
    CHECKPOINT = './checkpoint_epoch_900_plcc_0.8314.pth'  #example checkpoint
    BATCH_SIZE = 5
    LR = 0.000008
    WEIGHT_DECAY = 1e-5
    NUM_EPOCHS = 30
    DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    SEED = 42
    SPEARMAN_WEIGHT = 0.1  # Weight for Spearman loss

# Set random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)

# Fine-tuning and evaluation
def fine_tune_and_evaluate():
    set_seed(Config.SEED)

    # Data transformations
    data_transforms = transforms.Compose([
        transforms.Resize((224, 398)),
        transforms.ToTensor(),
    ])

    # Dataset and split
    dataset = KonvidVQADataset(
        frame_dir=Config.FRAME_DIR,
        mos_csv=Config.MOS_CSV,
        saliency_dir=Config.SALIENCY_DIR,
        transform=data_transforms,
        num_frames=8,
        alpha=0.5
    )
    total_indices = list(range(len(dataset)))
    random.shuffle(total_indices)
    train_indices = total_indices[:int(0.8 * len(total_indices))]
    test_indices = total_indices[int(0.8 * len(total_indices)):]

    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=1)
    test_loader = DataLoader(Subset(dataset, test_indices), batch_size=1, shuffle=False, num_workers=1)

    # Model setup
    model = VideoQualityModelSimpleFusion(
        spatial_feature_dim=2048,  
        temporal_feature_dim=2048,
        device=Config.DEVICE
    ).to(Config.DEVICE)

    spearman_criterion = SpearmanCorrelationLoss()
    l1_criterion = nn.L1Loss()

    optimizer = optim.Adam(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

    # Load checkpoint if available
    best_srocc = -1
    if os.path.exists(Config.CHECKPOINT):
        checkpoint = torch.load(Config.CHECKPOINT, map_location=Config.DEVICE)

        # Fix for DDP-wrapped state_dicts
        state_dict = checkpoint.get('model_state_dict', {})
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)

        optimizer_state = checkpoint.get('optimizer_state_dict', None)
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)

        best_srocc = checkpoint.get('best_srocc', -1)
        print("Checkpoint loaded.")
    else:
        print("No checkpoint found. Training from scratch.")

    for epoch in range(Config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.NUM_EPOCHS}")

        # Training
        model.train()
        running_loss = 0.0

        for frames, mos_labels, _ in tqdm(train_loader, desc="Training", leave=False):
            frames, mos_labels = frames.to(Config.DEVICE), mos_labels.to(Config.DEVICE)

            optimizer.zero_grad()

            # Forward pass
            quality_scores = model(frames)

            # Regression loss
            regression_loss = l1_criterion(quality_scores.squeeze(), mos_labels.squeeze())

            # Spearman correlation loss
            spearman_loss = spearman_criterion(quality_scores.squeeze(), mos_labels.squeeze())

            # Combined loss (without contrastive loss)
            loss = regression_loss + Config.SPEARMAN_WEIGHT * spearman_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        print(f"Training Loss: {avg_loss:.4f}")

        # Evaluation
        model.eval()
        all_true, all_pred = [], []

        with torch.no_grad():
            for frames, mos_labels, _ in tqdm(test_loader, desc="Evaluating", leave=False):
                frames, mos_labels = frames.to(Config.DEVICE), mos_labels.to(Config.DEVICE)

                predictions = model(frames).squeeze()

                if predictions.dim() == 0:  # If scalar, wrap in a list
                    predictions = predictions.unsqueeze(0)

                all_true.extend(mos_labels.cpu().numpy())
                all_pred.extend(predictions.cpu().numpy())

        # Metrics computation
        all_true, all_pred = np.array(all_true), np.array(all_pred)
        plcc = pearsonr(all_true, all_pred)[0]
        srocc = spearmanr(all_true, all_pred)[0]
        mae = mean_absolute_error(all_true, all_pred)

        print(f"PLCC: {plcc:.4f}, SROCC: {srocc:.4f}, MAE: {mae:.4f}")

        # Plot and save graph for predicted vs actual scores
        plt.figure(figsize=(6, 6))
        plt.scatter(all_true, all_pred, alpha=0.6, label=f'Epoch {epoch + 1}')
        plt.plot([all_true.min(), all_true.max()], [all_true.min(), all_true.max()], 'r--', lw=2, label='Ideal Fit')
        plt.xlabel("Actual MOS Scores")
        plt.ylabel("Predicted MOS Scores")
        plt.title(f"Predicted vs Actual Scores: KonVid1k")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"pred_vs_actual_epoch_{epoch + 1}.png")
        plt.close()

if __name__ == "__main__":
    fine_tune_and_evaluate()
