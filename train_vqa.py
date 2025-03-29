import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import numpy as np
import csv
from spatiotemp import *
from dataloader import KonvidVQADataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

class SiameseNetworkTrainer:
    def __init__(self, frame_dir, mos_csv, saliency_dir, device, rank=0):
        self.device = device
        self.rank = rank

        # Data transformations
        self.data_transforms = transforms.Compose([
            transforms.Resize((224, 398)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor()
        ])

        self.dataset = KonvidVQADataset(frame_dir=frame_dir, mos_csv=mos_csv, saliency_dir=saliency_dir,
                                        transform=self.data_transforms, num_frames=8, alpha=0.75)
        total_videos = min(len(self.dataset), 1000)
        selected_indices = list(range(total_videos))
        train_indices, val_indices = train_test_split(selected_indices, test_size=0.2, random_state=953)

        self.train_dataset = Subset(self.dataset, train_indices)
        self.val_dataset = Subset(self.dataset, val_indices)

        self.train_sampler = DistributedSampler(self.train_dataset, num_replicas=dist.get_world_size(), rank=rank, shuffle=True)
        self.val_sampler = DistributedSampler(self.val_dataset, num_replicas=dist.get_world_size(), rank=rank, shuffle=False)

        self.train_loader = DataLoader(self.train_dataset, sampler=self.train_sampler, batch_size=5, num_workers=5, drop_last=True)
        self.val_loader = DataLoader(self.val_dataset, sampler=self.val_sampler, batch_size=2, num_workers=5, drop_last=True)

        # Model setup
        self.model = VideoQualityModelSimpleFusion(
            device=self.device,
            spatial_feature_dim=2048,
            temporal_feature_dim=2048
        ).to(self.device)

        self.model = DDP(self.model, device_ids=[rank])

        # Loss functions
        self.regression_criterion = nn.L1Loss()
        self.rank_criterion = SpearmanCorrelationLoss()

        # Optimizer and Scheduler
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.000005, weight_decay=5e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=50, eta_min=1e-8)

        # CSV file to log losses
        self.loss_log_file = 'loss_log.csv'
        with open(self.loss_log_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'Regression Loss', 'Correlation Loss', 'PLCC', 'Val Regression Loss', 'Val Correlation Loss', 'Val PLCC'])

    def calculate_plcc(self, y_true, y_pred):
        """
        Calculate Pearson Linear Correlation Coefficient (PLCC)
        :param y_true: Ground truth values (MOS labels)
        :param y_pred: Predicted values (predicted MOS scores)
        :return: PLCC value
        """
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()

        corr_matrix = np.corrcoef(y_true, y_pred)
        return corr_matrix[0, 1]

    def validate(self):
        self.model.eval()
        val_regression_loss, val_corr_loss = 0.0, 0.0
        all_mos_labels = []
        all_predicted_scores = []

        with torch.no_grad():
            for frames, mos_labels, _ in self.val_loader:
                frames, mos_labels = frames.to(self.device), mos_labels.to(self.device)

                quality_scores = self.model(frames)

                # Regression loss
                regression_loss = self.regression_criterion(quality_scores.view(-1), mos_labels)
                val_regression_loss += regression_loss.item()

                # Append MOS labels and predictions for correlation calculation
                all_mos_labels.extend(mos_labels.cpu().numpy().flatten())
                all_predicted_scores.extend(quality_scores.cpu().numpy().flatten())

        # Calculate correlation loss
        all_mos_labels = np.array(all_mos_labels)
        all_predicted_scores = np.array(all_predicted_scores)
        val_corr_loss = self.rank_criterion(torch.tensor(all_predicted_scores), torch.tensor(all_mos_labels))
        val_plcc = self.calculate_plcc(all_mos_labels, all_predicted_scores)

        return val_regression_loss / len(self.val_loader), val_corr_loss.item(), val_plcc

    def train(self, num_epochs=15):
        for epoch in range(num_epochs):
            self.model.train()
            self.train_sampler.set_epoch(epoch)

            running_regression_loss = 0.0
            running_corr_loss = 0.0
            all_train_mos_labels = []
            all_train_predicted_scores = []

            self.optimizer.zero_grad()

            for batch_idx, (frames, mos_labels, _) in enumerate(self.train_loader):
                frames, mos_labels = frames.to(self.device), mos_labels.to(self.device)

                quality_scores = self.model(frames)

                # Regression loss
                regression_loss = self.regression_criterion(quality_scores.view(-1), mos_labels)

                # Correlation loss
                corr_loss = self.rank_criterion(quality_scores.view(-1), mos_labels)

                # Combined loss
                combined_loss = regression_loss + 0.1 * corr_loss
                combined_loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                running_regression_loss += regression_loss.item()
                running_corr_loss += corr_loss.item()

                all_train_mos_labels.extend(mos_labels.cpu().numpy().flatten())
                all_train_predicted_scores.extend(quality_scores.detach().cpu().numpy().flatten())

            train_plcc = self.calculate_plcc(np.array(all_train_mos_labels), np.array(all_train_predicted_scores))
            avg_val_regression_loss, avg_val_corr_loss, val_plcc = self.validate()

            if self.rank == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], "
                    f"Train Regression Loss: {running_regression_loss / len(self.train_loader):.4f}, "
                    f"Train Correlation Loss: {running_corr_loss / len(self.train_loader):.4f}, "
                    f"Train PLCC: {train_plcc:.4f}, "
                    f"Val Regression Loss: {avg_val_regression_loss:.4f}, "
                    f"Val Correlation Loss: {avg_val_corr_loss:.4f}, "
                    f"Val PLCC: {val_plcc:.4f}"
                )

                with open(self.loss_log_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        epoch + 1,
                        running_regression_loss / len(self.train_loader),
                        running_corr_loss / len(self.train_loader),
                        train_plcc,
                        avg_val_regression_loss,
                        avg_val_corr_loss,
                        val_plcc
                    ])
                if epoch % 10 == 0 and epoch > 800:
                    torch.save(
                        {
                            "model_state_dict": self.model.state_dict(),
                        },
                        f"checkpoint_epoch_{epoch + 1}_plcc_{val_plcc:.4f}.pth",
                    )
                    print(f"Checkpoint saved for epoch {epoch + 1}, PLCC: {val_plcc:.4f}")

            self.scheduler.step()

def main_worker(rank, world_size):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    trainer = SiameseNetworkTrainer(
        frame_dir='./KoNViD_1k_extracted_frames',
        mos_csv='./KoNViD_1k/KoNViD_1k_mos.csv',
        saliency_dir='./KoNViD_1k_saliency_maps/',
        device=device,
        rank=rank
    )
    trainer.train(num_epochs=1000)
    cleanup()

def train():
    world_size = torch.cuda.device_count()
    mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    train()
