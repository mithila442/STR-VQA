import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np
import scipy.io
from extract_frame import process_all_videos

class KonvidVQADataset(Dataset):
    def __init__(self, frame_dir, mos_csv, saliency_dir, transform=None, num_frames=8, selected_videos=None, alpha=0.5):
        self.frame_dir = frame_dir
        self.saliency_dir = saliency_dir
        self.transform = transform
        self.num_frames = num_frames
        self.selected_videos = selected_videos
        self.alpha=alpha

        self.saliency_transforms = transforms.Compose([
            transforms.Resize((224, 398)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.available_videos = self._get_available_videos()
        self.mos_df = pd.read_csv(mos_csv)
        self.valid_indices = self._filter_valid_videos()

    def __len__(self):
        return len(self.valid_indices)

    def _get_available_videos(self):
        available_videos = set(os.listdir(self.frame_dir))
        if self.selected_videos is not None:
            available_videos = available_videos.intersection(self.selected_videos)
        return available_videos

    def _filter_valid_videos(self):
        valid_indices = []
        for idx in range(len(self.mos_df)):
            flickr_id = str(int(self.mos_df.iloc[idx]['flickr_id']))
            if flickr_id in self.available_videos:
                valid_indices.append(idx)
        return valid_indices

    def __getitem__(self, idx):
        valid_idx = self.valid_indices[idx]
        flickr_id = str(int(self.mos_df.iloc[valid_idx]['flickr_id']))
        mos_score = self.mos_df.iloc[valid_idx]['mos']

        # Normalize MOS score
        mos_min = self.mos_df['mos'].min()
        mos_max = self.mos_df['mos'].max()
        mos_score = (mos_score - mos_min) / (mos_max - mos_min)

        frame_folder = os.path.join(self.frame_dir, flickr_id)
        saliency_folder = os.path.join(self.saliency_dir, flickr_id)

        # Load pre-extracted frames from disk
        frame_files = sorted([os.path.join(frame_folder, img) for img in os.listdir(frame_folder) if img.endswith('.png')])

        if len(frame_files) != self.num_frames:
            raise ValueError(f"Mismatch: Expected {self.num_frames} frames but found {len(frame_files)} for {flickr_id}")

        frames = []
        for frame_file in frame_files:
            frame = Image.open(frame_file).convert("RGB")
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)

        # Load and align saliency maps
        if not os.path.exists(saliency_folder):
            raise FileNotFoundError(f"Saliency folder does not exist: {saliency_folder}")

        saliency_maps = sorted([os.path.join(saliency_folder, img) for img in os.listdir(saliency_folder) if img.endswith('.png')])

        if len(saliency_maps) >= self.num_frames:
            saliency_indices = np.linspace(0, len(saliency_maps) - 1, self.num_frames, dtype=int)
            saliency_maps = [saliency_maps[i] for i in saliency_indices]
        else:
            saliency_maps.extend([saliency_maps[-1]] * (self.num_frames - len(saliency_maps)))

        frame_tensors = []
        for i, (frame, saliency_map) in enumerate(zip(frames, saliency_maps)):
            saliency = Image.open(saliency_map).convert('L')
            saliency = self.saliency_transforms(saliency)

            if torch.all(saliency == 0):
                print(f"Warning: Zero saliency map detected for {saliency_map}")
                saliency = torch.ones_like(saliency)

            saliency = saliency / max(torch.max(saliency), 1e-6)
            saliency = torch.clamp(saliency, min=0.1)

            weighted_frame = (1 - self.alpha) * frame + self.alpha * (frame * saliency)

            frame_tensors.append(weighted_frame)

        frames_tensor = torch.stack(frame_tensors, dim=1)  # Shape: (C, num_frames, H, W)

        return frames_tensor, torch.tensor(mos_score, dtype=torch.float32), flickr_id


class LiveVQADataset(Dataset):
    def __init__(self, frame_dir, mat_file, saliency_dir, transform=None, num_frames=8, alpha=0.5):
        self.frame_dir = frame_dir
        self.saliency_dir = saliency_dir
        self.transform = transform
        self.num_frames = num_frames
        self.alpha=alpha

        self.saliency_transforms = transforms.Compose([
            transforms.Resize((224, 398)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # Load LIVE-VQA dataset metadata
        mat_contents = scipy.io.loadmat(mat_file)
        self.video_list = [v[0][0] for v in mat_contents["video_list"]]  # Extract filenames
        self.mos_scores = mat_contents["mos"].flatten()  # Extract MOS scores

        self.available_videos = self._get_available_videos()
        self.valid_indices = self._filter_valid_videos()

    def __len__(self):
        return len(self.valid_indices)

    def _get_available_videos(self):
        available_videos = set(os.listdir(self.frame_dir))
        return available_videos

    def _filter_valid_videos(self):
        valid_indices = []
        for idx in range(len(self.video_list)):
            video_name = self.video_list[idx].split('.')[0]  # Remove extension
            if video_name in self.available_videos:
                valid_indices.append(idx)
        return valid_indices

    def __getitem__(self, idx):
        valid_idx = self.valid_indices[idx]
        video_name = self.video_list[valid_idx]
        mos_score = self.mos_scores[valid_idx]

        # Normalize MOS score
        mos_min = self.mos_scores.min()
        mos_max = self.mos_scores.max()
        mos_score = (mos_score - mos_min) / (mos_max - mos_min)

        frame_folder = os.path.join(self.frame_dir, video_name.split('.')[0])
        saliency_folder = os.path.join(self.saliency_dir, video_name.split('.')[0])

        # Load pre-extracted frames from disk
        frame_files = sorted([os.path.join(frame_folder, img) for img in os.listdir(frame_folder) if img.endswith('.png')])

        if len(frame_files) != self.num_frames:
            raise ValueError(f"Mismatch: Expected {self.num_frames} frames but found {len(frame_files)} for {video_name}")

        frames = []
        for frame_file in frame_files:
            frame = Image.open(frame_file).convert("RGB")
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)

        # Load and align saliency maps
        if not os.path.exists(saliency_folder):
            raise FileNotFoundError(f"Saliency folder does not exist: {saliency_folder}")

        saliency_maps = sorted([os.path.join(saliency_folder, img) for img in os.listdir(saliency_folder) if img.endswith('.png')])

        if len(saliency_maps) >= self.num_frames:
            saliency_indices = np.linspace(0, len(saliency_maps) - 1, self.num_frames, dtype=int)
            saliency_maps = [saliency_maps[i] for i in saliency_indices]
        else:
            saliency_maps.extend([saliency_maps[-1]] * (self.num_frames - len(saliency_maps)))

        frame_tensors = []
        for i, (frame, saliency_map) in enumerate(zip(frames, saliency_maps)):
            saliency = Image.open(saliency_map).convert('L')
            saliency = self.saliency_transforms(saliency)

            if torch.all(saliency == 0):
                print(f"Warning: Zero saliency map detected for {saliency_map}")
                saliency = torch.ones_like(saliency)

            saliency = saliency / max(torch.max(saliency), 1e-6)
            saliency = torch.clamp(saliency, min=0.1)

            weighted_frame = (1 - self.alpha) * frame + self.alpha * (frame * saliency)

            frame_tensors.append(weighted_frame)

        frames_tensor = torch.stack(frame_tensors, dim=1)  # Shape: (C, num_frames, H, W)

        return frames_tensor, torch.tensor(mos_score, dtype=torch.float32), video_name


class YouTubeUGCVQADataset(Dataset):
    def __init__(self, frame_dir, mos_csv, saliency_dir, transform=None, num_frames=8, selected_videos=None, alpha=0.5):
        self.frame_dir = frame_dir
        self.saliency_dir = saliency_dir
        self.transform = transform
        self.num_frames = num_frames
        self.selected_videos = selected_videos
        self.alpha= alpha

        self.saliency_transforms = transforms.Compose([
            transforms.Resize((224, 398)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.available_videos = self._get_available_videos()
        self.mos_df = pd.read_csv(mos_csv)
        self.valid_indices = self._filter_valid_videos()

    def __len__(self):
        return len(self.valid_indices)

    def _get_available_videos(self):
        available_videos = set(os.listdir(self.frame_dir))
        if self.selected_videos is not None:
            available_videos = available_videos.intersection(self.selected_videos)
        return available_videos

    def _filter_valid_videos(self):
        valid_indices = []
        for idx in range(len(self.mos_df)):
            video_id = self.mos_df.iloc[idx]['vid']  # Extracted frame folders match video IDs
            if video_id in self.available_videos:
                valid_indices.append(idx)
        return valid_indices

    def __getitem__(self, idx):
        valid_idx = self.valid_indices[idx]
        video_id = self.mos_df.iloc[valid_idx]['vid']
        mos_score = self.mos_df.iloc[valid_idx]['MOS full']

        # Normalize MOS score
        mos_min = self.mos_df['MOS full'].min()
        mos_max = self.mos_df['MOS full'].max()
        mos_score = (mos_score - mos_min) / (mos_max - mos_min)

        frame_folder = os.path.join(self.frame_dir, video_id)
        saliency_folder = os.path.join(self.saliency_dir, video_id)

        # Load pre-extracted frames
        frame_files = sorted([os.path.join(frame_folder, img) for img in os.listdir(frame_folder) if img.endswith('.png')])

        if len(frame_files) != self.num_frames:
            raise ValueError(f"Mismatch: Expected {self.num_frames} frames but found {len(frame_files)} for {video_id}")

        frames = []
        for frame_file in frame_files:
            frame = Image.open(frame_file).convert("RGB")
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)

        # Load and align saliency maps
        if not os.path.exists(saliency_folder):
            raise FileNotFoundError(f"Saliency folder does not exist: {saliency_folder}")

        saliency_maps = sorted([os.path.join(saliency_folder, img) for img in os.listdir(saliency_folder) if img.endswith('.png')])

        if len(saliency_maps) >= self.num_frames:
            saliency_indices = np.linspace(0, len(saliency_maps) - 1, self.num_frames, dtype=int)
            saliency_maps = [saliency_maps[i] for i in saliency_indices]
        else:
            saliency_maps.extend([saliency_maps[-1]] * (self.num_frames - len(saliency_maps)))

        frame_tensors = []
        for i, (frame, saliency_map) in enumerate(zip(frames, saliency_maps)):
            saliency = Image.open(saliency_map).convert('L')
            saliency = self.saliency_transforms(saliency)

            if torch.all(saliency == 0):
                print(f"Warning: Zero saliency map detected for {saliency_map}")
                saliency = torch.ones_like(saliency)

            saliency = saliency / max(torch.max(saliency), 1e-6)
            saliency = torch.clamp(saliency, min=0.1)

            weighted_frame = (1 - self.alpha) * frame + self.alpha * (frame * saliency)

            frame_tensors.append(weighted_frame)

        frames_tensor = torch.stack(frame_tensors, dim=1)  # Shape: (C, num_frames, H, W)

        return frames_tensor, torch.tensor(mos_score, dtype=torch.float32), video_id
