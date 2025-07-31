import torch
import math
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy.interpolate import splprep, splev
import os






class SeqDataset(Dataset):
    def __init__(self,src_path, tgt_path,src2_path, input_size=8, target_size=25):
        self.training_size = len(os.listdir(src_path))
        self.training_size2 = len(os.listdir(src2_path))
        self.src_files = [f'{src_path}/src{n}.pt' for n in range(self.training_size)]
        self.tgt_files = [f'{tgt_path}/tgt{n}.pt' for n in range(self.training_size)]
        self.csv_files = [f'{src2_path}/random_track{n}.csv' for n in range(self.training_size2)]
        self.input_size = input_size
        self.target_size = target_size

    def __len__(self):
        return len(self.src_files)

    def __getitem__(self, index):
        src = torch.load(self.src_files[index])  # Shape: [seq_len, feat_dim]
        tgt = torch.load(self.tgt_files[index])  # Shape: [seq_len, feat_dim]

        df = pd.read_csv(self.csv_files[index%self.training_size2], header=None)
        df.drop(df.columns[[3,4,5,6]],axis=1,inplace=True)
        df.columns = ['color', 'x', 'y']

        left2 = df[df['color'] == 'blue'][['x', 'y']].to_numpy() 
        left2 = np.concatenate([left2,np.ones((left2.shape[0], 1)), np.zeros((left2.shape[0], 1))], axis=1)
        right2 = df[df['color'] == 'yellow'][['x', 'y']].to_numpy()
        right2 = np.concatenate([right2, np.zeros((right2.shape[0], 1)), np.ones((right2.shape[0], 1))], axis=1)

        tgt2 = generate_mid_track(left2, right2)
        tgt2 = torch.from_numpy(tgt2)


        

        # Create masks
        left_mask = src[:, 2] == 1
        right_mask = src[:, 4] == 1

        # Apply masks and padding
        left = pad_and_mask(src, left_mask)
        right = pad_and_mask(src, right_mask)

        # Interleave
        left_selected = left[:, [0, 1,2, 4]]  # shape [N, 3]
        right_selected = right[:, [0,1,2, 4]]

        left2 = torch.from_numpy(left2)
        right2 = torch.from_numpy(right2)



        seq = fast_interleave(left_selected, right_selected)
        seq2 = fast_interleave(left2, right2)
        #seq = interleave(left, right)  # Shape: [new_seq_len, feat_dim]

        # Create sliding window samples
        samples = []
        if random.random() > 0.5:
            seq, tgt = reverse_track(seq, tgt)
            seq2, tgt2 = reverse_track(seq2, tgt2)
            
        k=0
        while k<tgt.size(0):
            target_chunk = tgt[k:k+self.target_size, :]
            start = target_chunk[0]
            dx = target_chunk[1][0] - target_chunk[0][0]
            dy = target_chunk[1][1] - target_chunk[0][1]
            theta = math.atan2(dy, dx)
            input_chunk = seq[is_inside_semicircle(start, theta, 8, seq)][:self.input_size,:]
            input_chunk, target_chunk = normalize_data(input_chunk, target_chunk, start, theta)
            input_chunk = add_noisy_data(input_chunk)
            samples.append((input_chunk[torch.randperm(input_chunk.size(0))] , target_chunk))
            k+=5

        k=0
        while k<tgt2.size(0):
            target_chunk = tgt2[k:k+self.target_size, :]
            start = target_chunk[0]
            dx = target_chunk[1][0] - target_chunk[0][0]
            dy = target_chunk[1][1] - target_chunk[0][1]
            theta = math.atan2(dy, dx)
            input_chunk = seq2[is_inside_semicircle(start, theta, 8, seq2)][:self.input_size,:]
            input_chunk, target_chunk = normalize_data(input_chunk, target_chunk, start, theta)
            input_chunk = add_noisy_data(input_chunk)
            samples.append((input_chunk[torch.randperm(input_chunk.size(0))] , target_chunk))
            k+=5

        return samples  # Return list of (input, target) pairs





def is_inside_semicircle(center, direction, radius, cones):
    vectors = cones[:, :2] - center[:2]
    angles = torch.atan2(vectors[:, 1], vectors[:, 0])
    dists = torch.norm(vectors, dim=1)
    
    start_angle = direction - math.pi/4
    end_angle = direction + math.pi/4

    angle_in_range = (angles >= start_angle) & (angles <= end_angle)
    dist_in_range = (dists <= radius) & (dists>1)

    return angle_in_range & dist_in_range

def resample_curve(points, n_points):
    """Resample a 2D curve to have exactly n_points using linear interpolation."""
    # Compute cumulative arc length
    deltas = np.diff(points, axis=0)
    dists = np.sqrt((deltas**2).sum(axis=1))
    cumulative_dist = np.insert(np.cumsum(dists), 0, 0)
    total_dist = cumulative_dist[-1]
    new_distances = np.linspace(0, total_dist, n_points)

    # Interpolate x and y separately
    x_interp = np.interp(new_distances, cumulative_dist, points[:, 0])
    y_interp = np.interp(new_distances, cumulative_dist, points[:, 1])
    return np.stack((x_interp, y_interp), axis=1)

def generate_mid_track(left, right, upsample_factor=5, smoothing=0.5):
    n_points = min(len(left), len(right))

    # Resample both tracks to same length
    left_resampled = resample_curve(left, n_points)
    right_resampled = resample_curve(right, n_points)

    # Midpoints
    midpoints = (left_resampled + right_resampled) / 2

    # Smooth and upsample using spline
    tck, u = splprep([midpoints[:, 0], midpoints[:, 1]], s=smoothing)
    u_fine = np.linspace(0, 1, n_points * upsample_factor)
    x_smooth, y_smooth = splev(u_fine, tck)

    return np.stack((x_smooth, y_smooth), axis=1)


def add_noisy_data(input_chunk):
  for i in range(input_chunk.shape[0]):
    if random.random()>0.9999:
      if input_chunk[i][2] == 1:
        input_chunk[i][2] = 0
      else:
        input_chunk[i][2] = 1

    input_chunk[i][0] += random.random()
    input_chunk[i][1] += random.random()

    if random.random()>0.97:
      input_chunk[i][0] = random.random()*10
      input_chunk[i][1] = random.random()*10

    if random.random()>0.97:
      input_chunk[i] = 0

  return input_chunk


def normalize_data(input_chunk, target_chunk, center, theta):
    # Ensure float32 dtype for PyTorch operations
    input_chunk = input_chunk.clone().float()
    target_chunk = target_chunk.clone().float()
    center = center.float()

    # Shift to origin
    noise = torch.tensor([
        np.random.uniform(-2, 2),
        np.random.uniform(0, 2)
    ], dtype=torch.float32)
    
    starting_point = center + noise

    target_chunk = target_chunk - starting_point
    input_chunk[:, :2] = input_chunk[:, :2] - starting_point[:2]

    # Compute angle


    # Rotate to make heading point up (along +y)
    rotation_angle = math.pi/2 - theta + random.uniform(-0.4,0.4)
    cos_a = math.cos(rotation_angle)
    sin_a = math.sin(rotation_angle)

    rotation_matrix = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], dtype=torch.float32)

    input_chunk[:, :2] = input_chunk[:, :2] @ rotation_matrix.T
    target_chunk[:, :2] = target_chunk[:, :2] @ rotation_matrix.T

    return input_chunk, target_chunk


def pad_and_mask(src, mask):
    seq_len, feat_dim = src.shape
    masked = src[mask]  # Extract valid elements
    max_len = masked.shape[0] if masked.shape[0] > 0 else 1
    padded = torch.zeros((max_len, feat_dim), device=src.device)
    padded[:masked.shape[0], :] = masked
    return padded


def fast_interleave(left_selected, right_selected):
    L, R = left_selected.shape[0], right_selected.shape[0]
    max_len = max(L, R)
    pad_left = F.pad(left_selected, (0, 0, 0, max_len - L))
    pad_right = F.pad(right_selected, (0, 0, 0, max_len - R))
    return torch.stack((pad_left, pad_right), dim=1).view(-1, 4)


def reverse_track(src, tgt):
    # Reverse both input (src) and target (tgt)
    src_reversed = torch.flip(src, dims=[0]).clone()
    tgt_reversed = torch.flip(tgt, dims=[0])

    # Masks
    left_mask = src_reversed[:, 2] == 1
    right_mask = src_reversed[:, 3] == 1

    # Swap using copies to avoid overlap issues
    new_left = src_reversed[:, 2].clone()
    new_right = src_reversed[:, 3].clone()

    new_left[left_mask] = 0
    new_right[left_mask] = 1

    new_right[right_mask] = 0
    new_left[right_mask] = 1

    src_reversed[:, 2] = new_left
    src_reversed[:, 3] = new_right

    return src_reversed, tgt_reversed



def collate_fn(batch):
    """
    Custom collate function to flatten samples and pad sequences for batching.
    """
    flattened_samples = [sample for sublist in batch for sample in sublist]  # Flatten list of lists
    input_batch, target_batch = zip(*flattened_samples)  # Separate inputs and targets

    # Convert to tensors and pad
    input_padded = pad_sequence(input_batch, batch_first=True, padding_value=0)
    target_padded = pad_sequence(target_batch, batch_first=True, padding_value=0)

    return input_padded, target_padded

def output_data_loader(src_path, tgt_path, src2_path):
    train_dataset = SeqDataset(src_path=src_path, tgt_path=tgt_path,src2_path=src2_path)


    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=1, collate_fn=collate_fn)
    return train_dataloader
