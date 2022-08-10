import os
import os.path as osp
import pandas as pd
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def prepare_dataset(fp_data, fp_label):
    """
    Prepare dataset for training and testing.
    """
    # Load data
    data = np.load(fp_data)
    label = np.load(fp_label)
    # Convert to tensor
    data = torch.from_numpy(data).float()
    label = torch.from_numpy(label).long()
    # load on GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    data = data.to(device)
    label = label.to(device)
    # Create dataset
    dataset = TensorDataset(data, label)
    return dataset

if __name__ == '__main__':
    dataroot='/home/nawake/sthv2/'
    out_dir = osp.join(dataroot, 'videomae/hand_crop_right')
    fp_data = osp.join(out_dir, 'feat_test.npy')
    fp_label = osp.join(out_dir, 'label_test.npy')
    dataset, dataloader = prepare_dataset(fp_data, fp_label)
    print('Done')
    train_features, train_labels = next(iter(dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")