import h5py
import torch
from torch.utils.data import Dataset
import json
import numpy as np
import random

class CalculusAugmentation:
    def __init__(self, config=None):
        if config is None:
            config = {}
        
        # Light augmentation
        self.brightness_prob = config.get('brightness_prob', 0.3)
        self.brightness_range = config.get('brightness_range', 0.1)
        
        self.contrast_prob = config.get('contrast_prob', 0.3)
        self.contrast_range = config.get('contrast_range', 0.1)
        
        self.noise_prob = config.get('noise_prob', 0.2)
        self.noise_std = config.get('noise_std', 0.01)
        
    def __call__(self, image, calculus_mask):
        """
        Apply light data augmentation.
        Args:
            image: [3, H, W] image tensor.
            calculus_mask: [2, H, W] mask tensor.
        Returns:
            augmented_image, augmented_mask
        """
        image = image.clone()
        calculus_mask = calculus_mask.clone()
        
        # Brightness adjustment
        if random.random() < self.brightness_prob:
            brightness_factor = 1 + random.uniform(-self.brightness_range, self.brightness_range)
            image = image * brightness_factor
            
        # Contrast adjustment
        if random.random() < self.contrast_prob:
            contrast_factor = 1 + random.uniform(-self.contrast_range, self.contrast_range)
            mean = image.mean(dim=[1, 2], keepdim=True)
            image = (image - mean) * contrast_factor + mean
            
        # Add Gaussian noise
        if random.random() < self.noise_prob:
            noise = torch.randn_like(image) * self.noise_std
            image = image + noise
            
        # Ensure values are within valid range
        image = torch.clamp(image, 0, 1)
        calculus_mask = torch.clamp(calculus_mask, 0, 1)
        
        return image, calculus_mask

class HDF5CalculusDataset(Dataset):
    """HDF5 dataset for calculus segmentation."""
    def __init__(self, hdf5_file, transform=False, mode=None, augment_config=None, split_file_path=None, view_indices=None):
        """
        Args:
            hdf5_file: Path to the HDF5 file.
            transform: Whether to apply data augmentation.
            mode: 'train', 'val', or 'test'.
            augment_config: Dictionary of augmentation configurations.
            split_file_path: Path to the dataset split file.
            view_indices: View filtering list.
        """
        self.transform = transform
        self.file_path = hdf5_file
        self.mode = mode

        # View filtering list, None means keep all views
        if view_indices is not None:
            if isinstance(view_indices, str):
                view_indices = [int(v) for v in view_indices.split(',') if v.strip()]
            self.view_indices = set(view_indices)
        else:
            self.view_indices = None

        self.hdf5_path = hdf5_file
        self.h5_file = None

        # Open HDF5 file temporarily to read the index and keys
        with h5py.File(hdf5_file, 'r') as f:
            all_samples = json.loads(f['sample_index'][()])
            self.available_fields = list(f.keys())
        
        # Filter samples based on split file
        if split_file_path:
            with open(split_file_path, 'r') as f:
                split_case_names = {line.strip() for line in f if line.strip()}
            
            self.samples = []
            for sample in all_samples:
                if sample['case_name'] in split_case_names:
                    self.samples.append(sample)
            print(f"Loaded {len(self.samples)} samples according to {split_file_path}")
        else:
            self.samples = all_samples
            print(f"Warning: No split file provided, loaded all {len(self.samples)} samples")

        # Filter by views
        if self.view_indices is not None:
            view_filtered_samples = [s for s in self.samples if s['view_idx'] in self.view_indices]
            print(f"Filtered to {len(view_filtered_samples)} samples based on view indices")
            self.samples = view_filtered_samples
        
        # Check required fields
        required_fields = ['images', 'calculus_masks']
        missing_fields = [field for field in required_fields if field not in self.available_fields]
        if missing_fields:
            raise ValueError(f"HDF5 file is missing required fields: {missing_fields}")
        
        # Init augmentation
        if mode == 'train' and transform:
            self.augmentation = CalculusAugmentation(augment_config)
        else:
            self.augmentation = None
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.hdf5_path, 'r')
            
        sample = self.samples[idx]
        sample_id = sample['id']
        
        image = np.array(self.h5_file['images'][sample_id])
        calc_mask_np = np.array(self.h5_file['calculus_masks'][sample_id])
        
        # calc_mask_np is [H, W], binary
        # Create a 2-channel mask where channel 0 is background and channel 1 is calculus
        calc_mask_2ch = np.stack([1 - calc_mask_np, calc_mask_np], axis=0)
        
        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1)  # [3, H, W]
        calculus_mask_tensor = torch.from_numpy(calc_mask_2ch).float()  # [2, H, W]
        
        if self.augmentation is not None:
            image_tensor, calculus_mask_tensor = self.augmentation(image_tensor, calculus_mask_tensor)
        
        result = {
            'image': image_tensor,
            'calculus_mask': calculus_mask_tensor,
            'case_name': sample['case_name'],
            'view_idx': sample['view_idx']
        }
            
        return result
    
    def __del__(self):
        if getattr(self, 'h5_file', None) is not None:
            self.h5_file.close()
