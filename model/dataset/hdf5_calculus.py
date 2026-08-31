import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import random

def convert_calculus_to_hdf5(npz_dir, h5_path, compression="gzip", views=7):
    """
    Converts a directory of calculus .npz files into a single HDF5 file.
    
    Args:
        npz_dir: The source data directory containing .npz files.
        h5_path: The path for the output HDF5 file.
        compression: The compression algorithm to use ("gzip" recommended).
        views: Number of views per case.
    """
    npz_dir = Path(npz_dir)
    h5_path = Path(h5_path)
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    
    npz_files = list(npz_dir.glob("*.npz"))
    
    samples = []
    for npz_file in tqdm(npz_files, desc="Scanning calculus .npz files"):
        case_name = npz_file.stem
        for view_idx in range(views):
            samples.append({
                'npz_path': str(npz_file),
                'view_idx': view_idx,
                'case_name': case_name
            })
            
    print(f"Found {len(samples)} samples, starting conversion...")
    
    with h5py.File(h5_path, 'w') as f:
        images_group = f.create_group('images')
        calc_masks_group = f.create_group('calculus_masks')
        
        sample_index = []
        cached_npz = {}
        
        for sample in tqdm(samples, desc="Converting calculus data"):
            npz_path = sample['npz_path']
            view_idx = sample['view_idx']
            case_name = sample['case_name']
            sample_id = f"{case_name}_view{view_idx}"
            
            sample_index.append({
                'id': sample_id,
                'case_name': case_name,
                'view_idx': view_idx
            })
            
            if npz_path not in cached_npz:
                cached_npz[npz_path] = np.load(npz_path)
            npz_data = cached_npz[npz_path]
            
            # Write images
            image_array = npz_data['images'][view_idx]
            images_group.create_dataset(
                sample_id,
                data=image_array,
                compression=compression
            )
            
            # Write calculus masks
            calc_mask = npz_data['calculus_masks'][view_idx]
            calc_masks_group.create_dataset(
                sample_id,
                data=calc_mask,
                compression=compression
            )
            
        f.create_dataset('sample_index', data=json.dumps(sample_index))
        
    print(f"Conversion complete! Data saved to {h5_path}")
    return h5_path

def generate_calculus_splits(npz_dir, split_dir, train_ratio=0.7, seed=42, h5_path=None):
    """
    Generates training and testing split files for the calculus dataset.
    
    Args:
        npz_dir: The source data directory containing .npz files.
        split_dir: The directory to save the split files.
        train_ratio: Ratio of cases to use for training.
        seed: Random seed for reproducibility.
        h5_path: Fallback HDF5 file to read case names from when no .npz files exist.
    """
    npz_dir = Path(npz_dir)
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    
    random.seed(seed)
    
    npz_files = list(npz_dir.glob("*.npz"))
    if len(npz_files) > 0:
        case_names = list(set(f.stem for f in npz_files))
    elif h5_path and Path(h5_path).exists():
        print(f"No .npz files in {npz_dir}. Reading case names from {h5_path}...")
        with h5py.File(h5_path, 'r') as f:
            if 'sample_index' in f:
                samples = json.loads(f['sample_index'][()])
                case_names = list(set(s['case_name'] for s in samples))
            else:
                case_names = []
    else:
        print(f"Warning: No .npz files in {npz_dir} and no valid h5_path. Splits will be empty.")
        case_names = []
    
    # Sort to ensure deterministic behavior before shuffling
    case_names.sort()
    random.shuffle(case_names)
    
    num_train = int(len(case_names) * train_ratio)
    train_cases = case_names[:num_train]
    test_cases = case_names[num_train:]
    
    train_file = split_dir / "training_calculus.txt"
    with open(train_file, 'w') as f:
        for case in sorted(train_cases):
            f.write(f"{case}\n")
            
    test_file = split_dir / "testing_calculus.txt"
    with open(test_file, 'w') as f:
        for case in sorted(test_cases):
            f.write(f"{case}\n")
            
    print(f"Splits generated: {len(train_cases)} train, {len(test_cases)} test.")
    print(f"Saved to {split_dir}")
