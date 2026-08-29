import os
import glob
import json
import argparse
import trimesh
import numpy as np
from multiprocessing import Pool
from functools import partial
from pathlib import Path

# Import our fault generators
from fault_generators import generate_cavity, generate_crack, generate_abrasion, generate_calculus

def process_single_sample(sample_dir, input_root, output_root):
    """
    Processes a single patient's directory.
    Example sample_dir: 'data_part_1/lower/O52P1SZT' or 'teeth3ds_sample/01F4JV8X'
    """
    sample_path = Path(sample_dir)
    patient_id = sample_path.name
    
    # Locate the .obj and .json (exclude __kpt.json)
    obj_files = list(sample_path.glob("*.obj"))
    json_files = [f for f in sample_path.glob("*.json") if "__kpt" not in f.name]
    
    if not obj_files or not json_files:
        return f"Skipping {sample_dir} (Missing .obj or .json)"
        
    obj_file = obj_files[0]
    json_file = json_files[0]
    
    # Determine jaw from filename (e.g. O52P1SZT_lower.obj)
    jaw = 'lower' if 'lower' in obj_file.name.lower() else 'upper'
    
    try:
        # Load mesh
        mesh = trimesh.load(str(obj_file), process=False)
        
        # Load labels
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        labels = np.array(data['labels'])
        
        # Define the fault generators
        # We are exclusively focusing the pipeline on generate_calculus 
        # to prevent noisy geometric baseline issues.
        fault_types = {
            # 'cavity': generate_cavity,
            # 'crack': generate_crack,
            # 'abrasion': generate_abrasion,
            'calculus': generate_calculus
        }
        
        for fault_name, generator_func in fault_types.items():
            # Generate fault
            modified_mesh, fault_mask = generator_func(mesh, labels, jaw=jaw)
            
            # Prepare output directory: synthetic_data/<fault_name>/<jaw>/<patient_id>
            out_dir = Path(output_root) / fault_name / jaw / patient_id
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # Save the modified mesh
            out_obj_path = out_dir / obj_file.name
            modified_mesh.export(str(out_obj_path))
            
            # Prepare and save modified JSON label
            # We copy original data and add fault labels
            new_data = data.copy()
            new_data['fault_labels'] = fault_mask.astype(int).tolist()
            new_data['fault_type'] = fault_name
            
            out_json_path = out_dir / json_file.name
            with open(out_json_path, 'w') as f:
                json.dump(new_data, f, separators=(',', ':'))
                
        return f"Successfully processed {patient_id} ({jaw})"
        
    except Exception as e:
        return f"Error processing {sample_dir}: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic dental faults dataset.")
    parser.add_argument('--input_dir', type=str, required=True, help="Root directory of raw dataset (e.g. data_part_1)")
    parser.add_argument('--output_dir', type=str, default="synthetic_data", help="Output directory for synthetic dataset")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of multiprocessing workers")
    args = parser.parse_args()

    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)
    
    if not input_root.exists():
        print(f"Error: Input directory {input_root} does not exist.")
        return

    # Read curation ledger to filter for "Clean" meshes
    ledger_path = "curation_ledger.csv"
    clean_patients = set()
    if os.path.exists(ledger_path):
        import csv
        with open(ledger_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('status') == 'Clean':
                    clean_patients.add(row.get('patient_id', row.get('Unnamed: 0')))
    
    # Find all patient directories (directories containing .obj and .json)
    # We look for directories that have a .obj file in them.
    obj_files = list(input_root.rglob("*.obj"))
    
    # Filter sample_dirs by clean_patients
    sample_dirs = []
    for obj in obj_files:
        if obj.parent.name in clean_patients or not clean_patients:
            sample_dirs.append(obj.parent)
    sample_dirs = list(set(sample_dirs))
    
    if clean_patients:
        print(f"Filtered to {len(sample_dirs)} 'Clean' samples based on ledger.")
    else:
        print(f"Warning: No 'Clean' samples found in ledger or ledger missing. Processing {len(sample_dirs)} samples.")
    
    # Set up multiprocessing
    process_func = partial(process_single_sample, input_root=str(input_root), output_root=str(output_root))
    
    with Pool(args.num_workers) as pool:
        results = pool.map(process_func, [str(d) for d in sample_dirs])
        
    for r in results:
        print(r)
        
    print("Dataset generation complete!")

if __name__ == "__main__":
    main()
