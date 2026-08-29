"""
Preprocess synthetic calculus meshes into multi-view 2D renderings with binary calculus masks.

Takes .obj + .json pairs (with fault_labels) from the synthetic generation pipeline
and renders them from multiple camera viewpoints, producing .npz files containing:
  - images: [N, 512, 512, 3] float32 RGB renderings
  - calculus_masks: [N, 512, 512] bool binary calculus masks (True = calculus)

Usage:
    python preprocess/preprocess_calculus.py --data_dir synthetic_calculus --save_dir preprocess_calculus_data
"""

import os
import torch
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    DirectionalLights,
    TexturesVertex
)
from pytorch3d.io import load_obj
import json
from pathlib import Path
from tqdm import tqdm
import glob
import argparse


class CalculusPreprocessor:
    """Renders synthetic calculus meshes into multi-view 2D images with calculus GT masks."""

    def __init__(self, device='cuda'):
        self.device = device
        self.image_size = 512
        self.camera_distance = 70
        self.background_color = (1, 1, 1)

        # Same 7 canonical viewpoints as the tooth preprocessing pipeline
        self.views = [
            [10, 0],
            [10, 45],
            [10, 315],
            [320, 0],
            [315, 45],
            [315, 315],
            [60, 0],
        ]
        self.setup_renderer()

    def setup_renderer(self):
        self.raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            cull_backfaces=True,
        )

    def get_view_xyz(self, views):
        """Convert (azim, elev) to (x,y,z) for both camera position and light direction."""
        camera_positions = []
        light_directions = []
        for view in views:
            elev = view[0] * np.pi / 180.0
            azim = view[1] * np.pi / 180.0
            x = np.cos(elev) * np.sin(azim)
            y = np.sin(elev)
            z = np.cos(elev) * np.cos(azim)
            pos = np.array([x, y, z])
            camera_positions.append(pos)
            light_directions.append(pos)
        return np.stack(camera_positions, 0), np.stack(light_directions, 0)

    def load_and_normalize_mesh(self, mesh_path):
        """Load OBJ mesh and center it at the origin."""
        verts, faces, aux = load_obj(mesh_path)
        faces = faces.verts_idx
        verts = verts.to(self.device)
        faces = faces.to(self.device)
        center = verts.mean(dim=0)
        verts = verts - center
        return verts, faces

    def load_fault_labels(self, label_path):
        """Load fault_labels (per-vertex binary calculus annotations) from JSON."""
        with open(label_path, 'r') as f:
            gt_data = json.load(f)

        fault_labels = np.array(gt_data.get('fault_labels', []))
        fault_type = gt_data.get('fault_type', 'unknown')

        if len(fault_labels) == 0:
            raise ValueError(f"No fault_labels found in {label_path}")

        if fault_type != 'calculus':
            print(f"Warning: fault_type is '{fault_type}', expected 'calculus'")

        return fault_labels

    def render_single_view(self, verts, faces, vertex_colors, view_R, view_T, light_direction):
        """Render a single view with given camera parameters."""
        cameras = FoVPerspectiveCameras(
            device=self.device,
            R=view_R,
            T=view_T,
            znear=0.01,
            zfar=100.0,
            aspect_ratio=1.0,
            fov=60.0
        )
        current_lights = DirectionalLights(
            device=self.device,
            direction=torch.tensor(np.array([light_direction]), device=self.device),
            ambient_color=((0.48, 0.48, 0.48),),
            diffuse_color=((0.65, 0.65, 0.65),),
            specular_color=((0.6, 0.6, 0.6),)
        )
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=self.raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device,
                cameras=cameras,
                lights=current_lights
            )
        )
        mesh = Meshes(
            verts=[verts],
            faces=[faces],
            textures=TexturesVertex(vertex_colors.unsqueeze(0))
        )
        images = renderer(mesh)
        rendered_image = images[0, ..., :3].clamp(0, 1)
        return rendered_image

    def generate_calculus_mask(self, verts, faces, fault_labels, cameras):
        """
        Rasterize per-vertex fault_labels to a 2D binary mask via face-level majority vote.

        Args:
            verts: [V, 3] mesh vertices
            faces: [F, 3] mesh faces
            fault_labels: [V] per-vertex binary labels (0 or 1)
            cameras: PyTorch3D camera for this viewpoint

        Returns:
            calculus_mask: [H, W] boolean mask (True = calculus)
        """
        vertex_labels = torch.from_numpy(fault_labels).to(self.device).float()

        # Per-face label = majority vote of its 3 vertex labels
        face_vert_labels = vertex_labels[faces]  # [F, 3]
        face_labels = (face_vert_labels.mean(dim=1) >= 0.5).long()  # [F]

        # Rasterize to get per-pixel face indices
        full_mesh = Meshes(
            verts=[verts],
            faces=[faces],
            textures=TexturesVertex(torch.ones_like(verts).unsqueeze(0))
        )
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=self.raster_settings)
        fragments = rasterizer(full_mesh)
        face_idxs = fragments.pix_to_face[0, :, :, 0]  # [H, W]

        # Map face indices to calculus labels
        calculus_map = torch.zeros_like(face_idxs, dtype=torch.bool)
        valid_mask = face_idxs >= 0
        calculus_map[valid_mask] = face_labels[face_idxs[valid_mask]].bool()

        return calculus_map.cpu().numpy()

    def process_single_case(self, obj_path, label_path, save_dir):
        """Process a single synthetic calculus mesh into multi-view renderings."""
        fname = Path(obj_path).stem
        os.makedirs(save_dir, exist_ok=True)

        # Load mesh and labels
        verts, faces = self.load_and_normalize_mesh(obj_path)
        fault_labels = self.load_fault_labels(label_path)

        if len(fault_labels) != verts.shape[0]:
            print(f"Warning: fault_labels length ({len(fault_labels)}) != vertex count ({verts.shape[0]}) for {fname}")
            return

        num_views = len(self.views)
        all_images = np.zeros((num_views, self.image_size, self.image_size, 3), dtype=np.float32)
        all_calculus_masks = np.zeros((num_views, self.image_size, self.image_size), dtype=bool)

        # Default gray vertex colors for rendering
        default_colors = torch.ones_like(verts, dtype=torch.float32) * 0.75

        _, light_directions = self.get_view_xyz(self.views)

        for i, (view, light_dir) in enumerate(zip(self.views, light_directions)):
            R, T = look_at_view_transform(
                dist=self.camera_distance,
                elev=view[0],
                azim=view[1],
                device=self.device
            )

            # Render RGB image
            img = self.render_single_view(verts, faces, default_colors, R, T, light_dir)
            all_images[i] = img.cpu().numpy()

            # Generate calculus mask
            cameras = FoVPerspectiveCameras(
                device=self.device,
                R=R, T=T,
                znear=0.01, zfar=100.0,
                aspect_ratio=1.0, fov=60.0
            )
            calculus_mask = self.generate_calculus_mask(verts, faces, fault_labels, cameras)
            all_calculus_masks[i] = calculus_mask

        # Save as .npz
        output_path = os.path.join(save_dir, f"{fname}.npz")
        np.savez(
            output_path,
            images=all_images,
            calculus_masks=all_calculus_masks,
        )
        print(f"Saved calculus preprocessed data to {output_path}")


def batch_process_calculus(data_dir, save_dir):
    """
    Batch process synthetic calculus meshes.

    Expected input structure:
        data_dir/
        └── calculus/
            ├── upper/
            │   └── <patient_id>/
            │       ├── <patient_id>_upper.obj
            │       └── <patient_id>_upper.json
            └── lower/
                └── <patient_id>/
                    ├── <patient_id>_lower.obj
                    └── <patient_id>_lower.json
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = CalculusPreprocessor(device=device)

    total_processed = 0

    # Check if data_dir has calculus/ subdirectory
    calculus_dir = os.path.join(data_dir, 'calculus')
    if os.path.exists(calculus_dir):
        data_dir = calculus_dir

    for jaw in ['upper', 'lower']:
        jaw_data_dir = os.path.join(data_dir, jaw)
        if not os.path.exists(jaw_data_dir):
            print(f"Skipping {jaw} (directory not found: {jaw_data_dir})")
            continue

        pc_dirs = sorted(glob.glob(f"{jaw_data_dir}/*"))
        print(f"Found {len(pc_dirs)} cases for {jaw} jaw")

        for pc_dir in tqdm(pc_dirs, desc=f"Processing {jaw}"):
            if not os.path.isdir(pc_dir):
                continue

            case = os.path.basename(pc_dir)
            obj_path = os.path.join(pc_dir, f"{case}_{jaw}.obj")
            label_path = os.path.join(pc_dir, f"{case}_{jaw}.json")

            if os.path.exists(obj_path) and os.path.exists(label_path):
                try:
                    processor.process_single_case(obj_path, label_path, save_dir)
                    total_processed += 1
                except Exception as e:
                    print(f"Error processing {case}_{jaw}: {e}")
            else:
                missing = []
                if not os.path.exists(obj_path):
                    missing.append(obj_path)
                if not os.path.exists(label_path):
                    missing.append(label_path)
                print(f"Skipping {case}_{jaw}: missing files {missing}")

    print(f"\nDone! Processed {total_processed} cases total.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess synthetic calculus meshes')
    parser.add_argument('--data_dir', required=True, type=str,
                        help='Input directory containing synthetic calculus meshes')
    parser.add_argument('--save_dir', required=True, type=str,
                        help='Output directory for preprocessed .npz files')
    args = parser.parse_args()

    batch_process_calculus(args.data_dir, args.save_dir)
