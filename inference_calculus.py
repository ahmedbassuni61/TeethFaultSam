import os
import argparse
import glob
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from model.calculus.calculus_system import CalculusSegmentationSystem
from config.config_calculus import CalculusConfig
from model.sam2.utils.transforms import SAM2Transforms

from pytorch3d.io import load_obj
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
from pytorch3d.structures import Meshes
import torch.nn.functional as F

class CalculusInferencePipeline:
    def __init__(self, checkpoint_path, config=None, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.config = config if config is not None else CalculusConfig()
        
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        
        self.model = CalculusSegmentationSystem(self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model.eval()
        
        self.transform = SAM2Transforms(
            resolution=1024,
            mask_threshold=0.0
        )
        
        self.views = [
            [10, 0], [10, 45], [10, 315], [320, 0],
            [315, 45], [315, 315], [60, 0],
        ]
        self.image_size = 512
        self.camera_distance = 70
        
        self.raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            cull_backfaces=True,
        )

    def _get_view_xyz(self):
        camera_positions = []
        light_directions = []
        for view in self.views:
            elev = view[0] * np.pi / 180.0
            azim = view[1] * np.pi / 180.0
            x = np.cos(elev) * np.sin(azim)
            y = np.sin(elev)
            z = np.cos(elev) * np.cos(azim)
            pos = np.array([x, y, z])
            camera_positions.append(pos)
            light_directions.append(pos)
        return np.stack(camera_positions, 0), np.stack(light_directions, 0)

    def _render_single_view(self, verts, faces, vertex_colors, view_R, view_T, light_direction):
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
        return rendered_image, cameras
        
    def _lift_to_3d(self, verts, faces, calculus_probs_2d, cameras_list):
        num_views = len(cameras_list)
        num_verts = verts.shape[0]
        
        accumulated_probs = torch.zeros(num_verts, device=self.device)
        weight_sum = torch.zeros(num_verts, device=self.device)
        
        mesh = Meshes(verts=[verts], faces=[faces])
        face_verts = faces.flatten()
        face_indices = torch.arange(len(faces), device=self.device).repeat_interleave(3)
        indices = torch.stack([face_verts, face_indices])
        values = torch.ones(len(face_verts), device=self.device)
        vert_to_faces_map = torch.sparse_coo_tensor(
            indices, values, 
            (num_verts, len(faces))
        ).coalesce()
        
        for i, camera in enumerate(cameras_list):
            rasterizer = MeshRasterizer(cameras=camera, raster_settings=self.raster_settings)
            fragments = rasterizer(mesh)
            
            face_idxs = fragments.pix_to_face[0, :, :, 0]
            visible_face_idxs = face_idxs[face_idxs >= 0].unique()
            
            if len(visible_face_idxs) == 0:
                continue
                
            visible_faces_mask = torch.zeros(len(faces), dtype=torch.float32, device=self.device)
            visible_faces_mask[visible_face_idxs] = 1.0
            
            vert_visibility = torch.sparse.mm(
                vert_to_faces_map, 
                visible_faces_mask.unsqueeze(1)
            ).squeeze(1)
            visible_verts = vert_visibility > 0
            
            projected = camera.transform_points_screen(verts.unsqueeze(0), image_size=(self.image_size, self.image_size))[0]
            xy = projected[:, :2]
            depths = projected[:, 2]
            
            valid_idx = torch.where(visible_verts & (xy[:, 0] >= 0) & (xy[:, 0] < self.image_size) & (xy[:, 1] >= 0) & (xy[:, 1] < self.image_size))[0]
            
            if len(valid_idx) == 0:
                continue
                
            valid_xy = xy[valid_idx]
            scaled_xy = valid_xy.clone()
            scaled_xy[:, 0] = 2.0 * (scaled_xy[:, 0] / (self.image_size - 1.0)) - 1.0
            scaled_xy[:, 1] = 2.0 * (scaled_xy[:, 1] / (self.image_size - 1.0)) - 1.0
            
            grid = scaled_xy.view(1, -1, 1, 2)
            
            probs = calculus_probs_2d[i].unsqueeze(0).unsqueeze(0).float() 
            sampled_probs = F.grid_sample(probs, grid, mode='bilinear', padding_mode='border', align_corners=True)
            sampled_probs = sampled_probs.view(-1)
            
            valid_depths = depths[valid_idx]
            depth_weight = 1.0 / valid_depths.clamp(min=1e-6)
            min_d, max_d = depth_weight.min(), depth_weight.max()
            if max_d > min_d:
                depth_weight = 1.0 + (depth_weight - min_d) / (max_d - min_d)
            else:
                depth_weight = torch.ones_like(depth_weight)
                
            accumulated_probs[valid_idx] += sampled_probs * depth_weight
            weight_sum[valid_idx] += depth_weight
            
        final_probs = accumulated_probs / weight_sum.clamp(min=1e-6)
        return final_probs.cpu().numpy()

    def predict_single(self, obj_path, json_path=None, output_dir=None):
        print(f"Processing {obj_path}...")
        verts, faces, _ = load_obj(obj_path)
        faces = faces.verts_idx
        verts = verts.to(self.device)
        faces = faces.to(self.device)
        center = verts.mean(dim=0)
        verts = verts - center
        
        default_colors = torch.ones_like(verts, dtype=torch.float32) * 0.75
        _, light_directions = self._get_view_xyz()
        
        all_images = []
        cameras_list = []
        
        for i, (view, light_dir) in enumerate(zip(self.views, light_directions)):
            R, T = look_at_view_transform(
                dist=self.camera_distance,
                elev=view[0],
                azim=view[1],
                device=self.device
            )
            img, cameras = self._render_single_view(verts, faces, default_colors, R, T, light_dir)
            all_images.append(img.cpu().numpy())
            cameras_list.append(cameras)
            
        images_np = np.stack(all_images, axis=0) # [N, H, W, 3]
        images_tensor = torch.from_numpy(images_np).permute(0, 3, 1, 2).to(self.device) # [N, 3, H, W]
        
        with torch.no_grad():
            if self.config.use_amp:
                with torch.autocast(device_type=self.device.type, dtype=self.config.get_torch_dtype()):
                    _, refined_masks, _ = self.model(images_tensor)
            else:
                _, refined_masks, _ = self.model(images_tensor)
                
            # Calculus class is index 1
            probs = F.softmax(refined_masks, dim=1)
            calculus_probs_2d = probs[:, 1, :, :] # [N, H, W]
            
        per_vertex_probs = self._lift_to_3d(verts, faces, calculus_probs_2d, cameras_list)
        per_vertex_labels = (per_vertex_probs >= 0.5).astype(int).tolist()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            case_name = Path(obj_path).stem
            out_json = os.path.join(output_dir, f"{case_name}.json")
            with open(out_json, 'w') as f:
                json.dump({
                    'fault_labels': per_vertex_labels,
                    'fault_type': 'calculus'
                }, f)
            print(f"Saved results to {out_json}")
            
        return per_vertex_labels

    def predict_batch(self, input_dir, output_dir):
        for jaw in ['lower', 'upper']:
            jaw_data_dir = os.path.join(input_dir, jaw)
            if not os.path.isdir(jaw_data_dir):
                jaw_data_dir = os.path.join(input_dir, 'calculus', jaw)
                if not os.path.isdir(jaw_data_dir):
                    continue
                    
            case_dirs = sorted([d for d in glob.glob(f"{jaw_data_dir}/*") if os.path.isdir(d)])
            for case_dir in tqdm(case_dirs, desc=f"Predicting {jaw}"):
                case = os.path.basename(case_dir)
                obj_path = os.path.join(case_dir, f"{case}_{jaw}.obj")
                
                if os.path.exists(obj_path):
                    current_output_dir = os.path.join(output_dir, jaw, case)
                    self.predict_single(obj_path, output_dir=current_output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="End-to-end inference for 3D calculus segmentation.")
    parser.add_argument('--input_dir', type=str, required=True, help="Path to input dataset.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to calculus checkpoint.")
    parser.add_argument('--output_dir', type=str, default='inference_results/calculus')
    args = parser.parse_args()
    
    pipeline = CalculusInferencePipeline(checkpoint_path=args.checkpoint)
    pipeline.predict_batch(args.input_dir, args.output_dir)
