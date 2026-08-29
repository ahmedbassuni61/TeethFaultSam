import numpy as np
from scipy.spatial import cKDTree

def generate_cavity(mesh, labels, jaw='upper', severity=0.08, max_cavities=3):
    modified_mesh = mesh.copy()
    fault_mask = np.zeros(len(mesh.vertices), dtype=bool)

    unique_teeth = np.unique(labels[labels > 0])
    if len(unique_teeth) == 0:
        return modified_mesh, fault_mask

    num_cavities = np.random.randint(1, max_cavities + 1)
    target_teeth = np.random.choice(
        unique_teeth,
        size=min(num_cavities, len(unique_teeth)),
        replace=False
    )

    tree = cKDTree(mesh.vertices)

    for target_tooth in target_teeth:
        tooth_indices = np.where(labels == target_tooth)[0]
        z_coords = mesh.vertices[tooth_indices, 2]
        
        if jaw == 'upper':
            crown_threshold = np.percentile(z_coords, 20)
            crown_indices = tooth_indices[z_coords < crown_threshold]
        else:
            crown_threshold = np.percentile(z_coords, 80)
            crown_indices = tooth_indices[z_coords > crown_threshold]

        if len(crown_indices) == 0:
            crown_indices = tooth_indices

        center_idx = np.random.choice(crown_indices)
        center = mesh.vertices[center_idx]
        center_normal = mesh.vertex_normals[center_idx]

        radius = max(2.0, 60.0 * severity)
        max_depth = max(1.0, 30.0 * severity)

        indices = tree.query_ball_point(center, r=radius)

        if len(indices) > 0:
            idx_array = np.array(indices)
            idx_array = idx_array[labels[idx_array] > 0]

            if len(idx_array) > 0:
                distances = np.linalg.norm(mesh.vertices[idx_array] - center, axis=1)
                sigma = radius / 2.5
                decay = np.exp(-(distances**2) / (2 * sigma**2))
                displacement = -center_normal * decay[:, np.newaxis] * max_depth
                modified_mesh.vertices[idx_array] += displacement
                fault_mask[idx_array] = True

    return modified_mesh, fault_mask

def generate_crack(mesh, labels, jaw='upper', depth=1.2, length=12.0, width=0.8):
    modified_mesh = mesh.copy()
    fault_mask = np.zeros(len(mesh.vertices), dtype=bool)

    teeth_indices = np.where(labels > 0)[0]
    gum_indices = np.where(labels == 0)[0]
    unique_teeth = np.unique(labels[labels > 0])

    if len(unique_teeth) == 0:
        return modified_mesh, fault_mask

    if len(gum_indices) > 0:
        mean_teeth_z = np.mean(mesh.vertices[teeth_indices, 2])
        mean_gum_z = np.mean(mesh.vertices[gum_indices, 2])
        teeth_point_up = mean_teeth_z > mean_gum_z
    else:
        teeth_point_up = (jaw == 'lower')

    num_affected = np.random.randint(2, 6)
    target_teeth = np.random.choice(unique_teeth, size=min(num_affected, len(unique_teeth)), replace=False)

    tree = cKDTree(mesh.vertices)

    for target_tooth in target_teeth:
        tooth_indices = np.where(labels == target_tooth)[0]
        z_coords = mesh.vertices[tooth_indices, 2]

        if teeth_point_up:
            crown_threshold = np.percentile(z_coords, 60)
            crown_indices = tooth_indices[z_coords > crown_threshold]
        else:
            crown_threshold = np.percentile(z_coords, 40)
            crown_indices = tooth_indices[z_coords < crown_threshold]

        if len(crown_indices) == 0:
            crown_indices = tooth_indices

        center_idx = np.random.choice(crown_indices)
        start_pt = mesh.vertices[center_idx]
        center_normal = mesh.vertex_normals[center_idx]

        tangent = np.cross(center_normal, [0, 0, 1])
        if np.linalg.norm(tangent) < 1e-5:
            tangent = np.cross(center_normal, [0, 1, 0])
        tangent /= np.linalg.norm(tangent)
        direction = tangent

        indices = tree.query_ball_point(start_pt, r=length)

        if len(indices) > 0:
            idx_array = np.array(indices)
            idx_array = idx_array[labels[idx_array] > 0]

            if len(idx_array) > 0:
                pts = mesh.vertices[idx_array]
                vecs = pts - start_pt
                projs = np.dot(vecs, direction)
                dists = np.linalg.norm(vecs - np.outer(projs, direction), axis=1)

                jagged_noise = np.sin(projs * 3.0) * 0.1
                effective_dists = np.abs(dists + jagged_noise)

                mask = (projs > -length/2) & (projs < length/2) & (effective_dists < width)
                crack_indices = idx_array[mask]

                if len(crack_indices) > 0:
                    profile_decay = 1.0 - (effective_dists[mask] / width)
                    length_decay = 1.0 - np.abs(projs[mask]) / (length/2)
                    combined_decay = np.clip(profile_decay * length_decay, 0, 1)

                    displacement = -center_normal * combined_decay[:, np.newaxis] * depth
                    modified_mesh.vertices[crack_indices] += displacement
                    fault_mask[crack_indices] = True

    return modified_mesh, fault_mask

def generate_abrasion(mesh, labels, jaw='upper', wear_level=0.5):
    modified_mesh = mesh.copy()
    fault_mask = np.zeros(len(mesh.vertices), dtype=bool)

    teeth_indices = np.where(labels > 0)[0]
    gum_indices = np.where(labels == 0)[0]
    unique_teeth = np.unique(labels[labels > 0])
    
    if len(unique_teeth) == 0:
        return modified_mesh, fault_mask

    if len(gum_indices) > 0:
        mean_teeth_z = np.mean(mesh.vertices[teeth_indices, 2])
        mean_gum_z = np.mean(mesh.vertices[gum_indices, 2])
        teeth_point_up = mean_teeth_z > mean_gum_z
    else:
        teeth_point_up = (jaw == 'lower')

    num_teeth_to_abrade = np.random.randint(2, 5)
    target_teeth = np.random.choice(unique_teeth, size=min(num_teeth_to_abrade, len(unique_teeth)), replace=False)

    for target_tooth in target_teeth:
        tooth_indices = np.where(labels == target_tooth)[0]
        z_coords = mesh.vertices[tooth_indices, 2]

        max_z = np.max(z_coords)
        min_z = np.min(z_coords)
        height = max_z - min_z

        if teeth_point_up:
            clip_z = max_z - wear_level * (height * 0.15)
            clip_mask = (mesh.vertices[:, 2] > clip_z) & (labels == target_tooth)
        else:
            clip_z = min_z + wear_level * (height * 0.15)
            clip_mask = (mesh.vertices[:, 2] < clip_z) & (labels == target_tooth)

        valid_clip_indices = np.where(clip_mask)[0]

        if len(valid_clip_indices) > 0:
            modified_mesh.vertices[valid_clip_indices, 2] = clip_z
            fault_mask[valid_clip_indices] = True

    return modified_mesh, fault_mask

def generate_calculus(mesh, labels, jaw='upper', buildup=0.05):
    modified_mesh = mesh.copy()
    fault_mask = np.zeros(len(mesh.vertices), dtype=bool)

    teeth_indices = np.where(labels > 0)[0]
    gum_indices = np.where(labels == 0)[0]

    if len(teeth_indices) == 0:
        return modified_mesh, fault_mask

    if len(gum_indices) > 0:
        mean_teeth_z = np.mean(mesh.vertices[teeth_indices, 2])
        mean_gum_z = np.mean(mesh.vertices[gum_indices, 2])
        teeth_point_up = mean_teeth_z > mean_gum_z
    else:
        teeth_point_up = (jaw == 'lower')

    z_coords = mesh.vertices[teeth_indices, 2]

    if teeth_point_up:
        gumline_threshold = np.percentile(z_coords, 20)
        gumline_indices = teeth_indices[z_coords < gumline_threshold]
    else:
        gumline_threshold = np.percentile(z_coords, 80)
        gumline_indices = teeth_indices[z_coords > gumline_threshold]

    if len(gumline_indices) == 0:
        gumline_indices = teeth_indices

    available_teeth = np.unique(labels[gumline_indices])

    num_affected = np.random.randint(2, 7)
    target_teeth = np.random.choice(available_teeth, size=min(num_affected, len(available_teeth)), replace=False)

    tree = cKDTree(mesh.vertices)

    for chosen_tooth in target_teeth:
        tooth_gumline_indices = gumline_indices[labels[gumline_indices] == chosen_tooth]

        if len(tooth_gumline_indices) == 0:
            continue

        center_idx = np.random.choice(tooth_gumline_indices)
        center = mesh.vertices[center_idx]

        radius = buildup * 60
        indices = tree.query_ball_point(center, r=radius)

        if len(indices) > 0:
            idx_array = np.array(indices)
            idx_array = idx_array[labels[idx_array] > 0]

            if len(idx_array) > 0:
                distances = np.linalg.norm(mesh.vertices[idx_array] - center, axis=1)
                noise = np.random.uniform(0.2, 1.8, size=len(idx_array))
                decay = np.exp(-(distances**2) / (2 * (radius/3)**2)) * noise

                normals = mesh.vertex_normals[idx_array]
                max_height = buildup * 10

                displacement = normals * decay[:, np.newaxis] * max_height
                modified_mesh.vertices[idx_array] += displacement
                fault_mask[idx_array] = True

    return modified_mesh, fault_mask
