# Fault Generators

This document details the four specific fault generation functions implemented in the synthetic faults pipeline. All functions operate on `trimesh` objects and utilize `scipy.spatial.cKDTree` for efficient spatial queries.

## 1. Cavities (`generate_cavity`)
Generates synthetic cavities on the crowns of multiple random teeth.
- **Mechanism:** Locates the crown based on jaw orientation. It selects a center point, calculates a dynamic radius based on severity, and displaces vertices inward along the normal using a Gaussian decay.
- **Parameters:** 
  - `jaw`: 'upper' or 'lower' to determine orientation.
  - `severity`: Controls the size (radius) and depth of the cavity.
  - `max_cavities`: Maximum number of teeth that will get a cavity.

## 2. Cracks (`generate_crack`)
Simulates cracks on the crowns of multiple random teeth.
- **Mechanism:** Determines a crack direction tangent to the normal vector of a chosen crown point. It applies jagged noise (`np.sin(projs * 3.0) * 0.1`) to make the fissure organic, and displaces vertices inward based on a combined width and length decay.
- **Parameters:**
  - `depth`: Depth of the crack.
  - `length`: Length of the crack.
  - `width`: Width of the crack.

## 3. Abrasion (`generate_abrasion`)
Simulates bruxism/wear by planar clipping of the occlusal plane on multiple teeth.
- **Mechanism:** Automatically detects if teeth point up or down. It selects a group of teeth and clips the highest (or lowest) Z-coordinates, effectively flattening the crowns.
- **Parameters:**
  - `wear_level`: Controls the percentage of the tooth height that is clipped.

## 4. Calculus / Tartar (`generate_calculus`)
Simulates tartar buildup via localized outward displacement near the gumline.
- **Mechanism:** Finds the global gumline by calculating Z-coordinate percentiles (20th or 80th depending on jaw orientation). It applies random noise and a Gaussian decay to displace vertices outward, mimicking tartar buildup.
- **Parameters:**
  - `buildup`: Controls the radius and maximum height of the outward displacement.
