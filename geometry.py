import numpy as np
import vtk
from typing import List, Any, Callable
import math
import numpy as np
from scipy.ndimage import distance_transform_edt, sobel, convolve, gaussian_filter
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
import SimpleITK as sitk
from sklearn.cluster import DBSCAN
from scipy.signal import butter, filtfilt


def voxel_to_world(image, voxel_coords, origin_sign=True):
    """
    Convert voxel coordinates to physical (world) coordinates in batch.
    
    Parameters:
        - image: SimpleITK.Image object
            The medical image containing metadata such as origin, spacing, and direction.
        - voxel_coords: numpy array of shape (n, 3) or (3,)
            A set of voxel coordinates (i, j, k) representing positions in the image grid.
        - origin_sign: bool, optional (default=True)
            If True, the origin offset will be added to the transformed coordinates.
            If False, only spacing and direction are applied.
    
    Returns:
        - world_coords: numpy array of shape (n, 3)
            The corresponding physical (world) coordinates in real-world space.
    """
    # Ensure voxel_coords is a NumPy array
    voxel_coords = np.atleast_2d(voxel_coords)  # Ensure (n, 3) shape

    # Extract image properties: origin, spacing, and direction matrix.
    origin = np.array(image.GetOrigin())  # (3,)
    spacing = np.array(image.GetSpacing())  # (3,)
    direction = np.array(image.GetDirection()).reshape(3, 3)  # (3, 3)

    # Transform voxel coordinates to physical coordinates (batch processing)
    world_coords = (voxel_coords[:, ::-1] * spacing) @ direction.T  # Apply spacing and direction

    # Add origin offset if origin_sign is True
    if origin_sign:
        world_coords += origin  # Broadcasting to (n, 3)

    return world_coords


def downsample_point_cloud(points: np.ndarray, max_points_num: int) -> np.ndarray:
    """
    Downsamples a point cloud to a target number of points using a voxel grid filter.

    If the number of points in the input cloud exceeds max_points_num, this function
    calculates an appropriate voxel size to reduce the point count to approximately
    the target number. All points within each active voxel are replaced by their centroid.

    Args:
        points (np.ndarray): 
            The input point cloud, an array of shape (N, 3), where N is the number of points.
        max_points_num (int): 
            The target maximum number of points for the downsampled cloud.

    Returns:
        np.ndarray: 
            The downsampled point cloud, with a shape of (M, 3), where M is approximately
            equal to or less than max_points_num. If no downsampling is needed, the
            original point cloud is returned.
    """
    # --- Step 1: Check if downsampling is necessary ---
    num_points = points.shape[0]
    if num_points <= max_points_num:
        # If the point cloud is already small enough, return it as is.
        return points

    # --- Step 2: Estimate the required voxel size ---
    # Calculate the bounding box volume of the point cloud.
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    bounding_box_volume = np.prod(max_coords - min_coords)
    
    # Handle the case of a flat or co-linear point cloud (zero volume).
    if bounding_box_volume == 0:
        # If volume is zero, we cannot estimate voxel size. Fallback to random sampling.
        # This is a robust fallback for degenerate cases.
        random_indices = np.random.choice(num_points, size=max_points_num, replace=False)
        return points[random_indices]

    # Estimate the voxel size needed to get the target number of points.
    # The idea is that the number of occupied voxels is related to the total volume
    # divided by the volume of a single voxel.
    voxel_volume_target = bounding_box_volume / max_points_num
    voxel_size = voxel_volume_target ** (1/3)

    # --- Step 3: Perform Voxel Grid Downsampling ---
    # Quantize point coordinates into integer voxel indices.
    # This assigns each point to a specific voxel in the grid.
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)

    # Use a dictionary to group points by their unique voxel index.
    # The key is the tuple of the voxel index, the value is a list of points in that voxel.
    voxel_dict = {}
    for i in range(num_points):
        key = tuple(voxel_indices[i])
        if key not in voxel_dict:
            voxel_dict[key] = []
        voxel_dict[key].append(points[i])

    # --- Step 4: Calculate the centroid for each voxel ---
    # The downsampled point cloud will consist of the centroids of all non-empty voxels.
    downsampled_points = np.array([
        np.mean(voxel_points, axis=0) for voxel_points in voxel_dict.values()
    ])

    return downsampled_points



def get_backlit_points(points, view_point, degree=1):
    """
    Identify backlit points from a set of 3D points relative to a specified viewpoint.

    This function calculates which points from the input are backlit (opposite the light source) when viewed 
    from a given viewpoint, considering a specified angle tolerance.

    Parameters:
    points (ndarray): An array of 3D points (shape: Nx3).
    view_point (ndarray): A single 3D viewpoint from which the points are observed (shape: 3,).
    degree (float): The angle in degrees used as a threshold to determine backlit points. Default is 2.5 degrees.

    Returns:
    ndarray: An array of points that are considered backlit.
    """
    # Calculate direction vectors from the viewpoint to each point
    view_direc = points - view_point

    # Compute the norm (magnitude) of each direction vector
    norms = np.linalg.norm(view_direc, axis=1)

    # Normalize direction vectors to unit vectors
    normalized_view_direc = view_direc / norms[:, np.newaxis]

    # Compute the "convolution" matrix as the dot product of direction vectors
    conv_mtx = np.dot(normalized_view_direc, normalized_view_direc.T)

    # Calculate the threshold for backlit condition using the given degree
    threshold = np.sin(np.deg2rad((90 - degree)))

    # Determine pairs of indices where the backlit condition is satisfied
    indices = np.where(conv_mtx > threshold)  # (row_indices, column_indices)

    backlit_indices = []
    checked_indices = []

    # Iterate over each point to determine if it is backlit
    for i in range(view_direc.shape[0]):
        if i in checked_indices:
            continue

        # Find indices of points deemed backlit relative to the current point
        tmp_indices = np.where(indices[0] == i)[0]
        checked_indices += indices[1][tmp_indices].tolist()

        # Determine the point with the maximum norm (distance)
        tmp_index = np.argmax(norms[indices[1][tmp_indices]])
        tmp_index = indices[1][tmp_indices[tmp_index]]

        # Add to backlit indices if not already included
        if tmp_index not in backlit_indices:
            backlit_indices.append(tmp_index)

    # Convert list of backlit indices to a numpy array
    backlit_indices = np.array(backlit_indices)

    # Return the backlit points
    return points[backlit_indices], backlit_indices



def calculate_surface_normals(image_array, sigma=None):
    """
    Computes surface normals for the target regions in a 3D binary image, optionally using Gaussian smoothing.

    Args:
        image_array (ndarray): A 3D binary NumPy array representing the scene, where 1 indicates the target areas 
                               and other values indicate non-target regions.
        sigma (float, optional): Standard deviation for Gaussian kernel used in smoothing. 
    
    Returns:
        ndarray: A 3D array of normal vectors corresponding to the target region's surface. Each vector is normalized 
                 to represent the local outward surface direction.
    """
    # Apply Gaussian smoothing if sigma is provided
    if sigma is not None:
        image_array = gaussian_filter(image_array.astype(float), sigma=sigma)
    
    # Compute gradients
    grad_x, grad_y, grad_z = np.gradient(image_array)

    # Compute magnitude of gradient
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2 + 1e-6)

    # Normalize the gradients to create unit normal vectors, adding a small epsilon to avoid division by zero
    grad_x /= grad_magnitude + 1e-6
    grad_y /= grad_magnitude + 1e-6
    grad_z /= grad_magnitude + 1e-6

    # Flip normals that are inward (facing target area), assuming higher values indicate inside
    normals = np.stack((grad_x, grad_y, grad_z), axis=-1)
    
    # Establish a mask for the borders
    border_mask = image_array > 0.5  # Slightly greater than 0 if smoothed
    normals[border_mask] *= -1  # Invert direction of normals inside the object

    return normals



def get_surface_points(image_array, target_val, obs_val, back_val):
    """
    Identifies and returns the coordinates of the surface points of the target region.

    Args:
        image_array (ndarray): A 3D NumPy array where:
            - Target region is marked with 'target_val',
            - Background is marked with 'back_val',
            - Obstacles are marked with 'obs_val'.

    Returns:
        surface_points (list): A list of coordinates (x, y, z) representing the surface points.
    """
    # Create a binary mask for the target region where the value equals 'target_val'
    target_region = (image_array == target_val)

    # Create a binary mask where the value is either 'back_val' or 'obs_val' indicating background or obstacles
    background_or_obstacle = np.isin(image_array, [back_val, obs_val])

    # Use a 3x3x3 kernel to examine neighboring voxels and identify adjacency
    kernel = np.ones((3, 3, 3), dtype=int)  # Kernel to evaluate all neighboring voxels

    # Count the number of neighboring voxels for each voxel in the target region
    # This helps to determine adjacency to background or obstacle areas
    neighbor_count = convolve(target_region.astype(int), kernel, mode='constant', cval=0)

    # Surface points are identified as those in the target region adjacent to any background or obstacle
    surface_points = np.argwhere((neighbor_count > 0) & target_region)

    return surface_points




def ray_tracing(image_array, surface_points, normals, obs_val, angle_range, ref_direc=None, epsilon=0.8, min_samples=1):
    """
    Cast rays from surface points along their respective normal directions and discard
    rays that intersect obstacles. Merge nearby rays to produce a sparser set of rays.

    Args:
        image_array (ndarray): A 3D NumPy array where:
            - Target regions are marked with `1`,
            - Background is marked with `0`,
            - Obstacles are marked with `-1`.
        
        surface_points (list): List of (x, y, z) coordinates that define the surface points
                               of the target region.

        normals (ndarray): A 3D array containing the normal vectors for each surface point.

        obs_val (int or float): Value representing obstacles in the image array.

        angle_range (float): The acceptable range of angles (in degrees) between the normal
                             and a reference direction.

        ref_direc (array-like, optional): A reference direction vector for angle checking.

        epsilon (float): Maximum distance for two rays to be considered in the same neighborhood.

        min_samples (int): Minimum number of rays needed to form a cluster.

    Returns:
        list: A collection of representative valid rays, each shown as a path of coordinate points
              from the surface to the background.
    """
    raw_rays = []

    # Process each surface point and trace a ray in the direction of its normal vector
    for point in surface_points:
        x, y, z = point  # Extract surface point coordinates
        normal = normals[x, y, z]  # Obtain the normal vector at this point
        if np.linalg.norm(normal) == 0:  # Skip if the normal vector magnitude is zero
            continue

        # Flag indicating whether the ray remains valid
        available = True
        t = 0  # Initialize the parameter for the ray equation

        # Check angle range condition if a reference direction is specified
        if ref_direc is not None:
            angle = calculate_angle_between_vectors(normal, ref_direc)
            if angle > angle_range:
                continue

        while True:
            # Determine the next point along the ray by advancing along the normal direction
            next_point = np.array([x, y, z]) + t * normal
            next_x, next_y, next_z = next_point.astype(int)  # Convert to integer indices

            # Terminate the ray if it exceeds image bounds
            if not (0 <= next_x < image_array.shape[0] and 0 <= next_y < image_array.shape[1] and 0 <= next_z < image_array.shape[2]):
                break

            # Stop the ray if it intersects an obstacle
            if image_array[next_x, next_y, next_z] == obs_val:
                available = False
                break

            # Progress to the next point along the ray
            t += 1

        # Save the raw ray data for later clustering if it is valid
        if available:
            raw_rays.append([x, y, z, normal[0], normal[1], normal[2]])

    # Convert raw rays to numpy array for clustering
    ray_coords = np.array(raw_rays)
    
    # Cluster the rays to reduce density
    if len(ray_coords) > 0:
        clustering = DBSCAN(eps=epsilon, min_samples=min_samples).fit(ray_coords[:, :3])
        sparse_rays = []
        
        for cluster in set(clustering.labels_):
            if cluster == -1:
                # Noise points or outliers
                continue
            indices = np.where(clustering.labels_ == cluster)[0]
            # Calculate the mean of the cluster to represent as a single sparse ray
            mean_point = np.mean(ray_coords[indices, :3], axis=0)
            mean_normal = np.mean(ray_coords[indices, 3:], axis=0)

            sparse_rays.append((mean_point, mean_normal))
    else:
        sparse_rays = []

    return sparse_rays



def get_rays_from_img_array(image_array, target_val, obs_val, back_val, angle_range, ref_direc, sigma=None):
    """
    Compute and trace rays within a 3D image array by analyzing surface normals of the target region.

    Args:
        image_array (ndarray): A 3D binary NumPy array representing the scene:
            - `1` marks the target areas,
            - `0` indicates background,
            - `-1` identifies obstacles.

        target_val (int or float): The value assigned to the target region.

        obs_val (int or float): The value representing obstacles in the array.

        back_val (int or float): The value representing background regions.

        angle_range (float): The allowed angular deviation in degrees when comparing rays to the 
                             reference direction.

        ref_direc (array-like): A reference direction vector used to evaluate the alignment of normal 
                                vectors and determine valid rays.

    Returns:
        list: A list of valid rays, each expressed as a sequence of 3D coordinates.
    """
    # Extract surface coordinates and their corresponding normal vectors
    normals = calculate_surface_normals(image_array, sigma)
    surface_points = get_surface_points(image_array, target_val, obs_val, back_val)

    # Perform ray tracing to identify and retrieve valid rays
    rays = ray_tracing(image_array, surface_points, normals, obs_val, angle_range, ref_direc)

    return rays




def calculate_angle_between_vectors(vector_a, vector_b):
    """
    Calculate the angle in degrees between two vectors using the dot product.

    Parameters:
    vector_a (numpy.ndarray): The first vector.
    vector_b (numpy.ndarray): The second vector.

    Returns:
    float: The angle between the two vectors in degrees.
    """
    # Ensure the vectors are numpy arrays
    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)
    
    # Calculate the dot product of the two vectors
    dot_product = np.dot(vector_a, vector_b)
    
    # Calculate the norms of the vectors
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    
    # Calculate the cosine of the angle using the dot product formula
    cos_theta = dot_product / (norm_a * norm_b)
    
    # Ensure the cosine value is within the valid range for arccos
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # Calculate the angle in radians
    angle_radians = np.arccos(cos_theta)
    
    # Convert the angle to degrees
    angle_degrees = np.degrees(angle_radians)
    
    return angle_degrees



def generate_dense_rays_from_foreground_with_normals(mask, theta_step, phi_step, obstacle_value):
    """
    Generate rays from foreground regions (1) in a binary mask (0 for background),
    with directions based on surface normals. Merges overlapping rays, and excludes
    rays that pass through obstacles (with value `obstacle_value`).

    Parameters:
        mask (np.ndarray): A 3D numpy array where 1 represents foreground (tumor),
                            and 0 represents background.
        theta_step (int): The step size for the polar angle (θ), in degrees. Default is 10 degrees.
        phi_step (int): The step size for the azimuthal angle (φ), in degrees. Default is 10 degrees.
        obstacle_value (int or float): The value representing obstacles in the mask (default is -1).

    Returns:
        list: A list of rays, each represented by a tuple (start_point, direction).
              - start_point is a tuple (x, y, z) representing the start of the ray.
              - direction is a tuple (dx, dy, dz) representing the direction of the ray.
    """
    # Convert angle steps from degrees to radians
    theta_step = np.radians(theta_step)
    phi_step = np.radians(phi_step)
    
    # Get indices of all foreground (1) voxels in the 3D mask
    foreground_indices = np.argwhere(mask == 1)
    
    rays = set()  # Use a set to store unique rays (start point, direction)
    
    # Generate spherical coordinates: theta from 0 to pi, phi from 0 to 2pi
    theta_vals = np.arange(0, np.pi, theta_step)  # Polar angle from 0 to pi
    phi_vals = np.arange(0, 2 * np.pi, phi_step)  # Azimuthal angle from 0 to 2pi
    
    # Create a meshgrid for all combinations of theta and phi
    theta_grid, phi_grid = np.meshgrid(theta_vals, phi_vals)
    
    # Convert spherical coordinates (theta, phi) to Cartesian directions (dx, dy, dz)
    dx = np.sin(theta_grid) * np.cos(phi_grid)
    dy = np.sin(theta_grid) * np.sin(phi_grid)
    dz = np.cos(theta_grid)
    
    # Flatten the direction arrays to create a list of directions
    directions = np.vstack((dx.flatten(), dy.flatten(), dz.flatten())).T

    # Function to compute the normal vector at a point (simple gradient approximation)
    def compute_normal(voxel, mask):
        x, y, z = voxel
        # Create a small 3x3x3 neighborhood around the point
        neighborhood = mask[max(0, x-1):x+2, max(0, y-1):y+2, max(0, z-1):z+2]
        
        # Simple gradient approximation (difference between neighboring points)
        grad_x = np.sum(neighborhood[1, 1, 1] - neighborhood[0, 1, 1])  # X-gradient
        grad_y = np.sum(neighborhood[1, 1, 1] - neighborhood[1, 0, 1])  # Y-gradient
        grad_z = np.sum(neighborhood[1, 1, 1] - neighborhood[1, 1, 0])  # Z-gradient
        
        # Return the normalized normal vector
        normal = np.array([grad_x, grad_y, grad_z])
        return normal / np.linalg.norm(normal)  # Normalize the vector

    # Iterate through all foreground voxels to generate rays
    for voxel in foreground_indices:
        x, y, z = voxel  # The current foreground voxel (start point of the ray)
        
        # Compute the normal vector at the surface point (the foreground voxel)
        normal = compute_normal(voxel, mask)

        # Normalize the direction (normal vector) for ray direction
        normal_tuple = tuple(normal)
        
        # Check if a ray with this direction already exists (merge overlapping rays)
        is_new_ray = True
        for existing_ray_start, existing_ray_direction in rays:
            if existing_ray_direction == normal_tuple:
                # Check if the two start points are collinear along the ray's direction
                vector_between = np.array([x, y, z]) - np.array(existing_ray_start)
                if are_collinear(vector_between, normal):
                    is_new_ray = False
                    break
        
        # If it's a new ray and not blocked by obstacles, add it to the set
        if is_new_ray and not is_ray_blocked((x, y, z), normal, mask, obstacle_value):
            rays.add(((x, y, z), normal_tuple))

            # Consider the reverse direction as well (negated normal vector)
            reverse_normal_tuple = tuple(-normal)
            
            # Check if a ray with this reverse direction already exists
            is_new_ray = True
            for existing_ray_start, existing_ray_direction in rays:
                if existing_ray_direction == reverse_normal_tuple:
                    # Check if the two start points are collinear along the ray's direction
                    vector_between = np.array([x, y, z]) - np.array(existing_ray_start)
                    if are_collinear(vector_between, normal):
                        is_new_ray = False
                        break
            
            # If it's a new reverse ray and not blocked by obstacles, add it to the set
            if is_new_ray and not is_ray_blocked((x, y, z), normal, mask, obstacle_value):
                rays.add(((x, y, z), reverse_normal_tuple))

    return list(rays)


# Function to check if two vectors are collinear
def are_collinear(v1, v2):
    """
    Check if two vectors v1 and v2 are collinear by verifying if their cross product is zero.
    """
    return np.allclose(np.cross(v1, v2), 0)


# Function to check if a ray passes through an obstacle
def is_ray_blocked(start, direction, mask, obstacle_value):
    """
    Check if a ray, starting at 'start' and moving in the direction 'direction',
    is blocked by an obstacle (value == obstacle_value) within the mask.
    """
    # Step size for checking along the ray
    step_size = 1  # Step size for checking along the ray
    max_steps = 100  # Maximum number of steps to prevent infinite loops

    for step in range(max_steps):
        point = np.array(start) + step * np.array(direction) * step_size
        # Ensure the point is within bounds
        if np.any(point < 0) or np.any(point >= np.array(mask.shape)):
            break
        # Check if the point is an obstacle
        if mask[tuple(np.round(point).astype(int))] == obstacle_value:
            return True  # Ray blocked by obstacle
    return False  # Ray not blocked




# def generate_dense_rays_from_foreground_with_obs(mask, theta_step, phi_step, obstacle_value):
#     """
#     Generate rays from foreground regions (1) in a binary mask (0 for background),
#     with dense ray directions using spherical coordinates. Merges overlapping rays,
#     and excludes rays that pass through obstacles (with value `obstacle_value`).

#     This function generates rays from each foreground voxel in the mask in all directions,
#     and avoids rays that are blocked by obstacles or overlapping with other rays.

#     Parameters:
#         mask (np.ndarray): A 3D numpy array where 1 represents foreground and 0 represents background.
#         theta_step (int): The step size for the polar angle (θ), in degrees. Default is 10 degrees.
#         phi_step (int): The step size for the azimuthal angle (φ), in degrees. Default is 10 degrees.
#         obstacle_value (int or float): The value representing obstacles in the mask (default is -1).

#     Returns:
#         list: A list of rays, each represented by a tuple (start_point, direction).
#               - start_point is a tuple (x, y, z) representing the start of the ray.
#               - direction is a tuple (dx, dy, dz) representing the direction of the ray.
#     """
#     # Convert angle steps from degrees to radians
#     theta_step = np.radians(theta_step)
#     phi_step = np.radians(phi_step)
    
#     # Get indices of all foreground (1) voxels in the 3D mask
#     foreground_indices = np.argwhere(mask == 1)
    
#     rays = set()  # Use a set to store unique rays (start point, direction)
    
#     # Generate spherical coordinates: theta from 0 to pi, phi from 0 to 2pi
#     theta_vals = np.arange(0, np.pi, theta_step)  # Polar angle from 0 to pi
#     phi_vals = np.arange(0, 2 * np.pi, phi_step)  # Azimuthal angle from 0 to 2pi
    
#     # Create a meshgrid for all combinations of theta and phi
#     theta_grid, phi_grid = np.meshgrid(theta_vals, phi_vals)
    
#     # Convert spherical coordinates (theta, phi) to Cartesian directions (dx, dy, dz)
#     dx = np.sin(theta_grid) * np.cos(phi_grid)
#     dy = np.sin(theta_grid) * np.sin(phi_grid)
#     dz = np.cos(theta_grid)
    
#     # Flatten the direction arrays to create a list of directions
#     directions = np.vstack((dx.flatten(), dy.flatten(), dz.flatten())).T

#     # Iterate through all foreground voxels to generate rays
#     for voxel in foreground_indices:
#         x, y, z = voxel  # The current foreground voxel (start point of the ray)
        
#         # For each foreground voxel, generate rays in the directions
#         for direction in directions:
#             direction = direction / np.linalg.norm(direction)
#             direction_tuple = tuple(direction)
            
#             # Check if a ray with this direction already exists (merge overlapping rays)
#             is_new_ray = True
#             for existing_ray_start, existing_ray_direction in rays:
#                 if existing_ray_direction == direction_tuple:
#                     # Check if the two start points are collinear along the ray's direction
#                     vector_between = np.array([x, y, z]) - np.array(existing_ray_start)
#                     if are_collinear(vector_between, direction):
#                         is_new_ray = False
#                         break
            
#             # If it's a new ray and not blocked by obstacles, add it to the set
#             if is_new_ray and not is_ray_blocked((x, y, z), direction, mask, obstacle_value):
#                 rays.add(((x, y, z), direction_tuple))
                
#                 # Consider the reverse direction as well (negated direction)
#                 direction_tuple = tuple(-direction)
                
#                 # Check if a ray with this reverse direction already exists
#                 is_new_ray = True
#                 for existing_ray_start, existing_ray_direction in rays:
#                     if existing_ray_direction == direction_tuple:
#                         # Check if the two start points are collinear along the ray's direction
#                         vector_between = np.array([x, y, z]) - np.array(existing_ray_start)
#                         if are_collinear(vector_between, direction):
#                             is_new_ray = False
#                             break
                
#                 # If it's a new reverse ray and not blocked by obstacles, add it to the set
#                 if is_new_ray and not is_ray_blocked((x, y, z), direction, mask, obstacle_value):
#                     rays.add(((x, y, z), direction_tuple))

#     return list(rays)


# def is_ray_blocked(start, direction, mask, obstacle_value):
#     """
#     Check if a ray, starting at 'start' and moving in the direction 'direction',
#     is blocked by an obstacle (value == obstacle_value) within the mask.
    
#     Parameters:
#         start (tuple): The starting point of the ray as (x, y, z).
#         direction (tuple): The direction of the ray as (dx, dy, dz).
#         mask (np.ndarray): The 3D mask array representing the scene.
#         obstacle_value (int or float): The value representing obstacles in the mask.

#     Returns:
#         bool: True if the ray is blocked by an obstacle, False otherwise.
#     """
#     # Step size for checking along the ray
#     step_size = 1  # Distance between each step along the ray
#     max_steps = 100  # Maximum number of steps to prevent infinite loops
    
#     # Check along the ray's path
#     for step in range(max_steps):
#         # Calculate the point along the ray
#         point = np.array(start) + step * np.array(direction) * step_size
        
#         # Ensure the point is within bounds of the mask
#         if np.any(point < 0) or np.any(point > np.array(mask.shape) - 1):
#             break
        
#         # Round the point and check if it hits an obstacle
#         if mask[tuple(np.round(point).astype(int))] == obstacle_value:
#             return True  # Ray is blocked by an obstacle
    
#     return False  # Ray is not blocked


# def are_collinear(v1, v2):
#     """
#     Check if two vectors v1 and v2 are collinear by verifying if their cross product is zero.
    
#     Parameters:
#         v1 (np.ndarray): The first vector.
#         v2 (np.ndarray): The second vector.

#     Returns:
#         bool: True if the vectors are collinear, False otherwise.
#     """
#     return np.allclose(np.cross(v1, v2), 0)



def compute_convex_hull_mask_from_array(input_array):
    """
    Compute the convex hull mask for a 3D binary array.
    
    Args:
        input_array (numpy.ndarray): A 3D NumPy array with binary values (0 and 1),
                                     where 1 represents the foreground and 0 represents the background.

    Returns:
        numpy.ndarray: A 3D NumPy array (same shape as input) where the convex hull of the foreground (value=1)
                       is marked as 1, and the rest is 0.
    """
    # Step 1: Extract foreground points (where input_array == 1)
    foreground_points = np.argwhere(input_array == 1)  # Coordinates of all foreground points
    
    if len(foreground_points) == 0:
        # If no foreground points exist, return a zero array
        return np.zeros_like(input_array, dtype=np.uint8)
    
    # Step 2: Compute the convex hull of the foreground points
    hull = ConvexHull(foreground_points)  # Calculate the convex hull using the points
    
    # Step 3: Generate a mask for the convex hull
    # Create a grid of all points in the 3D space
    grid = np.indices(input_array.shape).reshape(3, -1).T  # Shape (N, 3), where N = total number of voxels
    
    # Use Delaunay triangulation to find points inside the convex hull
    delaunay = Delaunay(foreground_points[hull.vertices])  # Triangulate using the convex hull vertices
    mask_points = delaunay.find_simplex(grid) >= 0  # True for points inside the convex hull
    
    # Create the convex hull mask
    convex_hull_mask = np.zeros_like(input_array, dtype=np.uint8)  # Initialize an empty mask
    convex_hull_mask[tuple(grid[mask_points].T)] = 1  # Mark the points inside the hull as 1
    
    return convex_hull_mask



def get_x_y_angle(direc):
    """
    Calculate the angles for the 3D direction vector with respect to the x and yz planes.

    Parameters:
        direc (numpy.ndarray): A 3D vector representing the direction of the seed. 
                               (direc[0], direc[1], direc[2]) corresponds to the (x, y, z) components.

    Returns:
        tuple: A tuple (x_a, y_a) where:
            - x_a (float): The angle between the direction vector and the yz-plane in radians.
            - y_a (float): The angle between the direction vector and the yz-plane (projected onto yz-plane).
    """
    # Step 1: Calculate the angle with respect to the yz-plane
    y_a = angle_with_yz_plane(direc)  # This calculates the angle between the direction vector and yz-plane
    
    # Step 2: Create the yz-plane projection of the direction vector
    yz_direc = np.array([0, direc[1], direc[2]])  # Project the vector onto yz-plane by setting x = 0
    yz_direc = yz_direc / np.linalg.norm(yz_direc)  # Normalize the yz projection to get a unit vector
    
    # Step 3: Calculate the angle between the yz-projected vector and the z-axis
    x_a = np.arccos(yz_direc[2])  # The angle with the z-axis, using the z-component of the yz-plane projection
    
    # Step 4: Adjust the angle based on the sign of the y-component of the original vector
    # If the y-component is positive, the angle is measured clockwise from the z-axis; 
    # otherwise, it's measured counterclockwise.
    if direc[1] > 0:
        x_a = 2 * np.pi - x_a  # Adjust the angle to ensure correct direction (counterclockwise if y > 0)
    
    # Return both angles: x_a (angle with respect to the z-axis in yz-plane) and y_a (angle with respect to the yz-plane)
    return x_a, y_a



def angle_with_yz_plane(v):
    """
    Calculate the angle between a 3D vector and its projection onto the yz-plane.

    Parameters:
        v (numpy.ndarray): A 3D vector (v[0], v[1], v[2]) representing the direction.

    Returns:
        float: The angle in radians between the 3D vector and its projection onto the yz-plane.
    """
    # Step 1: Calculate the projection of the vector onto the yz-plane by setting the x-component to 0
    v_yz_projection = np.array([0, v[1], v[2]])  # Project the vector onto yz-plane [0, vy, vz]
    
    # Step 2: Calculate the cosine of the angle between the original vector and the projected vector
    # Use the dot product formula: cos(θ) = (v . v_yz) / (||v|| * ||v_yz||)
    cos_theta = np.dot(v, v_yz_projection) / (np.linalg.norm(v) * np.linalg.norm(v_yz_projection))
    
    # Step 3: Clip the cosine value to the valid range of [-1, 1] to avoid numerical errors
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # Step 4: Calculate the angle using the arccosine of the cosine value
    return np.arccos(cos_theta)  # Return the angle in radians




def find_min_absolute_gradient_direction(array, point):
    """
    Finds the direction with the smallest absolute gradient change near a specified point in a 3D array.

    Parameters:
        array (numpy.ndarray): The 3D array to search.
        point (tuple): The (x, y, z) coordinates of the point.

    Returns:
        numpy.ndarray: The direction vector with the minimum absolute gradient change.
    """
    x, y, z = point

    # Calculate gradients in each direction
    grad_x = sobel(array, axis=0)
    grad_y = sobel(array, axis=1)
    grad_z = sobel(array, axis=2)
    
    # Define possible directions in the 3x3x3 neighborhood
    directions = [
        (dx, dy, dz)
        for dx in [-1, 0, 1]
        for dy in [-1, 0, 1]
        for dz in [-1, 0, 1]
        if not (dx == 0 and dy == 0 and dz == 0)
    ]

    min_absolute_change = float('inf')
    min_direction = np.array([1, 0, 0])

    # Calculate the absolute gradient change for each direction
    for dx, dy, dz in directions:
        neighbor = (x + dx, y + dy, z + dz)
        
        # Check if neighbor is within bounds
        if 0 <= neighbor[0] < array.shape[0] and 0 <= neighbor[1] < array.shape[1] and 0 <= neighbor[2] < array.shape[2]:
            # Calculate absolute gradient change
            abs_change = abs(grad_x[neighbor] - grad_x[x, y, z]) + \
                         abs(grad_y[neighbor] - grad_y[x, y, z]) + \
                         abs(grad_z[neighbor] - grad_z[x, y, z])

            # Update minimum if this direction has a lower absolute gradient change
            if abs_change < min_absolute_change:
                min_absolute_change = abs_change
                min_direction = np.array([dx, dy, dz])

    return min_direction



def set_ones_to_zero(input_array, num_to_remove):
    """
    Set a specified number of 1's in a 3D numpy array to 0's in a deterministic order.
    
    Parameters:
        input_array (numpy.ndarray): A 3D binary array (only 0's and 1's).
        num_to_remove (int): The number of 1's to set to 0.
    
    Returns:
        numpy.ndarray: The modified array with specified number of 1's set to 0.
    """
    # Get the indices where the value is 1
    ones_indices = np.argwhere(input_array == 1)
    
    # Ensure the number of 1's to remove does not exceed the number of 1's present
    num_to_remove = min(num_to_remove, len(ones_indices))
    
    # Remove the first `num_to_remove` ones by index order
    for idx in range(num_to_remove):
        input_array[tuple(ones_indices[idx])] = 0
    
    return input_array



def keep_ones(input_array, num_to_keep):
    """
    Keep a specified number of 1's in a 3D numpy array and set the rest to 0's.
    
    Parameters:
        input_array (numpy.ndarray): A 3D binary array (only 0's and 1's).
        num_to_keep (int): The number of 1's to keep, all others will be set to 0.
    
    Returns:
        numpy.ndarray: The modified array with only the specified number of 1's kept.
    """
    # Get the indices where the value is 1
    ones_indices = np.argwhere(input_array == 1)
    
    # Ensure the number of 1's to keep does not exceed the number of 1's present
    num_to_keep = min(num_to_keep, ones_indices.shape[0])
    
    # Create a new array of zeros with the same shape
    output_array = np.zeros_like(input_array)
    
    # Keep the first `num_to_keep` ones in the new array
    for idx in range(num_to_keep):
        output_array[tuple(ones_indices[idx])] = 1
    
    return output_array



def shrink_island_by_distance(input_array, target_percentage=0.98):
    """
    Shrinks a 3D island (1's surrounded by 0's) from the edge until its volume is reduced
    to the specified percentage of the original volume by removing the nearest points to the edge.

    Parameters:
        input_array (numpy.ndarray): 3D binary array with island represented by 1's.
        target_percentage (float): The target volume percentage (0.9 means shrink to 90% of the original volume).
        
    Returns:
        numpy.ndarray: The shrunken 3D binary island.
    """
    # Calculate the initial volume
    output_array = input_array.copy()
    original_volume = np.sum(output_array)
    
    # Calculate the target volume
    target_volume = int(original_volume * target_percentage)
    
    # Compute the distance transform (distance to nearest zero, i.e., edge)
    distance_map = distance_transform_edt(output_array)
    while np.sum(output_array) > target_volume:
        num2shrink = int(np.sum(output_array) - target_volume)
        distance_map = keep_ones(distance_map, num2shrink)
        output_array[distance_map==1] = 0
        distance_map = distance_transform_edt(output_array)
        
    return output_array



def generate_oriented_3d_gaussian(shape, center, direction, sigmas, translation=(0, 0, 0)):
    # Normalize the direction vector
    direction = np.array(direction).reshape(-1)
    direction = direction / np.linalg.norm(direction)

    # Generate coordinates and apply translation
    x = np.arange(shape[0]) - (center[0] - translation[0])
    y = np.arange(shape[1]) - (center[1] - translation[1])
    z = np.arange(shape[2]) - (center[2] - translation[2])
    x, y, z = np.meshgrid(x, y, z, indexing='ij')

    # Rotation matrix calculation
    x_axis = np.array([1, 0, 0])  # Assume the Gaussian's long axis originally aligns with the x-axis

    if np.abs(np.dot(x_axis, direction)) >= 0.99:  # If direction is almost aligned with the x-axis
        rotation_matrix = np.eye(3)  # No rotation needed
    else:
        # Calculate the angle and axis for rotation
        angle = np.arccos(np.dot(x_axis, direction))  # Angle between x-axis and direction
        axis = np.cross(x_axis, direction)  # Rotation axis
        axis = axis / np.linalg.norm(axis)  # Normalize the axis

        # Using Rodrigues' rotation formula to construct the rotation matrix
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        I = np.eye(3)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        rotation_matrix = I * cos_theta + np.outer(axis, axis) * (1 - cos_theta) + K * sin_theta

    # Rotate coordinates using the rotation matrix
    coords = np.stack([x, y, z]).reshape(3, -1)  # Reshape to 3xM
    rotated_coords = rotation_matrix @ coords  # Matrix multiplication for rotation
    rotated_coords = rotated_coords.reshape(3, *shape)  # Reshape back to 3xNxNxN

    # Unpack the sigmas and compute the Gaussian function
    sigma_x, sigma_y, sigma_z = sigmas
    gaussian = np.exp(-((rotated_coords[0]**2) / (2 * sigma_x**2) +
                        (rotated_coords[1]**2) / (2 * sigma_y**2) +
                        (rotated_coords[2]**2) / (2 * sigma_z**2)))
    
    return gaussian



def find_island_center(arr, sigma=3):
    """
    Find the center or the farthest point inside an irregularly shaped 'island' of 1s in a 3D array.

    Parameters:
        arr (numpy.ndarray): The input 3D binary array with values 0 and 1.
        sigma (float): The standard deviation for Gaussian smoothing. Higher sigma makes the result smoother.

    Returns:
        tuple: The coordinates of the point with the maximum Gaussian filtered value.
    """
    # Apply Gaussian filter to smooth the array
    smoothed = gaussian_filter(arr.astype(float), sigma=sigma)

    # Find the coordinates of the maximum value in the smoothed array
    center_coords = np.unravel_index(np.argmax(smoothed), arr.shape)
    
    return smoothed, center_coords



def find_island_center_adaptive_sigma(arr):
    """
    Find the center or the farthest point inside an irregularly shaped 'island' of 1s in a 3D array
    using an adaptive sigma based on distance transformation.

    Parameters:
        arr (numpy.ndarray): The input 3D binary array with values 0 and 1.

    Returns:
        tuple: The smoothed array and the coordinates of the point with the maximum Gaussian filtered value.
        
        
    # Example usage
    # arr is your 3D numpy array
    center_point, adaptive_sigma = find_island_center_adaptive_sigma(arr)
    print("Most likely center point:", center_point)
    print("Adaptive sigma used:", adaptive_sigma)
    """
    # Calculate distance transform of the binary array
    distance_map = distance_transform_edt(arr)
    
    # Set sigma based on the average or maximum distance (adjustable)
    sigma = max(1, np.mean(distance_map[arr == 1]) / 2)
    
    return find_island_center(arr, sigma)



def get_cylinder_polydata(center, direction, length, radius):
    """
    Generate a VTK polydata representation of a cylindrical seed with a given position, orientation, and dimensions.

    Parameters:
        center (array-like): The center position of the seed (x, y, z).
        direction (array-like): The direction vector indicating the seed's orientation.
        length (float): The length of the seed cylinder.
        radius (float): The radius of the cylindrical seed.

    Returns:
        vtk.vtkTransformPolyDataFilter: Transformed VTK polydata representing the cylindrical seed.
    """
    # Step 1: Create a cylinder source with the specified length and radius
    cylinder = vtk.vtkCylinderSource()
    cylinder.SetHeight(length)           # Set the cylinder height (i.e., length)
    cylinder.SetRadius(radius)           # Set the cylinder radius
    cylinder.SetResolution(96)           # Define the level of detail (higher for smoother cylinder)
    cylinder.Update()

    # Step 2: Normalize the direction vector for rotation calculations
    direction = direction.reshape(-1)
    direction = direction / np.linalg.norm(direction)
    
    # Step 3: Determine rotation axis and angle to align the cylinder with the specified direction
    # Initial cylinder points along the Y-axis; find rotation axis to align with `direction`
    rotation_axis = np.cross([0, -1, 0], direction)
    rotation_angle = math.degrees(math.acos(-direction[1]))  # Convert angle to degrees for VTK

    # Step 4: Set up the transformation: rotation and translation
    transform = vtk.vtkTransform()
    transform.PostMultiply()                              # Ensure transformations are applied in order
    transform.RotateWXYZ(rotation_angle, *rotation_axis)  # Rotate to align with the specified direction
    transform.Translate(*center) # - direction * length / 2)  # Translate to position centered at `center`
    transform.Update()

    # Step 5: Apply the transformation to the cylinder polydata
    transformed_polydata = vtk.vtkTransformPolyDataFilter()
    transformed_polydata.SetTransform(transform)
    transformed_polydata.SetInputConnection(cylinder.GetOutputPort())
    transformed_polydata.Update()  # Apply transformation

    return transformed_polydata




def perpendicular_vector(v):
    # Normalize the vector v
    v = np.array(v)
    v = v / np.linalg.norm(v)
    
    # Choose an arbitrary vector that is not parallel to v
    if abs(v[0]) < abs(v[1]) and abs(v[0]) < abs(v[2]):
        other = np.array([1, 0, 0])
    elif abs(v[1]) < abs(v[2]):
        other = np.array([0, 1, 0])
    else:
        other = np.array([0, 0, 1])
    
    # Calculate the cross product to get a perpendicular vector
    perp_vector = np.cross(v, other)
    perp_vector /= np.linalg.norm(perp_vector)  # Normalize the result
    
    return perp_vector


def get_trajectory_info(point, array, direction, target_value, background_value, obstacle_value):
    """
    Evaluate a trajectory within a multi-dimensional array, moving in a specified direction.
    
    Parameters:
    - point: A one-dimensional array representing the starting coordinates.
    - array: A multi-dimensional array to be traversed.
    - direction: A one-dimensional array specifying the direction of movement.
    - target_value: The value in the array representing the target.
    - background_value: The value representing the background.
    - obstacle_value: The value representing an obstacle.
    
    Returns:
    - obs_sign: A boolean indicating if an obstacle is encountered along the path.
    - target_lengths: A list of lengths for contiguous segments of the target value encountered.
    - background_lengths: A list of lengths for contiguous segments of the background value encountered.
    """
    
    # Reshape 'point' and 'direction' to one-dimensional arrays
    point = point.reshape(-1)
    direction = direction.reshape(-1)

    # Find the index of the largest absolute value in the direction array for normalization
    max_index = np.argmax(np.abs(direction)) 
    
    # Copy the starting point for updates
    update_point = np.copy(point)  
    
    # Normalize the direction for consistent movement scaling
    update_direction = direction / np.abs(direction[max_index])

    # Initialize variables to track the state
    obs_sign = False  # Flag to detect obstacles
    target_lengths = []  # List to store lengths of contiguous target segments
    background_lengths = []  # List to store lengths of contiguous background segments
    target_length = 0  # Length of the current target segment
    background_length = 0  # Length of the current background segment
    step = 0  # Step counter for movement

    # Check for obstacles and ensure the point is within bounds of the array
    while is_point_inside_array(update_point, array) and not obs_sign:
        int_coords = tuple(update_point.astype(int))  # Convert the point to integer coordinates
        step += 1
        if array[int_coords] == obstacle_value:
            obs_sign = True  # Set to True if an obstacle is encountered
        update_point = (point - step * update_direction).astype(np.float64)  # Move the point along the direction
    
    step = 0  # Reset step for the main traversal loop
    update_point = (point + step * update_direction).astype(np.float64)    # Continue traversing the array if no obstacle was encountered and the point is within bounds
    while is_point_inside_array(update_point, array) and not obs_sign:
        int_coords = tuple(update_point.astype(int))  # Convert the point to integer coordinates
        step += 1
        # Check the value at the current position and update the respective segment lengths
        if array[int_coords] == target_value:
            target_length += 1
            if background_length != 0:
                background_lengths.append(background_length)  # Save background length if a target is found
                background_length = 0  # Reset background length counter
        elif array[int_coords] == background_value:
            background_length += 1
            if target_length != 0:
                target_lengths.append(target_length)  # Save target length if a background is found
                target_length = 0  # Reset target length counter
        elif array[int_coords] == obstacle_value:
            if target_length != 0:
                target_lengths.append(target_length)  # Save target length before breaking if an obstacle is encountered
                target_length = 0
            break  # Stop if an obstacle is encountered
        
        # Move the point further in the direction by the step size
        update_point = (point + step * update_direction).astype(np.float64)
        
    return obs_sign, target_lengths, background_lengths  # Return the obstacle sign and lengths of target/background segments



def is_point_inside_array(point, array):
    """
    Check if the given point is within the bounds of the array.

    Parameters:
    - point: The coordinates of the point, should be a one-dimensional array.
    - array: The multi-dimensional array to check against.
    
    Returns:
    - bool: Returns True if the point is within the bounds of the array, otherwise returns False.
    """
    return np.all((point >= 0) & (point < np.array(array.shape)))




def projection_length(points, vector):
    """
    Calculate the total projection length of a set of points onto a given vector.
    
    Parameters:
    points (numpy.ndarray): A n x 3 numpy array representing n points in 3D space.
    vector (numpy.ndarray): A 1 x 3 numpy array representing the vector onto which the points are projected.
    
    Returns:
    float: The sum of the projection lengths of all points onto the given vector.
    
    Raises:
    ValueError: If the vector is a zero vector, as it cannot be normalized.
    """
    
    # Check if the vector is a zero vector by calculating its norm (magnitude)
    # The norm should not be zero; otherwise, it's impossible to normalize the vector
    if np.linalg.norm(vector) == 0:
        raise ValueError("The vector cannot be a zero vector.")
    
    # Normalize the vector to convert it to a unit vector
    # A unit vector is a vector with magnitude of 1 that keeps the same direction
    unit_vector = vector / np.linalg.norm(vector).reshape(-1)
    
    # Calculate the projection length for each point onto the normalized vector
    # This is done using the dot product between each point and the normalized vector
    # The dot product gives us the component of each point along the direction of the vector
    projections = np.dot(points.reshape(-1, 3), unit_vector)
    
    # Calculate the total length of all projections
    # Sum the individual projection lengths to get the total projection length
    length = calculate_range_difference(projections)
    
    # Return the total length of the projections
    return length



def calculate_range_difference(array):
    """
    Calculate the difference between the maximum and minimum values in an array.
    
    Parameters:
    array (numpy.ndarray): A numpy array containing numerical values.
    
    Returns:
    float: The difference between the maximum and minimum values in the array.
    """
    # Find the maximum value in the array
    max_value = np.max(array)
    
    # Find the minimum value in the array
    min_value = np.min(array)
    
    # Calculate the difference between the maximum and minimum values
    difference = max_value - min_value
    
    return difference



def line_to_line_distance(P0, d0, P1, d1):
    """
    Calculate the shortest distance between two lines in 3D space.

    Parameters:
    P0 -- A point on the first line.
    d0 -- The direction vector of the first line.
    P1 -- A point on the second line.
    d1 -- The direction vector of the second line.

    Returns:
    The shortest distance between the two lines.
    """
    # Ensure inputs are numpy arrays
    P0 = np.array(P0)
    d0 = np.array(d0)
    P1 = np.array(P1)
    d1 = np.array(d1)
    
    # Normalize direction vectors
    d0_norm = np.linalg.norm(d0)
    d1_norm = np.linalg.norm(d1)
    
    if d0_norm == 0 or d1_norm == 0:
        raise ValueError("Direction vectors must be non-zero")
    
    d0 = d0 / d0_norm
    d1 = d1 / d1_norm
    
    # Calculate cross product and normalize
    n = np.cross(d0, d1)
    denominator = np.linalg.norm(n)
    
    # If n is (close to) zero, the lines are parallel (and can be co-linear)
    if denominator < 1e-10:
        # Calculate distance between parallel lines
        return np.linalg.norm(np.cross(P1 - P0, d0)) / np.linalg.norm(d0)
    
    # Calculate shortest distance
    distance = np.abs(np.dot(n, P1 - P0)) / denominator
    return distance


def ray_to_ray_distance(P0, d0, P1, d1):
    """
    Calculate the shortest distance between two rays in 3D space.

    Parameters:
    P0 -- A point on the first ray (numpy array of shape (3,))
    d0 -- The direction vector of the first ray (numpy array of shape (3,))
    P1 -- A point on the second ray (numpy array of shape (3,))
    d1 -- The direction vector of the second ray (numpy array of shape (3,))

    Returns:
    The shortest distance between the two rays.
    """
    # Normalize direction vectors
    d0 = d0 / np.linalg.norm(d0)
    d1 = d1 / np.linalg.norm(d1)
    
    # Calculate vector between the two points
    P0P1 = P1 - P0
    
    # Calculate cross product and normalize
    n = np.cross(d0, d1)
    denominator = np.linalg.norm(n)
    
    # If n is (close to) zero, the rays are parallel
    if denominator < 1e-10:
        # Calculate the distance between the parallel rays
        # Project P0P1 onto the perpendicular direction of the ray
        t0 = np.dot(P0P1, d0)
        closest_point_on_ray1 = P0 + t0 * d0
        return np.linalg.norm(P1 - closest_point_on_ray1)
    
    # Calculate the parameters that minimize the distance between the two rays
    t0 = (np.dot(np.cross(P0P1, d1), n) / denominator**2)
    t1 = (np.dot(np.cross(P0P1, d0), n) / denominator**2)
    
    # Calculate the closest points on each ray
    closest_point_on_ray1 = P0 + max(t0, 0) * d0
    closest_point_on_ray2 = P1 + max(t1, 0) * d1
    
    # Calculate the shortest distance between the closest points
    distance = np.linalg.norm(closest_point_on_ray1 - closest_point_on_ray2)
    return distance


def ray_min_distance(p1, d1, p2, d2, epsilon=1e-6):
    """
    Calculate the minimum distance between two 3D rays.

    Parameters:
    - p1: The starting point of the first ray (a numpy array of shape (3,))
    - d1: The direction vector of the first ray (a numpy array of shape (3,))
    - p2: The starting point of the second ray (a numpy array of shape (3,))
    - d2: The direction vector of the second ray (a numpy array of shape (3,))
    - epsilon: A small value used for floating-point comparison (default: 1e-6)

    Returns:
    - min_distance: The minimum distance between the two rays
    """
    # Ensure inputs are numpy arrays
    p1 = np.array(p1)
    d1 = np.array(d1)
    p2 = np.array(p2)
    d2 = np.array(d2)

    # Calculate the cross product of the direction vectors
    n = np.cross(d1, d2)
    n_norm = np.linalg.norm(n)

    if n_norm < epsilon:
        # Rays are parallel
        m = np.cross(p1 - p2, d1)
        m_norm = np.linalg.norm(m)
        if m_norm < epsilon:
            # Rays are collinear
            t = np.dot(p2 - p1, d1) / np.dot(d1, d1)
            s = np.dot(p1 - p2, d2) / np.dot(d2, d2)
            if t <= 0 or s <= 0:
                return 0.0  # Rays overlap, minimum distance is 0
            else:
                return np.linalg.norm(p1 - p2)  # Rays do not overlap, return distance between start points
        else:
            # Rays are parallel but not collinear
            return m_norm / np.linalg.norm(d1)  # Return fixed distance between parallel rays
    else:
        # Rays are not parallel
        dp = p2 - p1
        a = np.dot(d1, d1)
        b = np.dot(d1, d2)
        c = np.dot(d2, d2)
        e = np.dot(d1, dp)
        f = np.dot(d2, dp)
        denom = a * c - b ** 2
        t_star = (e * c - f * b) / denom
        s_star = (a * f - b * e) / denom

        if t_star >= 0 and s_star >= 0:
            # Both parameters are non-negative, closest points are on the rays
            q1 = p1 + t_star * d1
            q2 = p2 + s_star * d2
            return np.linalg.norm(q1 - q2)
        elif t_star < 0 and s_star >= 0:
            # t* < 0, s* >= 0, closest point is on the first ray's starting point
            u_star = max(0, np.dot(p1 - p2, d2) / np.dot(d2, d2))
            q2 = p2 + u_star * d2
            return np.linalg.norm(p1 - q2)
        elif t_star >= 0 and s_star < 0:
            # t* >= 0, s* < 0, closest point is on the second ray's starting point
            u_star = max(0, np.dot(p2 - p1, d1) / np.dot(d1, d1))
            q1 = p1 + u_star * d1
            return np.linalg.norm(q1 - p2)
        else:
            # t* < 0, s* < 0, closest point is the distance between the starting points of the two rays
            return np.linalg.norm(p1 - p2)


def min_distance_to_lines(target_point, target_direction, lines):
    """
    Calculate the minimum distance from a target line to a set of lines.

    Parameters:
    target_point -- A point on the target line.
    target_direction -- The direction vector of the target line.
    lines -- A list of tuples, each containing (point, direction) for other lines.

    Returns:
    The minimum distance from the target line to the set of lines.
    """
    min_distance = float('inf')
    
    for point, direction in lines:
        distance = ray_min_distance(target_point, target_direction, point, direction)
        min_distance = min(min_distance, distance)
    
    return min_distance


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Apply a bandpass filter to the data.

    Parameters:
    data -- Input 1D numpy array.
    lowcut -- Low cutoff frequency for the bandpass filter.
    highcut -- High cutoff frequency for the bandpass filter.
    fs -- Sampling frequency.
    order -- Order of the filter (default is 5).

    Returns:
    Filtered numpy array.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist   
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y



def distance_filter(x, x1, x2, k):
    """
    Apply a distance-based filtering function to an input value.

    This function evaluates an input value `x` against two thresholds, `x1` and `x2`, 
    and produces an output based on its position relative to these thresholds:
    - For x < x1: Outputs a small constant value (1e-6).
    - For x1 <= x <= x2: Outputs a value that decreases smoothly from 1.0 as x increases.
    - For x > x2: The output continues decreasing at a slower rate.

    Parameters:
        x (float): The input value to evaluate.
        x1 (float): The first threshold. Values below this return a fixed small output (1e-6).
        x2 (float): The second threshold. Defines the end of the smooth transition zone.
        k (float, optional): Controls the steepness of the transition curve (default is 0.1).
                             Larger values of `k` produce a sharper transition.

    Returns:
        float: The filtered output value based on the input `x`.
    """
    if x < x1:
        return 0 # Return a very small constant value if x is less than x1
    else:
        # Smooth decreasing output for x between x1 and x2
        return 1 + k - (min(x, x2) - x1) / (x2 - x1)

    
    
    

def delete_elements_in_range(lst, start, end):
    """
    Remove elements in the range [start, end] (inclusive) from a sorted list.

    Parameters:
    lst (list): A sorted list of numbers in ascending order
    start (int): Starting value of the range
    end (int): Ending value of the range

    Returns:
    list: A new list after removing elements in the specified range
    """
    if not lst:
        return lst

    # Use list comprehension to filter out elements in the specified range
    return [x for x in lst if x < start or x > end]



def top_k(items: List[Any], scores: List[float], k: int, reverse: bool = True):
    """
    Return top-k items based on scores.

    Parameters
    ----------
    items : list
        List of items to rank.
    scores : list of float
        Scores corresponding to each item.
    k : int
        Number of top items to return.
    reverse : bool, default=True
        If True, higher scores are better.

    Returns
    -------
    list
        Top-k items (if k >= len(items), return all items sorted).
    """
    assert len(items) == len(scores), "Length of items and scores must match"

    # Pair items with scores
    paired = list(zip(items, scores))

    # Sort by score
    paired.sort(key=lambda x: x[1], reverse=reverse)

    # Take top-k, or fewer if list is shorter
    top_items = [x[0] for x in paired[:min(k, len(paired))]]

    return top_items
