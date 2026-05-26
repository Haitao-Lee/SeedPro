import SimpleITK as sitk
import numpy as np



def position_soft_method(seed, image_origin, image_size, image_spacing):
    """
    Generate a soft treatment plan based on a spherical distribution around a given seed point.

    Parameters:
    seed (np.ndarray): The seed point in 3D space, represented as [x, y, z].
    image_origin (np.ndarray): The physical origin of the image grid in 3D space, represented as [x, y, z].
    image_size (tuple): The size of the 3D image grid, represented as (x_dim, y_dim, z_dim).
    image_spacing (np.ndarray): The spacing between voxels in each dimension, represented as [x_spacing, y_spacing, z_spacing].

    Returns:
    np.ndarray: A 3D array representing the soft treatment plan, where each voxel's value corresponds 
                to the normalized overlap volume of a sphere centered at the seed point.

    Steps:
    1. Define a sphere with a fixed radius and compute its theoretical volume.
    2. Create a 3D grid of voxel centers based on the image size and spacing.
    3. Calculate the Euclidean distance from each voxel center to the seed point.
    4. Determine overlapping voxels within the sphere radius and compute their contribution to the sphere volume.
    5. Normalize the contribution of overlapping voxels to form the soft treatment plan.
    6. Transpose the resulting array to match the expected output orientation.

    Notes:
    - This function assumes the sphere radius is 4 units and calculates volumes in physical space based on 
      the provided grid spacing and origin.
    - Ensure that the input parameters are consistent in their coordinate systems and units.

    Example Usage:
    seed = np.array([10, 10, 10])
    image_origin = np.array([0, 0, 0])
    image_size = (50, 50, 50)
    image_spacing = np.array([1, 1, 1])
    result = position_soft_method(seed, image_origin, image_size, image_spacing)
    """
    sphere_radius = 4  # Radius of the sphere in physical units
    sphere_volume = (4 / 3) * np.pi * sphere_radius**3  # Volume of the sphere

    # Create a 3D grid for the image based on its size
    grid_shape = tuple(image_size)
    soft_treatment_plan = np.zeros(grid_shape)  # Initialize the treatment plan array
    grid_spacing = np.array(image_spacing)  # Spacing between grid points

    # Generate voxel center coordinates
    x_grid, y_grid, z_grid = np.meshgrid(
        np.arange(grid_shape[0]),
        np.arange(grid_shape[1]),
        np.arange(grid_shape[2]),
        indexing='ij'
    )
    voxel_centers = np.stack([x_grid, y_grid, z_grid], axis=-1) * grid_spacing + image_origin

    # Calculate the Euclidean distance from each voxel center to the seed point
    distances = np.linalg.norm(voxel_centers - seed[0:3], axis=-1)

    # Compute overlapping volumes for voxels within the sphere radius
    overlap_mask = distances <= sphere_radius
    overlapping_volume = overlap_mask * (4 / 3) * np.pi * ((sphere_radius - distances) ** 3)

    # Normalize the overlapping volume by the sphere volume
    normalized_volume = overlapping_volume / sphere_volume
    soft_treatment_plan += normalized_volume

    # Normalize the treatment plan to ensure sum of non-zero values equals 1
    non_zero_values = soft_treatment_plan[soft_treatment_plan > 0]
    if np.sum(non_zero_values) > 0:
        soft_treatment_plan[soft_treatment_plan > 0] /= np.sum(non_zero_values)

    # Transpose to match the desired output orientation
    soft_treatment_plan = np.transpose(soft_treatment_plan, (2, 1, 0))
    return soft_treatment_plan



def line_source_map(seed, direction, image_origin, image_size, image_spacing):
     # seed 粒子坐标，direction 粒子方向向量
        x = image_origin[0] + np.arange(image_size[0]) * image_spacing[0]
        y = image_origin[1] + np.arange(image_size[1]) * image_spacing[1]
        z = image_origin[2] + np.arange(image_size[2]) * image_spacing[2]

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        Vx = X - seed[0]
        Vy = Y - seed[1]
        Vz = Z - seed[2]

        distance_squared = Vx**2 + Vy**2 + Vz**2
        # print(np.where(distance_squared<0.01))
        distance_squared[np.where(distance_squared<0.01)] = 0.01
        v0 = np.array([0, 0, 1])
        L = 4.5
        
        norm_direction_vector = direction/ np.linalg.norm(direction[3])
        norm_direction_vector = np.array([norm_direction_vector[0], norm_direction_vector[1], -norm_direction_vector[2]])
        # 计算两个点的坐标
        A_prime = seed[0:3] - L/2 * norm_direction_vector
        B_prime = seed[0:3] + L/2 * norm_direction_vector
        V_PA = np.array([X - A_prime[0], Y - A_prime[1], Z - A_prime[2]])
        V_PB = np.array([X - B_prime[0], Y - B_prime[1], Z - B_prime[2]])

        # 计算向量的长度
        V_PA_magnitude = np.sqrt(np.sum(V_PA**2, axis=0))
        V_PB_magnitude = np.sqrt(np.sum(V_PB**2, axis=0))

        # 计算向量之间的夹角
        dot_product = np.sum(V_PA * V_PB, axis=0)
        cos_beta = np.abs(dot_product / (V_PA_magnitude * V_PB_magnitude))
        beta = np.arccos(np.clip(cos_beta, -1.0, 1.0))

        vectors_to_mid_point = np.stack((X - seed[0], Y - seed[1], Z - seed[2]), axis=-1)
        
        norm_vectors_to_mid_point = vectors_to_mid_point / np.linalg.norm(vectors_to_mid_point, axis=-1, keepdims=True)
        cos_theta = np.dot(norm_vectors_to_mid_point, norm_direction_vector)
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

        line_map = beta/ (np.sin(theta) * (distance_squared)**(-1) + 1e-5)
        line_map = np.transpose(line_map, (2, 1, 0))
        
        line_map = sitk.GetImageFromArray(line_map)
        line_map.SetOrigin(image_origin)
        line_map.SetSpacing(image_spacing)
        
        
        return line_map
