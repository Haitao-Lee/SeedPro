import vtk
import numpy as np
import geometry
import os
import SimpleITK as  sitk
import matplotlib.pyplot as plt
import threading
import utils


class DynamicVTKVisualizer:
    def __init__(self, seeds_poly, tumor_poly):
        """
        Initialize the VTK visualizer with seed and tumor geometry.

        Args:
            seeds_poly (vtk.vtkPolyData): The PolyData representing the seed points.
            tumor_poly (vtk.vtkPolyData): The PolyData representing the tumor geometry.
        """
        # Create a renderer for rendering graphical objects
        self.renderer = vtk.vtkRenderer()
        
        # Create a render window to display the scene
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)

        # Create an interactor for handling user input (e.g., mouse and keyboard events)
        rw_style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)
        self.interactor.SetInteractorStyle(rw_style)      
        
        # Store the seed and tumor geometry (PolyData)
        self.seeds_poly = seeds_poly
        self.tumor_poly = tumor_poly
        
        # Create mappers for visualizing the seed and tumor PolyData
        self.seeds_mapper = vtk.vtkPolyDataMapper()
        self.seeds_mapper.SetInputConnection(self.seeds_poly.GetOutputPort())
        
        self.tumor_mapper = vtk.vtkPolyDataMapper()
        self.tumor_mapper.SetInputData(self.tumor_poly)

        # Create actors to represent the seeds and tumor geometries in the scene
        self.seeds_actor = vtk.vtkActor()
        self.seeds_actor.SetMapper(self.seeds_mapper)
        self.seeds_actor.GetProperty().SetColor(230/255, 230/255, 77/255)

        
        self.tumor_actor = vtk.vtkActor()
        self.tumor_actor.SetMapper(self.tumor_mapper)
        self.tumor_actor.GetProperty().SetColor(0, 0, 0)
        self.tumor_actor.GetProperty().SetOpacity(0.6)
        
        # Add the actors (seeds and tumor) to the renderer
        self.renderer.AddActor(self.tumor_actor)
        self.renderer.AddActor(self.seeds_actor)

        # Set the background color of the rendering window (light blue)
        self.renderer.SetBackground(200 / 255, 200 / 255, 235 / 255)
        
        # Adjust the camera's view for a better initial perspective
        self.renderer.GetActiveCamera().Azimuth(30)  # Rotate the camera around the vertical axis
        self.renderer.GetActiveCamera().Elevation(30)  # Tilt the camera upward
        self.renderer.ResetCamera()

    def update_seeds(self, polydata):
        """
        Update the seed points in the visualization with new PolyData.

        Args:
            polydata (vtk.vtkPolyData): A new PolyData object containing updated seed geometry.
        """    
        # Update the seeds' PolyData with the new data
        self.seeds_poly.ShallowCopy(polydata)
        
        # Mark the updated PolyData as modified for re-rendering
        self.seeds_poly.Modified()
        
        # Trigger the rendering window to refresh the scene
        self.render_window.Render()

    def start(self):
        """
        Start the rendering loop and periodically update the seeds.
        The main thread is free to execute other logic.
        """
        # Start rendering in a separate thread
        render_thread = threading.Thread(target=self.start_rendering, daemon=True)
        render_thread.start()

        # Simulate updating the PolyData over time in the main thread
        import time
        for i in range(10):
            # Generate new points for demonstration (replace with actual data)
            points = np.random.rand(100, 3) * 10

            # Create a new PolyData from the points (replace with actual data update)
            new_seeds_poly = vtk.vtkPolyData()  # Create a new vtkPolyData from updated points
            
            # Update the visualizer with the new points
            self.update_seeds(new_seeds_poly)

            # Sleep for a short time before updating again
            time.sleep(1)



# # Example usage
# if __name__ == '__main__':
#     # Create the visualizer object
#     visualizer = DynamicVTKVisualizer()

#     # Start the rendering loop
#     import threading
#     threading.Thread(target=visualizer.start, daemon=True).start()

#     # Simulate external updates to PolyData
#     import time
#     for i in range(10):
#         # Generate new points
#         points = np.random.rand(100, 3) * 10

#         # Update the visualizer with the new points
#         visualizer.update_points(points)
#         time.sleep(1)



def get_seed_polydata(center, direction, length, radius):
    """
    Generate a VTK polydata representation of a cylindrical seed with specified position, direction, and dimensions.

    Parameters:
        center (array-like): The center position of the seed (x, y, z).
        direction (array-like): The direction vector of the seed's orientation.
        length (float): The length of the seed.
        radius (float): The radius of the cylindrical seed.

    Returns:
        vtk.vtkTransformPolyDataFilter: Transformed VTK polydata representing the cylindrical seed.
    """
    # Use geometry module's function to create and transform the cylindrical polydata
    return geometry.get_cylinder_polydata(center, direction, length, radius)


def nii_to_vtkpolydata(image):
    """
    Convert a binary 3D volume from a .nii.gz (NIfTI) file to a vtkPolyData surface mesh,
    considering the spacing and origin information from the NIfTI header using SimpleITK.

    Parameters:
        nii_file (str): Path to the .nii.gz NIfTI file containing the 3D binary volume.
                         The volume should contain values 0 for background and 1 for the target region.
    
    Returns:
        vtk.vtkPolyData: vtkPolyData object representing the surface mesh extracted from the binary volume.
    """
    
    # Step 1: Load the NIfTI file using SimpleITK
    image_array = sitk.GetArrayFromImage(image)  # Convert SimpleITK image to a NumPy array
    
    # Ensure the image is binary (0 for background, 1 for target region)
    image_array = np.where(image_array > 0.5, 1, 0)  # Convert to binary (thresholding)
    
    # Extract spacing and origin from the SimpleITK image
    spacing = image.GetSpacing()  # Get the spacing (x_spacing, y_spacing, z_spacing)
    origin = image.GetOrigin()    # Get the origin (physical position of the first voxel)

    # Step 2: Convert NumPy array to VTK image data
    vtk_image = vtk.vtkImageData()  # Create vtkImageData object
    depth, height, width = image_array.shape  # Get the dimensions of the 3D data
    vtk_image.SetDimensions(width, height, depth)  # Set the dimensions of the VTK image
    
    # Allocate scalar data with unsigned char type (single component)
    vtk_image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)  
    
    # Set the spacing and origin from the SimpleITK image to the VTK image data
    vtk_image.SetSpacing(spacing)  # Set spacing (voxel size in each direction)
    vtk_image.SetOrigin(origin)    # Set origin (physical position of the first voxel)

    # Copy each voxel value from the NumPy array to the VTK image
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                vtk_image.SetScalarComponentFromFloat(x, y, z, 0, image_array[z, y, x])

    # Step 3: Use Marching Cubes to extract the surface mesh (isosurface)
    marching_cubes = vtk.vtkMarchingCubes()  # Create an instance of the Marching Cubes algorithm
    marching_cubes.SetInputData(vtk_image)  # Set the input VTK image data
    marching_cubes.SetValue(0, 1)  # Extract the isosurface where voxel value = 1 (representing the target region)
    marching_cubes.Update()  # Perform the extraction

    # Step 4: Get the generated vtkPolyData surface mesh
    polydata = marching_cubes.GetOutput()  # Retrieve the output vtkPolyData from Marching Cubes
    return polydata  # Return the PolyData surface mesh



def seeds_to_polydata(seeds, radiation_volume, image, length, radius):
    """
    Convert a list of seed positions and directions to a vtkPolyData object for visualization.

    This function generates a `vtkPolyData` object representing the seeds in 3D space.
    Each seed consists of a position and a direction. The position is scaled to match
    the size and origin of the radiation volume, and a corresponding geometry (such as a 
    sphere or cylinder) is created for each seed.

    Parameters:
        seeds (list): A list of tuples, where each tuple contains:
                      - A position (tuple/list of 3 coordinates for x, y, z)
                      - A direction (tuple/list of 3 components for direction vector).
        radiation_volume (numpy.ndarray): The 3D array representing the radiation volume (used to scale positions).
        length (float): The length of the seed (e.g., for a cylindrical seed).
        radius (float): The radius of the seed (e.g., for a spherical seed).

    Returns:
        vtk.vtkPolyData: A vtkPolyData object containing the geometry of all seeds, ready for rendering.
    """
    seeds_poly = vtk.vtkAppendPolyData()  # Create a container to append the individual seed polydata

    for _, seed in enumerate(seeds):  # Iterate over the list of seeds
        # Scale seed positions to the radiation volume size and account for image spacing and origin
        pos = np.array([seed[0][2], seed[0][1], seed[0][0]]).reshape(-1) * \
              np.array([radiation_volume.shape[2], radiation_volume.shape[1], radiation_volume.shape[0]]) * \
              np.array(image.GetSpacing()) + np.array(image.GetOrigin()).reshape(-1)
        
        # Create a polydata representation for each seed
        seed_poly = get_seed_polydata(pos, seed[1], length, radius)
        
        # Add the seed's polydata to the collection
        seeds_poly.AddInputConnection(seed_poly.GetOutputPort())

    # Update the container with the appended polydata
    seeds_poly.Update()

    return seeds_poly  # Return the combined vtkPolyData object



def save_polydata_as_stl(polydata, output_filename):
    """
    Save a vtkPolyData object to an STL file.
    
    Parameters:
        polydata (vtkPolyData): The polydata to be saved.
        output_filename (str): The name of the output STL file.
    """
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Create a vtkSTLWriter object
    stl_writer = vtk.vtkSTLWriter()
    
    # Set the input to the writer
    stl_writer.SetInputConnection(polydata.GetOutputPort())
    
    # Set the output file name
    stl_writer.SetFileName(output_filename)
    
    # Write the polydata to an STL file
    stl_writer.Write()
    
    

def get_radiation_3d(radiation, target_volume, target_value, threshold=0.1, sample_fraction=0.1, save_path=None):
    """
    Visualize a 3D radiation field and target volume, with optional image saving.

    This function creates a 3D scatter plot showing radiation intensity and target volume distribution.
    Data points are filtered by a radiation threshold and sampled to improve visualization clarity.

    Args:
        radiation (numpy.ndarray): 
            A 3D array representing radiation intensity across spatial coordinates.
        target_volume (numpy.ndarray): 
            A 3D binary array indicating target regions (1 for target, 0 for non-target).
        target_value (int or float): 
            The value representing target regions in the target_volume array.
        threshold (float, optional): 
            Minimum radiation intensity for points to be visualized. Default is 0.1.
        sample_fraction (float, optional): 
            Fraction of data points to randomly sample for plotting. Default is 0.1.
        save_path (str, optional): 
            File path to save the plot. If None, the plot will not be saved. Default is None.

    Returns:
        None
    """
    
    # Step 1: Identify radiation points above the threshold
    x_rad, y_rad, z_rad = np.where(radiation >= threshold)
    
    # Step 2: Sample radiation points to reduce plot density
    if len(x_rad) > 0:
        sample_size = int(len(x_rad) * sample_fraction)  # Calculate the sample size
        indices = np.random.choice(len(x_rad), sample_size, replace=False)  # Randomly select indices
        x_rad, y_rad, z_rad = x_rad[indices], y_rad[indices], z_rad[indices]  # Apply sampling

    # Step 3: Identify target region points
    x_tar, y_tar, z_tar = np.where(target_volume == target_value)
    
    # Step 4: Sample target points to reduce plot density
    if len(x_tar) > 0:
        sample_size = int(len(x_tar) * sample_fraction)  # Calculate the sample size
        indices = np.random.choice(len(x_tar), sample_size, replace=False)  # Randomly select indices
        x_tar, y_tar, z_tar = x_tar[indices], y_tar[indices], z_tar[indices]  # Apply sampling

    # Step 5: Create a 3D plot for visualization
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Step 6: Plot radiation points with intensity-based coloring
    if len(x_rad) > 0:
        ax.scatter(
            x_rad, y_rad, z_rad, 
            c=radiation[x_rad, y_rad, z_rad],  # Color mapped to radiation intensity
            cmap='viridis',  # Gradient colormap for radiation
            marker='o',
            s=1,  # Marker size
            alpha=0.4,  # Transparency for clarity
            label="Radiation Volume"
        )

    # Step 7: Plot target volume points in red
    if len(x_tar) > 0:
        ax.scatter(
            x_tar, y_tar, z_tar, 
            c='r',  # Red color for target points
            marker='o',
            s=1,  # Marker size
            alpha=0.4,  # Transparency for clarity
            label="Target Volume"
        )

    # Step 8: Configure plot aesthetics
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    plt.legend()

    # Step 9: Save the plot if a save_path is provided
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)  # Create directory if it doesn't exist
        plt.savefig(save_path)  # Save the plot to the specified path

    # Step 10: Display the plot
    plt.show()


def visualize_rays_3d_with_obstacles_and_save(mask, rays, target_vulue = 1, obstacle_value=-1, filename="output.png"):
    """
    Visualize the 3D foreground mask, obstacles, and rays, and save the plot to a file.

    Parameters:
        mask (np.ndarray): A 3D binary mask where 1 represents foreground,
                            0 represents background, and obstacle_value indicates obstacles.
        rays (list): List of rays, each represented by a tuple (start_point, direction).
        obstacle_value (int or float): The value in the mask representing obstacles.
        filename (str): The filename to save the plot as.
    
    Example usage:
    # Create a sample 3D mask with foreground and obstacles
    mask = np.zeros((10, 10, 10))
    mask[4:6, 4:6, 4:6] = 1  # Set a small foreground region in the center
    mask[2:4, 2:4, 2:4] = -1  # Set obstacles in another region

    # Example rays, where rays are in the form of (start_point, direction)
    rays = [((5, 5, 5), (1, 0, 0)),  # A ray going along the x-axis
            ((5, 5, 5), (0, 1, 0)),  # A ray going along the y-axis
            ((5, 5, 5), (0, 0, 1))]  # A ray going along the z-axis

    # Visualize and save the result
    visualize_rays_3d_with_obstacles_and_save(mask, rays, filename="rays_visualization.png")
    """
    # Create a figure and 3D axes
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Get the indices of the foreground and obstacles in the mask
    foreground_indices = np.argwhere(mask == target_vulue)
    obstacle_indices = np.argwhere(mask == obstacle_value)

    # Plot foreground points
    ax.scatter(foreground_indices[:, 0], foreground_indices[:, 1], foreground_indices[:, 2],
               color='g', marker='o', label='Foreground', alpha=0.6)

    # Plot obstacle points
    ax.scatter(obstacle_indices[:, 0], obstacle_indices[:, 1], obstacle_indices[:, 2],
               color='r', marker='x', label='Obstacles', alpha=0.6)

    # Plot rays
    for (x, y, z), direction in rays:
        ax.quiver(x, y, z, -direction[0], -direction[1], -direction[2], length=5, color='b', arrow_length_ratio=0.1)

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Add a legend
    ax.legend()

    # Save the plot to a file
    plt.savefig(filename)

    # Show the plot
    plt.tight_layout()
    plt.show()



def save_numpy_as_nii(numpy_array, reference_img, output_path):
    """
    Save a 3D NumPy array as a .nii.gz file using a reference NIfTI image for spatial metadata.

    Parameters:
        numpy_array (np.ndarray): 
            The 3D NumPy array representing the radiation field or volumetric data.
        
        reference_img (SimpleITK.Image): 
            A reference NIfTI image used to copy spatial metadata (e.g., origin, spacing, direction).
        
        output_path (str): 
            Full path where the new .nii.gz file will be saved.

    Raises:
        ValueError: 
            If the shape of the NumPy array does not match the reference image.
    """
    # Step 1: Validate array shape
    if numpy_array.shape != sitk.GetArrayFromImage(reference_img).shape:
        raise ValueError("The shape of the NumPy array does not match the reference image.")
    
    # Step 2: Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        # print(f"Created output directory: {output_dir}")
    
    # Step 3: Convert NumPy array to SimpleITK image
    output_img = sitk.GetImageFromArray(numpy_array)
    
    # Step 4: Copy spatial metadata from reference image
    output_img.CopyInformation(reference_img)
    
    # Step 5: Save the image
    sitk.WriteImage(output_img, output_path)
    # print(f"Image saved successfully to {output_path}")




def save_points_as_stl(points, filename="output.stl", radius=1):
    """
    Save an array of 3D points as small spheres in an STL file using VTK.

    Args:
        points (np.ndarray): An (n, 3) array of 3D points.
        filename (str): Output STL file name.
        radius (float): Radius of each sphere.
    """
    append_filter = vtk.vtkAppendPolyData()

    for p in points:
        # Create a sphere
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(p.tolist())
        sphere.SetRadius(radius)
        sphere.SetThetaResolution(16)  # Control sphere resolution
        sphere.SetPhiResolution(16)

        sphere.Update()
        append_filter.AddInputData(sphere.GetOutput())

    # Merge all spheres
    append_filter.Update()

    # Write to STL file
    stl_writer = vtk.vtkSTLWriter()
    stl_writer.SetFileName(filename)
    stl_writer.SetInputData(append_filter.GetOutput())
    stl_writer.Write()