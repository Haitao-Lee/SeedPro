import utils
import numpy as np
import dose_pre.myDoseNet as myDoseNet
import torch
import torch.nn as nn
from scipy.ndimage import distance_transform_edt
import visualizer
import copy



def seed_plan(dose_image, radiation_volume, seed_info, in_lowest_dose, out_highest_dose, DVH_rate, direc_res, dl_params):
    """
    Generate an optimized radiotherapy seed placement plan using dose and volume constraints.

    This function iteratively places radiation seeds within the target area, ensuring minimum dose coverage 
    while adhering to the Dose-Volume Histogram (DVH) requirements. The optimization process incorporates 
    deep learning techniques for fine-tuning seed placement.

    Parameters:
        dose_image (nii.gz): 3D image (e.g., in NIfTI format) representing the target image for dose calculation.
        radiation_volume (numpy.ndarray): A binary 3D array representing the target volume for irradiation. 
                                          1 represents regions to be irradiated, and 0 represents non-target regions.
        seed_info (dict): Dictionary containing parameters for the radiation seed, including:
                          - 'length': Length of the seed.
                          - 'radius': Radius of the seed.
                          - 'seed_avr_dose': Average dose delivered by a single seed.
        in_lowest_dose (float): The minimum dose threshold for the target region to be considered treated.
        out_highest_dose (float): The maximum acceptable dose for the surrounding healthy tissue.
        DVH_rate (float): Desired Dose-Volume Histogram (DVH) rate, representing the percentage of the target 
                          volume that must meet or exceed the `in_lowest_dose`.
        direc_res (tuple): Angular resolution for seed direction sampling, with components:
                           (radial resolution, azimuthal resolution, polar resolution).
        dl_params (dict): Dictionary containing parameters for deep learning optimization, including:
                          - 'lr': Learning rate for optimization.
                          - 'epochs': Number of training epochs.
                          - 'patience': Early stopping patience.
                          - 'delta': Minimum improvement for early stopping.
                          - 'verbose': Verbosity level during training.
                          - 'device': Computational device (e.g., 'cpu' or 'cuda').
                          - 'loss_weights': Weights for different loss terms during optimization.
                          - 'search_region': Constraints defining the search region for optimization.

    Returns:
        tuple:
            - opti_planned_seeds (list): Optimized list of seed positions and directions after optimization.
            - opti_single_seed_radiations (list): Radiation distributions contributed by each seed in the optimized plan.
    """
    
    # Initialize seed parameters
    seed_sigma = (seed_info['length'] * 1, seed_info['radius'] * 3, seed_info['radius'] * 3)  # Seed radiation spread (in all directions)
    seed_avr_dose = seed_info['seed_avr_dose']  # Average radiation dose delivered by each seed

    # Initialize variables for tracking seed placement and optimization progress
    DVH_sign = False  # Flag indicating whether the DVH condition has been met
    cur_radiation = np.zeros(radiation_volume.shape)  # Current radiation distribution (initialized to zeros)
    single_seed_radiations = []  # List to store radiation fields from individual seeds
    planned_seeds = []  # List to store positions and directions of the placed seeds
    cur_DVH_rate = 0  # Current DVH rate (initialized to 0)
    mask_volume = (radiation_volume == 1).astype(float)  # Mask representing the target regions (where radiation is applied)
    volume = np.sum(radiation_volume == 1)  # Total volume of the target region (number of voxels)
    base_line_DVH = DVH_rate - dl_params['DVH_margin']  # Baseline DVH threshold, accounting for margin from parameters

    # Step 1: Seed placement loop
    # Iteratively place seeds until the desired DVH rate is achieved
    while not DVH_sign:
        # Calculate the next seed's position and direction based on current radiation coverage and target volume
        pos, direc, cur_seed_radiation, cur_radiation, cur_DVH_rate = utils.cal_next_seed_pos_direc(
            mask_volume, cur_radiation, in_lowest_dose, single_seed_radiations, seed_sigma, seed_avr_dose, direc_res)
        
        # Add the current seed's radiation contribution to the list of single seed radiations
        single_seed_radiations.append(cur_seed_radiation)
        # Record the position and direction of the placed seed
        planned_seeds.append([pos, direc])
        
        # Step 2: Check if the DVH condition has been satisfied
        DVH_sign = cur_DVH_rate >= base_line_DVH
        
        # Print progress: Current number of seeds and current DVH rate
        print(f'Current seeds: {len(planned_seeds)}, Current DVH rate: {cur_DVH_rate}')

    # Step 3: Optimization to reduce seed count while maintaining DVH rate
    # Optimize seed distribution by minimizing the number of seeds while preserving DVH requirements
    opti_single_seed_radiations, opti_planned_seeds, opti_radiation = utils.update_seeds(
        single_seed_radiations, planned_seeds)

    # Step 4: Deep learning-based re-optimization loop to refine seed placement
    DVH_sign = False
    replan_count = 0
    minus_sign = False  # Flag to check if removal of seeds is necessary
    seeds_variation = []  # Track variations in seed placement during optimization
    
    while not DVH_sign:
        # Initial re-optimization process (for the first iteration)
        if replan_count == 0:
            tmp_single_seed_radiations, tmp_planned_seeds, tmp_radiation = utils.update_seeds(
                opti_single_seed_radiations, opti_planned_seeds)
        
        # Recompute radiation coverage and DVH rate without any removed seed
        tmp_DVH_rate = np.sum(tmp_radiation * mask_volume > in_lowest_dose) / volume
        DVH_sign = tmp_DVH_rate >= DVH_rate  # Check if DVH condition is met
        best_single_seed_radiations = []  # List to store best radiation distributions
        best_planned_seeds = []  # List to store best seed positions and directions
        
        # Step 5: If DVH condition not satisfied, apply deep learning optimization
        if not DVH_sign:
            best_planned_seeds, best_single_seed_radiations, best_DVH_rate, seeds_variation = utils.deep_learning_optimization(
                tmp_planned_seeds, radiation_volume, cur_radiation, mask_volume, in_lowest_dose, out_highest_dose, 
                DVH_rate, volume, seed_sigma, seed_avr_dose, dl_params, seeds_variation)
            
            DVH_sign = best_DVH_rate > DVH_rate  # Check if optimization meets DVH requirement
        else:
            # If DVH condition is satisfied, update the optimized seed positions and radiation distributions
            best_single_seed_radiations, best_planned_seeds, _ = utils.update_seeds(
                tmp_single_seed_radiations, tmp_planned_seeds)

        # Step 6: Seed removal strategy if necessary (to reduce seed count)
        if not DVH_sign:
            if not minus_sign:
                # Update optimized seed positions and radiation values
                opti_single_seed_radiations, opti_planned_seeds, opti_radiation = utils.update_seeds(
                    best_single_seed_radiations, best_planned_seeds)

                # Calculate the next seed's position and direction
                pos, direc, cur_seed_radiation, cur_radiation, cur_DVH_rate = utils.cal_next_seed_pos_direc(
                    mask_volume, cur_radiation, in_lowest_dose, single_seed_radiations, seed_sigma, seed_avr_dose, direc_res)

                # Add the best seed's radiation contribution and check if DVH condition is met
                opti_single_seed_radiations.append(cur_seed_radiation)
                opti_planned_seeds.append([pos, direc])
                tmp_single_seed_radiations, tmp_planned_seeds, tmp_radiation = utils.update_seeds(
                    opti_single_seed_radiations, opti_planned_seeds)
            else:
                DVH_sign = True
        else:
            # If DVH condition is met, finalize the optimization and update seed positions
            opti_single_seed_radiations, opti_planned_seeds, opti_radiation = utils.update_seeds(
                best_single_seed_radiations, best_planned_seeds)
            if replan_count == 0:
                minus_sign = True
                
            if minus_sign:
                # Identify and remove the seed contributing the highest dose
                idx = utils.get_highest_pos_index(opti_planned_seeds, opti_radiation * mask_volume)[0]
                tmp_single_seed_radiations = copy.deepcopy(opti_single_seed_radiations)
                tmp_planned_seeds = copy.deepcopy(opti_planned_seeds)
                dl_params['lr'] = dl_params['lr'] / dl_params['lr_decay']

                # Remove the highest-dose seed and recompute radiation distribution
                tmp_single_seed_radiations.pop(idx)
                tmp_planned_seeds.pop(idx)
                tmp_radiation = np.sum(np.asarray(tmp_single_seed_radiations), axis=0)
                tmp_DVH_rate = np.sum(tmp_radiation * mask_volume > in_lowest_dose) / volume
                DVH_sign = False
        
        # Track the number of iterations for re-optimization
        replan_count += 1               

    # Final DVH rate after optimization
    opt_DVH_rate = np.sum(opti_radiation * mask_volume > in_lowest_dose) / volume  
    # Optionally visualize the optimized seed placements (e.g., saving a figure)
    utils.draw_radiations(radiation_volume, opti_single_seed_radiations, threshold=in_lowest_dose, sample_fraction = 1, save_path='./fig')
    
    return opti_planned_seeds, opti_single_seed_radiations



def seed_plan_v2(dose_image, radiation_volume, seed_info, in_lowest_dose, out_highest_dose, DVH_rate, direc_res, dl_params):
    """
    Generate an optimized radiotherapy seed placement plan using dose and volume constraints.

    This function iteratively places radiation seeds within the target area, ensuring minimum dose coverage 
    while adhering to the Dose-Volume Histogram (DVH) requirements. The optimization process incorporates 
    deep learning techniques for fine-tuning seed placement.

    Parameters:
        dose_image (nii.gz): 3D image (e.g., in NIfTI format) representing the target dose distribution for radiation.
        radiation_volume (numpy.ndarray): A binary 3D array representing the target volume for irradiation. 
                                          1 represents regions to be irradiated, and 0 represents non-target regions.
        seed_info (dict): Dictionary containing parameters for the radiation seed, including:
                          - 'length': Length of the seed.
                          - 'radius': Radius of the seed.
                          - 'seed_avr_dose': Average dose delivered by a single seed.
        in_lowest_dose (float): The minimum dose threshold for the target region to be considered treated.
        out_highest_dose (float): The maximum acceptable dose for the surrounding healthy tissue.
        DVH_rate (float): Desired Dose-Volume Histogram (DVH) rate, representing the percentage of the target 
                          volume that must meet or exceed the `in_lowest_dose`.
        direc_res (tuple): Angular resolution for seed direction sampling, with components:
                           (radial resolution, azimuthal resolution, polar resolution).
        dl_params (dict): Dictionary containing parameters for deep learning optimization, including:
                          - 'lr': Learning rate for optimization.
                          - 'epochs': Number of training epochs.
                          - 'patience': Early stopping patience.
                          - 'delta': Minimum improvement for early stopping.
                          - 'verbose': Verbosity level during training.
                          - 'device': Computational device (e.g., 'cpu' or 'cuda').
                          - 'loss_weights': Weights for different loss terms during optimization.
                          - 'search_region': Constraints defining the search region for optimization.

    Returns:
        tuple:
            - opti_planned_seeds (list): Optimized list of seed positions and directions after optimization.
            - opti_single_seed_radiations (list): Radiation distributions contributed by each seed in the optimized plan.
    """
    
    # Initialize seed parameters
    seed_sigma = (seed_info['length'] * 3, seed_info['radius'] * 20, seed_info['radius'] * 20)  # Seed radiation spread (in all directions)
    seed_avr_dose = seed_info['seed_avr_dose']  # Average radiation dose delivered by each seed

    # Initialize variables for tracking seed placement and optimization progress
    DVH_sign = False  # Flag indicating whether the DVH condition has been met
    cur_radiation = np.zeros(radiation_volume.shape)  # Current radiation distribution (initialized to zeros)
    single_seed_radiations = []  # List to store radiation fields from individual seeds
    planned_seeds = []  # List to store positions and directions of the placed seeds
    cur_DVH_rate = 0  # Current DVH rate (initialized to 0)
    mask_volume = (radiation_volume == 1).astype(float)  # Mask representing the target regions (where radiation is applied)
    volume = np.sum(radiation_volume == 1)  # Total volume of the target region (number of voxels)
    base_line_DVH = DVH_rate - dl_params['DVH_margin']  # Baseline DVH threshold, accounting for margin from parameters
    
    stride = sub_patch_size = input_shape = mask_volume.shape
    dose_cal_model = myDoseNet.myDoseNet(spatial_dims = dl_params['dose_spatial_dims'], 
                                         in_channels = dl_params['dose_in_channel'],
                                         out_channels = dl_params['dose_out_channel'], 
                                         features = dl_params['dose_cal_features'])
    
    if dl_params['multi_GPU']:
        dose_cal_model = torch.nn.DataParallel(dose_cal_model)
    dose_cal_model.load_state_dict(torch.load(dl_params['dose_model_path'], map_location=dl_params['device']))

    # Step 1: Seed placement loop
    # Iteratively place seeds until the desired DVH rate is achieved
    while not DVH_sign:
        # Calculate the next seed's position and direction based on current radiation coverage and target volume
        pos, direc, cur_seed_radiation, cur_radiation, cur_DVH_rate = utils.cal_next_seed_pos_direc_v2(
            dose_image, dose_cal_model, mask_volume, cur_radiation, in_lowest_dose, single_seed_radiations, direc_res
        )
        
        # Add the current seed's radiation contribution to the list of single seed radiations
        single_seed_radiations.append(cur_seed_radiation)
        # Record the position and direction of the placed seed
        planned_seeds.append([pos, direc])
        
        # Step 2: Check if the DVH condition has been satisfied
        DVH_sign = cur_DVH_rate >= base_line_DVH
        
        # Print progress: Current number of seeds and current DVH rate
        print(f'Current seeds: {len(planned_seeds)}, Current DVH rate: {cur_DVH_rate}')
        utils.draw_radiations(radiation_volume, single_seed_radiations, threshold=in_lowest_dose, save_path='./fig')



    # Step 3: Optimization to reduce seed count while maintaining DVH rate
    # Optimize seed distribution by minimizing the number of seeds while preserving DVH requirements
    opti_single_seed_radiations, opti_planned_seeds, opti_radiation = utils.update_seeds(
        single_seed_radiations, planned_seeds)

    # Step 4: Deep learning-based re-optimization loop to refine seed placement
    DVH_sign = False
    replan_count = 0
    minus_sign = False  # Flag to check if removal of seeds is necessary
    seeds_variation = []  # Track variations in seed placement during optimization
    
    while not DVH_sign:
        # Initial re-optimization process (for the first iteration)
        if replan_count == 0:
            tmp_single_seed_radiations, tmp_planned_seeds, tmp_radiation = utils.update_seeds(
                opti_single_seed_radiations, opti_planned_seeds)
        
        # Recompute radiation coverage and DVH rate without any removed seed
        tmp_DVH_rate = np.sum(tmp_radiation * mask_volume > in_lowest_dose) / volume
        DVH_sign = tmp_DVH_rate >= DVH_rate  # Check if DVH condition is met
        best_single_seed_radiations = []  # List to store best radiation distributions
        best_planned_seeds = []  # List to store best seed positions and directions
        
        # Step 5: If DVH condition not satisfied, apply deep learning optimization
        if not DVH_sign:
            best_planned_seeds, best_single_seed_radiations, best_DVH_rate, seeds_variation = utils.deep_learning_optimization(
                tmp_planned_seeds, radiation_volume, cur_radiation, mask_volume, in_lowest_dose, out_highest_dose, 
                DVH_rate, volume, seed_sigma, seed_avr_dose, dl_params, seeds_variation)
            
            DVH_sign = best_DVH_rate > DVH_rate  # Check if optimization meets DVH requirement
        else:
            # If DVH condition is satisfied, update the optimized seed positions and radiation distributions
            best_single_seed_radiations, best_planned_seeds, _ = utils.update_seeds(
                tmp_single_seed_radiations, tmp_planned_seeds)

        # Step 6: Seed removal strategy if necessary (to reduce seed count)
        if not DVH_sign:
            if not minus_sign:
                # Update optimized seed positions and radiation values
                opti_single_seed_radiations, opti_planned_seeds, opti_radiation = utils.update_seeds(
                    best_single_seed_radiations, best_planned_seeds)

                # Calculate the next seed's position and direction
                pos, direc, cur_seed_radiation, cur_radiation, cur_DVH_rate = utils.cal_next_seed_pos_direc_v2(
                    mask_volume, cur_radiation, in_lowest_dose, single_seed_radiations, seed_sigma, seed_avr_dose, direc_res)

                # Add the best seed's radiation contribution and check if DVH condition is met
                opti_single_seed_radiations.append(cur_seed_radiation)
                opti_planned_seeds.append([pos, direc])
                tmp_single_seed_radiations, tmp_planned_seeds, tmp_radiation = utils.update_seeds(
                    opti_single_seed_radiations, opti_planned_seeds)
            else:
                DVH_sign = True
        else:
            # If DVH condition is met, finalize the optimization and update seed positions
            opti_single_seed_radiations, opti_planned_seeds, opti_radiation = utils.update_seeds(
                best_single_seed_radiations, best_planned_seeds)
            if replan_count == 0:
                minus_sign = True
                
            if minus_sign:
                # Identify and remove the seed contributing the highest dose
                idx = utils.get_highest_pos_index(opti_planned_seeds, opti_radiation * mask_volume)[0]
                tmp_single_seed_radiations = opti_single_seed_radiations.copy()
                tmp_planned_seeds = opti_planned_seeds.copy()
                dl_params['lr'] = dl_params['lr'] / dl_params['lr_decay']

                # Remove the highest-dose seed and recompute radiation distribution
                tmp_single_seed_radiations.pop(idx)
                tmp_planned_seeds.pop(idx)
                tmp_radiation = np.sum(np.asarray(tmp_single_seed_radiations), axis=0)
                tmp_DVH_rate = np.sum(tmp_radiation * mask_volume > in_lowest_dose) / volume
                DVH_sign = False
        
        # Track the number of iterations for re-optimization
        replan_count += 1               

    # Final DVH rate after optimization
    opt_DVH_rate = np.sum(opti_radiation * mask_volume > in_lowest_dose) / volume  
    # Optionally visualize the optimized seed placements (e.g., saving a figure)   
    utils.draw_radiations(radiation_volume, opti_single_seed_radiations, threshold=in_lowest_dose, save_path='./fig')
    return opti_planned_seeds, opti_single_seed_radiations




def trajectory_plan(radiation_volume, target_val, obs_val, back_val, angle_range, ref_direc=None, sigma=None):
    """
    Plan optimal paths for particle implantation with minimal invasiveness, considering angular constraints.
    
    This function determines potential trajectories for implanting particles within a specified 3D volume. 
    The aim is to minimize the number of insertion points while adhering to specified angle constraints relative
    to a reference direction. It employs ray tracing to identify feasible paths through the volume.

    Parameters:
        radiation_volume (numpy.ndarray): A 3D array representing the space for particle implantation.
            - `target_val` identifies regions where particles should be implanted.
            - `obs_val` marks obstacles that must be avoided.
            - `back_val` denotes free space where paths can conclude.

        target_val (int or float): The value designating target regions for particle implantation.

        obs_val (int or float): The value used to represent obstacles in the volume.

        back_val (int or float): The value indicating background areas where valid paths can terminate.

        angle_range (float): Maximum allowable deviation in degrees between the trajectories and the reference direction.

        ref_direc (array-like, optional): An optional vector specifying the reference direction for trajectories.
            Paths must adhere to the specified angle constraints with respect to this direction if given.

        sigma (optional): Unspecified optional parameter for potential additional functionality.

    Returns:
        list: A list of dictionaries, each detailing a planned insertion trajectory:
            - 'start': Starting point of the trajectory (numpy.ndarray).
            - 'end': Ending point of the trajectory (numpy.ndarray).
            - 'particles': A list of indices indicating particles implanted along each trajectory.

    Notes:
        - The function visualizes and saves a 3D graphical representation of candidate trajectories and obstacles.
        - The visualization is saved as an image file ('./fig/rays.png').

    """
    # Generate potential trajectory paths in the specified volume
    candidate_trajectories = utils.generate_dense_rays_from_radiation_volume(
        radiation_volume, target_val, obs_val, back_val, angle_range, ref_direc, sigma
    )

    # Visualize the generated trajectories alongside obstacles and save the visualization
    visualizer.visualize_rays_3d_with_obstacles_and_save(
        radiation_volume, candidate_trajectories, obs_val, filename='./fig/rays.png'
    )

    sorted_trajectories = utils.sort_candidate_tracjectories(radiation_volume, candidate_trajectories)

    # Further processing of candidate_trajectories is needed
    pass



def init_plan(dose_image, radiation_volume, ref_direc, direc_resolution, extract_angle,
              target_value, background_value, obstacle_value, maximum_candidate_trajectories,
              min_depth=2):
    """
    Generate an initial set of candidate needle/catheter trajectories within a 3-D radiation volume.

    Workflow:
        1. Build a conical sampling grid around the reference direction.
        2. Extract candidate voxels that lie inside the cone and satisfy the target label.
        3. Initialise straight trajectories through those voxels and prune by minimum depth.
        4. (Optional) Sort or visualise results before returning.

    Parameters
    ----------
    dose_image : SimpleITK.Image
        Reference image providing voxel spacing and spatial metadata.
    radiation_volume : np.ndarray
        3-D label array (target, background, obstacle) used for geometric queries.
    ref_direc : np.ndarray, shape (3,)
        Unit vector that defines the central axis of the sampling cone.
    direc_resolution : list[float, float, int]
        [cone_half_angle, angular_step, n_rings] for direction sampling.
    extract_angle : float (radians)
        Half-angle of the cone inside which candidate voxels are collected.
    target_value : float
        Label value that identifies the target region.
    background_value : float
        Label value that identifies non-target soft tissue.
    obstacle_value : float
        Label value that identifies organs-at-risk or hard obstacles.
    maximum_candidate_trajectories : int
        Upper bound on the total number of trajectories to be generated.
    min_depth : float, optional
        Minimum path length (mm) required for a trajectory to be considered valid.

    Returns
    -------
    list[tuple]
        List of valid trajectories. Each tuple contains:
        (origin_voxel, direction_vector, depth, ...metadata...)
        Ready for downstream optimisation or refinement.
    """
    # ---- 1.  Build conical direction grid ----
    candidate_dirs = utils.get_cone(
        ref_direc, direc_resolution[0], direc_resolution[1], direc_resolution[2]
    )

    # ---- 2.  Extract candidate voxels inside cone ----
    max_points_num = maximum_candidate_trajectories // len(candidate_dirs)
    close_points, max_length = utils.get_close_points(
        dose_image, radiation_volume, ref_direc, target_value, extract_angle
    )

    # Relax cone angle if too few points
    while close_points.shape[0] > max_points_num:
        extract_angle *= 1.1
        close_points, max_length = utils.get_close_points(
            dose_image, radiation_volume, ref_direc, target_value, extract_angle
        )

    # ---- 3.  Initialise trajectories with depth filter ----
    init_trajectories = []
    for direc in candidate_dirs:
        traj_list = utils.init_trajectories_with_depth(
            close_points, radiation_volume, direc, target_value,
            background_value, obstacle_value, min_depth, max_length
        )
        init_trajectories += traj_list

    # ---- 4.  Optional post-processing (commented) ----
    # sorted_trajectories = utils.sort_candidate_trajectories_by_depth(init_trajectories)
    # visual_list = [t[:2] for t in init_trajectories]
    # visualizer.visualize_rays_3d_with_obstacles_and_save(
    #     radiation_volume, visual_list, target_value, obstacle_value,
    #     filename='./fig/rays.png'
    # )

    return init_trajectories



def optimal_plan(init_trajectories, radiation_volume, dose_image, dl_params, lower_bound, upper_bound, distance_rate, 
                 target_value, background_value, obstacle_value, infer_img_size, in_lowest_dose, out_highest_dose, 
                 DVH_rate, seed_info, iter_rate, image_normalize_min, image_normalize_max, image_normalize_scale):
    """
    Generate an optimized radiation treatment plan by selecting seed trajectories, placing seeds, and refining the plan 
    to ensure effective tumor coverage while minimizing radiation exposure to healthy tissues.

    Parameters:
        init_trajectories (list): A list of initial candidate trajectories for seed placement.
        radiation_volume (ndarray): A 3D array representing the radiation distribution within the treatment area.
        dose_image (SimpleITK.Image): A dose image used to extract voxel spacing for precise seed placement.
        lower_bound (float): Minimum allowable distance between seeds for placement validation.
        upper_bound (float): Maximum allowable distance between seeds for placement validation.
        distance_rate (float): A threshold ratio to filter and optimize trajectory selection.
        target_value (float): The value representing the tumor region in the radiation volume.
        background_value (float): The value representing non-tumor regions in the radiation volume.
        obstacle_value (float): The value representing obstacles (e.g., critical organs) in the radiation volume.
        in_lowest_dose (float): Minimum radiation dose required for tumor treatment (in Gray).
        out_highest_dose (float): Maximum allowable radiation dose for surrounding healthy tissues (in Gray).
        DVH_rate (float): Target Dose Volume Histogram (DVH) coverage rate for tumor regions.
        seed_info (tuple): A tuple containing properties of the seeds (e.g., size, length, radiation intensity).
        iter_rate (int): The iteration multiplier for refining seed placement and minimizing dangerous radiation exposure.
        image_normalize_min (float): Minimum value for normalizing image intensity.
        image_normalize_max (float): Maximum value for normalizing image intensity.
        image_normalize_scale (float): Scaling factor for image intensity normalization.

    Returns:
        tuple:
            - opti_res (list): The final optimized plan containing refined trajectories, seed placements, and radiation distributions.
            - init_planned_res (list): The initial plan before refinement, including trajectories, seeds, and radiation data.

    Stages:
        1. **Trajectory Selection and Initial Planning**: Iteratively select optimal trajectories and place seeds to achieve the target DVH rate.
        2. **Plan Refinement**: Remove ineffective seeds, refine trajectory placements, and ensure adequate radiation coverage.
        3. **Fine-tuning for Safety**: Adjust seed placements iteratively to minimize excessive radiation exposure to healthy tissue regions.
    """

    # --- Initialize Variables ---
    candidate_trajectories = copy.deepcopy(init_trajectories)
    init_planned_res = []  # Stores the initial planned trajectories, seeds, and radiation values
    cur_DVH_rate = 0  # Current DVH coverage rate
    cur_radiation = np.zeros_like(radiation_volume).astype(float)  # Initialize the radiation distribution field
    distance_map = distance_transform_edt((radiation_volume == target_value))  # Compute the distance map for the tumor region
    
    # Initialize the dose calculation model
    dose_cal_model = myDoseNet.myDoseNet(spatial_dims=dl_params['dose_spatial_dims'], 
                                         in_channels=dl_params['dose_in_channel'],
                                         out_channels=dl_params['dose_out_channel'], 
                                         features=dl_params['dose_cal_features'])
    
    if dl_params['multi_GPU']:
        dose_cal_model = torch.nn.DataParallel(dose_cal_model)  # Enable multi-GPU support if specified
    dose_cal_model.load_state_dict(torch.load(dl_params['dose_model_path'], map_location=dl_params['device']))
    dose_cal_model.to(dl_params['device'])
    
    # --- Stage 1: Trajectory Selection and Initial Planning ---
    selected_indices = []  # Store the indices of selected trajectories
    while cur_DVH_rate < DVH_rate:
        # Select the optimal trajectory based on current radiation distribution
        optimal_trajectory, selected_idx = utils.select_optimal_trajectory(
            candidate_trajectories,
            [traj for traj, _, _ in init_planned_res],
            cur_radiation,
            dose_image,
            lower_bound,
            upper_bound,
            distance_rate,
            in_lowest_dose,
            distance_map,
            seed_info,
            selected_indices
        )
        if optimal_trajectory is None:
            print("No seeds can be placed along the selected trajectory, The dose requirement is too high.")
            return init_planned_res, init_planned_res
        selected_indices.append(selected_idx)
        # Place seeds along the selected trajectory and calculate the radiation distribution
        optimal_seeds, cur_DVH_rate, cur_single_seed_radiations = utils.put_seeds(
            radiation_volume,
            dose_image,
            dose_cal_model,
            infer_img_size,
            cur_radiation,
            target_value,
            in_lowest_dose,
            optimal_trajectory,
            seed_info,
            DVH_rate,
            distance_map,
            image_normalize_min,
            image_normalize_max,
            image_normalize_scale
        )
        
        if len(optimal_seeds) == 0:
            print("No seeds can be placed along the selected trajectory, The dose requirement is too high.")
            return init_planned_res, init_planned_res
        
        # Append the selected trajectory and seed placements to the initial plan
        init_planned_res.append([optimal_trajectory, optimal_seeds, cur_single_seed_radiations])
        cur_radiation += np.sum(cur_single_seed_radiations, axis=0)  # Update the radiation distribution

    # --- Stage 2: Plan Refinement ---
    minus_res = copy.deepcopy(init_planned_res)
    # Remove seeds and radiation from the refined plan to prepare for re-planning
    for i in range(len(minus_res)):
        minus_res[i][1] = []
        minus_res[i][2] = []
    
    cur_DVH_rate = 0
    minus_radiation = np.zeros_like(radiation_volume)
    
    while cur_DVH_rate < DVH_rate:
        # Refine the plan by removing ineffective seeds and adjusting placements
        minus_res, updated_DVH_rate, minus_radiation, sign = utils.replan(
            minus_res,
            radiation_volume,
            minus_radiation,
            dose_image,
            dose_cal_model,
            infer_img_size,
            in_lowest_dose,
            target_value,
            background_value,
            obstacle_value,
            seed_info,
            distance_map,
            image_normalize_min,
            image_normalize_max,
            image_normalize_scale
        )
        if sign:
            cur_DVH_rate = updated_DVH_rate  # Update DVH rate if re-planning is successful
        else:
            minus_res = copy.deepcopy(init_planned_res)  # Reset if re-planning failed
            minus_radiation = cur_radiation
            break
        
    # --- Stage 3: Fine-tuning for Safety ---
    opti_res = copy.deepcopy(minus_res)
    all_seeds = []
    for _, (_, seeds, _) in enumerate(opti_res):
        all_seeds.extend(seeds)
    opti_radiation = copy.deepcopy(minus_radiation)
    iter_count = 0
    seed_num = sum(len(seeds) for _, seeds, _ in minus_res)
    
    while iter_count < iter_rate * seed_num:
        # Remove improper seeds that could cause excessive radiation exposure
        # rest_res, rest_radiation = utils.remove_unproper_seed(
        #     opti_res,
        #     radiation_volume,
        #     minus_radiation,
        #     out_highest_dose,
        #     target_value,
        #     background_value,
        #     obstacle_value
        # )
    
        # Remove seeds sequentially to minimize radiation exposure
        rest_res, rest_radiation = utils.remove_seed_sequentially(
            opti_res,
            all_seeds,
            iter_count % seed_num,
            opti_radiation,
        )
        
        # Add proper seeds to ensure sufficient tumor coverage and minimize radiation to healthy tissues
        add_res, add_radiation, sign = utils.add_proper_seed(
            rest_res,
            radiation_volume,
            rest_radiation,
            dose_image,
            dose_cal_model,
            infer_img_size,
            in_lowest_dose,
            out_highest_dose,
            target_value,
            background_value,
            obstacle_value,
            DVH_rate,
            seed_info,
            distance_map,
            image_normalize_min,
            image_normalize_max,
            image_normalize_scale
        )
        
        if sign:
            opti_res = copy.deepcopy(add_res)
            opti_radiation = copy.deepcopy(add_radiation)
        
        iter_count += 1 
        if iter_count % seed_num == 0:
            all_seeds = []
            for _, (_, seeds, _) in enumerate(opti_res):
                all_seeds.extend(seeds)

    opti_radiation = np.zeros_like(radiation_volume).astype(float)
    # Collect radiation data from all the optimized seeds
    seed_radiations = []
    for res in opti_res:
        for seed_radiation in res[2]:
            seed_radiations.append(seed_radiation)
            opti_radiation += seed_radiation
    
    # Visualize and save radiation distributions
    utils.draw_radiations(radiation_volume, seed_radiations, target_value, threshold=in_lowest_dose, save_path='./fig')
    visualizer.save_numpy_as_nii(opti_radiation, dose_image, './output/dose.nii.gz')
    
    return opti_res, init_planned_res


def optimal_plan_rf(
    init_trajectories,
    radiation_volume,
    dose_image,
    dl_params,
    rf_params,
    interval_rate,
    target_value,
    infer_img_size,
    in_lowest_dose,
    out_highest_dose,
    DVH_rate,
    seed_info,
    image_normalize_min,
    image_normalize_max,
    image_normalize_scale,
):
    """
    Hierarchical reinforcement-learning pipeline for prostate/LDR brachytherapy.
    
    Steps:
        1. Build distance map for fast collision query.
        2. Load & wrap CNN-based single-seed dose engine.
        3. Call hierarchical REINFORCE to return the optimal needle trajectory
           and seed positions that maximise target coverage while respecting
           healthy-tissue constraints.
    
    Parameters
    ----------
    init_trajectories : list[Trajectory]
        Initial candidate needle paths (start-voxel, direction, meta).
    radiation_volume : np.ndarray
        3-D segmentation mask (target_value labels tumour).
    dose_image : SimpleITK.Image
        Reference image yielding voxel spacing and world mapping.
    dl_params : dict
        {'dose_spatial_dims': int,
         'dose_in_channel': int,
         'dose_out_channel': int,
         'dose_cal_features': list[int],
         'multi_GPU': bool,
         'dose_model_path': str,
         'device': torch.device}
    rf_params : dict
        {'lr': float, 'gamma': float, 'episodes': int} for REINFORCE.
    interval_rate : float
        Step-density factor when generating sub-positions.
    target_value : float
        Voxel intensity that identifies tumour.
    infer_img_size : tuple[int, int, int]
        CNN input patch size (H, W, D).
    in_lowest_dose : float
        DVH coverage threshold [Gy].
    out_highest_dose : float
        OAR penalty threshold [Gy].
    DVH_rate : float
        Required target-cover fraction.
    seed_info : dict
        {'length': float, 'radius': float, ...} seed geometry & activity.
    image_normalize_{min,max,scale} : float
        Intensity pre-processing constants for CNN.

    Returns
    -------
    optimal_res : tuple
        (high_action, planned_positions, dose, DVH_rate) produced by
        hierarchical REINFORCE.
    """

    # ---- 1.  distance transform for fast exclusion-zone query ----
    distance_map = distance_transform_edt(radiation_volume == target_value)

    # ---- 2.  build & load single-seed dose network ----
    dose_cal_model = myDoseNet.myDoseNet(
        spatial_dims=dl_params['dose_spatial_dims'],
        in_channels=dl_params['dose_in_channel'],
        out_channels=dl_params['dose_out_channel'],
        features=dl_params['dose_cal_features'],
    )

    if dl_params['multi_GPU']:
        dose_cal_model = nn.DataParallel(dose_cal_model)

    dose_cal_model.load_state_dict(
        torch.load(dl_params['dose_model_path'], map_location=dl_params['device'])
    )
    dose_cal_model.to(dl_params['device'])
    dose_cal_model.eval()  # switch to inference mode

    # ---- 3.  hierarchical RL planner ----
    optimal_res, _ = utils.hierarchical_planning_rf(
        candidate_trajectories=copy.deepcopy(init_trajectories),
        seed_info=seed_info,
        interval_rate=interval_rate,
        rf_params=rf_params,
        radiation_volume=radiation_volume,
        dose_image=dose_image,
        dose_cal_model=dose_cal_model,
        infer_img_size=infer_img_size,
        target_value=target_value,
        in_lowest_dose=in_lowest_dose,
        out_highest_dose=out_highest_dose,
        DVH_rate=DVH_rate,
        distance_map=distance_map,
        image_normalize_min=image_normalize_min,
        image_normalize_max=image_normalize_max,
        image_normalize_scale=image_normalize_scale,
    )

    return optimal_res

        
        




        
    




    
    
    
    
    

    
    


    

    
    



