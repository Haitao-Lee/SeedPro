import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
import core
import utils
import numpy as np
import visualizer


def brachy_plan(args):
    """
    Generate a brachytherapy radiation treatment plan by optimizing seed trajectories and placements, 
    followed by 3D visualization and exporting the results as STL files.

    Parameters:
        args (Namespace): 
            Configuration object containing parameters for radiation planning, seed placement, 
            dose constraints, and file paths.

    Workflow:
        1. **Image and Radiation Data Preparation:**  
            - Load tumor image (NIfTI format) and extract voxel spacing.  
            - Generate the radiation planning volume based on specified thresholds.  
            - Determine a reference direction for trajectory initialization.  

        2. **Trajectory Initialization:**  
            - Generate candidate trajectories for seed placement based on geometry and dose constraints.  

        3. **Optimal Plan Generation:**  
            - Optimize seed placements along the selected trajectories to meet dose coverage requirements.  
            - Refine the plan iteratively to minimize radiation exposure to healthy tissues.  

        4. **3D Visualization and Export:**  
            - Convert seed positions and directions into 3D STL files for visualization and further analysis.  

    Returns:
        list:
            - plan_res (list): Final optimized seed placement plan, including trajectories, seeds, and dose information.
    
    Output Files:
        - STL files representing the 3D position and direction of each seed are saved in the `./output` folder.
    """

    # --- Stage 1: Image and Radiation Data Preparation ---
    # Load the tumor image
    dose_image = utils.normalize_dose_image(utils.read_nii_image(args.dose_image_path), args.image_normalize[0], args.image_normalize[1], args.image_normalize[0], args.image_normalize[1])
    # target_image = utils.read_nii_image(args.target_image_path)

    # Generate the radiation planning volume based on threshold values
    radiation_volume = utils.get_planning_volume_array(
        args.target_image_path,
        args.radiation_array_params['target_value'],
        args.radiation_array_params['obstacle_value'],
        args.radiation_array_params['background_value'],
    )
    
    # Determine the reference direction for trajectory planning
    # ref_direc = utils.get_reference_direction(
    #     radiation_volume,
    #     args.radiation_array_params['target_value']
    # )
    ref_direc = np.array([1, 0, 0])  # Manually set direction if needed
    
    # --- Stage 2: Trajectory Initialization ---
    init_tracjectories = core.init_plan(
        dose_image,
        radiation_volume,
        ref_direc,
        args.direc_resolution,
        args.radiation_array_params['backlit_angle'],
        args.radiation_array_params['target_value'],
        args.radiation_array_params['background_value'],
        args.radiation_array_params['obstacle_value'],
        args.radiation_array_params['maximum_candidate_trajectories'],
        args.seed_info['length']
    )
    
    # --- Stage 3: Optimal Plan Generation ---
    plan_res = core.optimal_plan_rf(
        init_tracjectories,
        radiation_volume,
        dose_image,
        args.dl_params,
        args.rf_params,
        args.distance_filtter['interval_rate'],
        args.radiation_array_params['target_value'],
        args.radiation_array_params['infer_img_size'],
        args.in_lowest_energy,
        args.out_highest_energy,
        args.DVH_rate,
        args.seed_info,
        args.image_normalize[0], 
        args.image_normalize[1],
        args.image_normalize[2]
    )
    
    # --- Stage 4: 3D Visualization and Export ---
    utils.create_folder_if_not_exists('./output_rf')
    
    planned_seeds = []
    planned_seed_doses = []
    for res in plan_res:
        planned_seeds.append(res[1])
        planned_seed_doses.append(res[2])
        # all_dose += sum(res[2])
    
    all_dose = np.zeros_like(radiation_volume).astype(np.float32)

    for i, seeds in enumerate(planned_seeds):
        for j, seed in enumerate(seeds):
            pos, direction = seed
            
            # Save the seed geometry as an STL file
            visualizer.save_polydata_as_stl(
                visualizer.get_seed_polydata(
                    pos,
                    direction,
                    args.seed_info['length'],
                    args.seed_info['radius']
                ),
                f'./output_rf/{args.case_name}/seed_{i}_{j}.stl'
            )
            all_dose += planned_seed_doses[i][j]
            visualizer.save_numpy_as_nii(all_dose, dose_image, f'./output_rf/{args.case_name}/dose_{i}_{j}.nii.gz')
            # visualizer.save_numpy_as_nii(planned_seed_doses[i][j], dose_image, f'./output_rf/{args.case_name}/dose_{i}_{j}.nii.gz')
            
    return plan_res


if __name__ == '__main__':
    brachy_plan(config.setting())
