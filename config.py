# coding = utf-8
import argparse
import os
import torch
import sys

# 添加上级目录到系统路径，便于模块导入
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def setting():
    """
    Parse and configure command-line arguments for the radiation treatment planning system.

    This function uses argparse to define and parse parameters for data paths, seed properties, 
    planning configurations, deep learning parameters, and visualization settings.

    Returns:
        argparse.Namespace: A namespace containing all configuration parameters.
    """
    parser = argparse.ArgumentParser(description="Radiation Treatment Planning Configuration")

    # --- Global Settings ---
    parser.add_argument('--root', 
                        default=os.path.dirname(os.path.abspath(__file__)).replace('\\', '/'),
                        help="Root directory of the project.")

    # --- Data Paths ---
    parser.add_argument('--case_name',
                        default='pt_163', 
                        help="Identifier for the patient case being processed.")
    
    parser.add_argument('--dose_image_path', 
                        default='./data/pt_163_ct.nii.gz', 
                        help="Path to the dose image file (e.g., CT scan).")
    
    parser.add_argument('--target_image_path', 
                        default='./data/pt_163_label.nii.gz', 
                        help="Path to the target region label image.")
    
    parser.add_argument('--risk_image_path', 
                        default='./data/CTdaimingya segmentation-models-segmentation-label.nii.gz', 
                        help="Path to the risk volume image (if available).")
    
    parser.add_argument('--guide_dire', 
                        default='./data/guide_dir.npy',
                        help="Path to the guide direction data file.")
    
    parser.add_argument('--image_normalize', 
                        default=[-1000, 3000, 255],
                        help="Path to the guide direction data file.")

    # --- Seed Parameters ---
    parser.add_argument('--seed_info', 
                        default={'radius': 0.4,
                                 'length': 4.5,
                                 'margin_rate':1,
                                 'num_of_seeds': [5, 10],
                                 'seed_avr_dose': 50}, 
                        help="Seed properties including radius, length, quantity, and average radiation dose per seed.")

    # --- Radiation Planning Parameters ---
    parser.add_argument('--radiation_array_params', 
                        default={'target_value': 2,
                                 'obstacle_value': 3,
                                 'background_value': 0,
                                 'backlit_angle': 0.5,
                                 'maximum_candidate_trajectories': 500,
                                 'min_depth_rate': 5,
                                 'smooth_sigma': 1,
                                 'infer_img_size':(32, 32, 32)},
                        help="Radiation array parameters for dose distribution and obstacle avoidance.")
    
    parser.add_argument('--in_lowest_energy', 
                        default=1, 
                        help="Minimum required radiation dose for the target region (in Gray).")
    
    parser.add_argument('--out_highest_energy', 
                        default=1, 
                        help="Maximum allowable radiation dose for non-target regions (in Gray).")
    
    parser.add_argument('--distance_filtter', 
                        default={'lower_bound': 0.8,
                                 'upper_bound': 10,
                                 'distance_rate': 0.8,
                                 'interval_rate': 2}, 
                        help="Distance filtering settings for seed placement.")
    
    parser.add_argument('--iter_rate', 
                        default=2, 
                        help="Iteration rate for radiation dose adjustment (in Gray).")
    
    parser.add_argument('--rays_res', 
                        default=[10, 20], 
                        help="Resolution settings for radiation ray tracing.")
    
    parser.add_argument('--obstacle_res', 
                        default=[10, 20], 
                        help="Resolution settings for obstacle detection during planning.")
    
    parser.add_argument('--DVH_rate', 
                        default=0.9, 
                        help="Desired Dose Volume Histogram (DVH) coverage rate.")
    
    parser.add_argument('--max_iter', 
                        default=4, 
                        help="Maximum number of iterations allowed for planning optimization.")
    
    parser.add_argument('--replan_rate', 
                        default=0.6, 
                        help="Threshold for triggering replanning during optimization.")
    
    parser.add_argument('--direc_resolution', 
                        default=[30, 3, 2], 
                        help="Resolution for directional sampling during radiation planning.")

    # --- Deep Learning Parameters ---
    parser.add_argument('--dl_params', 
                        default={'lr': 4e-4,
                                 'lr_decay': 1.5,
                                 'epochs': 1000,
                                 'patience': 200,
                                 'delta': 1e-6,
                                 'verbose': False,
                                 'search_region': 0.5,
                                 'loss_weights': [1, 1, 1],
                                 'DVH_margin': 0.05,
                                 'dose_model_path': './dose_pre/dose_model.pth',
                                 'dose_spatial_dims': 3,
                                 'dose_in_channel': 3,
                                 'dose_out_channel': 1,
                                 'dose_cal_features': (16, 32, 64, 128, 256, 32),
                                 'multi_GPU': False,
                                 'device': torch.device("cuda:1" if torch.cuda.is_available() else "cpu")},
                        help="Deep learning parameters for dose prediction and optimization.")
    
    parser.add_argument('--rf_params',  
                        default={'lr': 1e-3,
                                 'gamma': 0.99,
                                 'max_episodes': 400,
                                 'print_every': 100,
                                 'bandwidth':200,
                                 'hierarchical_optimization': True,
                                 'segmented_rewards':True,
                                 'flip_ratio': 1
                                 },
                        help="Reinforcement planning parameters for dose prediction and optimization.")

    # --- Visualization Parameters ---
    parser.add_argument('--color', 
                        default=[128 / 255, 174 / 255, 128 / 255,
                                 216 / 255, 101 / 255, 79 / 255,
                                 230 / 255, 220 / 255, 70 / 255,
                                 183 / 255, 156 / 255, 220 / 255,
                                 111 / 255, 184 / 255, 210 / 255,
                                 241 / 255, 214 / 255, 145 / 255],
                        help="Color settings for visualizing different regions in the output.")

    return parser.parse_args()
