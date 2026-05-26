import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import geometry


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.fc2 = nn.Linear(out_channels, out_channels)
        self.activation = nn.ReLU()
        self.skip_connection = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.skip_connection(x)  # Skip connection
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        x += residual  # Adding the residual to the output (skip connection)
        x = self.activation(x)
        return x



class BrachyPlanNet(nn.Module):
    def __init__(self, search_region, num_blocks=20):
        super(BrachyPlanNet, self).__init__()
        self.fc1 = nn.Linear(1, 72)  # Input -> Hidden layer 1
        self.weight = search_region
        # Create a sequence of residual blocks
        layers = []
        in_channels = 72
        for _ in range(num_blocks):
            layers.append(ResidualBlock(in_channels, 144))
            in_channels = 144  # Keep input and output channels the same for each block

        self.res_blocks = nn.Sequential(*layers)  # Stack all residual blocks

        self.fc2 = nn.Linear(144, 1)  # Hidden layer -> Output layer

    # most important
    def normalize(self, out, x):
            assert out.numel() % 3 == 0, "Input vector length must be a multiple of 3."
            out = out.view(-1, 3)  # Reshape to (n, 3)
            x = x.view(-1, 3)
            
            # Create masks for even/odd groups
            even_mask = torch.arange(out.size(0), device=out.device) % 2 == 0
            odd_mask = ~even_mask

            # Normalize even groups
            # even_norms = torch.linalg.norm(out[even_mask], dim=1, keepdim=True) + 1e-6  # Add small epsilon to avoid division by zero
            # even_norm_normalized = out[even_mask] / even_norms
            
            # Normalize odd groups
            odd_norms = torch.linalg.norm(out[odd_mask], dim=1, keepdim=True) + 1e-6  # Add small epsilon to avoid division by zero
            odd_norm_normalized = out[odd_mask] / odd_norms
            
            # Apply sigmoid to even groups
            sigmoid_normalized = torch.clamp(self.weight*(2*torch.sigmoid(out[even_mask])-1) + x[even_mask], 0, 1)

            # Combine results
            output_vector = torch.zeros_like(out)
            output_vector[even_mask] = sigmoid_normalized
            output_vector[odd_mask] = odd_norm_normalized
            output_vector = output_vector.view(-1, 1)

            return output_vector
        
    # def normalize(self, x):
    #     assert x.numel() % 3 == 0, "Input vector length must be a multiple of 3."
    #     x = x.view(-1, 3)  # Reshape to (n, 3)
        
    #     # Create masks for even/odd groups
    #     even_mask = torch.arange(x.size(0), device=x.device) % 2 == 0
    #     odd_mask = ~even_mask

    #     # Apply sigmoid to even groups
    #     sigmoid_normalized = torch.sigmoid(x[even_mask])

    #     # Normalize odd groups
    #     norms = torch.linalg.norm(x[odd_mask], dim=1, keepdim=True) + 1e-6  # Add small epsilon to avoid division by zero
    #     norm_normalized = x[odd_mask] / norms

    #     # Combine results
    #     output_vector = torch.zeros_like(x)
    #     output_vector[even_mask] = sigmoid_normalized
    #     output_vector[odd_mask] = norm_normalized
    #     output_vector = output_vector.view(-1, 1)
    #     # print(output_vector.shape)
    #     return output_vector

    def forward(self, x):
        x = x.view(-1, 1)  # Flatten the input (nx1) tensor
        out = self.fc1(x)  # First fully connected layer
        out = self.res_blocks(out)  # Pass through all residual blocks
        out = self.fc2(out)
        return self.normalize(out, x)
        
        

def initialize_to_zero_output(model):
    """
    Initialize the weights and biases of a model such that the output will always be zero.
    Works specifically for the BrachyPlanNet defined earlier.

    Args:
        model (nn.Module): The PyTorch model to initialize.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Use Xavier initialization (or He initialization) for the weights instead of zeroing them out
            init.xavier_normal_(module.weight)  # Use Xavier initialization (normal distribution)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.1)  # Set biases to a small positive value
        elif isinstance(module, ResidualBlock):
            # Initialize the skip connection in ResidualBlock
            if isinstance(module.skip_connection, nn.Linear):
                init.xavier_normal_(module.skip_connection.weight)  # Xavier initialization for skip connection
                nn.init.constant_(module.skip_connection.bias, 0.1)  # Small positive bias


def generate_oriented_3d_gaussian(shape, center, direction, sigmas, translation=(0, 0, 0)):
    """
    Generate a 3D Gaussian distribution oriented along a given direction using PyTorch.

    Parameters:
        shape (tuple): The shape of the 3D volume (depth, height, width).
        center (torch.Tensor): The center of the Gaussian distribution in normalized coordinates [0, 1].
        direction (torch.Tensor): The orientation vector of the Gaussian distribution.
        sigmas (torch.Tensor): Standard deviations (σx, σy, σz) along each axis.
        translation (tuple): A 3D translation vector (dx, dy, dz).

    Returns:
        torch.Tensor: A 3D tensor representing the oriented Gaussian distribution.
    """
    # Normalize direction
    direction = direction / torch.norm(direction)

    # Generate a 3D grid of coordinates
    grid_z, grid_y, grid_x = torch.meshgrid(
        torch.arange(shape[0], dtype=torch.float32, device=center.device),
        torch.arange(shape[1], dtype=torch.float32, device=center.device),
        torch.arange(shape[2], dtype=torch.float32, device=center.device),
        indexing='ij'
    )
    grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # Shape: (depth, height, width, 3)

    # Apply translation and center adjustment
    center = center * torch.tensor(shape, dtype=torch.float32, device=center.device)
    translation = torch.tensor(translation, dtype=torch.float32, device=center.device)
    coords = grid - (center + translation).view(1, 1, 1, 3)  # Shape: (depth, height, width, 3)

    # Compute rotation matrix using Rodrigues' formula
    x_axis = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=center.device)
    if torch.abs(torch.dot(x_axis, direction)) >= 0.99:
        rotation_matrix = torch.eye(3, device=center.device)  # No rotation needed
    else:
        # Calculate rotation angle and axis
        angle = torch.acos(torch.dot(x_axis, direction))
        axis = torch.cross(x_axis, direction)
        axis = axis / torch.norm(axis)

        # Rodrigues' rotation matrix
        cos_theta = torch.cos(angle)
        sin_theta = torch.sin(angle)
        K = torch.tensor([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ], dtype=torch.float32, device=center.device)
        rotation_matrix = cos_theta * torch.eye(3, device=center.device) + \
                          (1 - cos_theta) * torch.outer(axis, axis) + \
                          sin_theta * K

    # Rotate coordinates+
    rotated_coords = torch.einsum('ij,xyzj->xyzi', rotation_matrix, coords)  # Shape: (depth, height, width, 3)

    # Compute Gaussian
    sigma_tensor = sigmas.view(1, 1, 1, 3)
    gaussian = torch.exp(-0.5 * (rotated_coords / sigma_tensor) ** 2).prod(dim=-1)

    return gaussian



def simple_single_dose_calculation(shape, pos, direc, seed_sigma, seed_avr_dose, device):
    """
    PyTorch version of simple_single_dose_calculation.
    """
    pos = torch.tensor(pos, dtype=torch.float32, device=device)  # Move to GPU
    direc = torch.tensor(direc, dtype=torch.float32, device=device)
    seed_sigma = torch.tensor(seed_sigma, dtype=torch.float32, device=device)
    
    # Generate oriented Gaussian
    gaussian = generate_oriented_3d_gaussian(shape, pos, direc, seed_sigma)

    # Scale by seed's average dose
    return gaussian * seed_avr_dose




class DoseOptimizationLoss(torch.nn.Module):
    def __init__(self, seed_sigma, radiation_volume, in_lowest_dose, out_highest_dose, DVH_rate, seed_avr_dose, device, weights):
        """
        Initializes the DoseOptimizationLoss class for dose optimization in radiation therapy.
        
        This class defines the custom loss function used in dose optimization for radiation therapy, 
        which evaluates the effectiveness of seed placement in terms of dose distribution and 
        the dose-volume histogram (DVH) coverage rate.

        Args:
            seed_sigma (tuple): Standard deviations (σx, σy, σz) of the Gaussian radiation distribution 
                                 for each seed, defining the spread of the radiation in the 3D space.
            radiation_volume (ndarray): A 3D binary mask array representing the target volume 
                                         that should be irradiated (1 for irradiated area, 0 for non-irradiated area).
            in_lowest_dose (float): Minimum dose threshold that should be reached for the treatment to be effective.
            out_highest_dose (float): Maximum dose threshold for areas outside the treatment target.
            DVH_rate (float): Desired dose-volume histogram coverage rate, representing the fraction of the target 
                              volume that should receive at least the `in_lowest_dose`.
            seed_avr_dose (float): The average dose delivered by a single seed.
            device (torch.device): The device on which computations will be carried out (e.g., 'cpu' or 'cuda').
            weights (list): A list of weights to balance different components of the loss function, such as:
                            [margin_weight, unsafe_weight, overdose_weight].
        """
        super(DoseOptimizationLoss, self).__init__()

        # Initialize parameters and tensors, move them to the specified device (CPU or GPU)
        self.seed_sigma = torch.tensor(seed_sigma, dtype=torch.float32, device=device, requires_grad=False)  # Radiation spread in 3D space
        self.radiation_volume = torch.tensor(radiation_volume, dtype=torch.float32, device=device, requires_grad=False)  # Target volume to irradiate
        self.outside_mask = torch.tensor(1 - geometry.compute_convex_hull_mask_from_array(radiation_volume), dtype=torch.float32, device=device, requires_grad=False)  # Mask for regions outside the target volume
        # self.outside_mask = torch.tensor(1 - radiation_volume, dtype=torch.float32, device=device, requires_grad=False)  # Mask for regions outside the target volume
        self.in_lowest_dose = torch.tensor(in_lowest_dose, dtype=torch.float32, device=device, requires_grad=False)  # Minimum acceptable dose for the target volume
        self.out_highest_dose = torch.tensor(out_highest_dose, dtype=torch.float32, device=device, requires_grad=False)  # Maximum allowable dose outside target
        self.seed_avr_dose = torch.tensor(seed_avr_dose, dtype=torch.float32, device=device, requires_grad=False)  # Average dose of a single seed
        self.num_target_volume = torch.sum(self.radiation_volume).float()  # Total number of points in the target volume
        self.num_outside_volume = torch.sum(self.outside_mask).float()  # Total number of points in the non-target volume
        self.target_rate = self.num_outside_volume/self.num_target_volume  # Ratio of outside volume to target volume (used later for loss calculation)
        self.elu = nn.ELU(alpha=1.0)  # Exponential Linear Unit activation (used for penalty scaling)
        self.loss_weights = weights  # Weights for the different components of the loss function
        self.DVH_rate = torch.tensor(DVH_rate, dtype=torch.float32, device=device, requires_grad=False)  # Target DVH rate
        self.device = device  # Store the device for later use
        self.scaler = 1e5  # Scaling factor for the loss function
        
        # Generate a 3D grid of coordinates for the radiation volume to calculate the seed's impact on the grid
        grid_x, grid_y, grid_z = torch.meshgrid(
            torch.arange(self.radiation_volume.shape[0], dtype=torch.float32, device=device),
            torch.arange(self.radiation_volume.shape[1], dtype=torch.float32, device=device),
            torch.arange(self.radiation_volume.shape[2], dtype=torch.float32, device=device),
            indexing='ij'  # Use 'ij' indexing for grid coordinates (depth, height, width)
        )
        
        # Stack the 3D coordinates into a tensor of shape (depth, height, width, 3), representing the (x, y, z) coordinates
        self.grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)

    def forward(self, x):
        """
        Forward pass to compute the loss function for dose optimization.
        
        This function computes the loss based on the radiation distribution generated by the seed positions
        and directions provided in the input tensor `x`. The loss function considers multiple components:
        - Margin penalty: Difference between the desired DVH rate and the actual DVH rate.
        - Unsafe units: Regions outside the treatment target that exceed the maximum allowable dose.
        - Overdose penalty: Total radiation relative to the contribution of the seeds.

        Args:
            x (Tensor): Input tensor representing the positions and directions of seeds. Each seed consists 
                        of 6 values: 3 for the position (x, y, z), and 3 for the direction (unit vector).

        Returns:
            Tensor: The computed loss value, which is used to guide the optimization of seed placement.
        """
        # Initialize the radiation grid to zeros (shape: (depth, height, width))
        radiation = torch.zeros(self.radiation_volume.shape, dtype=torch.float32, device=self.device, requires_grad=True)

        # Define the x-axis direction (for rotation calculation, fixed for simplicity)
        x_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device, requires_grad=False)

        # Iterate over each seed (each seed is represented by 6 values: 3 for position, 3 for direction)
        for i in range(x.shape[0] // 6):
            # Extract the position (scaled to the radiation grid size) and direction of the seed
            center = x[6 * i:6 * i + 3] * (torch.tensor(self.radiation_volume.shape, device=self.device).view(-1, 1))
            coords = self.grid - center.view(1, 1, 1, 3)  # Calculate the coordinates relative to the seed's position
            
            # Extract the direction (unit vector)
            direction = x[6 * i + 3:6 * i + 6].view(-1)
            
            # Compute the rotation matrix to align the seed's direction with the desired orientation
            if torch.abs(torch.dot(x_axis, direction)) >= 0.99:
                rotation_matrix = torch.eye(3, device=self.device)  # Identity matrix if the direction is close to the x-axis
            else:
                # Calculate the angle and axis for the rotation matrix (using Rodrigues' rotation formula)
                angle = torch.acos(torch.dot(x_axis, direction))
                axis = torch.cross(x_axis, direction)
                axis = axis / (torch.norm(axis))  # Normalize the axis of rotation

                # Construct the rotation matrix using Rodrigues' formula
                cos_theta = torch.cos(angle)
                sin_theta = torch.sin(angle)
                K = torch.stack([torch.stack([torch.tensor(0, device=self.device), -axis[2], axis[1]]),
                                 torch.stack([axis[2], torch.tensor(0, device=self.device), -axis[0]]),
                                 torch.stack([-axis[1], axis[0], torch.tensor(0, device=self.device)])])
                rotation_matrix = cos_theta * torch.eye(3, device=self.device) + \
                                  (1 - cos_theta) * torch.outer(axis, axis) + \
                                  sin_theta * K
            
            # Apply the rotation matrix to the coordinates (rotate the grid points)
            coords = coords.view(-1, 3).T  # Flatten coordinates to shape (3, N)
            rotated_coords = torch.matmul(rotation_matrix, coords)  # Perform the rotation
            rotated_coords = rotated_coords.T.view(*self.grid.shape)  # Reshape back to (depth, height, width, 3)

            # Calculate the Gaussian distribution based on the rotated coordinates
            sigma_x, sigma_y, sigma_z = self.seed_sigma
            gaussian = torch.exp(-((rotated_coords[..., 0] ** 2) / (2 * sigma_x ** 2) +
                                   (rotated_coords[..., 1] ** 2) / (2 * sigma_y ** 2) +
                                   (rotated_coords[..., 2] ** 2) / (2 * sigma_z ** 2)))

            # Add the seed's radiation contribution to the total radiation grid
            radiation = radiation + self.seed_avr_dose * gaussian
          
        # Compute effective radiation within the target volume
        effective_radiation = radiation * self.radiation_volume
        
        # Compute radiation outside the target volume (using the outside mask)
        outside_radiation = radiation * self.outside_mask     

        # Calculate the number of effective units (dose >= in_lowest_dose) in the target volume
        effective_units = torch.sum(torch.sigmoid((effective_radiation - self.in_lowest_dose))).float()

        # Calculate the number of unsafe units (dose > out_highest_dose) outside the target volume
        unsafe_rate = torch.sum(torch.sigmoid(100*(outside_radiation - self.out_highest_dose))).float()/self.num_target_volume

        # Calculate the current dose-volume histogram (DVH) rate
        cur_DVH_rate = effective_units / (self.num_target_volume)

        # Compute the margin penalty (the difference between the target DVH rate and the current DVH rate)
        margin_penalty = self.DVH_rate - cur_DVH_rate

        # Compute overdose penalty (total radiation relative to total contribution from seeds)
        overdose_penalty = 1 - torch.sum(effective_radiation) / (torch.sum(radiation))

        # Combine all components of the loss function (margin penalty, unsafe units, overdose penalty)
        loss = (
            self.loss_weights[0] * margin_penalty 
            +
            self.loss_weights[2] * overdose_penalty 
            +
            self.loss_weights[1] * unsafe_rate
            # (self.elu(self.scaler * (cur_DVH_rate - self.DVH_rate)) + 1) 
            # * 
            # (   
            #     self.loss_weights[1] * unsafe_rate                  
            # ) 
            # / self.scaler
        )

        # Return the final loss value
        return loss




class early_stop():
    """
    A class to implement early stopping during training, which halts the training process
    when validation loss does not improve beyond a certain threshold (`delta`) for a specified
    number of consecutive epochs (`patience`).

    Attributes:
        patience (int): The number of epochs to wait after the last improvement in validation loss.
                        Training will stop if no improvement is observed for this many epochs.
        verbose (bool): Whether to print messages during the early stopping process.
        delta (float): Minimum change in the monitored value to qualify as an improvement.
        counter (int): Counts the number of epochs since the last improvement in validation loss.
        best_score (float): Tracks the best validation loss observed during training.
        early_stop (bool): Flag indicating whether early stopping has been triggered.
        val_loss_min (float): Tracks the minimum validation loss value.
        down_patience_sign (bool): Flag to reduce the patience dynamically if the best score becomes negative.

    Methods:
        __call__(val_loss):
            Checks the validation loss for improvement and updates the stopping condition.
            If the stopping condition is met, the `early_stop` flag is set to True.
    """

    def __init__(self, patience, verbose, delta):
        """
        Initializes the early stopping object with the specified parameters.

        Parameters:
            patience (int): Number of epochs to wait without improvement before stopping.
            verbose (bool): Whether to print detailed messages about early stopping.
            delta (float): Minimum improvement required to reset the counter.
        """
        self.patience = patience  # Number of epochs to wait for an improvement
        self.verbose = verbose  # Whether to output early stopping information
        self.counter = 0  # Counter for epochs without improvement
        self.best_score = np.inf  # Best score seen so far (initialized to infinity)
        self.early_stop = False  # Flag to indicate if early stopping should be triggered
        self.val_loss_min = np.inf  # Minimum validation loss observed so far
        self.delta = delta  # Minimum improvement threshold
        self.down_patience_sign = False  # Flag to dynamically reduce patience if needed

    def __call__(self, val_loss):
        """
        Checks the validation loss to determine if training should stop early.

        Parameters:
            val_loss (float): Current epoch's validation loss.

        Returns:
            bool: True if early stopping is triggered, False otherwise.
        """
        # Print the current loss and best loss for debugging purposes
        print("loss={}, best loss={}".format(val_loss, self.best_score))
        score = val_loss  # Current validation loss is treated as the score to monitor

        if self.best_score is None:
            # Initialize the best score if it has not been set yet
            self.best_score = score
            return False

        elif score < self.best_score - self.delta:
            # Improvement detected: update the best score and reset the counter
            self.best_score = score
            self.counter = 0

            # Dynamically adjust patience if the best score becomes negative
            if self.best_score < 0 and not self.down_patience_sign:
                self.patience = int(self.patience / 2)  # Reduce patience by a factor of 10
                self.down_patience_sign = True
            return False

        else:
            # No significant improvement: increment the counter
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')

            if self.counter >= self.patience:
                # Trigger early stopping if the patience threshold is reached
                self.early_stop = True
            return True