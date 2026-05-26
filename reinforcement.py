import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
from gymnasium import spaces
from tqdm import tqdm
import utils
from numba import njit, prange
from typing import Tuple



# ==========================================================
# 1.  Reward Calculator (with seed cache - kept)
# ==========================================================
# ----------  JIT  compiled helper  ----------
@njit(parallel=True, fastmath=True, error_model="numpy", cache=True)
def _dvh_oar_jit(
    dose: np.ndarray,
    target_idx: np.ndarray,
    non_target_idx: np.ndarray,
    thr_target: float,
    thr_oar: float,
    count_out: bool = True
):
    """
    Multi-core DVH & OAR damage calculator.

    Parameters
    ----------
    dose : np.ndarray
        Flattened float32 dose grid.
    target_idx : np.ndarray
        Flat indices of target voxels.
    non_target_idx : np.ndarray
        Flat indices of OAR voxels.
    thr_target : float
        DVH threshold [Gy].
    thr_oar : float
        OAR penalty threshold [Gy].
    count_out : bool, optional
        If False, skip OAR loop (default True).

    Returns
    -------
    dvh : float
        Target coverage fraction.
    oar : float | None
        OAR overdose fraction; None if `count_out` is False.
    """
    n_target = target_idx.size
    target_sum = 0.0

    # ---- parallel over target voxels ----
    for i in prange(n_target):
        if dose[target_idx[i]] > thr_target:
            target_sum += 1.0
    dvh = target_sum / n_target

    oar = None
    if count_out:
        oar_sum = 0.0
        # ---- parallel over OAR voxels ----
        for i in prange(non_target_idx.size):
            if dose[non_target_idx[i]] > thr_oar:
                oar_sum += 1.0
        oar = n_target / max(0.1, oar_sum)

    return dvh, oar


# class SeedPlacementReward:
#     """
#     Reward function for seed placement in brachytherapy planning.
#     """
#     def __init__(self,
#                  dose_cal_model,
#                  dose_image,
#                  radiation_volume,
#                  target_value,
#                  in_lowest_dose,
#                  out_highest_dose,
#                  infer_img_size,
#                  seed_info,
#                  image_normalize_min,
#                  target_valueimage_normalize_max,   # REVIEW: typo left for compatibility
#                  image_normalize_scale,
#                  DVH_rate):
#         self.dose_cal_model = dose_cal_model
#         self.dose_image = dose_image
#         self.radiation_volume = radiation_volume
#         self.target_value = target_value
#         self.in_lowest_dose = in_lowest_dose
#         self.out_highest_dose = out_highest_dose
#         self.infer_img_size = infer_img_size
#         self.seed_info = seed_info
#         self.image_normalize_min = image_normalize_min
#         self.target_valueimage_normalize_max = target_valueimage_normalize_max
#         self.image_normalize_scale = image_normalize_scale

#         # Precompute masks and sizes once
#         self.mask_volume = (self.radiation_volume == self.target_value).astype(float)
#         self.total_v = np.prod(self.radiation_volume.shape)
#         self.target_v = np.sum(self.mask_volume)
#         self.DVH_rate = DVH_rate
#         self.seed_cache = {}
#         self.target_idx = np.where(self.mask_volume > 0)
#         self.non_target_idx = np.where(self.mask_volume == 0)

#     # ------------------------------------------------------
#     def forward(self, traj, cur_radiation, direction, seed_point, protect_OAR = True):
#         """
#         Compute reward for placing a seed along the given trajectory.
#         Returns:
#             reward (float), updated dose (np.ndarray), new DVH rate (float), seed_radiation (np.ndarray)
#         """
#         cur_seed_radiation = None
#         key = tuple(np.round(seed_point).astype(int))

#         # Try seed cache
#         cur_seed_radiation = self.seed_cache.get(key, None)

#         # If not cached, try matching against trajectory cached seeds (keeps original behavior)
#         if cur_seed_radiation is None:
#             for cached_seed, cached_dose in zip(traj[2], traj[3]):
#                 # cached_seed[0] assumed to be position vector like seed_point
#                 if np.linalg.norm(np.array(cached_seed[0]).reshape(-1) - seed_point) < 1e-3:
#                     cur_seed_radiation = cached_dose
#                     break

#         # If still None, compute using provided function
#         if cur_seed_radiation is None:
#             cur_seed_radiation = utils.single_seed_dose_calculation_dl(
#                 np.array(seed_point).astype(int).reshape(-1),
#                 direction,
#                 self.dose_image,
#                 self.dose_cal_model,
#                 self.infer_img_size,
#                 self.seed_info,
#                 self.image_normalize_min,
#                 self.target_valueimage_normalize_max,
#                 self.image_normalize_scale
#             )
#             self.seed_cache[key] = cur_seed_radiation

#         # accumulate radiation (non in-place for safety)
#         cur_radiation = cur_radiation + cur_seed_radiation

#         # Use precomputed idx for speed
#         target_voxels = cur_radiation[self.target_idx]
#         # compute DVH rate
#         cur_DVH_rate = np.mean(target_voxels > self.in_lowest_dose)
#         if not protect_OAR:
#             reward = cur_DVH_rate
#         else:
#             non_target_voxels = cur_radiation[self.non_target_idx]
#             out_damage = np.count_nonzero(non_target_voxels > self.out_highest_dose) / max(1.0, self.target_v)
#             # reward formula preserved
#             reward = min(cur_DVH_rate, self.DVH_rate) + ((cur_DVH_rate - self.DVH_rate) >= 0) * ((1 - out_damage))
#         return reward, cur_radiation, cur_DVH_rate, cur_seed_radiation



def percentile_value_in_mask(image: np.ndarray, mask: np.ndarray, ratio: float = 0.1):
    """
    Compute the pixel value threshold corresponding to the lowest `ratio` proportion
    (e.g., bottom 10%) of pixel intensities within a masked region.
    
    Args:
        image (np.ndarray): Grayscale image.
        mask (np.ndarray): Binary mask (same shape as image).
        ratio (float): Ratio of lowest values to consider (e.g., 0.1 for 10%).
    
    Returns:
        float: The pixel value at the cutoff (bottom ratio).
    """
    # Extract pixel values in the mask region
    masked_values = image[mask > 0]
    
    if masked_values.size == 0:
        raise ValueError("Mask region is empty.")
    
    # Sort values from large to small
    sorted_values = np.sort(masked_values)[::-1]
    
    # Get the index corresponding to the bottom `ratio` of sorted values
    index = int(len(sorted_values) * (1 - ratio))
    index = np.clip(index, 0, len(sorted_values) - 1)
    
    return sorted_values[index]



# ----------  Reward class  ----------
class SeedPlacementReward:
    """
    Incremental reward for seed placement in brachytherapy.
    Uses CNN-based single-seed dose + position cache + JIT metrics.
    """
    def __init__(self,
                 dose_cal_model: nn.Module,
                 dose_image: np.ndarray,
                 radiation_volume: np.ndarray,
                 target_value: float,
                 in_lowest_dose: float,
                 out_highest_dose: float,
                 infer_img_size: tuple,
                 seed_info: dict,
                 image_normalize_min: float,
                 target_valueimage_normalize_max: float,  # typo kept
                 image_normalize_scale: float,
                 DVH_rate: float
                 ) -> None:

        self.dose_cal_model = dose_cal_model
        self.dose_image = dose_image
        self.radiation_volume = radiation_volume
        self.target_value = target_value
        self.in_lowest_dose = in_lowest_dose
        self.out_highest_dose = out_highest_dose
        self.infer_img_size = infer_img_size
        self.seed_info = seed_info
        self.image_normalize_min = image_normalize_min
        self.target_valueimage_normalize_max = target_valueimage_normalize_max
        self.image_normalize_scale = image_normalize_scale
        self.DVH_rate = DVH_rate

        # one-time masks
        self.mask_volume = (radiation_volume == target_value).astype(float)
        self.non_target_mask  = (radiation_volume != target_value).astype(float)
        self.seed_cache: dict[tuple, np.ndarray] = {}

        # flat indices for JIT speed
        self.target_idx = np.where(self.mask_volume.ravel() > 0)[0].astype(np.int32)
        self.target_v = self.target_idx.size
        self.non_target_idx = np.where(self.mask_volume.ravel() == 0)[0].astype(np.int32)
        self.non_target_v = self.non_target_idx.size
        
        

    # ------------------------------------------------------------------
    def forward(self,
                traj: list,
                cur_radiation: np.ndarray,
                direction: np.ndarray,
                seed_point: np.ndarray,
                protect_OAR: bool = True):
        """
        Compute reward for placing one seed.
        Returns: reward, updated_dose, DVH_rate, seed_dose_map
        """
        seed_point_int=np.array(seed_point).astype(int)
        key = tuple(seed_point_int)

        # 1.  look-up cache
        cur_seed_radiation = self.seed_cache.get(key, None)

        # 2.  trajectory cache fall-back
        if cur_seed_radiation is None:
            for cached_seed, cached_dose in zip(traj[2], traj[3]):
                if np.linalg.norm(np.asarray(cached_seed[0]).ravel() - seed_point) < 1e-3:
                    cur_seed_radiation = cached_dose
                    self.seed_cache[key] = cur_seed_radiation.copy()
                    break

        # 3.  CNN inference if still missing
        if cur_seed_radiation is None:
            cur_seed_radiation = utils.single_seed_dose_calculation_dl(
                seed_point_int.reshape(-1),
                direction,
                self.dose_image,
                self.dose_cal_model,
                self.infer_img_size,
                self.seed_info,
                self.image_normalize_min,
                self.target_valueimage_normalize_max,
                self.image_normalize_scale
            )
            self.seed_cache[key] = cur_seed_radiation.copy()

        # 4.  accumulate dose (out-of-place)
        cur_radiation = cur_radiation + cur_seed_radiation

        # 5.  JIT metrics
        

        # 6.  reward formula (unchanged)
        if not protect_OAR:
            cur_DVH_rate, out_damage = _dvh_oar_jit(
                cur_radiation.ravel(), self.target_idx, self.non_target_idx,
                self.in_lowest_dose, self.out_highest_dose, count_out = False
            )
            reward = cur_DVH_rate
        else:
            cur_DVH_rate, out_damage = _dvh_oar_jit(
                cur_radiation.ravel(), self.target_idx, self.non_target_idx,
                self.in_lowest_dose, self.out_highest_dose, count_out = True
            )
            reward = min(cur_DVH_rate, self.DVH_rate) + \
                     ((cur_DVH_rate - self.DVH_rate) >= 0) * (out_damage)

        return reward, cur_radiation, cur_DVH_rate, cur_seed_radiation


# ==========================================================
# 2.  Policy Network
# ==========================================================
class PolicyNet(nn.Module):
    """Simple MLP policy for discrete actions."""
    def __init__(self, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.logits = nn.Linear(hidden, n_actions)

    def forward(self, s):
        return self.logits(self.net(s))

    def act(self, s):
        logits = self(s)
        dist = Categorical(logits=logits)
        a = dist.sample()
        return a, dist.log_prob(a)


# ==========================================================
# 3.  REINFORCE Agent (vectorized loss)
# ==========================================================
class REINFORCE:
    def __init__(self, n_actions, lr=1e-2, gamma=0.99, device="cpu"):
        self.device = torch.device(device)
        self.policy = PolicyNet(n_actions).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.log_probs = []
        self.rewards = []

    # ------------------------------------------------------
    def select_action(self, state, mask=None):
        """Return action index (int) and store log_prob."""
        # state is small (shape (1,)) -- keep as float32
        state_array = np.array(state, dtype=np.float32)
        s = torch.from_numpy(state_array).unsqueeze(0).to(self.device)
        logits = self.policy(s)  # shape [1, n_actions]

        if mask is not None:
            # mask is numpy boolean array of length n_actions
            # convert once to a tensor on correct device
            mask_tensor = torch.from_numpy(mask.astype(bool)).to(self.device)
            # set invalid logits to a large negative value
            logits = logits.clone()
            logits[..., ~mask_tensor] = -1e10

        dist = Categorical(logits=logits)
        a = dist.sample()
        self.log_probs.append(dist.log_prob(a))
        return a.item()

    # ------------------------------------------------------
    def record_reward(self, r):
        self.rewards.append(r)

    # ------------------------------------------------------
    def finish_episode(self):
        """Compute discounted returns and update policy."""
        if not self.rewards:
            # nothing to update
            self.log_probs.clear()
            return

        # compute discounted returns
        returns = []
        R = 0.0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # normalize if more than 1 value
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        log_probs_tensor = torch.stack(self.log_probs)
        loss = - (log_probs_tensor * returns).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.log_probs.clear()
        self.rewards.clear()


# ==========================================================
# 4.  Low-level Environment
# ==========================================================
class LowLevelEnv(gym.Env):
    """
    Low-level env: pick candidate positions along a fixed needle path.
    """
    def __init__(self, candidate_positions):
        super().__init__()
        self.candidate_positions = candidate_positions
        self.used_mask = np.ones(len(candidate_positions), dtype=bool)  # True = available
        self.cur_step = 0
        self.done = False
        self.action_space = spaces.Discrete(len(candidate_positions))
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

    # ------------------------------------------------------
    def reset(self):
        self.used_mask[:] = True
        self.cur_step = 0
        self.done = False
        return np.array([0.0], dtype=np.float32)

    # ------------------------------------------------------
    def step(self, action):
        # mark used and advance
        self.used_mask[action] = False
        self.cur_step += 1
        if not self.used_mask.any():
            self.done = True
        # keep same return signature as original code expects: (state, done, {})
        return np.array([0.0], dtype=np.float32), self.done, {}


# ==========================================================
# 5.  High-level Environment
# ==========================================================
class HighLevelEnv(gym.Env):
    """
    High-level env: select trajectory group, then run low-level RL to place seeds.
    """
    def __init__(self,
                 target_level_traj,
                 high_level_ranges,
                 low_level_ranges,
                 range_length,
                 target_level,
                 dose_image,
                 radiation_volume,
                 seed_info,
                 reward_calculator,
                 DVH_rate,
                 protect_OAR):
        super().__init__()
        self.level = target_level
        self.range_length = range_length
        self.cum_sizes = np.cumsum(self.range_length)
        self.action_space = spaces.Discrete(len(high_level_ranges))
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        self.high_level_ranges   = high_level_ranges
        self.target_level_traj   = target_level_traj
        self.low_level_ranges    = low_level_ranges
        self.DVH_rate            = DVH_rate
        self.protect_OAR         = protect_OAR

        self.group_idx = None
        self.dose_image  = dose_image
        self.seed_info   = seed_info
        self.spacing     = np.array(self.dose_image.GetSpacing()).reshape(-1)

        self.planned_positions = [[] for _ in range(self.level)]
        self.planned_seed_radiations = [[] for _ in range(self.level)]
        self.planned_directions = [np.array([0, 0, 1]) for _ in range(self.level)]
        self.cur_radiation = np.zeros_like(radiation_volume)
        self.reward_calculator = reward_calculator
        
        # Precompute candidate positions (img coords and world coords) per group and level
        # Structure:
        # candidate_img_positions[(group_idx, lv)] = np.array([img_pos for length in effective_range])
        # candidate_world_positions[(group_idx, lv)] = np.array([...]) (same shape)
        self.candidate_img_positions = {}
        self.candidate_world_positions = {}
        self._precompute_candidate_positions()
      
    # ------------------------------------------------------    
    def _precompute_candidate_positions(self):
        """Precompute and cache image/world positions for every group/level/length."""
        for group_idx, group in enumerate(self.target_level_traj):
            for lv in range(self.level):
                traj = group[lv][1]
                point = np.array(traj[0]).reshape(-1)
                direction = np.array(traj[1]).reshape(-1)
                direction = direction / np.linalg.norm(direction)
                max_idx = np.argmax(np.abs(direction))
                update_dir = direction / np.abs(direction[max_idx])

                effective_range = self.low_level_ranges[group_idx][lv]
                # compute img positions
                img_positions = np.stack([point + update_dir * length for length in effective_range], axis=0)
                # try to use batch transform if available; fallback to loop
                try:
                    world_positions = np.array(utils.position_transform(self.dose_image, img_positions))
                    # position_transform may return list of tuples; normalize to array of points
                    # assume returns array-like shape (N, 3) or list of length N each a 3-vector
                    if world_positions.ndim == 3:
                        # sometimes transforms return [[(x,y,z)]] shape; try to squeeze
                        world_positions = world_positions.squeeze()
                except Exception:
                    # fallback: transform point-by-point
                    world_positions = np.array([utils.position_transform(self.dose_image, img_pos)[0] for img_pos in img_positions])

                self.candidate_img_positions[(group_idx, lv)] = img_positions
                self.candidate_world_positions[(group_idx, lv)] = world_positions

    # ------------------------------------------------------
    def reset(self):
        """Reset before new episode."""
        self.group_idx = None
        self.planned_positions = [[] for _ in range(self.level)]
        self.planned_seed_radiations = [[] for _ in range(self.level)]
        self.planned_directions = [np.array([0, 0, 1]) for _ in range(self.level)]
        self.cur_radiation[:] = 0.0
        return np.array([0.0], dtype=np.float32)

    # ------------------------------------------------------
    def generate_low_level_state_space(self, action):
        """Flatten all low-level candidates for the chosen trajectory group."""
        # keep original logic for group_idx computation
        self.group_idx = np.searchsorted(self.cum_sizes, action, side="right") // self.level
        merged = []
        for lv in range(self.level):
            merged.extend(self.low_level_ranges[self.group_idx][lv])
        return merged

    # ------------------------------------------------------
    def generate_mask(self):
        """
        Build boolean mask for low-level actions.
        True  = position is available (not inside exclusion zone).
        False = position is forbidden.
        Uses precomputed candidate_world_positions for speed.
        """
        trajs = self.target_level_traj[self.group_idx]
        mask = []
        # iterate levels
        for lv in range(self.level):
            cur_traj = trajs[lv][1]
            point = np.array(cur_traj[0]).reshape(-1)
            world_p = utils.position_transform(self.dose_image, point)[0]
            effective_range = self.low_level_ranges[self.group_idx][lv]

            # start all True
            lv_mask = np.ones(len(effective_range), dtype=bool)

            # If there are existing planned positions at this lv, vectorize distance checks
            if self.planned_positions[lv]:
                all_pos = self.candidate_world_positions[(self.group_idx, lv)]  # shape (N, 3)
                # For each already planned position, compute blocked indices
                planned_arr = np.array(self.planned_positions[lv])  # shape (M,3)
                # compute distances from planned positions to world_p
                # but original logic uses dist between planned_pos and world_p to set a start/end, then
                # compares candidate distances to world_p against that start/end.
                # So do this vectorized:
                dists_planned_to_world_p = np.linalg.norm(planned_arr - world_p, axis=1)  # (M,)
                # compute candidate distances to world_p once
                candidate_dists = np.linalg.norm(all_pos - world_p, axis=1)  # (N,)
                # For each planned, set mask false where candidate_dists in (start, end)
                for dist in dists_planned_to_world_p:
                    start = dist - self.seed_info['length']
                    end = dist + self.seed_info['length']
                    # vectorized boolean update
                    lv_mask[(candidate_dists > start) & (candidate_dists < end)] = False

            mask.extend(lv_mask)
        return np.array(mask)

    # ------------------------------------------------------
    def update_planned_position(self, action, high_level=False):
        """
        Convert action (high or low) to world position, store it, and return incremental reward.
        Uses precomputed candidate positions.
        """
        if high_level:
            lv = np.searchsorted(self.cum_sizes, action, side="right") % self.level
        else:
            lv = np.searchsorted(np.cumsum(self.range_length[self.group_idx * self.level:]), action, side="right") % self.level

        traj = self.target_level_traj[self.group_idx][lv][1]
        point = np.array(traj[0]).reshape(-1)
        direction = np.array(traj[1]).reshape(-1)
        direction = direction / np.linalg.norm(direction)
        max_idx = np.argmax(np.abs(direction))
        update_dir = direction / np.abs(direction[max_idx])

        # select length depending on action (high/low)
        if high_level:
            length = self.high_level_ranges[action]
            # compute img position (not necessarily in candidate lists)
            img_position = np.array(update_dir * length + point)
            # get world position via transform
            world_position = utils.position_transform(self.dose_image, img_position)[0]
        else:
            # for low-level action, action is index in flattened low-level state space
            length = self.low_level_state_space[action]
            # use cached candidate positions for this group/lv
            # find index inside effective_range
            effective_range = self.low_level_ranges[self.group_idx][lv]
            # length may appear multiple times but indices align; find index position
            # attempt to find first index matching length
            try:
                idx_in_lv = list(effective_range).index(length)
            except ValueError:
                # fallback compute
                img_position = np.array(update_dir * length + point)
                world_position = utils.position_transform(self.dose_image, img_position)[0]
            else:
                img_position = self.candidate_img_positions[(self.group_idx, lv)][idx_in_lv]
                world_position = self.candidate_world_positions[(self.group_idx, lv)][idx_in_lv]

        reward, self.cur_radiation, cur_DVH_rate, cur_seed_radiation = self.reward_calculator.forward(
            traj=self.target_level_traj[self.group_idx][lv],
            cur_radiation=self.cur_radiation.copy(),
            direction=direction,
            seed_point=img_position,
            protect_OAR=self.protect_OAR
        )

        self.planned_positions[lv].append(world_position)
        self.planned_seed_radiations[lv].append(cur_seed_radiation)
        self.planned_directions[lv] = np.array(utils.direction_transform(self.dose_image, direction))[0]
        return reward, cur_DVH_rate
    
    # ------------------------------------------------------
    def planned_position2planned_res(self):
        """
        Convert internally stored planned seed positions (world coordinates) into
        the canonical 'planned_res' format used downstream.
        """
        planned_res = []

        for lv in range(self.level):
            seeds = []
            single_seed_radiations = []

            for position, single_seed_radiation in zip(
                self.planned_positions[lv], self.planned_seed_radiations[lv]
            ):
                seeds.append([position, self.planned_directions[lv]])
                single_seed_radiations.append(single_seed_radiation)

            trajectory_def = self.target_level_traj[self.group_idx][lv][1]
            planned_res.append([trajectory_def, seeds, single_seed_radiations])

        return planned_res

    # ------------------------------------------------------
    def step(self, action, device="cpu"):
        """
        Execute high-level action, then run full low-level episode.
        Returns:
            total_reward (float), plan_tuple (high_action, planned_positions)
        """
        self.low_level_state_space = self.generate_low_level_state_space(action)
        low_env = LowLevelEnv(self.low_level_state_space)
        low_agent = REINFORCE(n_actions=len(self.low_level_state_space), device=device)

        state = low_env.reset()

        # Place first (high-level) seed
        total_reward, _ = self.update_planned_position(action, high_level=True)

        # Low-level rollout
        mask = None
        while not low_env.done:
            mask = self.generate_mask()
            if mask.any():
                a = low_agent.select_action(state, mask=mask)
                state, _, _ = low_env.step(a)
                r, cur_DVH_rate = self.update_planned_position(a, high_level=False)
                low_agent.record_reward(r)
                total_reward += r
                if cur_DVH_rate > self.DVH_rate:
                    low_env.done = True
            else:
                low_env.done = True

        planned_res = self.planned_position2planned_res()
        return total_reward, planned_res, cur_DVH_rate, mask, self.group_idx, self.low_level_state_space, low_env, low_agent



def randomly_flip_false(mask: np.ndarray, flip_ratio: float = 0.5, seed: int = None):
    """
    Randomly flip a fraction of False values in a boolean mask to True.

    Parameters
    ----------
    mask : np.ndarray
        Boolean array of any shape.
    flip_ratio : float
        Fraction of False values to flip to True (0 < flip_ratio <= 1).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        New boolean array with the specified fraction of False flipped to True.
    """
    if seed is not None:
        np.random.seed(seed)
    
    mask = mask.copy()  # avoid modifying original
    false_idx = np.where(mask == False)  # get all False indices
    n_flip = int(len(false_idx[0]) * flip_ratio)
    
    if n_flip > 0:
        flip_idx = np.random.choice(len(false_idx[0]), size=n_flip, replace=False)
        mask[tuple(idx[flip_idx] for idx in false_idx)] = True
    
    return mask


def DVH2Rewards(plan_res, radiation_volume, target_value, out_highest_dose, cur_DVH_rate, DVH_rate):
    """
    Compute DVH rate and reward for a given plan result.
    """
    # Accumulate total radiation from all seeds
    total_radiation = np.zeros_like(radiation_volume).astype(float)
    for _, _, single_seed_radiations in plan_res:
        for single_seed_radiation in single_seed_radiations:
            total_radiation += single_seed_radiation

    # Compute DVH rate
    mask_volume = (radiation_volume == target_value).astype(float)
    non_target_idx = np.where(mask_volume != target_value)
    target_idx = np.where(mask_volume == target_value)

    non_target_voxels = total_radiation[non_target_idx]
    out_damage = target_idx.size * cur_DVH_rate / np.count_nonzero(non_target_voxels > out_highest_dose)

    reward = min(cur_DVH_rate, DVH_rate) + ((cur_DVH_rate - DVH_rate) >= 0) * ((out_damage))
    
    return reward


def generate_baseline_state_space(low_level_state_spaces, level, idx):
    merged = []
    for lv in range(level):
        merged.extend(low_level_state_spaces[idx][lv])
    return merged



def generate_plan_res(dose_image, elem):
    planned_res = []
    for _, traj, seeds, single_seed_radiations, _ in elem:
        direction = utils.direction_transform(dose_image, np.array(traj[1]).reshape(-1))
        world_seeds = []
        for _, seed in enumerate(seeds):
            world_seeds.append([utils.position_transform(dose_image, seed[0]), direction])
        planned_res.append([traj, world_seeds, single_seed_radiations])
    return planned_res


# ==========================================================
# 6.  Top-level optimisation loop (kept logic, improved small parts)
# ==========================================================
def reinforcement_planning(
        rf_params,
        dose_cal_model,
        dose_image,
        radiation_volume,
        target_value,
        target_level_traj,
        high_level_state_spaces,
        low_level_state_spaces,
        range_length,
        target_level,
        seed_info,
        in_lowest_dose,
        out_highest_dose,
        infer_img_size,
        image_normalize_min,
        image_normalize_max,
        image_normalize_scale,
        DVH_rate):
    """
    Hierarchical reinforcement learning driver for a single patient case.
    Logic preserved; internal calls use optimized env and caches.
    """
    # Select device based on the dose model
    device = next(dose_cal_model.parameters()).device

    # Build high-level policy agent
    high_agent = REINFORCE(
        n_actions=len(high_level_state_spaces),
        lr=rf_params['lr'],
        gamma=rf_params['gamma'],
        device=device
    )

    # Reward calculator for seed placement
    reward_calculator = SeedPlacementReward(
        dose_cal_model, dose_image, radiation_volume, target_value,
        in_lowest_dose, out_highest_dose, infer_img_size, seed_info,
        image_normalize_min, image_normalize_max,
        image_normalize_scale, DVH_rate
    )

    # Environment wrapper for hierarchical planning
    env = HighLevelEnv(
        target_level_traj, high_level_state_spaces, low_level_state_spaces,
        range_length, target_level, dose_image, radiation_volume,
        seed_info, reward_calculator, DVH_rate, rf_params['segmented_rewards']
    )

    # Initialize bookkeeping
    best_reward = -np.inf
    best_DVH_rate = 0.0
    best_plan = None
    best_group_idx = None
    best_low_level_state_space = None
    best_low_env = None
    best_low_agent = None
    best_D90 = 0
    
    print("Baseline planning...")
    pbar = tqdm(total=len(target_level_traj), desc="Basic trajectories")
    for idx, elem in enumerate(target_level_traj):
        trajs_radiations = np.zeros_like(radiation_volume, dtype=np.float32)
        for _, traj, _, _, cur_seeds_radiations in elem:
            trajs_radiations += cur_seeds_radiations
        if rf_params['segmented_rewards']:
            target_sum = np.sum(trajs_radiations * reward_calculator.mask_volume > in_lowest_dose)
            cur_DVH_rate = target_sum / reward_calculator.target_v
            non_target_sum = np.count_nonzero(
                (trajs_radiations * reward_calculator.non_target_mask > out_highest_dose)
            )
            cur_out_damage = reward_calculator.target_v / max(0.1, non_target_sum)
            cur_reward =  min(cur_DVH_rate, DVH_rate) + ((cur_DVH_rate >= DVH_rate) * cur_out_damage)
        else:
            target_sum = np.sum(trajs_radiations * reward_calculator.mask_volume > in_lowest_dose)
            cur_reward = cur_DVH_rate = target_sum / reward_calculator.target_v

        D90 = percentile_value_in_mask(trajs_radiations, reward_calculator.mask_volume)


        if cur_reward > best_reward:
            best_reward = cur_reward
            best_DVH_rate = cur_DVH_rate
            best_plan = generate_plan_res(dose_image, elem)
            best_group_idx = idx
            best_low_level_state_space = generate_baseline_state_space(
                low_level_state_spaces, target_level, idx
            )
            best_D90 = D90

        pbar.set_postfix({
            "curr_reward": f"{cur_reward:.3f}",
            "best_reward": f"{best_reward:.3f}",
            "best_V100": f"{best_DVH_rate:.5f}",
            "best_D90": f"{best_D90:.5f}"
        })

        pbar.update(1)

    pbar.close()
    
    best_low_env = LowLevelEnv(best_low_level_state_space)
    best_low_agent = REINFORCE(n_actions=len(best_low_level_state_space), device=device)
    
    # return best_plan, best_reward
    
    import time
    start_time = time.time()
    print("Reinforcement planning...")
    # ===============================
    # Hierarchical optimization mode
    # ===============================
    if rf_params['hierarchical_optimization']:
        pbar = tqdm(range(rf_params['max_episodes'] // 2), total=rf_params['max_episodes'] // 2, ncols=120, desc="Episodes")
        # --------- Global optimization phase ---------
        for ep in pbar:
            state = env.reset()
            high_action = high_agent.select_action(state)

            low_reward, plan, cur_DVH_rate, mask, group_idx, low_level_state_space, low_env, low_agent = env.step(high_action, device=device)

            high_agent.record_reward(low_reward)
            high_agent.finish_episode()

            # Keep best solution so far (compare last episode reward of low_agent)
            if low_agent.rewards and low_agent.rewards[-1] > best_reward:
                best_plan = plan
                best_DVH_rate = cur_DVH_rate
                best_group_idx = group_idx
                best_low_level_state_space = low_level_state_space
                best_low_env = low_env
                best_low_agent = low_agent
                best_reward = low_agent.rewards[-1]
            
            # if ep == pbar.total - 1:
            #     best_reward = DVH2Rewards(best_plan, radiation_volume, target_value, out_highest_dose, best_DVH_rate, DVH_rate)

            pbar.set_postfix({
                "curr_reward": f"{(low_agent.rewards[-1] if low_agent.rewards else 0.0):.3f}",
                "best_reward": f"{best_reward:.3f}",
                "best_DVH": f"{best_DVH_rate:.3f}"
            })
            low_agent.finish_episode()

        # --------- Local refinement phase ---------        
        planned_directions = [np.array([0, 0, 1]) for _ in range(target_level)]
        best_cunsum = np.cumsum(range_length[best_group_idx * target_level:])
        best_traj = target_level_traj[best_group_idx]
        best_sub_state_space = low_level_state_spaces[best_group_idx]

        pbar = tqdm(range(rf_params['max_episodes'] // 2), total=rf_params['max_episodes'] // 2, ncols=120, desc="Episodes")
        # ------------------------------------------------------
        # Precompute trajectory info and position transforms (cache)
        # ------------------------------------------------------
        traj_cache = []
        for lv in range(target_level):
            traj = best_traj[lv][1]
            point = np.array(traj[0]).reshape(-1)
            direction = np.array(traj[1]).reshape(-1).astype(np.float64)
            direction /= np.linalg.norm(direction)
            max_idx = np.argmax(np.abs(direction))
            update_dir = direction / np.abs(direction[max_idx])
            world_p = utils.position_transform(dose_image, point)[0]
            traj_cache.append((best_traj[lv], point, direction, update_dir, world_p, best_sub_state_space[lv]))

        # Precompute (lv, length) → world_position (use env's cached positions if possible)
        pos_cache = {}
        for lv, (_, point, _, update_dir, _, effective_range) in enumerate(traj_cache):
            # use env cache if available
            try:
                positions = env.candidate_world_positions[(best_group_idx, lv)]
                lengths = best_sub_state_space[lv]
                for idx, length in enumerate(lengths):
                    pos_cache[(lv, length)] = positions[idx]
            except Exception:
                # fallback: compute
                for length in effective_range:
                    img_position = point + update_dir * length
                    pos_cache[(lv, length)] = utils.position_transform(dose_image, img_position)[0]

        # ------------------------------------------------------
        # Training loop with cache
        # ------------------------------------------------------
        for ep in pbar:
            planned_positions = [[] for _ in range(target_level)]
            planned_seed_radiations = [[] for _ in range(target_level)]

            cur_radiation = np.zeros_like(radiation_volume)
            state = best_low_env.reset()

            # Run rollout in low-level environment
            while not best_low_env.done:
                mask = []
                for lv in range(target_level):
                    _, _, _, update_dir, world_p, effective_range = traj_cache[lv]
                    lv_mask = np.ones(len(effective_range), dtype=bool)

                    if planned_positions[lv]:
                        all_pos = np.array([pos_cache[(lv, x)] for x in effective_range])
                        # compute distances vectorized
                        dists = np.linalg.norm(all_pos - world_p, axis=1)
                        for planned_pos in planned_positions[lv]:
                            dist = np.linalg.norm(planned_pos - world_p)
                            start, end = dist - seed_info['length'], dist + seed_info['length']
                            lv_mask[(dists > start) & (dists < end)] = False

                    mask.extend(lv_mask)

                mask = np.array(mask)
                if np.any(mask):
                    a = best_low_agent.select_action(state, mask=mask)
                    state, _, _ = best_low_env.step(a)
                    lv = np.searchsorted(best_cunsum, a, side="right") % target_level
                    traj, point, direction, update_dir, *_ = traj_cache[lv]

                    length = best_low_level_state_space[a]
                    img_position = point + update_dir * length
                    world_position = pos_cache[(lv, length)]

                    reward, cur_radiation, cur_DVH_rate, cur_seed_radiation = reward_calculator.forward(
                        traj=traj,
                        cur_radiation=cur_radiation,
                        direction=direction,
                        seed_point=img_position,
                        protect_OAR=rf_params['segmented_rewards']
                    )

                    best_low_agent.record_reward(reward)
                    planned_positions[lv].append(world_position)
                    planned_seed_radiations[lv].append(cur_seed_radiation)
                    planned_directions[lv] = np.array(utils.direction_transform(dose_image, direction))[0]

                    if cur_DVH_rate > DVH_rate:
                        best_low_env.done = True
                else:
                    best_low_env.done = True

            # Build final planned result for this rollout
            planned_res = []
            for lv in range(target_level):
                seeds = [[pos, planned_directions[lv]] for pos in planned_positions[lv]]
                single_seed_radiations = planned_seed_radiations[lv]
                trajectory_def = best_traj[lv][1]
                planned_res.append([trajectory_def, seeds, single_seed_radiations])

            # Update best solution if improved
            if best_low_agent.rewards and best_low_agent.rewards[-1] > best_reward:
                best_reward = best_low_agent.rewards[-1]
                best_plan = planned_res
                best_DVH_rate = cur_DVH_rate
                best_mask = []

                for lv, (_, _, _, _, world_p, effective_range) in enumerate(traj_cache):
                    lv_mask = np.ones(len(effective_range), dtype=bool)

                    if planned_positions[lv]:
                        all_pos = np.array([pos_cache[(lv, x)] for x in effective_range])
                        for planned_pos in planned_positions[lv]:
                            dist = np.linalg.norm(planned_pos - world_p)
                            start, end = dist - seed_info['length'], dist + seed_info['length']
                            dists = np.linalg.norm(all_pos - world_p, axis=1)
                            lv_mask[(dists > start) & (dists < end)] = False

                    best_mask.extend(lv_mask)

            pbar.set_postfix({
                "curr_reward": f"{(best_low_agent.rewards[-1] if best_low_agent.rewards else 0.0):.3f}",
                "best_reward": f"{best_reward:.3f}",
                "best_DVH": f"{best_DVH_rate:.3f}"
            })
            best_low_agent.finish_episode()

    # ===============================
    # Standard optimization mode
    # ===============================
    else:
        pbar = tqdm(range(rf_params['max_episodes']), total=rf_params['max_episodes'], ncols=120, desc="Episodes")
        for ep in pbar:
            state = env.reset()
            high_action = high_agent.select_action(state)

            low_reward, plan, cur_DVH_rate, mask, group_idx, low_level_state_space, low_env, low_agent = env.step(high_action, device=device)

            high_agent.record_reward(low_reward)
            high_agent.finish_episode()

            if low_agent.rewards and low_agent.rewards[-1] > best_reward:
                best_reward = low_agent.rewards[-1]
                best_plan = plan
                best_DVH_rate = cur_DVH_rate

            pbar.set_postfix({
                "curr_reward": f"{(low_agent.rewards[-1] if low_agent.rewards else 0.0):.3f}",
                "best_reward": f"{best_reward:.3f}",
                "best_DVH": f"{best_DVH_rate:.3f}"
            })
    end_time = time.time()
    print(f"Total time for reinforcement learning: {end_time - start_time:.2f} seconds")
            
    return best_plan, best_reward