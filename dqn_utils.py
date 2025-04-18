"""
Utility functions for saving/loading models, seeding, replay step handling, epsilon schedules, logging, and GIF generation.
Includes:
- save_checkpoint: persist model state
- seed_everything: fix random seeds for reproducibility
- handle_step: wrap experience storage and state update
- linearly_decaying_epsilon: schedule for exploration rate
- write_info_file: log experiment configuration
- generate_gif: save episode frames as an animated GIF
"""
import numpy as np
import torch
import os
import sys
from imageio import mimsave
import cv2

# Persist model training state to disk
def save_checkpoint(state, filename='model.pkl'):
    """
    Save model and optimizer state dict to file for checkpointing.
    Args:
        state (dict): contains 'info', 'optimizer', 'cnt', 'policy_net_state_dict', 'target_net_state_dict', 'perf'
        filename (str): target filepath
    """
    print("starting save of model %s" %filename)
    torch.save(state, filename)
    print("finished save of model %s" %filename)

# Seed all relevant random number generators for deterministic runs
def seed_everything(seed=1234):
    """
    Set seed for numpy, torch, CUDA, and Python hash seed.
    Ensures reproducible experiments.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# Handle adding a new experience to the replay buffer
def handle_step(random_state, cnt, S_hist, S_prime, action, reward, finished, k_used, acts, episodic_reward, replay_buffer, checkpoint='', n_ensemble=1, bernoulli_p=1.0):
    """
    Process one environment step:
      - Sample mask for ensemble heads via bernoulli
      - Add transition to replay_buffer
      - Update state history queue and counter
    Returns:
        cnt (int): updated step count
        S_hist (list): updated frame history
        batch: minibatch ready flag or data
        episodic_reward (float): cumulative reward
    """
    exp_mask = random_state.binomial(1, bernoulli_p, n_ensemble).astype(np.uint8)
    experience =  [S_prime, action, reward, finished, exp_mask, k_used, acts, cnt]
    batch = replay_buffer.send((checkpoint, experience))
    S_hist.pop(0)
    S_hist.append(S_prime)
    episodic_reward += reward
    cnt+=1
    return cnt, S_hist, batch, episodic_reward

# Schedule for epsilon-greedy exploration decay
def linearly_decaying_epsilon(decay_period, step, warmup_steps, epsilon):
    """
    Compute epsilon for current step following linear decay schedule:
      - Epsilon starts at 1 until warmup
      - Decays from 1->epsilon over decay_period
      - Clamped at epsilon thereafter
    Args:
        decay_period (float), step (int), warmup_steps (int), epsilon (float final)
    Returns:
        float: current epsilon
    """
    steps_left = decay_period + warmup_steps - step
    bonus = (1.0 - epsilon) * steps_left / decay_period
    bonus = np.clip(bonus, 0., 1. - epsilon)
    return epsilon + bonus

# Write experiment info parameters to a text file
def write_info_file(info, model_base_filepath, cnt):
    """
    Serialize the 'info' dict to an _info.txt file for record-keeping.
    """
    info_filename = model_base_filepath + "_%010d_info.txt"%cnt
    info_f = open(info_filename, 'w')
    for (key,val) in info.items():
        info_f.write('%s=%s\n'%(key,val))
    info_f.close()

# Generate a GIF from stored episode frames
def generate_gif(base_dir, step_number, frames_for_gif, reward, name='', results=[]):
    """
    Resize frames, assemble into GIF, and save to base_dir.
    Args:
        base_dir (str), step_number (int), frames_for_gif (list of images), reward (float), name (str suffix)
    """
    for idx, frame_idx in enumerate(frames_for_gif):
        frames_for_gif[idx] = cv2.resize(frame_idx, (320, 220)).astype(np.uint8)

    if len(frames_for_gif[0].shape) == 2:
        name+='gray'
    else:
        name+='color'
    gif_fname = os.path.join(base_dir, "ATARI_step%010d_r%04d_%s.gif"%(step_number, int(reward), name))

    print("WRITING GIF", gif_fname)
    mimsave(gif_fname, frames_for_gif, duration=1/30)
