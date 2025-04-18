"""
Environment wrapper using ALEInterface to interact with Atari ROMs.
Handles frame preprocessing, grayscale conversion, stacking frame history,
frame skipping, life-based termination, and reset/step functionality.
"""
import os
from collections import deque
import numpy as np
import ale_py
from ale_py import ALEInterface
import cv2

# Convert RGB frame to grayscale and resize to square output
def cv_preprocess_frame(observ, output_size):
    """Convert RGB to grayscale and resize via nearest neighbor interpolation"""
    gray = cv2.cvtColor(observ, cv2.COLOR_RGB2GRAY)
    output = cv2.resize(gray, (output_size, output_size), interpolation=cv2.INTER_NEAREST)
    return output

class Environment(object):
    """Main environment class interfacing with ALE (Atari Learning Environment)"""
    def __init__(self,
                 rom_file,
                 frame_skip=4,
                 num_frames=4,
                 frame_size=84,
                 no_op_start=30,
                 rand_seed=393,
                 dead_as_end=True,
                 max_episode_steps=18000,
                 autofire=False):
        """
        Args:
            rom_file (str): path to Atari ROM file
            frame_skip (int): repeat the same action this many frames
            num_frames (int): number of history frames to stack for state
            frame_size (int): size to resize frames (frame_size x frame_size)
            no_op_start (int): random no-op actions at episode start
            rand_seed (int): seed for reproducibility
            dead_as_end (bool): treat life loss as terminal
            max_episode_steps (int): cap on steps per episode
        """
        self.max_episode_steps = max_episode_steps
        self.random_state = np.random.RandomState(rand_seed+15)
        self.ale = self._init_ale(rand_seed, rom_file)
        # normally (160, 210)
        self.actions = self.ale.getMinimalActionSet()

        self.frame_skip = frame_skip
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.no_op_start = no_op_start
        self.dead_as_end = dead_as_end

        self.total_reward = 0
        screen_width, screen_height = self.ale.getScreenDims()
        self.prev_screen = np.zeros(
            (screen_height, screen_width, 3), dtype=np.uint8)
        self.frame_queue = deque(maxlen=num_frames)
        self.end = True

    @staticmethod
    def _init_ale(rand_seed, rom_file):
        """
        Configure ALEInterface settings and load the ROM.
        Returns an ALEInterface instance for game simulation.
        """
        assert os.path.exists(rom_file), f'{rom_file} does not exist.'
        ale = ALEInterface()

        # Set settings compatible with ale_py 0.10.2
        ale.setInt("random_seed", rand_seed)
        ale.setInt("frame_skip", 1)
        ale.setFloat("repeat_action_probability", 0.0)
        ale.setBool("color_averaging", False)

        # Load the ROM file
        ale.loadROM(rom_file)
        return ale

    @property
    def num_actions(self):
        """Number of discrete actions available in this Atari environment"""
        return len(self.actions)

    def _get_current_frame(self):
        """
        Preprocess current and previous screen frames to produce a single grayscale observation.
        Applies max-pooling over RGB channels then resizing.
        """
        # Get current screen
        screen = self.ale.getScreenRGB()

        # Ensure consistent dimensions by checking and reshaping if necessary
        if self.prev_screen.shape != screen.shape:
            # Reshape prev_screen to match current screen dimensions
            self.prev_screen = np.zeros(screen.shape, dtype=np.uint8)

        max_screen = np.maximum(self.prev_screen, screen)
        frame = cv_preprocess_frame(max_screen, self.frame_size)
        return frame

    def close(self):
        """Clean up ALE resources when done."""
        del(self.ale)

    def reset(self):
        """
        Reset the game environment:
          - Clear frame queue and reward
          - Run random no-op actions
          - Populate initial stacked frames
        Returns initial state as np.array of stacked frames.
        """
        self.steps = 0
        self.end = False
        self.plot_frames = []
        self.gray_plot_frames = []
        for _ in range(self.num_frames - 1):
            self.frame_queue.append(
                np.zeros((self.frame_size, self.frame_size), dtype=np.uint8))

        # steps are in steps the agent sees
        self.ale.reset_game()
        self.total_reward = 0
        self.prev_screen = np.zeros(self.prev_screen.shape, dtype=np.uint8)

        n = self.random_state.randint(0, self.no_op_start)
        for i in range(n):
            if i == n - 1:
                self.prev_screen = self.ale.getScreenRGB()
            self.ale.act(0)

        self.frame_queue.append(self._get_current_frame())
        self.plot_frames.append(self.prev_screen)
        a = np.array(self.frame_queue)
        out = np.concatenate((a[0],a[1],a[2],a[3]),axis=0).T
        self.gray_plot_frames.append(out)
        if self.ale.game_over():
            print("Unexpected game over in reset", self.reset())
        return np.array(self.frame_queue)

    def step(self, action_idx):
        """
        Execute action for frame_skip frames:
          - Accumulate reward
          - Detect life loss or game over
          - Update frame queue and return new state
        Returns:
          state (np.array): stacked frames
          reward (float): sum of rewards over skips
          life_dead (bool): True if life lost or terminal
          end (bool): True if episode ended
        """
        assert not self.end
        reward = 0
        old_lives = self.ale.lives()

        for i in range(self.frame_skip):
            if i == self.frame_skip - 1:
                self.prev_screen = self.ale.getScreenRGB()
            r = self.ale.act(self.actions[action_idx])
            reward += r
        dead = (self.ale.lives() < old_lives)
        if self.dead_as_end and dead:
            lives_dead = True
        else:
            lives_dead = False

        if self.ale.game_over():
            self.end = True
            lives_dead = True
        self.steps +=1
        if self.steps >= self.max_episode_steps:
            self.end = True
            lives_dead = True
        self.frame_queue.append(self._get_current_frame())
        self.total_reward += reward
        a = np.array(self.frame_queue)
        self.prev_screen = self.ale.getScreenRGB()
        self.gray_plot_frames.append(np.concatenate((a[0],a[1],a[2],a[3]),axis=0))
        self.plot_frames.append(self.prev_screen)
        return np.array(self.frame_queue), reward, lives_dead, self.end


if __name__ == '__main__':
    # Quick performance test of environment stepping speed
    import time
    env = Environment('roms/breakout.bin', 4, 4, 84, 30, 33, True)
    print('starting with game over?', env.ale.game_over())
    random_state = np.random.RandomState(304)

    state = env.reset()
    i = 0
    times = [0.0 for a in range(1000)]
    total_reward = 0
    for t in times:
        action = random_state.randint(0, env.num_actions)
        st = time.time()
        state, reward, end, do_reset = env.step(action)
        total_reward += reward
        times[i] = time.time()-st
        i += 1
        if end:
            print(i,"END")
        if do_reset:
            print(i,"RESET")
            state = env.reset()
    print('total_reward:', total_reward)
    print('total steps:', i)
    print('mean time', np.mean(times))
    print('max time', np.max(times))
