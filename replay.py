"""
Replay memory buffer for storing and sampling transitions in Bootstrapped DQN.
Implements circular buffer of frames, actions, rewards, terminal flags, and per-head masks.
Supports saving/loading buffer state and sampling minibatches.
"""
import numpy as np
import time

class ReplayMemory:
    """Fixed-size circular replay buffer with per-ensemble-head masking"""
    def __init__(self, size=1000000, frame_height=84, frame_width=84,
                 agent_history_length=4, batch_size=32, num_heads=1, bernoulli_probability=1.0):
        """
        Args:
            size: max number of stored transitions
            frame_height, frame_width: dimensions of each frame
            agent_history_length: number of stacked frames per state
            batch_size: number of samples per minibatch
            num_heads: number of ensemble heads (for bootstrapping)
            bernoulli_probability: probability an experience is assigned to a head
        """
        self.bernoulli_probability = bernoulli_probability
        assert(self.bernoulli_probability > 0)
        self.size = size
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.agent_history_length = agent_history_length
        self.count = 0
        self.current = 0
        self.num_heads = num_heads
        # Pre-allocate arrays for frames, actions, rewards, masks, and terminal flags
        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.active_heads = np.empty(self.size, dtype=np.float32)

        self.frames = np.empty((self.size, self.frame_height, self.frame_width), dtype=np.uint8)
        self.terminal_flags = np.empty(self.size, dtype=np.bool)
        self.masks = np.empty((self.size, self.num_heads), dtype=np.bool)

        # Pre-allocate arrays for sampled states and next_states
        self.states = np.empty((batch_size, self.agent_history_length,
                                self.frame_height, self.frame_width), dtype=np.uint8)
        self.new_states = np.empty((batch_size, self.agent_history_length,
                                    self.frame_height, self.frame_width), dtype=np.uint8)
        self.indices = np.empty(batch_size, dtype=np.int32)
        self.random_state = np.random.RandomState(393)
        if self.num_heads == 1:
            assert(self.bernoulli_probability == 1.0)

    def save_buffer(self, filepath):
        """
        Save current replay buffer arrays and metadata to a .npz file.
        """
        st = time.time()
        print("starting save of buffer to %s"%filepath, st)
        np.savez(filepath,
                 frames=self.frames, actions=self.actions, rewards=self.rewards,
                 terminal_flags=self.terminal_flags, active_heads = self.active_heads, masks=self.masks,
                 count=self.count, current=self.current,
                 agent_history_length=self.agent_history_length,
                 frame_height=self.frame_height, frame_width=self.frame_width,
                 num_heads=self.num_heads, bernoulli_probability=self.bernoulli_probability,
                 )
        print("finished saving buffer", time.time()-st)

    def load_buffer(self, filepath):
        """
        Load replay buffer state from a .npz file into memory arrays.
        """
        st = time.time()
        print("starting load of buffer from %s"%filepath, st)
        npfile = np.load(filepath)
        self.frames = npfile['frames']
        self.actions = npfile['actions']
        self.rewards = npfile['rewards']
        self.terminal_flags = npfile['terminal_flags']
        self.active_heads = npfile['active_heads']
        self.masks = npfile['masks']
        self.count = npfile['count']
        self.current = npfile['current']
        self.agent_history_length = npfile['agent_history_length']
        self.frame_height = npfile['frame_height']
        self.frame_width = npfile['frame_width']
        self.num_heads = npfile['num_heads']
        self.bernoulli_probability = npfile['bernoulli_probability']
        if self.num_heads == 1:
            assert(self.bernoulli_probability == 1.0)
        print("finished loading buffer", time.time()-st)
        print("loaded buffer current is", self.current)

    def add_experience(self, action, frame, reward, terminal, active_head):
        """
        Add a single transition to the buffer with random masking for ensemble heads.
        Args:
            action (int): index of the action taken
            frame (np.ndarray): processed frame of shape (H, W)
            reward (float): reward received
            terminal (bool): whether episode ended
            active_head: head index used for this transition
        """
        if frame.shape != (self.frame_height, self.frame_width):
            raise ValueError('Dimension of frame is wrong!')
        self.actions[self.current] = action
        self.frames[self.current, ...] = frame
        self.rewards[self.current] = reward
        self.terminal_flags[self.current] = terminal
        self.active_heads[self.current] = active_head

        mask = self.random_state.binomial(1, self.bernoulli_probability, self.num_heads)
        self.masks[self.current] = mask
        self.count = max(self.count, self.current+1)
        self.current = (self.current + 1) % self.size

    def _get_state(self, index):
        """
        Retrieve a stacked state (history_length frames) ending at given index.
        """
        if self.count == 0:
            raise ValueError("The replay memory is empty!")
        if index < self.agent_history_length - 1:
            raise ValueError("Index must be min 3")
        return self.frames[index-self.agent_history_length+1:index+1, ...]

    def _get_valid_indices(self, batch_size):
        """
        Sample batch_size random indices that are valid for constructing states.
        Ensures episodes do not cross terminal boundaries.
        """
        if batch_size != self.indices.shape[0]:
             self.indices = np.empty(batch_size, dtype=np.int32)

        for i in range(batch_size):
            while True:
                index = self.random_state.randint(self.agent_history_length, self.count - 1)
                if index < self.agent_history_length:
                    continue
                if index >= self.current and index - self.agent_history_length <= self.current:
                    continue
                # dont add if there was a terminal flag in previous
                # history_length steps
                if self.terminal_flags[index - self.agent_history_length:index].any():
                    continue
                break
            self.indices[i] = index

    def get_minibatch(self, batch_size):
        """
        Return a minibatch of transitions (states, actions, rewards, next_states, terminals, heads, masks).
        Args:
            batch_size (int): number of samples
        Returns:
            tuple of numpy arrays: (states, actions, rewards, next_states, terminal_flags, active_heads, masks)
        """
        if batch_size != self.states.shape[0]:
            self.states = np.empty((batch_size, self.agent_history_length,
                                    self.frame_height, self.frame_width), dtype=np.uint8)
            self.new_states = np.empty((batch_size, self.agent_history_length,
                                        self.frame_height, self.frame_width), dtype=np.uint8)

        if self.count < self.agent_history_length:
            raise ValueError('Not enough memories to get a minibatch')

        self._get_valid_indices(batch_size)

        for i, idx in enumerate(self.indices):
            self.states[i] = self._get_state(idx - 1)
            self.new_states[i] = self._get_state(idx)
        return self.states, self.actions[self.indices], self.rewards[self.indices], self.new_states, self.terminal_flags[self.indices], self.active_heads[self.indices], self.masks[self.indices]



