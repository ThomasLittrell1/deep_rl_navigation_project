import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.optim as optim
from scipy.stats import rankdata

from src.model import QNetwork

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 5e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network
DEFAULT_PRIORITY = 1  # Priority to put on an experience before it is sampled
PRIORITY_ALPHA = 0.1  # How much prioritization is used in sample (0 = uniform)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# noinspection PyUnresolvedReferences
class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, td_target_type="DQN"):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.td_target_type = td_target_type

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(
            state, action, reward, next_state, done, priority=DEFAULT_PRIORITY
        )

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.0):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        ids, states, actions, rewards, next_states, dones, priorities = experiences

        criterion = torch.nn.MSELoss()
        optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        optimizer.zero_grad()

        if self.td_target_type == "DQN":
            # compute the Q target using the Q-target network
            best_next_Q = (
                self.qnetwork_target.forward(next_states)
                .detach()
                .max(1)[0]
                .unsqueeze(1)
            )
        elif self.td_target_type == "Double DQN":
            # select best action using current network
            best_next_actions = (
                self.qnetwork_local.forward(next_states)
                .detach()
                .max(1)[1]
                .reshape(-1, 1)
            )

            # Use the target network to evaluate the best actions
            best_next_Q = (
                self.qnetwork_target.forward(next_states)
                .detach()
                .gather(1, best_next_actions)
            )
        else:
            raise ValueError(f"Invalid td target method {self.td_target_type}")

        Q_target = rewards + gamma * best_next_Q * (1 - dones)

        Q_current = self.qnetwork_local.forward(states).gather(1, actions)
        loss = criterion(Q_current, Q_target)
        loss.backward()
        optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

        # Update the replay buffer with new priorities
        new_priorities = rankdata(
            (Q_target - Q_current).detach().abs().reshape(-1).numpy()
        )
        for experience in zip(
            ids,
            states,
            actions,
            rewards,
            next_states,
            dones,
            priorities,
            new_priorities,
        ):
            self.memory.update_priority(*experience)

    @staticmethod
    def soft_update(local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )


# noinspection PyUnresolvedReferences
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.next_id = 0  # keep a running counter for IDs in the experience buffer

    def add(self, state, action, reward, next_state, done, priority):
        """Add a new experience to memory."""
        e = Experience(self.next_id, state, action, reward, next_state, done, priority)
        self.memory.append(e)
        self.next_id += 1

    def update_priority(
        self, id, state, action, reward, next_state, done, old_priority, new_priority
    ):
        old_experience = Experience(
            id, state, action, reward, next_state, done, old_priority
        )
        new_experience = Experience(
            id, state, action, reward, next_state, done, new_priority
        )
        self.memory.remove(old_experience)
        self.memory.append(new_experience)

    def get_sample_weights(self):
        priorities = np.array([e.priority for e in self.memory]) ** PRIORITY_ALPHA
        denom = priorities.sum()
        return priorities / denom

    def sample(self):
        """Randomly sample a batch of experiences from memory."""

        weights = self.get_sample_weights()
        experiences: List[Experience] = random.choices(
            population=self.memory, weights=weights, k=self.batch_size
        )

        ids = (
            torch.from_numpy(np.vstack([e.idx for e in experiences if e is not None]))
            .float()
            .to(device)
        )
        states = (
            torch.from_numpy(np.vstack([e.state for e in experiences if e is not None]))
            .float()
            .to(device)
        )
        states = (
            torch.from_numpy(np.vstack([e.state for e in experiences if e is not None]))
            .float()
            .to(device)
        )
        actions = (
            torch.from_numpy(
                np.vstack([e.action for e in experiences if e is not None])
            )
            .long()
            .to(device)
        )
        rewards = (
            torch.from_numpy(
                np.vstack([e.reward for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        next_states = (
            torch.from_numpy(
                np.vstack([e.next_state for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        dones = (
            torch.from_numpy(
                np.vstack([e.done for e in experiences if e is not None]).astype(
                    np.uint8
                )
            )
            .float()
            .to(device)
        )
        priorities = (
            torch.from_numpy(
                np.vstack([e.priority for e in experiences if e is not None]).astype(
                    np.uint8
                )
            )
            .float()
            .to(device)
        )
        return ids, states, actions, rewards, next_states, dones, priorities

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class Experience:
    """
    Use an experience class instead of a namedtuple so we can overwrite the equality
    method for experience updating
    """

    def __init__(self, idx, state, action, reward, next_state, done, priority):
        self.idx = idx
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.priority = priority

    def __eq__(self, other):
        return self.idx == other.idx
