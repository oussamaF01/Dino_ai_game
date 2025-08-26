import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import os
from datetime import datetime

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import os
from datetime import datetime

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class DQN(nn.Module):
    """Deep Q-Network for feature-based observations"""
    
    def __init__(self, state_size: int, action_size: int, hidden_sizes: List[int] = [128, 64]):
        super(DQN, self).__init__()
        
        layers = []
        input_size = state_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU()
            ])
            input_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(input_size, action_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        return self.network(state)

class ConvDQN(nn.Module):
    """Convolutional DQN for pixel-based observations"""
    
    def __init__(self, action_size: int, frame_stack: int = 4):
        super(ConvDQN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(frame_stack, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate conv output size
        conv_out_size = self._get_conv_out_size(84, 84, frame_stack)
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, action_size)
    
    def _get_conv_out_size(self, h, w, c):
        """Calculate the output size of convolutional layers"""
        with torch.no_grad():
            x = torch.zeros(1, c, h, w)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            return x.view(1, -1).size(1)
    
    def forward(self, state):
        # Normalize pixel values to [0, 1]
        x = state.float() / 255.0
        
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

class ReplayBuffer:
    """Experience replay buffer for DQN training"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample random batch from buffer"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """DQN Agent for Dinosaur Game"""
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        use_conv: bool = False,
        frame_stack: int = 4,
        lr: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100000,
        batch_size: int = 32,
        target_update: int = 1000,
        device: Optional[str] = None
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.use_conv = use_conv
        self.frame_stack = frame_stack
        self.lr = lr
        self.gamma = gamma
        self.epsilon_start = epsilon_start  # Store the initial epsilon value
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Networks
        if use_conv:
            self.q_network = ConvDQN(action_size, frame_stack).to(self.device)
            self.target_network = ConvDQN(action_size, frame_stack).to(self.device)
        else:
            self.q_network = DQN(state_size, action_size).to(self.device)
            self.target_network = DQN(state_size, action_size).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Training tracking
        self.steps_done = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        
        # Copy weights to target network
        self.update_target_network()
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Choose action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        # Convert state to tensor
        if self.use_conv:
            # For pixel observations: (H, W, C) -> (1, C, H, W)
            state_tensor = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0).to(self.device)
        else:
            # For feature observations: (features,) -> (1, features)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get Q-values
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self):
        """Train the network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        experiences = self.memory.sample(self.batch_size)
        
        # Prepare batch tensors
        if self.use_conv:
            # Pixel observations
            states = torch.FloatTensor([e.state for e in experiences]).permute(0, 3, 1, 2).to(self.device)
            next_states = torch.FloatTensor([e.next_state for e in experiences]).permute(0, 3, 1, 2).to(self.device)
        else:
            # Feature observations
            states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
            next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        dones = torch.BoolTensor([e.done for e in experiences]).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        
        # Update target network
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.update_target_network()
        
        # Track loss
        self.losses.append(loss.item())
    
    def save_model(self, filepath: str):
        """Save model and training state"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'epsilon_start': self.epsilon_start,  # Save epsilon_start
            'epsilon_end': self.epsilon_end,
            'epsilon_decay': self.epsilon_decay,
            'steps_done': self.steps_done,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'losses': self.losses,
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load model and training state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        # Load epsilon_start if available (for backwards compatibility)
        self.epsilon_start = checkpoint.get('epsilon_start', 1.0)
        self.epsilon_end = checkpoint.get('epsilon_end', 0.01)
        self.epsilon_decay = checkpoint.get('epsilon_decay', 0.995)
        self.steps_done = checkpoint['steps_done']
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_lengths = checkpoint['episode_lengths']
        self.losses = checkpoint['losses']
    
    def plot_training_progress(self, save_path: Optional[str] = None):
        """Plot training progress"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        if self.episode_rewards:
            axes[0, 0].plot(self.episode_rewards)
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Total Reward')
            
            # Moving average
            if len(self.episode_rewards) >= 100:
                moving_avg = np.convolve(self.episode_rewards, np.ones(100)/100, mode='valid')
                axes[0, 0].plot(range(99, len(self.episode_rewards)), moving_avg, 'r-', alpha=0.7, label='Moving Average (100)')
                axes[0, 0].legend()
        
        # Episode lengths
        if self.episode_lengths:
            axes[0, 1].plot(self.episode_lengths)
            axes[0, 1].set_title('Episode Lengths')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Steps')
        
        # Training loss
        if self.losses:
            axes[1, 0].plot(self.losses)
            axes[1, 0].set_title('Training Loss')
            axes[1, 0].set_xlabel('Training Step')
            axes[1, 0].set_ylabel('MSE Loss')
        
        # Epsilon decay
        epsilons = [self.epsilon_start * (self.epsilon_decay ** i) for i in range(self.steps_done)]
        axes[1, 1].plot(epsilons)
        axes[1, 1].set_title('Epsilon Decay')
        axes[1, 1].set_xlabel('Steps')
        axes[1, 1].set_ylabel('Epsilon')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def train_agent(
    env,
    agent: DQNAgent,
    n_episodes: int = 1000,
    max_steps: int = 10000,
    save_every: int = 100,
    model_dir: str = "models"
):
    """Train the DQN agent"""
    
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Starting training for {n_episodes} episodes...")
    print(f"Device: {agent.device}")
    print(f"Epsilon start: {agent.epsilon}")
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(max_steps):
            # Choose action
            action = agent.act(state, training=True)
            
            # Take action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Train agent
            agent.replay()
            
            # Update state
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # Record episode
        agent.episode_rewards.append(total_reward)
        agent.episode_lengths.append(steps)
        
        # Print progress
        if episode % 10 == 0:
            avg_reward = np.mean(agent.episode_rewards[-100:]) if len(agent.episode_rewards) >= 100 else np.mean(agent.episode_rewards)
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}, Score: {info.get('score', 0)}")
        
        # Save model
        if episode % save_every == 0 and episode > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(model_dir, f"dino_dqn_ep{episode}_{timestamp}.pth")
            agent.save_model(model_path)
            print(f"Model saved: {model_path}")
    
    return agent

def test_agent(env, agent: DQNAgent, n_episodes: int = 10, render: bool = True):
    """Test the trained agent"""
    print(f"Testing agent for {n_episodes} episodes...")
    
    episode_scores = []
    episode_rewards = []
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        
        while True:
            if render:
                env.render()
            
            # Choose action (no exploration)
            action = agent.act(state, training=False)
            
            # Take action
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                score = info.get('score', 0)
                episode_scores.append(score)
                episode_rewards.append(total_reward)
                print(f"Episode {episode + 1}: Score = {score}, Total Reward = {total_reward:.2f}")
                break
    
    print(f"\nTest Results:")
    print(f"Average Score: {np.mean(episode_scores):.2f} ± {np.std(episode_scores):.2f}")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Best Score: {max(episode_scores)}")
    
    return episode_scores, episode_rewards


def train_agent(
    env,
    agent: DQNAgent,
    n_episodes: int = 1000,
    max_steps: int = 10000,
    save_every: int = 100,
    model_dir: str = "models"
):
    """Train the DQN agent"""
    
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Starting training for {n_episodes} episodes...")
    print(f"Device: {agent.device}")
    print(f"Epsilon start: {agent.epsilon}")
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(max_steps):
            # Choose action
            action = agent.act(state, training=True)
            
            # Take action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Train agent
            agent.replay()
            
            # Update state
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # Record episode
        agent.episode_rewards.append(total_reward)
        agent.episode_lengths.append(steps)
        
        # Print progress
        if episode % 10 == 0:
            avg_reward = np.mean(agent.episode_rewards[-100:]) if len(agent.episode_rewards) >= 100 else np.mean(agent.episode_rewards)
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}, Score: {info.get('score', 0)}")
        
        # Save model
        if episode % save_every == 0 and episode > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(model_dir, f"dino_dqn_ep{episode}_{timestamp}.pth")
            agent.save_model(model_path)
            print(f"Model saved: {model_path}")
    
    return agent

def test_agent(env, agent: DQNAgent, n_episodes: int = 10, render: bool = True):
    """Test the trained agent"""
    print(f"Testing agent for {n_episodes} episodes...")
    
    episode_scores = []
    episode_rewards = []
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        
        while True:
            if render:
                env.render()
            
            # Choose action (no exploration)
            action = agent.act(state, training=False)
            
            # Take action
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                score = info.get('score', 0)
                episode_scores.append(score)
                episode_rewards.append(total_reward)
                print(f"Episode {episode + 1}: Score = {score}, Total Reward = {total_reward:.2f}")
                break
    
    print(f"\nTest Results:")
    print(f"Average Score: {np.mean(episode_scores):.2f} ± {np.std(episode_scores):.2f}")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Best Score: {max(episode_scores)}")
    
    return episode_scores, episode_rewards

if __name__ == "__main__":
    print("DQN Agent for Dinosaur Game")
    print("This module provides the DQN implementation.")
    print("Import and use with your environment for training.")
    
    print("DQN Agent for Dinosaur Game")
    print("This module provides the DQN implementation.")
    print("Import and use with your environment for training.")