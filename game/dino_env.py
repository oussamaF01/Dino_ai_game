import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from typing import Optional, Any, Dict, Tuple
from collections import deque
import cv2
import sys
import os

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from dino_game import DinoGame
except ImportError:
    try:
        from src.game.dino_game import DinoGame
    except ImportError:
        try:
            from .dino_game import DinoGame
        except ImportError:
            print("Could not import DinoGame. Please ensure dino_game.py is in the same directory.")
            raise


class SimplifiedDinosaurGameEnv(gym.Env):
    """
    Simplified Dinosaur Game Environment using feature-based observations
    
    Action Space: Discrete(3) - [no action, jump, duck]
    Observation Space: Box(-inf, inf, (8,), float32) - Feature vector
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array", None],
        "render_fps": 60,
    }
    
    def __init__(self, render_mode: Optional[str] = None, max_steps: int = 10000):
        super().__init__()
        
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0
        
        # Initialize pygame if rendering
        if render_mode == "human":
            pygame.init()
            pygame.display.init()
        
        # Initialize the base dinosaur game
        try:
            self.game = DinoGame()
            print("✓ DinoGame initialized successfully")
        except Exception as e:
            print(f"✗ Failed to initialize DinoGame: {e}")
            raise
        
        # Action space: 0=no action, 1=jump, 2=duck
        self.action_space = spaces.Discrete(3)
        
        # Observation space: Feature vector
        # [dino_y, dino_jumping, dino_ducking, obstacle_distance, obstacle_y, 
        #  obstacle_width, obstacle_height, game_speed]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(8,),
            dtype=np.float32
        )
        
        # Game state tracking
        self.previous_score = 0
        self.steps_alive = 0
        
        # Reward parameters
        self.survival_reward = 0.1
        self.score_reward_scale = 1.0
        self.death_penalty = -100.0
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Reset the game using the correct method
        try:
            self.game.reset_game()
            print("✓ Game reset successfully")
        except Exception as e:
            print(f"Warning: Game reset failed: {e}")
            
        self.current_step = 0
        self.previous_score = 0
        self.steps_alive = 0
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment"""
        self.current_step += 1
        self.steps_alive += 1
        
        # Execute action in the game using take_action method
        try:
            self.game.take_action(action)
            
            # Update the game state using the correct method
            self.game.update_game()
        except Exception as e:
            print(f"Warning: Game update failed: {e}")
        
        # Get new observation
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        game_state = self.game.get_game_state()
        terminated = game_state.get('game_over', False)
        truncated = self.current_step >= self.max_steps
        
        # Update tracking variables
        self.previous_score = game_state.get('score', 0)
        
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment"""
        if self.render_mode == "human":
            try:
                # Make sure pygame display is initialized
                if not pygame.get_init():
                    pygame.init()
                if not pygame.display.get_init():
                    pygame.display.init()
                
                # Update the game display
                self.game.draw()
                pygame.display.flip()  # Make sure the display updates
                
                # Handle pygame events to prevent window from becoming unresponsive
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pass  # Don't close during training
                
                # Control frame rate
                self.game.clock.tick(self.game.fps)
                
            except Exception as e:
                print(f"Warning: Render failed: {e}")
        elif self.render_mode == "rgb_array":
            return self._get_rgb_array()
    
    def close(self):
        """Clean up resources"""
        try:
            if pygame.get_init():
                pygame.quit()
        except:
            pass
    
    def _get_observation(self) -> np.ndarray:
        """Extract simplified features from game state"""
        try:
            # Get game state using the game's method
            state = self.game.get_game_state()
            
            # Normalize features for better training
            # Dino Y position (normalized by screen height)
            dino_y_norm = state.get('dino_y', 300.0) / self.game.height
            
            # Boolean states
            dino_jumping = float(state.get('dino_jumping', False))
            dino_ducking = float(state.get('dino_ducking', False))
            
            # Obstacle features (normalized)
            obstacle_distance = state.get('obstacle_distance', 999.0)
            obstacle_distance_norm = min(obstacle_distance / self.game.width, 1.0)  # Normalize by screen width
            
            obstacle_y_norm = state.get('obstacle_y', 350.0) / self.game.height
            obstacle_width_norm = state.get('obstacle_width', 0.0) / 100.0  # Assume max width ~100px
            obstacle_height_norm = state.get('obstacle_height', 0.0) / 100.0  # Assume max height ~100px
            
            # Game speed (normalized)
            game_speed_norm = state.get('game_speed', 6.0) / 15.0  # Max speed ~15
            
            observation = np.array([
                dino_y_norm,
                dino_jumping,
                dino_ducking,
                obstacle_distance_norm,
                obstacle_y_norm,
                obstacle_width_norm,
                obstacle_height_norm,
                game_speed_norm
            ], dtype=np.float32)
            
            return observation
            
        except Exception as e:
            print(f"Warning: Failed to get observation: {e}")
            # Return default observation
            return np.zeros(8, dtype=np.float32)
    
    def _get_rgb_array(self) -> np.ndarray:
        """Get RGB array from the game surface"""
        try:
            if hasattr(self.game, 'screen') and self.game.screen is not None:
                # Make sure the screen is up to date
                self.game.draw()
                
                # Convert pygame surface to numpy array
                frame = pygame.surfarray.array3d(self.game.screen)
                frame = np.transpose(frame, (1, 0, 2))  # Pygame uses (width, height, channels)
                return frame
        except Exception as e:
            print(f"Warning: Failed to get RGB array: {e}")
        
        # Return a black frame if screen is not available
        return np.zeros((400, 800, 3), dtype=np.uint8)
    
    def _calculate_reward(self) -> float:
        """Calculate reward for the current step"""
        try:
            # Get current game state
            state = self.game.get_game_state()
            current_score = state.get('score', 0)
            
            # Base survival reward
            reward = self.survival_reward
            
            # Score increase reward
            if current_score > self.previous_score:
                reward += (current_score - self.previous_score) * self.score_reward_scale
            
            # Death penalty
            if state.get('game_over', False):
                reward += self.death_penalty
            
            # Small bonus for getting closer to obstacles (risk-taking)
            obstacle_distance = state.get('obstacle_distance', 999.0)
            if obstacle_distance < 200:  # If obstacle is close
                proximity_bonus = (200 - obstacle_distance) / 1000.0  # Small bonus
                reward += proximity_bonus
        
        except Exception as e:
            print(f"Warning: Failed to calculate reward: {e}")
            reward = self.survival_reward  # Default survival reward
        
        return reward
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the current state"""
        try:
            state = self.game.get_game_state()
            
            return {
                "score": state.get('score', 0),
                "steps_alive": self.steps_alive,
                "game_over": state.get('game_over', False),
                "current_step": self.current_step,
                "game_speed": state.get('game_speed', 6.0),
                "obstacle_distance": state.get('obstacle_distance', 999.0),
                "dino_y": state.get('dino_y', 300.0),
                "dino_jumping": state.get('dino_jumping', False),
                "dino_ducking": state.get('dino_ducking', False),
            }
        except Exception as e:
            print(f"Warning: Failed to get info: {e}")
            return {
                "score": 0,
                "steps_alive": self.steps_alive,
                "game_over": False,
                "current_step": self.current_step,
            }


class DinosaurGameEnv(gym.Env):
    """
    Pixel-based Dinosaur Game Environment with frame stacking
    
    Action Space: Discrete(3) - [no action, jump, duck]
    Observation Space: Box(0, 255, (84, 84, 4), uint8) - Stacked grayscale frames
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array", None],
        "render_fps": 60,
    }
    
    def __init__(self, render_mode: Optional[str] = None, frame_stack: int = 4, max_steps: int = 10000):
        super().__init__()
        
        self.render_mode = render_mode
        self.frame_stack = frame_stack
        self.max_steps = max_steps
        self.current_step = 0
        
        # Initialize pygame if rendering
        if render_mode == "human":
            pygame.init()
            pygame.display.init()
        
        # Initialize the base dinosaur game
        try:
            self.game = DinoGame()
            print("✓ DinoGame initialized successfully")
        except Exception as e:
            print(f"✗ Failed to initialize DinoGame: {e}")
            raise
            
        # Action space: 0=no action, 1=jump, 2=duck
        self.action_space = spaces.Discrete(3)
        
        # Observation space: Stack of grayscale frames (84x84)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(84, 84, frame_stack),
            dtype=np.uint8
        )
        
        # Frame stacking for temporal information
        self.frame_buffer = deque(maxlen=frame_stack)
        
        # Game state tracking
        self.previous_score = 0
        self.steps_alive = 0
        
        # Reward parameters
        self.survival_reward = 0.1
        self.score_reward_scale = 1.0
        self.death_penalty = -100.0
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Reset the game
        try:
            self.game.reset_game()
            print("✓ Game reset successfully")
        except Exception as e:
            print(f"Warning: Game reset failed: {e}")
            
        self.current_step = 0
        self.previous_score = 0
        self.steps_alive = 0
        
        # Initialize frame buffer with initial frames
        initial_frame = self._get_processed_frame()
        self.frame_buffer.clear()
        for _ in range(self.frame_stack):
            self.frame_buffer.append(initial_frame)
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment"""
        self.current_step += 1
        self.steps_alive += 1
        
        # Execute action in the game
        try:
            self.game.take_action(action)
            
            # Update the game state
            self.game.update_game()
        except Exception as e:
            print(f"Warning: Game update failed: {e}")
        
        # Get new observation
        new_frame = self._get_processed_frame()
        self.frame_buffer.append(new_frame)
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        game_state = self.game.get_game_state()
        terminated = game_state.get('game_over', False)
        truncated = self.current_step >= self.max_steps
        
        # Update tracking variables
        self.previous_score = game_state.get('score', 0)
        
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment"""
        if self.render_mode == "human":
            try:
                # Make sure pygame display is initialized
                if not pygame.get_init():
                    pygame.init()
                if not pygame.display.get_init():
                    pygame.display.init()
                
                # Update the game display
                self.game.draw()
                pygame.display.flip()
                
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pass  # Don't close during training
                
                # Control frame rate
                self.game.clock.tick(self.game.fps)
                
            except Exception as e:
                print(f"Warning: Render failed: {e}")
        elif self.render_mode == "rgb_array":
            return self._get_rgb_array()
    
    def close(self):
        """Clean up resources"""
        try:
            if pygame.get_init():
                pygame.quit()
        except:
            pass
    
    def _get_processed_frame(self) -> np.ndarray:
        """Get processed grayscale frame from the game"""
        try:
            # Get the current frame from the game
            frame = self._get_rgb_array()
            
            # Convert to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Resize to 84x84
            resized_frame = cv2.resize(gray_frame, (84, 84), interpolation=cv2.INTER_AREA)
            
            return resized_frame
        except Exception as e:
            print(f"Warning: Failed to process frame: {e}")
            return np.zeros((84, 84), dtype=np.uint8)
    
    def _get_rgb_array(self) -> np.ndarray:
        """Get RGB array from the game surface"""
        try:
            if hasattr(self.game, 'screen') and self.game.screen is not None:
                # Make sure the screen is up to date
                self.game.draw()
                
                # Convert pygame surface to numpy array
                frame = pygame.surfarray.array3d(self.game.screen)
                frame = np.transpose(frame, (1, 0, 2))  # Pygame uses (width, height, channels)
                return frame
        except Exception as e:
            print(f"Warning: Failed to get RGB array: {e}")
        
        # Return a black frame if screen is not available
        return np.zeros((400, 800, 3), dtype=np.uint8)
    
    def _get_observation(self) -> np.ndarray:
        """Stack frames to create observation"""
        try:
            stacked_frames = np.stack(list(self.frame_buffer), axis=-1)
            return stacked_frames
        except Exception as e:
            print(f"Warning: Failed to get observation: {e}")
            return np.zeros((84, 84, self.frame_stack), dtype=np.uint8)
    
    def _calculate_reward(self) -> float:
        """Calculate reward for the current step"""
        try:
            # Get current game state
            state = self.game.get_game_state()
            current_score = state.get('score', 0)
            
            # Base survival reward
            reward = self.survival_reward
            
            # Score increase reward
            if current_score > self.previous_score:
                reward += (current_score - self.previous_score) * self.score_reward_scale
            
            # Death penalty
            if state.get('game_over', False):
                reward += self.death_penalty
        
        except Exception as e:
            print(f"Warning: Failed to calculate reward: {e}")
            reward = self.survival_reward  # Default survival reward
        
        return reward
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the current state"""
        try:
            state = self.game.get_game_state()
            
            return {
                "score": state.get('score', 0),
                "steps_alive": self.steps_alive,
                "game_over": state.get('game_over', False),
                "current_step": self.current_step,
                "game_speed": state.get('game_speed', 6.0),
                "obstacle_distance": state.get('obstacle_distance', 999.0),
            }
        except Exception as e:
            print(f"Warning: Failed to get info: {e}")
            return {
                "score": 0,
                "steps_alive": self.steps_alive,
                "game_over": False,
                "current_step": self.current_step,
            }


if __name__ == "__main__":
    print("Testing Fixed Dinosaur Game Environment...")
    
    try:
        # Test simplified environment creation
        env = SimplifiedDinosaurGameEnv(render_mode="human")
        print("✓ Simplified Environment created successfully")
        
        # Test reset
        obs, info = env.reset()
        print(f"✓ Reset successful - Observation shape: {obs.shape}")
        print(f"  Initial observation: {obs}")
        print(f"  Initial info: {info}")
        
        # Test a few steps with rendering
        total_reward = 0
        for i in range(100):  # Run longer to see the game
            action = env.action_space.sample()  # Random actions
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Render the game
            env.render()
            
            if i % 20 == 0:
                print(f"  Step {i+1}: action={action}, reward={reward:.2f}, score={info.get('score', 0)}")
            
            if terminated or truncated:
                print(f"  Episode ended at step {i + 1}")
                break
        
        print(f"✓ Test completed - Total reward: {total_reward:.2f}")
        env.close()
        
        print("\n" + "="*50)
        print("Fixed environment is working correctly!")
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()