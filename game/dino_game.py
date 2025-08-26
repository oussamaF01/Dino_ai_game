import pygame
import random
import sys
from typing import List, Tuple, Optional
from .game_elements import Dino, Obstacle, Ground, Cloud

class DinoGame:
    def __init__(self, width: int = 800, height: int = 400):
        pygame.init()
        
        # Screen setup
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Chrome Dino Game")
        
        # Colors
        self.bg_color = (247, 247, 247)  # Light gray background
        self.text_color = (83, 83, 83)   # Dark gray text
        
        # Game settings
        self.clock = pygame.time.Clock()
        self.fps = 60
        self.ground_y = height - 50
        
        # Game state
        self.game_over = False
        self.score = 0
        self.high_score = 0
        self.game_speed = 6
        
        # Game objects
        self.dino = Dino(50, self.ground_y - 47)
        self.ground = Ground(width, 50, self.ground_y)
        self.obstacles: List[Obstacle] = []
        self.clouds: List[Cloud] = []
        
        # Spawning timers
        self.obstacle_timer = 0
        self.obstacle_spawn_time = random.randint(80, 150)
        self.cloud_timer = 0
        self.cloud_spawn_time = random.randint(200, 400)
        
        # Score timer
        self.score_timer = 0
        
        # Font
        self.font = pygame.font.Font(None, 36)
        
        # Game state
        self.running = True
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if self.game_over:
                        self.reset_game()
                    else:
                        self.dino.jump()
                elif event.key == pygame.K_DOWN:
                    self.dino.duck(True)
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_DOWN:
                    self.dino.duck(False)
    
    def take_action(self, action: int):
        """Take action for AI control"""
        # Actions: 0 = do nothing, 1 = jump, 2 = duck
        if action == 1:
            self.dino.jump()
        elif action == 2:
            self.dino.duck(True)
        else:
            self.dino.duck(False)
    
    def spawn_obstacle(self):
        """Spawn new obstacle"""
        obstacle_type = random.choice(["cactus", "cactus", "cactus", "bird"])  # More cacti
        obstacle = Obstacle(self.width, self.ground_y - 35, obstacle_type)
        self.obstacles.append(obstacle)
    
    def spawn_cloud(self):
        """Spawn new cloud"""
        cloud_y = random.randint(50, 200)
        cloud = Cloud(self.width, cloud_y)
        self.clouds.append(cloud)
    
    def check_collisions(self) -> bool:
        """Check if dino collides with obstacles"""
        dino_rect = self.dino.get_rect()
        
        for obstacle in self.obstacles:
            obstacle_rect = obstacle.get_rect()
            # Simple rectangle collision with slight tolerance
            if dino_rect.colliderect(obstacle_rect):
                # Add some tolerance to make game more fair
                tolerance = 5
                adjusted_dino = pygame.Rect(
                    dino_rect.x + tolerance,
                    dino_rect.y + tolerance,
                    dino_rect.width - 2 * tolerance,
                    dino_rect.height - 2 * tolerance
                )
                adjusted_obstacle = pygame.Rect(
                    obstacle_rect.x + tolerance,
                    obstacle_rect.y + tolerance,
                    obstacle_rect.width - 2 * tolerance,
                    obstacle_rect.height - 2 * tolerance
                )
                if adjusted_dino.colliderect(adjusted_obstacle):
                    return True
        return False
    
    def update_game(self):
        """Update game state"""
        if self.game_over:
            return
        
        # Update score
        self.score_timer += 1
        if self.score_timer >= 6:  # Increase score every 6 frames
            self.score += 1
            self.score_timer = 0
        
        # Increase game speed over time
        if self.score > 0 and self.score % 100 == 0:
            self.game_speed = min(self.game_speed + 0.1, 12)
        
        # Update game objects
        self.dino.update()
        self.ground.update()
        
        # Update obstacles
        for obstacle in self.obstacles[:]:
            obstacle.update()
            if obstacle.is_off_screen():
                self.obstacles.remove(obstacle)
        
        # Update clouds
        for cloud in self.clouds[:]:
            cloud.update()
            if cloud.is_off_screen():
                self.clouds.remove(cloud)
        
        # Spawn obstacles
        self.obstacle_timer += 1
        if self.obstacle_timer >= self.obstacle_spawn_time:
            self.spawn_obstacle()
            self.obstacle_timer = 0
            self.obstacle_spawn_time = random.randint(80, 200)
        
        # Spawn clouds
        self.cloud_timer += 1
        if self.cloud_timer >= self.cloud_spawn_time:
            self.spawn_cloud()
            self.cloud_timer = 0
            self.cloud_spawn_time = random.randint(200, 400)
        
        # Check collisions
        if self.check_collisions():
            self.game_over = True
            if self.score > self.high_score:
                self.high_score = self.score
    
    def draw(self):
        """Draw everything"""
        # Clear screen
        self.screen.fill(self.bg_color)
        
        # Draw clouds
        for cloud in self.clouds:
            cloud.draw(self.screen)
        
        # Draw ground
        self.ground.draw(self.screen)
        
        # Draw obstacles
        for obstacle in self.obstacles:
            obstacle.draw(self.screen)
        
        # Draw dino
        self.dino.draw(self.screen)
        
        # Draw score
        score_text = self.font.render(f"Score: {self.score:05d}", True, self.text_color)
        self.screen.blit(score_text, (self.width - 200, 30))
        
        high_score_text = self.font.render(f"High: {self.high_score:05d}", True, self.text_color)
        self.screen.blit(high_score_text, (self.width - 200, 70))
        
        # Draw game over screen
        if self.game_over:
            game_over_text = self.font.render("GAME OVER", True, self.text_color)
            restart_text = self.font.render("Press SPACE to restart", True, self.text_color)
            
            # Center the text
            go_rect = game_over_text.get_rect(center=(self.width // 2, self.height // 2))
            restart_rect = restart_text.get_rect(center=(self.width // 2, self.height // 2 + 40))
            
            self.screen.blit(game_over_text, go_rect)
            self.screen.blit(restart_text, restart_rect)
        
        pygame.display.flip()
    
    def reset_game(self):
        """Reset game to initial state"""
        self.game_over = False
        self.score = 0
        self.game_speed = 6
        
        # Reset dino
        self.dino = Dino(50, self.ground_y - 47)
        
        # Clear obstacles and clouds
        self.obstacles.clear()
        self.clouds.clear()
        
        # Reset timers
        self.obstacle_timer = 0
        self.obstacle_spawn_time = random.randint(80, 150)
        self.cloud_timer = 0
        self.cloud_spawn_time = random.randint(200, 400)
        self.score_timer = 0
    
    def get_game_state(self) -> dict:
        """Get current game state for AI"""
        # Find nearest obstacle
        nearest_obstacle = None
        min_distance = float('inf')
        
        for obstacle in self.obstacles:
            if obstacle.x > self.dino.x:  # Only consider obstacles ahead
                distance = obstacle.x - self.dino.x
                if distance < min_distance:
                    min_distance = distance
                    nearest_obstacle = obstacle
        
        state = {
            'dino_y': self.dino.y,
            'dino_ground_y': self.ground_y - 47,
            'dino_jumping': self.dino.is_jumping,
            'dino_ducking': self.dino.is_ducking,
            'game_speed': self.game_speed,
            'score': self.score,
            'game_over': self.game_over
        }
        
        if nearest_obstacle:
            state.update({
                'obstacle_distance': min_distance,
                'obstacle_type': nearest_obstacle.type,
                'obstacle_y': nearest_obstacle.y,
                'obstacle_width': nearest_obstacle.width,
                'obstacle_height': nearest_obstacle.height
            })
        else:
            state.update({
                'obstacle_distance': 999,
                'obstacle_type': 'none',
                'obstacle_y': self.ground_y,
                'obstacle_width': 0,
                'obstacle_height': 0
            })
        
        return state
    
    def get_reward(self, prev_score: int) -> float:
        """Calculate reward for AI training"""
        if self.game_over:
            return -100  # Big penalty for dying
        
        reward = 0.1  # Small reward for staying alive
        
        if self.score > prev_score:
            reward += 1  # Reward for increasing score
        
        return reward
    
    def run(self):
        """Main game loop"""
        while self.running:
            self.handle_events()
            self.update_game()
            self.draw()
            self.clock.tick(self.fps)
        
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = DinoGame()
    game.run()