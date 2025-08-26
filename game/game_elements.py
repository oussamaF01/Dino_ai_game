import pygame
import random
from typing import List, Tuple

class Dino:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.ground_y = y
        self.width = 44
        self.height = 47
        
        # Physics
        self.velocity_y = 0
        self.gravity = 0.6
        self.jump_strength = -12
        self.is_jumping = False
        self.is_ducking = False
        
        # Animation
        self.running_sprites = []
        self.ducking_sprites = []
        self.current_sprite = 0
        self.animation_timer = 0
        self.animation_speed = 5
        
        self._create_sprites()
    
    def _create_sprites(self):
        """Create simple colored rectangles as sprites (you can replace with actual images)"""
        # Running animation (2 frames)
        for i in range(2):
            surface = pygame.Surface((self.width, self.height))
            surface.fill((83, 83, 83))  # Dark gray dino
            self.running_sprites.append(surface)
        
        # Ducking animation (2 frames)
        for i in range(2):
            surface = pygame.Surface((self.width + 15, self.height // 2))
            surface.fill((83, 83, 83))
            self.ducking_sprites.append(surface)
    
    def jump(self):
        """Make the dino jump"""
        if not self.is_jumping and not self.is_ducking:
            self.velocity_y = self.jump_strength
            self.is_jumping = True
    
    def duck(self, ducking: bool):
        """Make the dino duck"""
        if not self.is_jumping:
            self.is_ducking = ducking
    
    def update(self):
        """Update dino physics and animation"""
        # Handle jumping physics
        if self.is_jumping:
            self.y += self.velocity_y
            self.velocity_y += self.gravity
            
            # Land on ground
            if self.y >= self.ground_y:
                self.y = self.ground_y
                self.velocity_y = 0
                self.is_jumping = False
        
        # Update animation
        self.animation_timer += 1
        if self.animation_timer >= self.animation_speed:
            self.animation_timer = 0
            self.current_sprite = (self.current_sprite + 1) % 2
    
    def draw(self, screen: pygame.Surface):
        """Draw the dino"""
        if self.is_ducking:
            sprite = self.ducking_sprites[self.current_sprite]
            screen.blit(sprite, (self.x, self.y + self.height // 2))
        else:
            sprite = self.running_sprites[self.current_sprite]
            screen.blit(sprite, (self.x, self.y))
    
    def get_rect(self) -> pygame.Rect:
        """Get collision rectangle"""
        if self.is_ducking:
            return pygame.Rect(self.x, self.y + self.height // 2, 
                             self.width + 15, self.height // 2)
        return pygame.Rect(self.x, self.y, self.width, self.height)


class Obstacle:
    def __init__(self, x: int, y: int, obstacle_type: str = "cactus"):
        self.x = x
        self.y = y
        self.type = obstacle_type
        self.speed = 6
        
        if obstacle_type == "cactus":
            self.width = random.choice([17, 34, 51])  # Different cactus sizes
            self.height = 35
        elif obstacle_type == "bird":
            self.width = 46
            self.height = 40
            self.y = y - random.choice([0, 20, 40])  # Birds at different heights
        
        self._create_sprite()
    
    def _create_sprite(self):
        """Create obstacle sprite"""
        self.sprite = pygame.Surface((self.width, self.height))
        if self.type == "cactus":
            self.sprite.fill((0, 128, 0))  # Green cactus
        else:  # bird
            self.sprite.fill((0, 0, 0))    # Black bird
    
    def update(self):
        """Update obstacle position"""
        self.x -= self.speed
    
    def draw(self, screen: pygame.Surface):
        """Draw the obstacle"""
        screen.blit(self.sprite, (self.x, self.y))
    
    def get_rect(self) -> pygame.Rect:
        """Get collision rectangle"""
        return pygame.Rect(self.x, self.y, self.width, self.height)
    
    def is_off_screen(self) -> bool:
        """Check if obstacle is off screen"""
        return self.x + self.width < 0


class Ground:
    def __init__(self, width: int, height: int, ground_y: int):
        self.width = width
        self.height = height
        self.ground_y = ground_y
        self.x1 = 0
        self.x2 = width
        self.speed = 6
        
        # Create ground sprite with simple pattern
        self.sprite = pygame.Surface((width, height))
        self.sprite.fill((211, 211, 211))  # Light gray
        
        # Add some simple texture lines
        for i in range(0, width, 20):
            pygame.draw.line(self.sprite, (169, 169, 169), (i, 0), (i, height))
    
    def update(self):
        """Update ground scrolling"""
        self.x1 -= self.speed
        self.x2 -= self.speed
        
        # Reset positions for infinite scroll
        if self.x1 <= -self.width:
            self.x1 = self.width
        if self.x2 <= -self.width:
            self.x2 = self.width
    
    def draw(self, screen: pygame.Surface):
        """Draw the ground"""
        screen.blit(self.sprite, (self.x1, self.ground_y))
        screen.blit(self.sprite, (self.x2, self.ground_y))


class Cloud:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.width = 46
        self.height = 14
        self.speed = 1
        
        # Create cloud sprite
        self.sprite = pygame.Surface((self.width, self.height))
        self.sprite.fill((200, 200, 200))  # Light gray cloud
    
    def update(self):
        """Update cloud position"""
        self.x -= self.speed
    
    def draw(self, screen: pygame.Surface):
        """Draw the cloud"""
        screen.blit(self.sprite, (self.x, self.y))
    
    def is_off_screen(self) -> bool:
        """Check if cloud is off screen"""
        return self.x + self.width < 0