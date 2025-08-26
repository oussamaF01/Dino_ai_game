from gymnasium.envs.registration import register
from dino_env import DinosaurGameEnv, SimplifiedDinosaurGameEnv

# Register the pixel-based environment
register(
    id='DinosaurGame-v0',
    entry_point='dino_env:DinosaurGameEnv',
    max_episode_steps=10000,
    kwargs={
        'frame_stack': 4,
        'max_steps': 10000
    }
)

# Register the feature-based environment
register(
    id='DinosaurGameSimple-v0',
    entry_point='dino_env:SimplifiedDinosaurGameEnv',
    max_episode_steps=10000,
    kwargs={
        'max_steps': 10000
    }
)

# Register variants with different configurations
register(
    id='DinosaurGame-v1',
    entry_point='dino_env:DinosaurGameEnv',
    max_episode_steps=5000,
    kwargs={
        'frame_stack': 2,
        'max_steps': 5000
    }
)

register(
    id='DinosaurGameSimple-v1',
    entry_point='dino_env:SimplifiedDinosaurGameEnv',
    max_episode_steps=5000,
    kwargs={
        'max_steps': 5000
    }
)

print("Registered environments:")
print("- DinosaurGame-v0: Full pixel-based environment with 4 frame stack")
print("- DinosaurGameSimple-v0: Simplified feature-based environment")
print("- DinosaurGame-v1: Shorter episodes, 2 frame stack")
print("- DinosaurGameSimple-v1: Simplified with shorter episodes")