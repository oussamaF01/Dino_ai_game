#!/usr/bin/env python3
"""
Chrome Dino Game - Main Runner
Run this to test the game manually before implementing AI.
"""

import sys
import os

# Add src to path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from game import DinoGame

def main():
    """Run the Chrome Dino game"""
    print("Starting Chrome Dino Game...")
    print("Controls:")
    print("- SPACE: Jump")
    print("- DOWN ARROW: Duck")
    print("- SPACE when game over: Restart")
    print("\nPress Ctrl+C to quit\n")
    
    try:
        game = DinoGame(width=800, height=400)
        game.run()
    except KeyboardInterrupt:
        print("\nGame interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error running game: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()