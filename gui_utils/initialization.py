# -*- coding: utf-8 -*-
import pygame
import sys

# Store fonts globally within this module after initialization
fonts = {}

def setup_pygame(width, height, caption):
    """Initializes Pygame, sets up the screen, loads fonts, and creates a clock."""
    global fonts
    try:
        pygame.init()
    except pygame.error as e:
        print(f"Fatal Error: Pygame initialization failed: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        screen = pygame.display.set_mode((width, height))
    except pygame.error as e:
        print(f"Fatal Error: Failed to set display mode ({width}x{height}): {e}", file=sys.stderr)
        pygame.quit()
        sys.exit(1)

    pygame.display.set_caption(caption)

    # Load fonts
    try:
        fonts['small'] = pygame.font.SysFont(None, 20)
        fonts['medium'] = pygame.font.SysFont(None, 24)
        fonts['large'] = pygame.font.SysFont(None, 30)
        fonts['grid_cell'] = pygame.font.SysFont(None, 28) # Adjust size as needed
    except pygame.error as e:
         print(f"Fatal Error: Failed to load system font: {e}", file=sys.stderr)
         pygame.quit()
         sys.exit(1)
    except FileNotFoundError:
         print(f"Fatal Error: System font not found. Check Pygame font support.", file=sys.stderr)
         pygame.quit()
         sys.exit(1)

    clock = pygame.time.Clock()

    print("Pygame initialized successfully.")
    return screen, clock, fonts # Return fonts dictionary