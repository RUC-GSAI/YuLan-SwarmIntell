# -*- coding: utf-8 -*-
import pygame

# --- Color Definitions ---
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
RED = (255, 0, 0)
EMPTY_CELL_COLOR = (30, 30, 30)

# Agent Colors (Consider making this configurable or dynamically generated if needed)
PYGAME_AGENT_COLORS = [
    (255, 85, 85), (255, 0, 0), (255, 255, 85), (255, 255, 0),
    (85, 255, 85), (0, 255, 0), (85, 255, 255), (0, 255, 255),
    (85, 85, 255), (0, 0, 255), (255, 85, 255), (255, 0, 255)
]

# Object Colors on the Grid
OBJECT_COLORS = {
    'P': WHITE,          # Example: Pellet?
    'Y': (0, 200, 200),  # Example: Yellow Object
    'W': RED,            # Example: Wall? Warning?
    'B': (255, 255, 0),  # Example: Blue Object? (Looks Yellow here!) - CHECK THIS
    'X': (200, 0, 200),  # Example: Obstacle?
    'A': (0, 150, 0)     # Example: Generic Agent (if not colored)
}

# Text and Line Colors
AGENT_TEXT_COLOR = BLACK
MENTION_LINE_COLOR = (200, 200, 0, 180) # RGBA for potential transparency
MENTION_TEXT_COLOR = WHITE
GRAPH_NODE_COLOR = GRAY
GRAPH_EDGE_COLOR = (100, 100, 255)
SCORE_LINE_COLOR = (0, 200, 0)

# Slider Colors
SLIDER_TRACK_COLOR = (80, 80, 80)
SLIDER_HANDLE_COLOR = (200, 200, 200)
SLIDER_TEXT_COLOR = WHITE
SLIDER_HANDLE_DRAG_COLOR = (255, 255, 100)

# --- Drawing Configuration ---
MESSAGE_SNIPPET_LENGTH = 25 # Max characters for mention line message snippets