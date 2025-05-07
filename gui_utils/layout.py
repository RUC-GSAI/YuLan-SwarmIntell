# -*- coding: utf-8 -*-
import pygame

def calculate_grid_layout(screen_width, screen_height):
    """Calculates the layout rectangles for different UI elements."""
    layout = {}
    padding = 15
    main_width = screen_width * 0.6
    side_width = screen_width * 0.35
    graph_height = screen_height * 0.35
    score_height = screen_height * 0.25
    bottom_bar_height = 60 # Space for slider + status text

    # Ensure main grid height doesn't become negative
    main_grid_height = max(10, screen_height - 2 * padding - bottom_bar_height)

    layout['global_grid'] = pygame.Rect(padding, padding, main_width - padding, main_grid_height)

    right_col_x = main_width + padding
    current_y = padding
    layout['mention_graph'] = pygame.Rect(right_col_x, current_y, side_width - padding, graph_height)
    current_y += graph_height + padding
    layout['score_curve'] = pygame.Rect(right_col_x, current_y, side_width - padding, score_height)
    current_y += score_height + padding

    # Calculate remaining height for messages, ensuring it's not negative
    remaining_height = screen_height - current_y - padding - bottom_bar_height
    layout['messages'] = pygame.Rect(right_col_x, current_y, side_width - padding, max(50, remaining_height)) # Min height 50

    # Slider layout
    slider_width = 300
    slider_height = 40 # Includes space for labels below the track
    slider_y = screen_height - bottom_bar_height + 5 # Position it within the bottom bar area
    slider_x = (screen_width - slider_width) // 2 # Center horizontally
    layout['speed_slider'] = pygame.Rect(slider_x, slider_y, slider_width, slider_height)

    return layout