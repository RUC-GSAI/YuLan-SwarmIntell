# -*- coding: utf-8 -*-
import pygame
from . import constants as C # Import constants from the same package

def get_slider_position(slider_rect, num_steps, index):
    """Calculates the X coordinate for a slider handle at a given index."""
    padding = 15 # Horizontal padding within the slider rect
    track_width = slider_rect.width - 2 * padding
    if num_steps <= 1:
        return slider_rect.centerx # Center if only one step
    # Ensure track_width is positive before division
    if track_width <= 0:
        return slider_rect.left + padding # Default to start if track is too small

    step_width = track_width / (num_steps - 1)
    x = slider_rect.left + padding + index * step_width
    return int(x)

def get_slider_index_from_pos(slider_rect, num_steps, pos_x):
    """Calculates the slider index corresponding to a mouse X coordinate."""
    padding = 15
    track_width = slider_rect.width - 2 * padding
    track_start_x = slider_rect.left + padding

    # Clamp relative_x to the track bounds
    relative_x = max(0, min(pos_x - track_start_x, track_width))

    if num_steps <= 1 or track_width <= 0:
        return 0 # Only one possible index or invalid track

    # Calculate the fractional index and round to the nearest integer index
    fractional_index = (relative_x / track_width) * (num_steps - 1)
    index = round(fractional_index)

    # Ensure the index is within the valid range [0, num_steps - 1]
    return int(max(0, min(index, num_steps - 1)))