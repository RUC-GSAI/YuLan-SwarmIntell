# -*- coding: utf-8 -*-
import pygame
import re
import math
from . import constants as C # Use alias C for brevity

# --- Drawing Functions ---

def draw_grid(surface, grid_data, agent_coords_map, agent_color_map, target_rect, cell_font):
    """Draws the main simulation grid."""
    cell_rects = {} # Store center coordinates of agent cells: {agent_id: (cx, cy)}
    if not grid_data or not isinstance(grid_data, list) or not grid_data[0]:
        pygame.draw.rect(surface, C.GRAY, target_rect, 1)
        try:
            err_text = cell_font.render("Invalid Grid Data", True, C.RED)
            surface.blit(err_text, (target_rect.x + 5, target_rect.y + 5))
        except Exception as e:
             print(f"Warn: Failed to render grid error text: {e}")
        return cell_rects

    rows, cols = len(grid_data), len(grid_data[0])
    if rows == 0 or cols == 0:
        return cell_rects # Empty grid

    # Calculate cell size based on available space
    cell_width = target_rect.width / cols
    cell_height = target_rect.height / rows
    cell_size = min(cell_width, cell_height) # Use the smaller dimension for square cells

    # Calculate drawing area dimensions and centering offset
    grid_draw_width = cell_size * cols
    grid_draw_height = cell_size * rows
    start_x = target_rect.x + (target_rect.width - grid_draw_width) / 2
    start_y = target_rect.y + (target_rect.height - grid_draw_height) / 2

    for r in range(rows):
        for c in range(cols):
            cell_rect = pygame.Rect(start_x + c * cell_size, start_y + r * cell_size, cell_size, cell_size)
            char = str(grid_data[r][c]) if grid_data[r][c] is not None and str(grid_data[r][c]).strip() else '.'

            bg_color = C.BLACK
            text_content = None
            text_color = C.WHITE

            # Check if an agent is at this coordinate
            agent_id_here = agent_coords_map.get((r, c))
            agent_specific_color = agent_color_map.get(agent_id_here) if agent_id_here else None

            if agent_id_here and agent_specific_color:
                # Agent rendering logic
                agent_num_match = re.search(r'\d+$', agent_id_here)
                # Display last 2 digits of agent number, or '?'
                text_content = agent_num_match.group()[-2:] if agent_num_match else '?'
                # Differentiate background if agent is on an 'empty' cell vs an object
                if char == 'a': # Assuming 'a' might mean agent on empty space
                    bg_color = C.EMPTY_CELL_COLOR # Show empty cell bg
                    text_color = agent_specific_color # Color the number
                else:
                    bg_color = agent_specific_color # Agent color is background
                    text_color = C.AGENT_TEXT_COLOR # Use standard text color for number
                # Store the center of the cell for this agent (for mention lines)
                cell_rects[agent_id_here] = cell_rect.center

            elif char in C.OBJECT_COLORS:
                # Object rendering logic
                bg_color = C.OBJECT_COLORS[char]
                text_content = char # Display the object character
                # Choose contrasting text color (simple heuristic)
                text_color = C.BLACK if char in ('B', 'Y', 'P') else C.WHITE # Adjust based on OBJECT_COLORS
            elif char == '.':
                 bg_color = C.EMPTY_CELL_COLOR # Empty cell background
            else:
                 # Unknown character rendering
                 bg_color = (50, 0, 50) # Default color for unknowns
                 text_content = char

            # Draw cell background and border
            pygame.draw.rect(surface, bg_color, cell_rect)
            pygame.draw.rect(surface, C.GRAY, cell_rect, 1) # Border

            # Draw text content if any
            if text_content:
                try:
                    text_surf = cell_font.render(text_content, True, text_color)
                    text_rect = text_surf.get_rect(center=cell_rect.center)
                    surface.blit(text_surf, text_rect)
                except Exception as e:
                    print(f"Warn: Grid text render failed for '{text_content}': {e}")

    return cell_rects # Return the map of agent IDs to cell centers

def draw_arrow(surface, color, start_pos, end_pos, arrow_size=12, line_width=2):
    """Draws a line with an arrowhead."""
    try:
        # Ensure integer coordinates for drawing
        start_pos_i = (int(start_pos[0]), int(start_pos[1]))
        end_pos_i = (int(end_pos[0]), int(end_pos[1]))

        dx = end_pos_i[0] - start_pos_i[0]
        dy = end_pos_i[1] - start_pos_i[1]
        length = math.hypot(dx, dy)

        if length < 1: return # Avoid division by zero or tiny arrows

        # Normalize direction vector
        norm_dx = dx / length
        norm_dy = dy / length

        # Calculate end point of the line segment (shorten to make space for arrowhead)
        line_end_x = end_pos_i[0] - norm_dx * (arrow_size * 0.7) # Adjust factor as needed
        line_end_y = end_pos_i[1] - norm_dy * (arrow_size * 0.7)

        # Draw the line part
        # Handle RGBA color for transparency if provided
        line_color_rgba = color[:3] + (color[3],) if len(color) > 3 else color[:3] + (255,)
        pygame.draw.line(surface, line_color_rgba, start_pos_i, (int(line_end_x), int(line_end_y)), line_width)

        # Calculate arrowhead points
        angle = math.atan2(dy, dx) # Angle of the line
        angle1 = angle + math.pi + math.pi / 6 # Angle for one side of the arrowhead
        angle2 = angle + math.pi - math.pi / 6 # Angle for the other side

        p1 = (end_pos_i[0] + arrow_size * math.cos(angle1), end_pos_i[1] + arrow_size * math.sin(angle1))
        p2 = (end_pos_i[0] + arrow_size * math.cos(angle2), end_pos_i[1] + arrow_size * math.sin(angle2))

        arrow_points = [end_pos_i, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1]))]

        # Draw the arrowhead polygon
        pygame.draw.polygon(surface, color, arrow_points)

    except Exception as e:
        print(f"Error drawing arrow {start_pos}->{end_pos}: {e}")


def draw_mention_lines(surface, mentions_list, messages_this_round, agent_cell_centers, msg_font):
    """Draws arrows and message snippets for agent mentions on the grid."""
    if not mentions_list or not agent_cell_centers:
        return

    arrow_color = C.MENTION_LINE_COLOR
    text_color = C.MENTION_TEXT_COLOR
    snippet_len = C.MESSAGE_SNIPPET_LENGTH

    for sender_id, mentioned_id in mentions_list:
        sender_pos = agent_cell_centers.get(sender_id)
        mentioned_pos = agent_cell_centers.get(mentioned_id)

        # Only draw if both agents are visible and distinct
        if sender_pos and mentioned_pos and sender_pos != mentioned_pos:
            try:
                # Draw the arrow first
                draw_arrow(surface, arrow_color, sender_pos, mentioned_pos)

                # Prepare message snippet
                message = messages_this_round.get(sender_id, "")
                if message and isinstance(message, str):
                    # Clean and shorten message
                    display_msg = (message[:snippet_len] + '...') if len(message) > snippet_len else message
                    display_msg = display_msg.replace('\n', ' ').strip() # Remove newlines

                    if display_msg:
                        # Render the text snippet
                        try:
                            msg_surf = msg_font.render(display_msg, True, text_color)
                            # Position near the middle of the arrow, slightly offset upwards
                            mid_x = (sender_pos[0] + mentioned_pos[0]) / 2
                            mid_y = (sender_pos[1] + mentioned_pos[1]) / 2
                            msg_rect = msg_surf.get_rect(center=(int(mid_x), int(mid_y - 10)))

                            # Draw a semi-transparent background for readability
                            bg_rect = msg_rect.inflate(6, 4)
                            bg_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA) # Enable alpha
                            bg_surf.fill((0, 0, 0, 160)) # Black background, semi-transparent
                            surface.blit(bg_surf, bg_rect.topleft)

                            # Draw the text on top of the background
                            surface.blit(msg_surf, msg_rect)
                        except Exception as render_err:
                            print(f"Warn: Failed to render mention text '{display_msg}': {render_err}")
            except Exception as e:
                print(f"Error processing mention line {sender_id}->{mentioned_id}: {e}")

def draw_mention_graph(surface, agent_ids, mentions_list, target_rect, font):
    """Draws a graph showing mention relationships in a dedicated panel."""
    try:
        # Background and border
        pygame.draw.rect(surface, (20, 20, 20), target_rect) # Dark background
        pygame.draw.rect(surface, C.GRAY, target_rect, 1) # Border

        # Title
        title_surf = font.render("Mention Graph", True, C.WHITE)
        surface.blit(title_surf, (target_rect.x + 5, target_rect.y + 5))

        if not agent_ids: return # No agents to draw

        nodes = {} # agent_id -> (x, y) position
        num_agents = len(agent_ids)

        # Define graph drawing area (inset from target_rect)
        graph_area = target_rect.inflate(-20, -30) # Padding
        graph_area.top += title_surf.get_height() # Move below title

        # Simple circular layout
        radius = min(graph_area.width, graph_area.height) * 0.4 # Radius of the circle
        center_x, center_y = graph_area.centerx, graph_area.centery
        angle_step = 2 * math.pi / num_agents if num_agents > 0 else 0

        # Sort agents by number for consistent layout
        def sort_key(aid):
            match = re.search(r'\d+$', aid)
            return int(match.group()) if match else 0
        sorted_ids = sorted(list(agent_ids), key=sort_key)

        # Calculate node positions
        for i, agent_id in enumerate(sorted_ids):
            angle = i * angle_step - math.pi / 2 # Start from top (-pi/2)
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            nodes[agent_id] = (int(x), int(y))

        # Draw edges (mention lines)
        if mentions_list:
            for sender_id, mentioned_id in mentions_list:
                p1 = nodes.get(sender_id)
                p2 = nodes.get(mentioned_id)
                if p1 and p2:
                    pygame.draw.line(surface, C.GRAPH_EDGE_COLOR, p1, p2, 1)

        # Draw nodes and labels
        node_radius = 6
        for agent_id, pos in nodes.items():
            pygame.draw.circle(surface, C.GRAPH_NODE_COLOR, pos, node_radius)
            # Label with agent number
            match = re.search(r'\d+$', agent_id)
            label_text = match.group() if match else '?'
            label_surf = font.render(label_text, True, C.WHITE)
            label_rect = label_surf.get_rect(center=(pos[0], pos[1] - node_radius - 8)) # Position above node
            surface.blit(label_surf, label_rect)

    except Exception as e:
        print(f"Error drawing mention graph: {e}")


def draw_score_curve(surface, score_history, target_rect, font):
    """Draws the score progression over rounds."""
    try:
        # Background and border
        pygame.draw.rect(surface, (20, 20, 20), target_rect)
        pygame.draw.rect(surface, C.GRAY, target_rect, 1)

        # Title
        title_surf = font.render("Score Over Rounds", True, C.WHITE)
        surface.blit(title_surf, (target_rect.x + 5, target_rect.y + 5))

        # Filter out non-numeric scores if any slipped through
        valid_scores = [(r, s) for r, s in score_history if isinstance(s, (int, float))]

        if not valid_scores: return # Nothing to plot

        # Define plotting area
        padding = 25
        draw_area = target_rect.inflate(-padding*2, -padding*2)
        draw_area.top += title_surf.get_height() # Adjust for title

        # Get data ranges
        rounds = [r for r, s in valid_scores]
        scores = [s for r, s in valid_scores]

        max_r_hist = max(rounds) if rounds else 0
        # Use the maximum round seen so far for scaling the X-axis consistently
        # If only one point, use at least 1 for the range, otherwise use max round.
        max_r = max(1, max_r_hist)

        min_s = min(scores) if scores else 0
        max_s = max(scores) if scores else 1
        # Avoid division by zero if all scores are the same
        if max_s == min_s:
            max_s = min_s + 1 # Create a small range

        # Function to map (round, score) to screen coordinates
        def map_pt(r, s):
            x = draw_area.left + (r / max_r * draw_area.width if max_r > 0 else 0)
            # Invert Y axis for screen coordinates (0 is top)
            y = draw_area.bottom - ((s - min_s) / (max_s - min_s) * draw_area.height if (max_s - min_s) != 0 else 0)
            return int(x), int(y)

        # Draw axes
        pygame.draw.line(surface, C.GRAY, draw_area.bottomleft, draw_area.bottomright, 1) # X axis (Rounds)
        pygame.draw.line(surface, C.GRAY, draw_area.bottomleft, draw_area.topleft, 1) # Y axis (Score)

        # Plot the points
        points = [map_pt(r, s) for r, s in valid_scores]
        if len(points) == 1:
             # Draw a horizontal line if only one point exists
             pygame.draw.line(surface, C.SCORE_LINE_COLOR, (draw_area.left, points[0][1]), (draw_area.right, points[0][1]), 2)
        elif len(points) >= 2:
            pygame.draw.lines(surface, C.SCORE_LINE_COLOR, False, points, 2) # False = not closed polygon

        # Draw labels for axes ranges
        # Y-axis labels (Min/Max Score)
        lbl_max_s = font.render(f"{max_s:.1f}", True, C.WHITE)
        surface.blit(lbl_max_s, (draw_area.left + 5, draw_area.top))
        lbl_min_s = font.render(f"{min_s:.1f}", True, C.WHITE)
        surface.blit(lbl_min_s, (draw_area.left + 5, draw_area.bottom - font.get_height()))

        # X-axis label (Max Round shown)
        current_max_r = max(rounds) if rounds else 0 # The actual latest round in the history
        lbl_max_r = font.render(f"R{current_max_r}", True, C.WHITE)
        r_rect = lbl_max_r.get_rect(bottomright=(draw_area.right - 5, draw_area.bottom + 15)) # Position below X axis end
        surface.blit(lbl_max_r, r_rect)

    except Exception as e:
        print(f"Error drawing score curve: {e}")


def draw_messages(surface, messages_dict, agent_color_map, target_rect, font, max_lines=20):
    """Draws the latest agent messages in a scrolling panel."""
    try:
        # Background and border
        pygame.draw.rect(surface, (10, 10, 10), target_rect) # Very dark background
        pygame.draw.rect(surface, C.GRAY, target_rect, 1) # Border

        # Title
        title_surf = font.render("Agent Messages", True, C.WHITE)
        surface.blit(title_surf, (target_rect.x + 5, target_rect.y + 5))

        if not messages_dict: return # No messages to display

        line_height = font.get_linesize()
        # Content area inset from target_rect
        content_rect = target_rect.inflate(-15, -15) # Smaller padding for text area
        content_rect.top += title_surf.get_height() + 5 # Position below title
        content_rect.height = max(10, content_rect.height) # Ensure non-negative height

        start_x, max_width = content_rect.left, content_rect.width
        lines_to_render = [] # List of {'id': surf, 'msg': surf} dicts

        # Sort agents by number for consistent message order
        def sort_key(aid):
            match = re.search(r'\d+$', aid)
            return int(match.group()) if match else 0
        sorted_agent_ids = sorted(messages_dict.keys(), key=sort_key)

        # Process messages for word wrapping
        for agent_id in sorted_agent_ids:
            msg = messages_dict.get(agent_id, "")
            cleaned = ' '.join(str(msg).split()) # Normalize whitespace
            if not cleaned: continue # Skip empty messages

            color = agent_color_map.get(agent_id, C.WHITE) # Agent's color or default
            id_prefix = f"{agent_id}: "
            id_surf = font.render(id_prefix, True, color)
            id_width = id_surf.get_width()
            available_width = max_width - id_width # Width available for message text

            if available_width <= 0: # Handle cases where ID is too long
                lines_to_render.append({'id': id_surf, 'msg': None}) # Just show the ID
                continue

            words = cleaned.split(' ')
            current_line = ""
            line_count = 0 # Track lines per agent for indenting subsequent lines

            for word in words:
                test_line = current_line + word + " "
                test_surf = font.render(test_line, True, C.WHITE) # Use white for width test

                if test_surf.get_width() <= available_width:
                    current_line = test_line # Word fits, add to current line
                else:
                    # Word doesn't fit, finalize previous line (if any)
                    if current_line.strip():
                        lines_to_render.append({
                            'id': id_surf if line_count == 0 else None, # Only show ID on first line
                            'msg': font.render(current_line.strip(), True, C.WHITE)
                        })
                        line_count += 1
                    current_line = "" # Reset line

                    # Handle the word that didn't fit:
                    # Does the single word itself exceed the available width?
                    current_word_surf = font.render(word + " ", True, C.WHITE)
                    if current_word_surf.get_width() > available_width:
                        # Word needs to be broken: Simple character-based break
                        chars_in_word = len(word)
                        est_chars_fit = int(chars_in_word * (available_width / current_word_surf.get_width())) if current_word_surf.get_width() > 0 else 0
                        cutoff = max(1, est_chars_fit -1) # Break near the estimate, add hyphen

                        lines_to_render.append({
                           'id': id_surf if line_count == 0 else None,
                            'msg': font.render(word[:cutoff] + '-', True, C.WHITE)
                        })
                        line_count += 1
                        current_line = word[cutoff:] + " " # Start next line with remainder
                    else:
                        # Word fits on a new line by itself
                         current_line = word + " "

            # Add the last line if it has content
            if current_line.strip():
                lines_to_render.append({
                    'id': id_surf if line_count == 0 else None,
                    'msg': font.render(current_line.strip(), True, C.WHITE)
                })

        # Draw the processed lines (most recent first, bottom-up)
        lines_drawn = 0
        original_clip = surface.get_clip() # Save current clipping region
        surface.set_clip(content_rect) # Clip drawing to the content area

        # Calculate indent based on a reasonably long agent ID
        indent_surf = font.render("Agent_XXX: ", True, C.WHITE) # Use dummy for width
        indent_width = indent_surf.get_width()

        for line_info in reversed(lines_to_render):
            if lines_drawn >= max_lines: break # Limit visible lines

            msg_surf = line_info['msg']
            id_surf = line_info['id']

            # Calculate Y position for the line
            line_y = content_rect.bottom - (lines_drawn + 1) * line_height
            if line_y < content_rect.top: break # Stop if drawing above the visible area

            if id_surf:
                # First line for this agent: draw ID and message
                surface.blit(id_surf, (start_x, line_y))
                if msg_surf:
                    surface.blit(msg_surf, (start_x + id_width, line_y))
            elif msg_surf:
                # Subsequent line for this agent: draw message indented
                 surface.blit(msg_surf, (start_x + indent_width, line_y))

            lines_drawn += 1

        surface.set_clip(original_clip) # Restore original clipping region

    except Exception as e:
        print(f"Error drawing messages panel: {e}")


def draw_speed_slider(surface, slider_rect, font, current_index, speed_multipliers, is_dragging):
    """Draws the speed control slider."""
    try:
        # Slider Track
        track_height = 6
        track_y = slider_rect.centery - track_height // 2
        padding = 15 # Horizontal padding inside slider_rect for the track
        track_rect = pygame.Rect(slider_rect.left + padding, track_y, slider_rect.width - 2 * padding, track_height)
        pygame.draw.rect(surface, C.SLIDER_TRACK_COLOR, track_rect, border_radius=3)

        num_steps = len(speed_multipliers)
        if num_steps == 0: return None # Cannot draw slider without options

        # Ticks and Labels
        label_y = track_rect.bottom + 5 # Position labels below the track
        for i, speed in enumerate(speed_multipliers):
            # Use the slider helper function to find tick position
            from .slider import get_slider_position # Import here to avoid circular dependency if moved
            tick_x = get_slider_position(slider_rect, num_steps, i)

            # Draw tick marks
            pygame.draw.line(surface, C.GRAY, (tick_x, track_rect.top - 3), (tick_x, track_rect.bottom + 3), 1)

            # Draw labels for key points (first, last, 1.0x)
            if i == 0 or i == num_steps - 1 or speed == 1.0:
                label_text = f"{speed}x"
                label_surf = font.render(label_text, True, C.SLIDER_TEXT_COLOR)
                # Center label text under the tick mark
                label_rect = label_surf.get_rect(center=(tick_x, label_y + font.get_height() // 2))
                surface.blit(label_surf, label_rect)

        # Slider Handle (Thumb)
        handle_width = 10
        handle_height = 20
        handle_color = C.SLIDER_HANDLE_DRAG_COLOR if is_dragging else C.SLIDER_HANDLE_COLOR

        # Get current handle position
        from .slider import get_slider_position # Re-import for clarity or ensure it's imported above
        handle_x = get_slider_position(slider_rect, num_steps, current_index)
        handle_rect = pygame.Rect(0, 0, handle_width, handle_height)
        handle_rect.center = (handle_x, track_rect.centery) # Center handle vertically on track

        pygame.draw.rect(surface, handle_color, handle_rect, border_radius=3)
        pygame.draw.rect(surface, C.BLACK, handle_rect, 1, border_radius=3) # Outline

        return handle_rect # Return the handle's rect for collision detection

    except Exception as e:
        print(f"Error drawing speed slider: {e}")
        return None # Return None if drawing failed