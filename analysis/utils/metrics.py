import numpy as np
import math
from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity

# Action vectors constants
MOVE_ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY'] # STAY is often treated specially
ACTUAL_MOVE_ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT'] # For metrics like move attempts
COORDINATION_ACTIONS = ACTUAL_MOVE_ACTIONS + ['STAY']
ACTION_VECTORS = {
    'UP': np.array([0, -1]),
    'DOWN': np.array([0, 1]),
    'LEFT': np.array([-1, 0]),
    'RIGHT': np.array([1, 0]),
    'STAY': np.array([0, 0])
}

def shannon_entropy(data):
    """Calculate Shannon entropy for a list of items"""
    if not data: return 0.0
    counts = Counter(data)
    total_count = len(data)
    probabilities = [count / total_count for count in counts.values() if count > 0]
    if not probabilities: return 0.0
    return -sum(p * math.log2(p) for p in probabilities)

def manhattan_distance(pos1, pos2):
    """Calculate Manhattan distance between two positions"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def calculate_directional_entropy(actions_in_round):
    """Calculate entropy of movement directions"""
    move_actions = [a for a in actions_in_round if a in ACTUAL_MOVE_ACTIONS]
    return shannon_entropy(move_actions)

def calculate_stillness_proportion(actions_in_round):
    """Calculate proportion of STAY actions"""
    if not actions_in_round:
        return 0.0
    stay_count = actions_in_round.count('STAY')
    return stay_count / len(actions_in_round)

def calculate_message_length_metrics(round_messages):
    """Calculate mean and standard deviation of message lengths"""
    if not round_messages:
        return 0.0, 0.0
    message_lengths = [len(msg) for msg in round_messages]
    mean_length = np.mean(message_lengths)
    std_length = np.std(message_lengths) if len(message_lengths) >= 2 else 0.0
    return mean_length, std_length

def calculate_message_content_metrics(round_messages):
    """Calculate question proportion and digit character proportion in messages"""
    if not round_messages:
        return 0.0, 0.0
    
    question_mark_count = 0
    total_chars_in_round = 0
    digit_chars_in_round = 0
    
    for msg in round_messages:
        if "?" in msg:
            question_mark_count += 1
        for char in msg:
            total_chars_in_round += 1
            if char.isdigit():
                digit_chars_in_round += 1
    
    prop_question_sentences = question_mark_count / len(round_messages) if round_messages else 0.0
    prop_digit_chars = digit_chars_in_round / total_chars_in_round if total_chars_in_round > 0 else 0.0
    
    return prop_question_sentences, prop_digit_chars

def calculate_info_homogeneity(round_messages, embedding_model=None, embedding_cache=None):
    """Calculate information homogeneity using embeddings"""
    if not embedding_model or not round_messages or len(round_messages) < 2:
        return 0.0
    
    try:
        unique_messages = list(set(round_messages))
        if len(unique_messages) < 2:
            return 1.0  # Perfect homogeneity if only one unique message
        
        # Use embedding cache if available
        embeddings = []
        if embedding_cache:
            for msg in unique_messages:
                embedding = embedding_cache.get_embedding(msg)
                if embedding is not None:
                    embeddings.append(embedding)
        else:
            embeddings = embedding_model.encode(unique_messages, show_progress_bar=False, convert_to_numpy=True)
        
        if len(embeddings) < 2:
            return 1.0
            
        embeddings = np.array(embeddings)
        cos_sim_matrix = cosine_similarity(embeddings)
        upper_triangle_indices = np.triu_indices_from(cos_sim_matrix, k=1)
        pairwise_sims = cos_sim_matrix[upper_triangle_indices]
        return np.nanmean(pairwise_sims) if len(pairwise_sims) > 0 else 0.0
    except Exception as e:
        print(f"Error calculating info_homogeneity: {e}")
        return 0.0

def calculate_norm_mean_message_embedding(round_messages, embedding_model=None, embedding_cache=None):
    """Calculate the norm of the mean message embedding vector"""
    if not embedding_model or not round_messages:
        return 0.0
    
    try:
        unique_messages = list(set(round_messages))
        if not unique_messages:
            return 0.0
            
        # Use embedding cache if available
        embeddings = []
        if embedding_cache:
            for msg in unique_messages:
                embedding = embedding_cache.get_embedding(msg)
                if embedding is not None:
                    embeddings.append(embedding)
        else:
            embeddings = embedding_model.encode(unique_messages, show_progress_bar=False, convert_to_numpy=True)
        
        # Check if embeddings is empty - handles both lists and NumPy arrays
        if isinstance(embeddings, list) and not embeddings:
            return 0.0
        elif isinstance(embeddings, np.ndarray) and embeddings.size == 0:
            return 0.0
            
        mean_embedding_vector = np.mean(embeddings, axis=0)
        return np.linalg.norm(mean_embedding_vector)
    except Exception as e:
        print(f"Error calculating norm_mean_message_embedding: {e}")
        return 0.0

def calculate_avg_moving_distance(positions, prev_positions):
    """Calculate average moving distance for agents"""
    cumulative_distances = {}
    for agent_id, curr_pos in positions.items():
        if agent_id in prev_positions:
            prev_pos = prev_positions[agent_id]
            distance_moved = manhattan_distance(prev_pos, curr_pos)
            cumulative_distances[agent_id] = distance_moved
    
    if cumulative_distances:
        return sum(cumulative_distances.values()) / len(cumulative_distances)
    return 0.0

def calculate_exploration_rate(game_steps, current_step_index):
    """Calculate number of unique cells explored so far"""
    explored_cells = set()
    for i in range(current_step_index + 1):
        if i < len(game_steps) and 'agents' in game_steps[i]:
            for agent in game_steps[i].get('agents', []):
                if isinstance(agent, dict) and 'x' in agent and 'y' in agent:
                    try:
                        explored_cells.add((int(agent['x']), int(agent['y'])))
                    except (ValueError, TypeError):
                        continue
    return len(explored_cells)

def calculate_coordination_metrics(actions_in_round):
    """Calculate dominant action proportion and polarization index"""
    dominant_action_prop = 0.0
    polarization_index = 0.0
    
    actions_for_coordination = [a for a in actions_in_round if a in COORDINATION_ACTIONS]
    if actions_for_coordination:
        # Dominant action proportion
        action_counts = Counter(actions_for_coordination)
        max_freq = action_counts.most_common(1)[0][1] if action_counts else 0
        dominant_action_prop = max_freq / len(actions_for_coordination)
        
        # Polarization index
        sum_vector = np.zeros(2)
        for action in actions_for_coordination:
            sum_vector += ACTION_VECTORS.get(action, np.array([0, 0]))
        avg_vector = sum_vector / len(actions_for_coordination)
        polarization_index = np.linalg.norm(avg_vector)
    
    return dominant_action_prop, polarization_index

def calculate_local_structure_preservation(positions, prev_positions):
    """Calculate local structure preservation count"""
    count = 0
    agent_ids = list(positions.keys())
    
    if len(agent_ids) >= 2 and prev_positions:
        for i in range(len(agent_ids)):
            for j in range(i+1, len(agent_ids)):
                agent_i = agent_ids[i]
                agent_j = agent_ids[j]
                if agent_i in prev_positions and agent_j in prev_positions and agent_i in positions and agent_j in positions:
                    prev_dist = manhattan_distance(prev_positions[agent_i], prev_positions[agent_j])
                    curr_dist = manhattan_distance(positions[agent_i], positions[agent_j])
                    # Check if agents maintained adjacency
                    if prev_dist == 1 and curr_dist == 1:
                        count += 1
    
    return count

def calculate_agent_push_events(positions, prev_positions, prev_actions):
    """Calculate number of agent push events"""
    count = 0
    
    if not prev_positions or not prev_actions:
        return count
    
    for agent_A_id, pos_A_prev in prev_positions.items():
        if agent_A_id not in prev_actions or agent_A_id not in positions:
            continue
            
        action_A = prev_actions.get(agent_A_id)
        if action_A not in ACTUAL_MOVE_ACTIONS:
            continue
            
        for agent_B_id, pos_B_prev in prev_positions.items():
            if agent_A_id == agent_B_id or agent_B_id not in positions:
                continue
            
            # Check if agents were adjacent
            if manhattan_distance(pos_A_prev, pos_B_prev) != 1:
                continue
                
            # Calculate intended position after A's move
            intended_A_pos = tuple(np.array(pos_A_prev) + ACTION_VECTORS[action_A])
            
            # Check if A moved into B's previous position
            if intended_A_pos == pos_B_prev and positions[agent_A_id] == pos_B_prev:
                # Calculate expected position for B if pushed
                expected_B_pos = tuple(np.array(pos_B_prev) + ACTION_VECTORS[action_A])
                
                # Check if B actually moved in the expected direction
                if positions[agent_B_id] == expected_B_pos:
                    count += 1
    
    return count

def calculate_all_metrics(
    step_data, messages_by_round, game_steps, current_step_index, 
    agent_log=None, embedding_model=None, embedding_cache=None
):
    """Calculate all metrics for a specific frame"""
    metrics = {}
    try:
        # Basic info retrieval 
        round_num = int(step_data['round'])
        grid_data = step_data.get('grid', [])
        agents_list = step_data.get('agents', [])
        
        # Position analysis
        positions = {}
        for agent in agents_list:
            if isinstance(agent, dict) and all(k in agent for k in ('x','y','id')):
                try:
                    agent_id = str(agent['id'])
                    positions[agent_id] = (int(agent['x']), int(agent['y']))
                except (ValueError, TypeError): 
                    continue
        
        # Previous positions for comparison
        prev_positions = {}
        if current_step_index > 0 and game_steps:
            prev_step = game_steps[current_step_index-1]
            if isinstance(prev_step, dict) and 'agents' in prev_step:
                for agent in prev_step.get('agents', []):
                    if isinstance(agent, dict) and all(k in agent for k in ('x','y','id')):
                        try:
                            agent_id = str(agent['id'])
                            prev_positions[agent_id] = (int(agent['x']), int(agent['y']))
                        except (ValueError, TypeError):
                            continue

        # Action analysis
        actions_in_round = []
        if agent_log:
            for entry in agent_log:
                if isinstance(entry, dict) and 'round' in entry and entry['round'] == round_num and 'action' in entry:
                    action = entry.get('action')
                    if action:
                        actions_in_round.append(action)
        
        # Message analysis
        round_messages = []
        for agent_id, msg in messages_by_round.get(round_num, {}).items():
            if msg:
                round_messages.append(str(msg))
        
        # Calculate all metrics
        
        # 1. Directional entropy
        metrics['directional_entropy'] = calculate_directional_entropy(actions_in_round)
        
        # 2. Stillness proportion
        metrics['stillness_proportion'] = calculate_stillness_proportion(actions_in_round)
        
        # 3. Message length metrics
        mean_length, std_length = calculate_message_length_metrics(round_messages)
        metrics['mean_message_length'] = mean_length
        metrics['std_message_length'] = std_length
        
        # 4. Message content metrics
        prop_question, prop_digit = calculate_message_content_metrics(round_messages)
        metrics['prop_question_sentences'] = prop_question
        metrics['prop_digit_chars'] = prop_digit
        
        # 5. Embedding-based metrics
        metrics['info_homogeneity'] = calculate_info_homogeneity(round_messages, embedding_model, embedding_cache)
        metrics['norm_mean_message_embedding'] = calculate_norm_mean_message_embedding(round_messages, embedding_model, embedding_cache)
        
        # 6. Movement distance
        metrics['avg_moving_distance'] = calculate_avg_moving_distance(positions, prev_positions)
            
        # 7. Exploration rate
        metrics['exploration_rate'] = calculate_exploration_rate(game_steps, current_step_index)
        
        # 8. Coordination metrics
        dominant_action, polarization = calculate_coordination_metrics(actions_in_round)
        metrics['dominant_action_prop'] = dominant_action
        metrics['polarization_index'] = polarization
        
        # 9. Local structure preservation
        metrics['local_structure_preservation_count'] = calculate_local_structure_preservation(positions, prev_positions)
        
        # 10. Agent push events
        prev_actions = {}
        if current_step_index > 0 and agent_log:
            prev_round = int(game_steps[current_step_index-1].get('round', 0))
            for entry in agent_log:
                if isinstance(entry, dict) and 'round' in entry and entry['round'] == prev_round and 'action' in entry and 'agent_id' in entry:
                    prev_actions[entry['agent_id']] = entry['action']
                    
        metrics['agent_push_events'] = calculate_agent_push_events(positions, prev_positions, prev_actions)
        
        return metrics
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        import traceback
        traceback.print_exc()
        return {}
