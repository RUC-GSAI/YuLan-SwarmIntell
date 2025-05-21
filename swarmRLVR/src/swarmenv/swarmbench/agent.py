from src.swarmenv.framework.agent import Agent

import numpy as np


def view2str(view_full, offset):
    i1 = offset[0] - view_full.shape[0] // 2
    j1 = offset[1] - view_full.shape[1] // 2
    view_full[view_full.shape[0] // 2, view_full.shape[1] // 2] = 'Y'
    view_str = "   " + " ".join(f'{i+j1:>3}' for i in range(view_full.shape[1])) + "\n"
    for i in range(view_full.shape[0]):
        view_str += f"{i+i1:>3}  "
        for j in range(view_full.shape[1]):
            view_str += f"{view_full[i, j]:<4}"
        view_str += "\n"
    return view_str


class SwarmAgent(Agent):
    def __init__(self, name, brain, sys_prompt, memory=5):
        super().__init__(name, brain, sys_prompt)
        self.history = []
        self.mesh = None
        self.views = []
        self.memory = memory
        self.escaped = False
        self.valid_actions = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]
        self.prompt = ''

    async def decision(self, obs):
        if self.escaped:
            return {}
        return await super().decision(obs)

    def gen_prompt(self, obs, msgs):
        if obs is None:
            return ""

        task = obs.get('task', None)
        round_num = obs.get('round', 0)
        view = obs.get('view', None)
        name = obs.get('name', self.name)
        pos = obs.get('position', None)
        x, y = pos['x'], pos['y']
        self.views.append(view2str(view, (y, x)))

        view_str = '\n'.join(f'{step_name}:\n{s}' for step_name, s in
                             zip(['Current Step'] + [f'{i} Steps Before' for i in range(1, self.memory)], self.views[-self.memory:][::-1]))

        history_str = ""
        for entry in self.history[-self.memory:]:
            history_str += f"Round {entry['round']}: Action: {entry['action']}"
            if 'message' in entry:
                history_str += f", Message: \"{entry['message']}\""
            history_str += "\n"

        messages_str = ""
        for agent_name, msg in msgs.items():
            messages_str += f"Message: \"{msg}\"\n"

        task_desc = task.desc()

        task_obs = task.task_obs(self)
        task_obs_str = '\n'.join(f'{k}: {v}' for k, v in task_obs.items())

        self.prompt = f"""You are Agent {name}, operating in a multi-agent environment. Your goal is to complete the task through exploration and collaboration.

Task description:
{task_desc}

Round: {round_num}

Your recent {self.memory}-step vision (not the entire map):
{view_str}

Your current observation:
{task_obs_str}

Message you received:
{messages_str}

Your action history:
{history_str}

Symbol legend:
- Number: An agent whose id is this number (do not mistake column no. and line no. as agent id).
- Y: Yourself. Others see you as your id instead of "Y".
- W: Wall.
- B: Pushable obstacle (requires at least 5 agents pushing in the same direction).
- .: Empty space (you can move to this area).
- *: Area outside the map.
And other symbols given in task description (if any).

Available actions:
1. UP: Move up
2. DOWN: Move down
3. LEFT: Move left
4. RIGHT: Move right
5. STAY: Stay in place
6. MSG: Send a message
And other actions given in task description (if any).

Physics rules:
1. Your own weight is 1, and you can exert a force of up to 2.
2. An object (including yourself) can only be pushed if the total force in one direction is greater than or equal to its weight.
3. Static objects like W (walls) cannot be pushed; only B can be pushed.
4. Force can be transmitted, but only between directly adjacent objects. That means, if an agent is applying force in a direction, you can push that agent from behind to help.
5. Only pushing is allowed - there is no pulling or lateral dragging. In other words, to push an object to the right, you must be on its left side and take the RIGHT action to apply force.

Message rules:
1. A message is a string including things you want to tell other agents.
2. Your message can be received by all agents within your view, and you can receive messages from all agents within your view.
3. Messages are broadcast-based. The source of a message is anonymous.
4. Write only what's necessary in your message. Avoid any ambiguity in your message.
5. Messages is capped to no more than 120 characters, exceeding part will be replaced by "...".

Other rules:
1. Coordinates are represented as (i, j), where i is the row index and j is the column index. Your vision uses global coordinates, so please use global coordinates.
2. The direction of increasing i is downward, and increasing j is to the right.
3. Objects that are completely outside the map (marked with "*") will be removed.

Please think carefully and choose your next action. You will need to collaborate with other agents to successfully complete the task.

Your response should include:
1. Analysis of the current situation
2. Your decision and reasoning
3. The message to be left (if any)

End your response clearly with your chosen action: "ACTION: [YOUR_ACTION]" and/or "MSG: [Your message (no line breaks).]"
"""

        return self.prompt
    
    def to_action(self, response):
        action = None
        message = None
        
        action_patterns = ["ACTION:", "Action:", "action:"]
        for pattern in action_patterns:
            if pattern in response:
                action_part = response.split(pattern, 1)[1].split("\n", 1)[0].strip()
                for valid_action in self.valid_actions:
                    if valid_action in action_part:
                        action = valid_action
                        break
        
        msg_patterns = ["MSG:", "Msg:", "msg:"]
        for pattern in msg_patterns:
            if pattern in response:
                msg_part = response.split(pattern, 1)[1]
                if "\n" in msg_part:
                    message = msg_part.split("\n", 1)[0].strip()
                else:
                    message = msg_part.strip()
                if len(message) > 120:
                    message = message[:120] + "..."
        
        self.history.append({
            'round': len(self.history) + 1,
            'action': action if action else 'NONE',
            'message': message if message else ''
        })
        
        result = {
            'prompt': self.prompt,
            'response': response,
        }
        
        if action:
            result['move'] = action

        if message:
            result['speak'] = message
        print(f'{self.name} {action if action else "NONE":<8}{message}')
        result['api_call_time'] = 0
        return result
