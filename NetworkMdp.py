import numpy as np
from bokeh.plotting import figure, save, show

# Types of nodes in the network
DESTINATION = -1    # Destination of our packet
ACTIVE = 0          # Node that's active and can forward packets
INACTIVE = 1        # Node that's inactive and cannot forward packets

# Routing actions
RIGHT = 0
LEFT = 1
UP = 2
DOWN = 3


class NetworkMdp:
    def __init__(self, map_file):
        self.actions = {
            RIGHT: [0, 1],
            LEFT: [0, -1],
            UP: [1, 0],
            DOWN: [-1, 0]
        }

        self.states = np.loadtxt(open(map_file, "rb"), delimiter=" ")
        self.state_values = np.zeros(self.states.shape)
        self.policies = np.random.randint(0, 4, self.states.shape)

    # Plot the network and its policies
    def render(self, method, filename):
        x_pos = []
        y_pos = []
        color = []
        arrow = []
        arrow_dir = {DOWN: 0.0, UP: np.pi, RIGHT: -np.pi / 2.0, LEFT: np.pi / 2.0}

        n_feat = 0
        for i in range(self.states.shape[0]):
            for j in range(self.states.shape[1]):
                if self.states[i, j] == INACTIVE:
                    x_pos.append(f"{j}")
                    y_pos.append(f"{i}")
                    color.append("#000000")
                    n_feat += 1
                elif self.states[i, j] == DESTINATION:
                    x_pos.append(f"{j}")
                    y_pos.append(f"{i}")
                    color.append("#00FF00")
                    n_feat += 1

        for i in range(self.states.shape[0]):
            for j in range(self.states.shape[1]):
                if self.states[i, j] == ACTIVE:
                    x_pos.append(f"{j}")
                    y_pos.append(f"{i}")
                    color.append(f"#FFFFFF")
                    arrow.append(arrow_dir[self.policies[i, j]])

        fig = figure(
            title=f"{method}: Network Policy",
            x_range=[f"{i}" for i in np.arange(self.states.shape[0])],
            y_range=[f"{i}" for i in np.flip(np.arange(self.states.shape[1]))]
        )
        fig.rect(x_pos, y_pos, color=color, width=1, height=1)
        fig.triangle(x_pos[n_feat:], y_pos[n_feat:], angle=arrow, size=400 / np.sum(self.states.shape), color="#FF0000",
                     alpha=0.5)

        save(fig, filename)
        show(fig)

    # Given a current state and an action taken, return the next state and the given reward
    def next_state(self, current_state, action):
        next_state = (current_state[0] + action[0], current_state[1] + action[1])
        if not self.in_bounds(next_state):
            # Can't route off of the network (no node exists here), stay in the current state
            next_state = current_state

        if self.state(next_state) == INACTIVE:
            # Can't route to this node (it's inactive), stay in the current state and exit
            next_state = current_state

        # Reward is -1 unless the action takes us to the goal (in which case the reward is 0)
        reward = -1
        if self.state(next_state) == DESTINATION:
            reward = 0

        return [next_state, reward]

    # Check if a state is in bounds, i.e. on the grid
    def in_bounds(self, state):
        if state[0] < 0 or state[0] >= self.states.shape[0]:
            return False
        elif state[1] < 0 or state[1] >= self.states.shape[1]:
            return False
        else:
            return True

    # Get the type of node (e.g. destination, inactive, active) for a given state
    def state(self, state):
        return self.states[state[0], state[1]]

    # Get the policy for a given state
    def policy(self, state):
        return self.policies[state[0], state[1]]

    # Set the policy for a given state
    def set_policy(self, state, action):
        self.policies[state[0], state[1]] = action

    # Convenience function for getting the value function of a given state
    def state_value(self, state):
        return self.state_values[state[0], state[1]]

