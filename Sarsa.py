import numpy as np
import NetworkMdp
import TrafficGenerator
import random


class Sarsa:
    def __init__(self, mapfile, topology, epsilon=0.05, alpha=0.9, gamma=0.9):
        self.network = NetworkMdp.NetworkMdp(mapfile, topology)
        self.q = np.zeros((self.network.nodes.shape[0], self.network.nodes.shape[1], len(NetworkMdp.actions)))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def send_packet(self, origin, max_hops=100):
        current_node = origin
        action = self.get_action(current_node)

        hops = 0
        while self.network.node(current_node) != NetworkMdp.DESTINATION:
            next_node, reward = self.network.next_node(current_node, NetworkMdp.actions[action])
            next_action = self.get_action(next_node)

            q_current = self.q[current_node[0]][current_node[1]][action]
            q_next = self.q[next_node[0]][next_node[1]][next_action]
            new_q = q_current + self.alpha * (reward + self.gamma * q_next - q_current)
            self.q[current_node[0]][current_node[1]][action] = new_q

            self.network.set_policy(current_node, action)

            current_node = next_node
            action = next_action

            hops = hops + 1

            if hops > max_hops:
                return -1

        return hops

    def get_action(self, node):
        if random.uniform(0, 1) <= self.epsilon:
            # Randomly return an action
            return random.randint(0, len(NetworkMdp.actions) - 1)

        best_q = float('-inf')
        best_actions = []
        for action in NetworkMdp.actions.keys():
            q = self.q[node[0]][node[1]][action]

            if q > best_q:
                best_q = q
                best_actions.clear()
                best_actions.append(action)
            elif q == best_q:
                best_actions.append(action)

        if len(best_actions) == 0:
            # No actions found that maximize the Q function, randomly pick one
            return np.random.randint(0, len(NetworkMdp.actions) - 1)
        else:
            # Randomly pick an action from the best actions that maximized the Q function
            return best_actions[np.random.randint(0, len(best_actions))]


def main():
    sarsa = Sarsa("mesh4x4.txt", NetworkMdp.TORUS)

    traffic = TrafficGenerator.TrafficGenerator(sarsa)
    traffic.simulate(10000, "SARSA", False)


if __name__ == '__main__':
    main()
