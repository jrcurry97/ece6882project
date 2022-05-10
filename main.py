import NetworkMdp
import DynamicMethods
import numpy as np
import random
import matplotlib.pyplot as plt


class Sarsa:
    def __init__(self, network):
        self.q = np.zeros((network.nodes.shape[0], network.nodes.shape[1], len(NetworkMdp.actions)))
        self.network = network

    def send_packet(self, origin, epsilon, alpha, gamma):
        current_node = origin
        action = self.get_action(current_node, epsilon)

        hops = 0
        while self.network.node(current_node) != NetworkMdp.DESTINATION:
            next_node, reward = self.network.next_node(current_node, NetworkMdp.actions[action])
            next_action = self.get_action(next_node, epsilon)

            q_current = self.q[current_node[0]][current_node[1]][action]
            q_next = self.q[next_node[0]][next_node[1]][next_action]
            new_q = q_current + alpha * (reward + gamma * q_next - q_current)
            self.q[current_node[0]][current_node[1]][action] = new_q

            self.network.set_policy(current_node, action)

            current_node = next_node
            action = next_action

            hops = hops + 1

            if hops > 100:
                return -1

        return hops

    def get_action(self, node, epsilon):
        if random.uniform(0, 1) <= epsilon:
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
    network = NetworkMdp.NetworkMdp("mesh4x4.txt")
    sarsa = Sarsa(network)

    orig_map = np.copy(network.nodes)
    ratio = []
    for ep in range(0, 10000):
        if ep % 100 == 0:
            print("Episode " + str(ep))

        origin = (random.randint(0, network.nodes.shape[0] - 1), random.randint(0, network.nodes.shape[1] - 1))
        hops = sarsa.send_packet(origin, 0.01, 0.9, 0.9)
        distance = abs(3 - origin[0]) + abs(2 - origin[1])

        if distance > 0:
            ratio.append(hops / distance)

        if ep % 500 == 0:
            network.nodes = np.copy(orig_map)
            network.nodes[random.randint(0, 3)][random.randint(0, 3)] = NetworkMdp.INACTIVE
            network.nodes[2][3] = NetworkMdp.DESTINATION

    #network.render("SARSA", "sarsa.html")
    plt.plot(ratio)
    plt.show()


if __name__ == '__main__':
    main()
