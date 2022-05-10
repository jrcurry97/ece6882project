import numpy as np
import NetworkMdp
import TrafficGenerator


class DynamicMethods:

    def __init__(self, mapfile, topology):
        self.network = NetworkMdp.NetworkMdp(mapfile, topology)

    # Evaluate the chosen policy and update the value function for each state
    def policy_eval(self, discount, theta):
        nabla = theta
        while nabla >= theta:
            max_diff = 0

            old_values = np.copy(self.network.values)
            for x in range(0, self.network.nodes.shape[0]):
                for y in range(0, self.network.nodes.shape[1]):
                    state = (x, y)

                    if self.network.node(state) != NetworkMdp.INACTIVE:
                        old_value = old_values[state]
                        action = NetworkMdp.actions[self.network.policy(state)]

                        # Loop through all the next possible states and calculate the total value function
                        next_state, reward = self.network.next_node(state, action)
                        self.network.values[state] = reward + discount * old_values[next_state]
                        max_diff = max(max_diff, abs(old_value - self.network.values[state]))
                    else:
                        self.network.values[state] = -1000
            nabla = max_diff

    # Find the optimal policy using the policy iteration method
    def policy_iteration(self, discount, theta, max_k=15):
        for k in range(max_k):
            print("Iteration " + str(k))

            # Evaluate the policy
            self.policy_eval(discount, theta)

            last_policy = np.copy(self.network.policies)

            # Loop through each state
            for x in range(0, self.network.nodes.shape[0]):
                for y in range(0, self.network.nodes.shape[1]):
                    state = (x, y)

                    if self.network.node(state) != NetworkMdp.INACTIVE:
                        # Now find the action(s) that maximize the value function
                        max_value = float('-inf')
                        best_a = []
                        for action_id in NetworkMdp.actions.keys():
                            action = NetworkMdp.actions[action_id]

                            # Loop through all the next possible states and calculate the total value function
                            next_state, reward = self.network.next_node(state, action)
                            value = reward + discount * self.network.value(next_state)

                            if value > max_value:
                                # New maximum value found
                                max_value = value
                                best_a.clear()
                                best_a.append(action_id)
                            elif value == max_value:
                                # This action has the same value as our current maximum, so this is an option as well
                                best_a.append(action_id)

                        if len(best_a) == 0:
                            # No actions found that maximize the value function, randomly pick one for our policy
                            self.network.set_policy(state, np.random.randint(0, 4))
                        else:
                            # Randomly pick an action from the best actions that maximized the value function
                            self.network.set_policy(state, best_a[np.random.randint(0, len(best_a))])

            if (last_policy == self.network.policies).all():
                # No change from the previous policy, we can stop here
                print("No policy change, exiting early")
                break

            last_policy = np.copy(self.network.policies)

    # Find the optimal policy using the value iteration method
    def value_iteration(self, discount, theta):
        nabla = theta
        while nabla >= theta:
            max_diff = 0

            old_values = np.copy(self.network.values)
            for x in range(0, self.network.nodes.shape[0]):
                for y in range(0, self.network.nodes.shape[1]):
                    state = (x, y)

                    if self.network.node(state) != NetworkMdp.INACTIVE:
                        old_value = old_values[state]

                        # Now find the action(s) that maximize the value function
                        max_value = float('-inf')
                        best_a = []
                        for action_id in NetworkMdp.actions.keys():
                            action = NetworkMdp.actions[action_id]

                            # Loop through all the next possible states and calculate the total value function
                            next_state, reward = self.network.next_node(state, action)
                            value = reward + discount * self.network.value(next_state)

                            if value > max_value:
                                # New maximum value found
                                max_value = value
                                best_a.clear()
                                best_a.append(action_id)
                            elif value == max_value:
                                # This action has the same value as our current maximum, so this is an option as well
                                best_a.append(action_id)

                        self.network.values[state] = max_value
                        max_diff = max(max_diff, abs(old_value - self.network.values[state]))

                        if len(best_a) == 0:
                            # No actions found that maximize the value function, randomly pick one for our policy
                            self.network.set_policy(state, np.random.randint(0, 4))
                        else:
                            # Randomly pick an action from the best actions that maximized the value function
                            self.network.set_policy(state, best_a[np.random.randint(0, len(best_a))])

            nabla = max_diff
            print(nabla)

    def send_packet(self, origin, max_hops=100):
        return self.network.send_packet(origin, max_hops)


def main():
    # Find the optimal policy using value iteration
    dp = DynamicMethods("mesh4x4.txt", NetworkMdp.TORUS)
    dp.value_iteration(0.9, 0.001)

    # Generate random traffic
    traffic = TrafficGenerator.TrafficGenerator(dp)
    traffic.simulate(10000, "Value Iteration", True)


if __name__ == '__main__':
    main()
