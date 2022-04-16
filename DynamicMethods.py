import numpy as np
import NetworkMdp

# Evaluate the chosen policy and update the value function for each state
def policy_eval(network, discount, theta):
    nabla = theta
    while nabla >= theta:
        max_diff = 0

        old_values = np.copy(network.state_values)
        for x in range(0, network.states.shape[0]):
            for y in range(0, network.states.shape[1]):
                state = (x, y)

                if network.state(state) != NetworkMdp.INACTIVE:
                    old_value = old_values[state]
                    action = network.actions[network.policy(state)]

                    # Loop through all the next possible states and calculate the total value function
                    next_state, reward = network.next_state(state, action)
                    network.state_values[state] = reward + discount * old_values[next_state]
                    max_diff = max(max_diff, abs(old_value - network.state_values[state]))
                else:
                    network.state_values[state] = -1000
        nabla = max_diff

# Find the optimal policy using the policy iteration method
def policy_iteration(network, discount, theta, max_k=15):
    for k in range(max_k):
        print("Iteration " + str(k))

        # Evaluate the policy
        network.policy_eval(discount, theta)

        last_policy = np.copy(network.policies)

        # Loop through each state
        for x in range(0, network.states.shape[0]):
            for y in range(0, network.states.shape[1]):
                state = (x, y)

                if network.state(state) != NetworkMdp.INACTIVE:
                    # Now find the action(s) that maximize the value function
                    max_value = float('-inf')
                    best_a = []
                    for action_id in network.actions.keys():
                        action = network.actions[action_id]

                        # Loop through all the next possible states and calculate the total value function
                        next_state, reward = network.next_state(state, action)
                        value = reward + discount * network.state_value(next_state)

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
                        network.set_policy(state, np.random.randint(0, 4))
                    else:
                        # Randomly pick an action from the best actions that maximized the value function
                        network.set_policy(state, best_a[np.random.randint(0, len(best_a))])

        if (last_policy == network.policies).all():
            # No change from the previous policy, we can stop here
            print("No policy change, exiting early")
            break

        last_policy = np.copy(network.policies)

# Find the optimal policy using the value iteration method
def value_iteration(network, discount, theta):
    nabla = theta
    while nabla >= theta:
        max_diff = 0

        old_values = np.copy(network.state_values)
        for x in range(0, network.states.shape[0]):
            for y in range(0, network.states.shape[1]):
                state = (x, y)

                if network.state(state) != NetworkMdp.INACTIVE:
                    old_value = old_values[state]

                    # Now find the action(s) that maximize the value function
                    max_value = float('-inf')
                    best_a = []
                    for action_id in network.actions.keys():
                        action = network.actions[action_id]

                        # Loop through all the next possible states and calculate the total value function
                        next_state, reward = network.next_state(state, action)
                        value = reward + discount * network.state_value(next_state)

                        if value > max_value:
                            # New maximum value found
                            max_value = value
                            best_a.clear()
                            best_a.append(action_id)
                        elif value == max_value:
                            # This action has the same value as our current maximum, so this is an option as well
                            best_a.append(action_id)

                    network.state_values[state] = max_value
                    max_diff = max(max_diff, abs(old_value - network.state_values[state]))

                    if len(best_a) == 0:
                        # No actions found that maximize the value function, randomly pick one for our policy
                        network.set_policy(state, np.random.randint(0, 4))
                    else:
                        # Randomly pick an action from the best actions that maximized the value function
                        network.set_policy(state, best_a[np.random.randint(0, len(best_a))])

        nabla = max_diff
        print(nabla)
