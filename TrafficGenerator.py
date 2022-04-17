import NetworkMdp
import matplotlib.pyplot as plt
import random
import numpy as np


class TrafficGenerator:
    def __init__(self, solver):
        self.solver = solver

    def simulate(self, packet_count, disable_nodes=True):
        destination = (2, 3)
        ratio = []

        orig_network = np.copy(self.solver.network.nodes)
        for packet in range(1, packet_count):
            # Randomly choose origin node
            origin = (random.randint(0, self.solver.network.nodes.shape[0] - 1), random.randint(0, self.solver.network.nodes.shape[1] - 1))

            # Send the packet
            hops = self.solver.send_packet(origin)

            # Calculate optimal distance
            distance = abs(destination[0] - origin[0]) + abs(destination[1] - origin[1])

            # Account for a node sending a packet to itself
            if distance == 0:
                ratio.append(1.0)

            # Account for timeouts
            elif (distance > 0) and (hops == NetworkMdp.TIMEOUT):
                ratio.append(100.0)

            else:
                ratio.append(hops / distance)

            if disable_nodes:
                # Every 1000 packets, disable 1-3 nodes
                if packet % 1000 == 0:
                    self.solver.network.nodes = np.copy(orig_network)

                    for i in range(0, random.randint(0, 3)):
                        inactive_x = random.randint(0, self.solver.network.nodes.shape[0] - 1)
                        inactive_y = random.randint(0, self.solver.network.nodes.shape[1] - 1)
                        self.solver.network.nodes[inactive_x][inactive_y] = NetworkMdp.INACTIVE

                    self.solver.network.nodes[2][3] = NetworkMdp.DESTINATION

        plt.plot(ratio)
        plt.show()
