import NetworkMdp
import DynamicMethods


def main():
    network = NetworkMdp.NetworkMdp("mesh4x4.txt")

    DynamicMethods.value_iteration(network, 0.9, 0.001)
    network.render("Value Iteration", "val_iter.html")


if __name__ == '__main__':
    main()
