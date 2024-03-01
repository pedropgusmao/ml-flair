import flwr as fl
from flwr.server.strategy import FedAvg
from client import MyFlowerClient


def client_fn(cid: str):
    # Return a standard Flower client
    return MyFlowerClient().to_client()


# Launch the simulation
hist = fl.simulation.start_simulation(
    client_fn=client_fn,  # A function to run a _virtual_ client when required
    num_clients=200,  # Total number of clients available
    config=fl.server.ServerConfig(num_rounds=3),  # Specify number of FL rounds
    strategy=FedAvg(),  # A Flower strategy
)
