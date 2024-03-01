from collections import OrderedDict

import flwr as fl
import torch

from utils import FlairDataset, test, train


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id: str, h5p_file: str, path_to_label_mapping) -> None:
        super().__init__()
        self.client_id: str = client_id
        self.h5p_file: str = h5p_file
        self.path_to_label_mapping: str = path_to_label_mapping

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        num_epochs = 2
        trainset = FlairDataset(
            h5p_file="dataset/flair.h5",
            partition="train",
            client_id=self.client_id,
            path_to_label_mapping="flair_labels_to_index.pkl",
            label_type="coarse",
        )
        self.set_parameters(parameters)
        train(net, trainloader, epochs=1)
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        return float(loss), len(testloader.dataset), {"accuracy": float(accuracy)}
