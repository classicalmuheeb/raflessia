from typing import Callable, Dict, Tuple

import flwr as fl
import torch
from flwr.common.typing import NDArrays, Scalar
from torch.utils.data import DataLoader
from model import train, test, set_parameters, get_parameters, network
from dataset import load_datasets
from logging import ERROR, INFO
from flwr.common.logger import configure, log

class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: str,
        net: torch.nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        device: torch.device,
        num_epochs: int,
        learning_rate: float,
    ):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        "get model weights"
        log(INFO, f"[Client {self.cid}] get_parameters")
        return get_parameters(self.net)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Pass customized values from config"""
        server_round = config['server_round']
        local_epochs = config['local_epochs']
        log(INFO, f"[Client {self.cid}, round {server_round}] fit, config: {config}")

        """Implements distributed fit function for a given client."""
        set_parameters(self.net, parameters)
        train(
            self.net,
            self.trainloader,
            self.device,
            epochs=local_epochs,
            learning_rate=self.learning_rate,
        )
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        """Implements distributed evaluation for a given client."""
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}

def gen_client_fn(
    device: torch.device,
    iid: bool,
    balance: bool,
    num_clients: int,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
) -> Tuple[Callable[[str], FlowerClient], DataLoader]:
    """Generates the client function that creates the Flower Clients"""

    trainloaders, valloaders, testloader = load_datasets(
        iid=iid, balance=balance, num_clients=num_clients, batch_size=batch_size
    )

    def client_fn(cid: str) -> FlowerClient:
        """Create a Flower client representing a single organization."""

        # Load model
        net = network.to(device)

        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]

        # Create a  single Flower client representing a single organization
        return FlowerClient(
            cid, net, trainloader, valloader, device, num_epochs, learning_rate
        )
    return client_fn, testloader
