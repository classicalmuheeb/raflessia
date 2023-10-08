from typing import Tuple, List
from collections import OrderedDict
import numpy as np
from torch.nn.parameter import Parameter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import DataLoader
from flwr.common.typing import NDArrays, Scalar, Parameters
from flwr.common import ndarrays_to_parameters
from logging import INFO, ERROR
from flwr.common.logger import log

""" Specify resnet model """
network = models.resnet152(num_classes=10)

def train( 
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    proximal_mu: float,
) -> None:
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    global_params = [val.detach().clone() for val in net.parameters()]
    net.train()
    for _ in range(epochs):
        net = _training_loop(
            net, global_params, trainloader, device, criterion, optimizer, proximal_mu
        )

def _training_loop( 
    net: nn.Module,
    global_params: List[Parameter],
    trainloader: DataLoader,
    device: torch.device,
    criterion: torch.nn.CrossEntropyLoss,
    optimizer: torch.optim.Adam,
    proximal_mu: float
) -> nn.Module:

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        proximal_term = 0.0
        for local_weights, global_weights in zip(net.parameters(), global_params):
            proximal_term += (local_weights - global_weights).norm(2)
        loss = criterion(net(images), labels) + (proximal_mu / 2) * proximal_term
        loss.backward()
        optimizer.step()
    return net


def test(
    net: nn.Module, testloader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    """Evaluate the network on the entire test set."""

    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    if len(testloader.dataset) == 0:
        log(ERROR, "Testloader can't be 0, exiting...")
    loss /= len(testloader.dataset)
    accuracy = correct / total
    log(INFO, f"Evaluation loss: {loss}, Accurracy: {accuracy}")
    return loss, accuracy

def get_parameters(net: nn.Module) -> NDArrays:
    return [val.cpu().numpy() for name, val in net.state_dict().items() if "num_batches_tracked" not in name]

def set_parameters(net: nn.Module, parameters: NDArrays):
    keys = [k for k in net.state_dict().keys() if "num_batches_tracked" not in k]
    params_dict = zip(keys, parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def get_initial_parameters(net: nn.Module) -> Parameters:
    "get model weights as a list of NumPy ndarrays"
    weights = [val.cpu().numpy() for name, val in net.state_dict().items() if "num_batches_tracked" not in name]
    parameters = ndarrays_to_parameters(weights)
    return parameters