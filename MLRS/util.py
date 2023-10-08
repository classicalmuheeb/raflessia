from typing import Tuple, Optional
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, List, Dict, Optional, Tuple, Callable
from model import set_parameters, network, test

import matplotlib.pyplot as plt
import numpy as np

from flwr.common import Metrics
from flwr.common.typing import NDArrays, Scalar
from flwr.server.history import History

#aggregate function for weighted average for evaluation
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": int(sum(accuracies)) / int(sum(examples))}

def gen_evaluate_fn(
    testloader: DataLoader, device: torch.device
) -> Callable[
    [int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]
]:

    def evaluate(
        server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        net = network

        #Use entire test set for evaluation 
        set_parameters(net, parameters) #Update model with the latest parameters
        net.to(device)
        loss, accuracy = test(net, testloader, device=device)
        # return statistics
        return loss, {"accuracy": accuracy}
    return evaluate

def fit_config(server_round: int):
    config = {
        "server_round": server_round,
        "local_epochs": 1,   
    }
    return config

# (optional) specify ray config
ray_config = {"include_dashboard": False, 
             "_system_config": {"automatic_object_spilling_enabled": False}, 
             }

"""ploting functions adapted from https://github.com/adap/flower/blob/main/baselines/flwr_baselines/publications/fedavg_mnist/utils.py"""
def plot_metric_from_history(
    hist: History,
    suffix: Optional[str] = "",
    strategy: str = None,
) -> None:
    metric_type = "centralized"
    metric_dict = (
        hist.metrics_centralized
        if metric_type == "centralized"
        else hist.metrics_distributed
    )
    rounds, values = zip(*metric_dict["accuracy"])
    fig = plt.figure()
    axis = fig.add_subplot(111)
    plt.plot(np.asarray(rounds), np.asarray(values), label=strategy)
    plt.title(f"{metric_type.capitalize()} Validation - MLRS")
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")

    # Set the apect ratio to 1.0
    xleft, xright = axis.get_xlim()
    ybottom, ytop = axis.get_ylim()
    axis.set_aspect(abs((xright - xleft) / (ybottom - ytop)) * 1.0)

    plt.savefig(Path('results') / Path(f"{metric_type}_metrics{suffix}.png"))
    plt.close()
