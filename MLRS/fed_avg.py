import flwr as fl
import numpy as np
import argparse
import torch
from client import gen_client_fn
from model import network, get_initial_parameters
from pathlib import Path
from util import plot_metric_from_history, gen_evaluate_fn, weighted_average, fit_config, ray_config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 1
LEARNING_RATE = 1e-1

def main() -> None:
    parser = argparse.ArgumentParser(description='FL experiments')
    parser.add_argument('--num_clients', type=int, default=10, help='The number of clients for the FL experiment')
    parser.add_argument('--fraction_fit', type=float, default=0.5, help = 'Specify what fraction of clients should be sampled for an FL fitting round.')
    parser.add_argument('--num_rounds', type=int, default=5, help = 'Specify the number of FL rounds')
    parser.add_argument('--batch_size', type=int, default=32, help='Specify batch size')
    parser.add_argument('--iid', type=bool, default=False, help='Specify if dataset should be iid')
    parser.add_argument('--balance', type=bool, default=False, help='Specify if dataset should be balanced')
    parser.add_argument('--strategy', type=str, default='FedAvg', help='Specify strategy')

    args = parser.parse_args()

    client_resources = {'num_cpus': 1}

    if DEVICE.type == "cuda":
        client_resources["num_gpus"] = 1

    client_fn, testloader = gen_client_fn(
        num_epochs=NUM_EPOCHS,
        batch_size=args.batch_size,
        device=DEVICE,
        num_clients=args.num_clients,
        iid=args.iid,
        balance=args.balance,
        learning_rate=LEARNING_RATE,
    )

    evaluate_fn = gen_evaluate_fn(testloader, DEVICE)

    FedAvgStrategy = fl.server.strategy.FedAvg(
        fraction_fit=1, #the fraction of available clients to use for evaluation of the model at each round of Federated Learning
        min_fit_clients=1, #the minimum number of clients that should be selected for training at each round of Federated learning
        min_available_clients=args.num_clients, #the minimum number of clients that should be available for participation in each round of federated learning.
        evaluate_fn=evaluate_fn, #the function to be used for evaluating the model on the client devices.
        initial_parameters=get_initial_parameters(network),
        evaluate_metrics_aggregation_fn=weighted_average, #function to be used for aggregating the evaluation metrics across all the clients.
        on_fit_config_fn=fit_config,
        accept_failures=False,
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.num_clients,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=FedAvgStrategy,
        ray_init_args=ray_config,
        client_resources=client_resources
    )

    result_str: str = (
        f"_C={args.num_clients}"
        f"_B={args.batch_size}"
        f"_E={NUM_EPOCHS}"
        f"_R={args.num_rounds}"
    )

    np.save(
        Path('results') / Path(f'history{result_str}'), history, )
    
    plot_metric_from_history(
        history,
        result_str,
        args.strategy,
    )

if __name__ == "__main__":
    main()
