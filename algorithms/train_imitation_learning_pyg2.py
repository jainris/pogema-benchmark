import argparse
import pickle
import pathlib
import numpy as np
import wandb

import multiprocessing as mp
from itertools import compress

from pogema import GridConfig

import torch
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.loader import DataLoader

from convert_to_imitation_dataset import (
    add_imitation_dataset_args,
    generate_graph_dataset,
    get_imitation_dataset_file_name,
)
from generate_hypergraphs import (
    add_hypergraph_generation_args,
    generate_hypergraph_indices,
    get_hypergraph_file_name,
)
from generate_pos import get_pos_file_name
from run_expert import (
    get_expert_dataset_file_name,
    get_expert_algorithm_and_config,
    run_expert_algorithm,
    add_expert_dataset_args,
)
from imitation_dataset_pyg import MAPFGraphDataset, MAPFHypergraphDataset

from agents import run_model_on_grid, get_model
from grid_config_generator import (
    grid_config_generator_factory,
    generate_grid_config_from_env,
)
from generate_target_vec import get_target_vec_file_name, generate_target_vec

from ranking_losses import PairwiseLogisticLoss


def add_training_args(parser):
    parser.add_argument("--validation_fraction", type=float, default=0.15)
    parser.add_argument("--test_fraction", type=float, default=0.15)
    parser.add_argument("--num_training_oe", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--imitation_learning_model", type=str, default="MAGAT")
    parser.add_argument("--cnn_mode", type=str, default="basic-CNN")
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--num_gnn_layers", type=int, default=3)
    parser.add_argument("--num_attention_heads", type=int, default=1)
    parser.add_argument("--attention_mode", type=str, default="GAT_modified")
    parser.add_argument("--edge_dim", type=int, default=None)
    parser.add_argument("--model_residuals", type=str, default=None)
    parser.add_argument(
        "--use_edge_weights", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--use_edge_attr", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--hyperedge_feature_generator", type=str, default="gcn")

    parser.add_argument("--agent_network_type", type=str, default="single")

    parser.add_argument("--model2_imitation_learning_model", type=str, default="MAGAT")
    parser.add_argument("--model2_embedding_size", type=int, default=128)
    parser.add_argument("--model2_num_gnn_layers", type=int, default=3)
    parser.add_argument("--model2_num_attention_heads", type=int, default=1)
    parser.add_argument("--model2_attention_mode", type=str, default="GAT_modified")
    parser.add_argument("--model2_edge_dim", type=int, default=None)
    parser.add_argument("--model2_model_residuals", type=str, default=None)
    parser.add_argument(
        "--model2_use_edge_weights",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--model2_use_edge_attr", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--model2_hyperedge_feature_generator", type=str, default="gcn")

    parser.add_argument("--load_partial_parameters_path", type=str, default=None)
    parser.add_argument("--replace_model", type=str, default=None)
    parser.add_argument("--parameters_to_load", type=str, default="all")
    parser.add_argument("--parameters_to_freeze", type=str, default=None)

    parser.add_argument("--lr_start", type=float, default=1e-3)
    parser.add_argument("--lr_end", type=float, default=1e-6)
    parser.add_argument("--num_epochs", type=int, default=300)

    parser.add_argument("--validation_every_epochs", type=int, default=4)
    parser.add_argument(
        "--run_online_expert", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--save_intmd_checkpoints", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints")

    parser.add_argument(
        "--skip_validation", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--skip_validation_accuracy",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--model_seed", type=int, default=42)
    parser.add_argument("--initial_val_size", type=int, default=128)
    parser.add_argument("--threshold_val_success_rate", type=float, default=0.9)
    parser.add_argument("--num_run_oe", type=int, default=500)
    parser.add_argument("--run_oe_after", type=int, default=0)
    parser.add_argument(
        "--recursive_oe", action=argparse.BooleanOptionalAction, default=False
    )

    parser.add_argument(
        "--load_positions_separately",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--train_on_terminated_agents",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--run_expert_in_separate_fork",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--use_target_vec", type=str, default=None)
    parser.add_argument("--collision_shielding", type=str, default="naive")
    parser.add_argument("--action_sampling", type=str, default="deterministic")

    parser.add_argument(
        "--train_only_for_relevance",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    return parser


def main():
    parser = argparse.ArgumentParser(description="Train imitation learning model.")
    parser = add_expert_dataset_args(parser)
    parser = add_imitation_dataset_args(parser)
    parser = add_hypergraph_generation_args(parser)
    parser = add_training_args(parser)

    args = parser.parse_args()
    print(args)

    assert args.save_termination_state

    if args.device == -1:
        device = torch.device("cuda")
    elif args.device is not None:
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")

    num_agents = int(args.robot_density * args.map_h * args.map_w)

    rng = np.random.default_rng(args.dataset_seed)
    seeds = rng.integers(10**10, size=args.num_samples)

    _grid_config_generator = grid_config_generator_factory(
        map_type=args.map_type,
        map_w=args.map_w,
        map_h=args.map_h,
        num_agents=num_agents,
        obstacle_density=args.obstacle_density,
        obs_radius=args.obs_radius,
        collision_system=args.collision_system,
        on_target=args.on_target,
        min_dist=args.min_dist,
        max_episode_steps=args.max_episode_steps,
    )

    grid_config = _grid_config_generator(seeds[0])

    expert_algorithm, inference_config = get_expert_algorithm_and_config(args)

    torch.manual_seed(args.model_seed)
    model, hypergraph_model, dataset_kwargs = get_model(args, device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr_start, weight_decay=1e-5)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=args.lr_end
    )

    dense_dataset = None
    hyper_edge_indices = None
    target_vecs = None
    relevances = None

    file_name = get_imitation_dataset_file_name(args)

    print("Loading Dataset.............")
    path = pathlib.Path(args.dataset_dir, "processed_dataset", file_name)
    with open(path, "rb") as f:
        dense_dataset = pickle.load(f)
    if args.load_positions_separately:
        print("Loading Agent Positions.....")
        file_name = get_pos_file_name(args)
        path = pathlib.Path(args.dataset_dir, "positions", file_name)
        with open(path, "rb") as f:
            agent_pos = pickle.load(f)
        dense_dataset = (*dense_dataset, agent_pos)
    if hypergraph_model:
        print("Loading Hypergraphs.........")
        file_name = get_hypergraph_file_name(args)
        path = pathlib.Path(args.dataset_dir, "hypergraphs", file_name)
        with open(path, "rb") as f:
            hyper_edge_indices = pickle.load(f)
    if args.use_target_vec is not None:
        print("Loading Target Vecs.........")
        file_name = get_target_vec_file_name(args)
        path = pathlib.Path(args.dataset_dir, "target_vec", file_name)
        with open(path, "rb") as f:
            target_vecs = pickle.load(f)
    if args.pibt_expert_relevance_training:
        print("Loading Relevance Scores....")
        file_name = get_expert_dataset_file_name(args)
        path = pathlib.Path(f"{args.dataset_dir}", "pibt_relevance", f"{file_name}")
        with open(path, "rb") as f:
            relevances = pickle.load(f)
        relevances = torch.from_numpy(relevances)

    if args.train_only_for_relevance:
        assert (
            args.pibt_expert_relevance_training
        ), "Need the relevance data to train for relevance."
        loss_function = PairwiseLogisticLoss()
    else:
        loss_function = torch.nn.CrossEntropyLoss()

    wandb.init(
        project="hyper-mapf-pogema",
        name=args.run_name,
        config=vars(args),
        entity="jainris",
    )

    # Data split
    train_id_max = int(
        args.num_samples * (1 - args.validation_fraction - args.test_fraction)
    )
    validation_id_max = train_id_max + int(args.num_samples * args.validation_fraction)

    def _divide_dataset(start, end):
        mask = torch.logical_and(dense_dataset[4] >= start, dense_dataset[4] < end)
        hindices, relevs = None, None
        if hyper_edge_indices is not None:
            hindices = list(compress(hyper_edge_indices, mask))
        if relevances is not None:
            relevs = relevances[mask]
        return tuple(gd[mask] for gd in dense_dataset), hindices, relevs

    train_dataset, train_hindices, train_relevances = _divide_dataset(0, train_id_max)
    validation_dataset, validation_hindices, validation_relevances = _divide_dataset(
        train_id_max, validation_id_max
    )
    # test_dataset = _divide_dataset(validation_id_max, torch.inf)

    if hypergraph_model:
        train_dataset = MAPFHypergraphDataset(
            train_dataset,
            train_hindices,
            target_vec=target_vecs,
            use_target_vec=args.use_target_vec,
            **dataset_kwargs,
        )
        validation_dataset = MAPFHypergraphDataset(
            validation_dataset,
            validation_hindices,
            target_vec=target_vecs,
            use_target_vec=args.use_target_vec,
            **dataset_kwargs,
        )
    else:
        train_dataset = MAPFGraphDataset(
            train_dataset,
            target_vec=target_vecs,
            use_target_vec=args.use_target_vec,
            **dataset_kwargs,
        )
        validation_dataset = MAPFGraphDataset(
            validation_dataset,
            target_vec=target_vecs,
            use_target_vec=args.use_target_vec,
            **dataset_kwargs,
        )
    train_dl = DataLoader(train_dataset, batch_size=args.batch_size)
    validation_dl = DataLoader(validation_dataset, batch_size=args.batch_size)

    best_validation_success_rate = 0.0
    best_validation_accuracy = 0.0
    best_val_file_name = "best_low_val.pt"
    best_val_acc_file_name = "best_acc_val.pt"
    checkpoint_path = pathlib.Path(f"{args.checkpoints_dir}", best_val_file_name)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    cur_validation_id_max = min(train_id_max + args.initial_val_size, validation_id_max)

    oe_graph_dataset = None
    oe_grid_configs = []
    oe_hypergraph_indices = []
    oe_target_vecs = None

    def multiprocess_run_expert(
        queue,
        expert,
        grid_config,
        save_termination_state,
        additional_data_func=None,
    ):
        expert_results = run_expert_algorithm(
            expert,
            grid_config=grid_config,
            save_termination_state=save_termination_state,
            additional_data_func=additional_data_func,
        )
        queue.put((*expert_results, grid_config))

    move_results = np.array(grid_config.MOVES)

    def get_hypergraph_indices(env, **kwargs):
        return generate_hypergraph_indices(
            env,
            hypergraph_greedy_distance=args.hypergraph_greedy_distance,
            hypergraph_num_steps=args.hypergraph_num_steps,
            move_results=move_results,
            generate_graph=args.generate_graph_from_hyperedges,
        )

    queue = mp.Queue()

    print("Starting Training....")
    for epoch in range(args.num_epochs):
        total_loss = 0.0
        tot_correct = 0
        num_samples = 0
        n_batches = 0

        model = model.train()
        for data in train_dl:
            data = data.to(device)
            optimizer.zero_grad()

            out = model(data.x, data)
            target_actions = data.y

            if not args.train_on_terminated_agents:
                out = out[~data.terminated]
                target_actions = target_actions[~data.terminated]
            loss = loss_function(out, target_actions)

            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            tot_correct += (
                torch.sum(torch.argmax(out, dim=-1) == target_actions).detach().cpu()
            )
            num_samples += out.shape[0]
            n_batches += 1

        if oe_graph_dataset is not None:
            if hypergraph_model:
                oe_dl = DataLoader(
                    MAPFHypergraphDataset(
                        oe_graph_dataset,
                        oe_hypergraph_indices,
                        target_vec=oe_target_vecs,
                        use_target_vec=args.use_target_vec,
                        **dataset_kwargs,
                    ),
                    batch_size=args.batch_size,
                )
            else:
                oe_dl = DataLoader(
                    MAPFGraphDataset(
                        oe_graph_dataset,
                        target_vec=oe_target_vecs,
                        use_target_vec=args.use_target_vec,
                        **dataset_kwargs,
                    ),
                    batch_size=args.batch_size,
                )

            for data in oe_dl:
                data = data.to(device)
                optimizer.zero_grad()

                out = model(data.x, data)
                target_actions = data.y

                if not args.train_on_terminated_agents:
                    out = out[~data.terminated]
                    target_actions = target_actions[~data.terminated]
                loss = loss_function(out, target_actions)

                total_loss += loss.item()

                loss.backward()
                optimizer.step()

                target_actions = target_actions.reshape((*out.shape[:-1], -1))
                target_actions = torch.argmax(out, dim=-1)
                tot_correct += (
                    torch.sum(torch.argmax(out, dim=-1) == target_actions)
                    .detach()
                    .cpu()
                )
                num_samples += out.shape[0]
                n_batches += 1
        lr_scheduler.step()

        print(
            f"Epoch {epoch}, Mean Loss: {total_loss / n_batches}, Mean Accuracy: {tot_correct / num_samples}"
        )

        results = {
            "train_loss": total_loss / n_batches,
            "train_accuracy": tot_correct / num_samples,
        }
        if (not args.skip_validation) and (
            (epoch + 1) % args.validation_every_epochs == 0
        ):
            model = model.eval()

            num_completed = 0

            print("-------------------")
            print("Starting Validation")

            if not args.skip_validation_accuracy:
                val_correct = 0
                val_samples = 0

                for data in validation_dl:
                    data = data.to(device)
                    out = model(data.x, data)
                    target_actions = data.y

                    if not args.train_on_terminated_agents:
                        out = out[~data.terminated]
                        target_actions = target_actions[~data.terminated]
                    target_actions = target_actions.reshape((*out.shape[:-1], -1))
                    target_actions = torch.argmax(out, dim=-1)
                    val_correct += (
                        torch.sum(torch.argmax(out, dim=-1) == target_actions)
                        .detach()
                        .cpu()
                    )
                    val_samples += out.shape[0]
                val_accuracy = val_correct / val_samples
                results = results | {"validation_accuracy": val_accuracy}
                if val_accuracy > best_validation_accuracy:
                    best_validation_accuracy = val_accuracy
                    checkpoint_path = pathlib.Path(
                        args.checkpoints_dir, best_val_acc_file_name
                    )
                    torch.save(model.state_dict(), checkpoint_path)

            for graph_id in range(train_id_max, cur_validation_id_max):
                success, env, observations = run_model_on_grid(
                    model=model,
                    device=device,
                    grid_config=_grid_config_generator(seeds[graph_id]),
                    args=args,
                    dataset_kwargs=dataset_kwargs,
                    hypergraph_model=hypergraph_model,
                    use_target_vec=args.use_target_vec,
                )

                if success:
                    num_completed += 1
                print(
                    f"Validation Graph {graph_id - train_id_max}/{validation_id_max - train_id_max}, "
                    f"Current Success Rate: {num_completed / (graph_id - train_id_max + 1)}"
                )
            success_rate = num_completed / (graph_id - train_id_max)
            results = results | {"validation_success_rate": success_rate}

            if args.save_intmd_checkpoints:
                checkpoint_path = pathlib.Path(
                    f"{args.checkpoints_dir}", f"epoch_{epoch}.pt"
                )
                torch.save(model.state_dict(), checkpoint_path)

            if success_rate > best_validation_success_rate:
                best_validation_success_rate = success_rate
                if success_rate >= args.threshold_val_success_rate:
                    print("Success rate passed threshold -- Increasing Validation Size")
                    args.threshold_val_success_rate = 1.1
                    cur_validation_id_max = validation_id_max
                    best_val_file_name = "best.pt"
                    best_validation_success_rate = 0.0
                checkpoint_path = pathlib.Path(
                    f"{args.checkpoints_dir}", best_val_file_name
                )
                torch.save(model.state_dict(), checkpoint_path)

            print("Finshed Validation")
            print("------------------")

            if args.run_online_expert and (epoch + 1 >= args.run_oe_after):
                print("---------------------")
                print("Running Online Expert")

                rng = np.random.default_rng(args.dataset_seed + epoch + 1)
                if args.recursive_oe:
                    oe_ids = rng.integers(
                        train_id_max + len(oe_grid_configs), size=args.num_run_oe
                    )
                else:
                    oe_ids = rng.integers(train_id_max, size=args.num_run_oe)

                oe_dataset = []
                oe_hindices = []

                for i, graph_id in enumerate(oe_ids):
                    print(f"Running model on {i}/{args.num_run_oe} ", end="")
                    if graph_id > train_id_max:
                        grid_config = oe_grid_configs[graph_id - train_id_max]
                    else:
                        grid_config = _grid_config_generator(seeds[graph_id])
                    success, env, observations = run_model_on_grid(
                        model=model,
                        device=device,
                        grid_config=grid_config,
                        args=args,
                        dataset_kwargs=dataset_kwargs,
                        hypergraph_model=hypergraph_model,
                        max_episodes=args.max_episode_steps,
                        use_target_vec=args.use_target_vec,
                    )

                    if not success:
                        print(f"-- Running OE ", end="")
                        expert = expert_algorithm(inference_config)

                        additional_data_func = (
                            get_hypergraph_indices if hypergraph_model else None
                        )
                        grid_config = generate_grid_config_from_env(env)

                        all_actions, all_observations, all_terminated = (
                            None,
                            None,
                            None,
                        )
                        expert_results = None
                        hindices = []

                        if args.run_expert_in_separate_fork:
                            p = mp.Process(
                                target=multiprocess_run_expert,
                                args=(
                                    queue,
                                    expert,
                                    grid_config,
                                    args.save_termination_state,
                                    additional_data_func,
                                ),
                            )
                            p.start()

                            while p.is_alive():
                                try:
                                    expert_results = queue.get(timeout=3)
                                    p.join()
                                    break
                                except:
                                    p.join(timeout=0.5)
                                    if p.exitcode is not None:
                                        break
                        else:
                            multiprocess_run_expert(
                                queue,
                                expert,
                                grid_config,
                                args.save_termination_state,
                                additional_data_func,
                            )
                            expert_results = queue.get()

                        if expert_results is not None:
                            if hypergraph_model:
                                (
                                    all_actions,
                                    all_observations,
                                    all_terminated,
                                    hindices,
                                    grid_config,
                                ) = expert_results
                            else:
                                (
                                    all_actions,
                                    all_observations,
                                    all_terminated,
                                    grid_config,
                                ) = expert_results
                            if all(all_terminated[-1]):
                                print(f"-- Success")
                                oe_dataset.append(
                                    (all_observations, all_actions, all_terminated)
                                )
                                oe_hindices.extend(hindices)
                                oe_grid_configs.append(grid_config)
                            else:
                                print(f"-- Fail")
                        else:
                            print(f"-- Error")
                    else:
                        print(f"-- Success")
                while queue.qsize() > 0:
                    # Popping remaining elements, although no elements should remain
                    expert_results = queue.get()
                    hindices = []
                    if hypergraph_model:
                        (
                            all_actions,
                            all_observations,
                            all_terminated,
                            hindices,
                            grid_config,
                        ) = expert_results
                    else:
                        all_actions, all_observations, all_terminated, grid_config = (
                            expert_results
                        )
                    oe_dataset.append((all_observations, all_actions, all_terminated))
                    oe_hindices.extend(hindices)
                    oe_grid_configs.append(grid_config)

                if len(oe_dataset) > 0:
                    print(f"Adding {len(oe_dataset)} OE grids to the dataset")
                    oe_hypergraph_indices.extend(oe_hindices)
                    new_oe_graph_dataset = generate_graph_dataset(
                        dataset=oe_dataset,
                        comm_radius=args.comm_radius,
                        obs_radius=args.obs_radius,
                        num_samples=None,
                        save_termination_state=True,
                        use_edge_attr=dataset_kwargs["use_edge_attr"],
                        print_prefix=None,
                    )
                    if args.use_target_vec is not None:
                        new_oe_target_vec = generate_target_vec(
                            dataset=oe_dataset, num_samples=None, print_prefix=None
                        )
                        if oe_target_vecs is None:
                            oe_target_vecs = new_oe_target_vec
                        else:
                            oe_target_vecs = torch.concat(
                                [oe_target_vecs, new_oe_target_vec], dim=0
                            )
                    if oe_graph_dataset is None:
                        oe_graph_dataset = new_oe_graph_dataset
                    else:
                        oe_graph_dataset = tuple(
                            torch.concat(
                                [oe_graph_dataset[i], new_oe_graph_dataset[i]], dim=0
                            )
                            for i in range(len(oe_graph_dataset))
                        )
                print("Finished Online Expert")
                print("----------------------")

        wandb.log(results)
    checkpoint_path = pathlib.Path(f"{args.checkpoints_dir}", f"last.pt")
    torch.save(model.state_dict(), checkpoint_path)


if __name__ == "__main__":
    mp.set_start_method("fork")  # TODO: Maybe add this as an cmd line option
    main()
