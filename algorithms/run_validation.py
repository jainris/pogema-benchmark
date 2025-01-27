import argparse
import pathlib
import numpy as np
import wandb
from collections import OrderedDict

import multiprocessing as mp

import torch

from training_args import add_training_args
from convert_to_imitation_dataset import add_imitation_dataset_args
from generate_hypergraphs import add_hypergraph_generation_args
from run_expert import add_expert_dataset_args

from agents import run_model_on_grid, get_model
from grid_config_generator import grid_config_generator_factory


def load_from_legacy_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for key in state_dict.keys():
        if key.split(".")[0] == "gnns":
            k2 = key.split(".")
            k2[1] = f"{k2[1]}.gnn"
            new_state_dict[".".join(k2)] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key]
    return new_state_dict


def main():
    parser = argparse.ArgumentParser(description="Validate imitation learning model.")
    parser = add_expert_dataset_args(parser)
    parser = add_imitation_dataset_args(parser)
    parser = add_hypergraph_generation_args(parser)
    parser = add_training_args(parser)
    parser.add_argument("--additional_run_name", type=str, default=None)
    parser.add_argument("--project_name", type=str, default="hyper-mapf-pogema")
    parser.add_argument("--num_validation_maps", type=int, default=None)

    args = parser.parse_args()
    print(args)

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

    torch.manual_seed(args.model_seed)
    model, hypergraph_model, dataset_kwargs = get_model(args, device)

    run_name = args.run_name
    if args.additional_run_name is not None:
        run_name = run_name + "_" + args.additional_run_name

    wandb.init(
        project=args.project_name,
        name=run_name,
        config=vars(args) | {"validation_run": True},
        entity="jainris",
    )

    # Data split
    train_id_max = int(
        args.num_samples * (1 - args.validation_fraction - args.test_fraction)
    )
    if args.num_validation_maps is not None:
        validation_id_max = train_id_max + args.num_validation_maps
    else:
        validation_id_max = train_id_max + int(
            args.num_samples * args.validation_fraction
        )

    print("Starting Validation....")
    for epoch in range(
        args.validation_every_epochs - 1, args.num_epochs, args.validation_every_epochs
    ):
        checkpoint_path = pathlib.Path(args.checkpoints_dir, f"epoch_{epoch}.pt")
        state_dict = torch.load(checkpoint_path, map_location=device)
        try:
            model.load_state_dict(state_dict)
        except:
            # Loading from legacy state dict
            state_dict = load_from_legacy_state_dict(state_dict)
            model.load_state_dict(state_dict)
        model = model.eval()
        num_completed = 0
        for graph_id in range(train_id_max, validation_id_max):
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
                f"Epoch {epoch}, Validation Graph {graph_id - train_id_max}/{validation_id_max - train_id_max}, "
                f"Current Success Rate: {num_completed / (graph_id - train_id_max + 1)}"
            )
        success_rate = num_completed / (graph_id - train_id_max)
        results = {"validation_success_rate": success_rate}
        print("------------------")

        wandb.log(results, step=epoch)


if __name__ == "__main__":
    mp.set_start_method("fork")  # TODO: Maybe add this as an cmd line option
    main()
