import argparse
import pickle
import pathlib
import numpy as np
import sys
import wandb

from multiprocessing import Process, Queue

from pogema import pogema_v0, GridConfig

from lacam.inference import LacamInference, LacamInferenceConfig

sys.path.append("./magat_pathplanning")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from graphs.weights_initializer import weights_init
import utils.graphUtils.graphML as gml
import utils.graphUtils.graphTools
from torchsummaryX import summary
from graphs.models.resnet_pytorch import *

from convert_to_imitation_dataset import generate_graph_dataset
from run_expert import (
    DATASET_FILE_NAME_KEYS,
    run_expert_algorithm,
    add_expert_dataset_args,
)
from train_imitation_learning import add_training_args
from train_imitation_learning import DecentralPlannerGATNet, run_model_on_grid
from test_expert import get_expert_file_name, EXPERT_FILE_NAME_KEYS


def main():
    parser = argparse.ArgumentParser(description="Train imitation learning model.")
    parser = add_expert_dataset_args(parser)
    parser = add_training_args(parser)

    parser.add_argument(
        "--test_in_distribution", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--get_validation_results", action=argparse.BooleanOptionalAction, default=False
    )

    parser.add_argument("--test_map_type", type=str, default="RandomGrid")
    parser.add_argument("--test_map_h", type=int, default=20)
    parser.add_argument("--test_map_w", type=int, default=20)
    parser.add_argument("--test_robot_density", type=float, default=0.025)
    parser.add_argument("--test_obstacle_density", type=float, default=0.1)
    parser.add_argument("--test_max_episode_steps", type=int, default=128)
    parser.add_argument("--test_obs_radius", type=int, default=3)
    parser.add_argument("--test_collision_system", type=str, default="soft")
    parser.add_argument("--test_on_target", type=str, default="nothing")

    parser.add_argument("--test_num_samples", type=int, default=2000)
    parser.add_argument("--test_dataset_seed", type=int, default=42)
    parser.add_argument("--test_dataset_dir", type=str, default="dataset")

    parser.add_argument("--test_comm_radius", type=int, default=7)
    parser.add_argument("--model_epoch_num", type=int, default=None)

    parser.add_argument("--test_name", type=str, default="in_distribution")
    parser.add_argument(
        "--test_wrt_expert", action=argparse.BooleanOptionalAction, default=False
    )

    args = parser.parse_args()
    print(args)

    assert args.save_termination_state

    if args.device == -1:
        device = torch.device("cuda")
    elif args.device is not None:
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")

    if args.test_wrt_expert:
        print("Loading Expert Solutions........")
        dict_args = vars(args)
        if args.test_in_distribution:
            train_id_max = int(
                args.num_samples * (1 - args.validation_fraction - args.test_fraction)
            )
            validation_id_max = train_id_max + int(
                args.num_samples * args.validation_fraction
            )
            if args.get_validation_results:
                dict_args["skip_n"] = train_id_max
                dict_args["subsample_n"] = validation_id_max - train_id_max
            else:
                dict_args["skip_n"] = validation_id_max
                dict_args["subsample_n"] = None
        else:
            for key in EXPERT_FILE_NAME_KEYS:
                if f"test_{key}" in dict_args:
                    dict_args[key] = dict_args[f"test_{key}"]
            dict_args["skip_n"] = None
            dict_args["subsample_n"] = None
        file_name = get_expert_file_name(dict_args)
        path = pathlib.Path(
            f"{args.dataset_dir}", "test_expert_metrics", f"{file_name}"
        )
        with open(path, "rb") as f:
            expert_success, expert_makespan, expert_total_flowtime = pickle.load(f)

    if args.test_in_distribution:
        num_agents = int(args.robot_density * args.map_h * args.map_w)
        comm_radius = args.comm_radius
        obs_radius = args.obs_radius

        if args.map_type == "RandomGrid":
            assert args.map_h == args.map_w, (
                f"Expect height and width of random grid to be the same, "
                f"but got height {args.map_h} and width {args.map_w}"
            )

            train_id_max = int(
                args.num_samples * (1 - args.validation_fraction - args.test_fraction)
            )
            validation_id_max = train_id_max + int(
                args.num_samples * args.validation_fraction
            )

            rng = np.random.default_rng(args.dataset_seed)
            seeds = rng.integers(10**10, size=args.num_samples)
            if args.get_validation_results:
                seeds = seeds[train_id_max:validation_id_max]
            else:
                seeds = seeds[validation_id_max:]

            if args.test_wrt_expert:

                def _grid_config_generator(seed, makespan):
                    return GridConfig(
                        num_agents=num_agents,  # number of agents
                        size=args.map_w,  # size of the grid
                        density=args.obstacle_density,  # obstacle density
                        seed=seed,  # set to None for random
                        # obstacles, agents and targets
                        # positions at each reset
                        max_episode_steps=3 * makespan,  # horizon
                        obs_radius=args.obs_radius,  # defines field of view
                        observation_type="MAPF",
                        collision_system=args.collision_system,
                        on_target=args.on_target,
                    )

            else:

                def _grid_config_generator(seed):
                    return GridConfig(
                        num_agents=num_agents,  # number of agents
                        size=args.map_w,  # size of the grid
                        density=args.obstacle_density,  # obstacle density
                        seed=seed,  # set to None for random
                        # obstacles, agents and targets
                        # positions at each reset
                        max_episode_steps=args.max_episode_steps,  # horizon
                        obs_radius=args.obs_radius,  # defines field of view
                        observation_type="MAPF",
                        collision_system=args.collision_system,
                        on_target=args.on_target,
                    )

        else:
            raise ValueError(f"Unsupported map type: {args.map_type}.")
    else:
        num_agents = int(args.test_robot_density * args.test_map_h * args.test_map_w)
        comm_radius = args.test_comm_radius
        obs_radius = args.test_obs_radius

        if args.test_map_type == "RandomGrid":
            assert args.test_map_h == args.test_map_w, (
                f"Expect height and width of random grid to be the same, "
                f"but got height {args.test_map_h} and width {args.test_map_w}"
            )

            rng = np.random.default_rng(args.test_dataset_seed)
            seeds = rng.integers(10**10, size=args.test_num_samples)

            if args.test_wrt_expert:

                def _grid_config_generator(seed, makespan):
                    return GridConfig(
                        num_agents=num_agents,  # number of agents
                        size=args.test_map_w,  # size of the grid
                        density=args.test_obstacle_density,  # obstacle density
                        seed=seed,  # set to None for random
                        # obstacles, agents and targets
                        # positions at each reset
                        max_episode_steps=3 * makespan,  # horizon
                        obs_radius=args.test_obs_radius,  # defines field of view
                        observation_type="MAPF",
                        collision_system=args.test_collision_system,
                        on_target=args.test_on_target,
                    )

            else:

                def _grid_config_generator(seed):
                    return GridConfig(
                        num_agents=num_agents,  # number of agents
                        size=args.test_map_w,  # size of the grid
                        density=args.test_obstacle_density,  # obstacle density
                        seed=seed,  # set to None for random
                        # obstacles, agents and targets
                        # positions at each reset
                        max_episode_steps=args.test_max_episode_steps,  # horizon
                        obs_radius=args.test_obs_radius,  # defines field of view
                        observation_type="MAPF",
                        collision_system=args.test_collision_system,
                        on_target=args.test_on_target,
                    )

        else:
            raise ValueError(f"Unsupported map type: {args.test_map_type}.")

    if args.imitation_learning_model == "MAGAT":
        model = DecentralPlannerGATNet(
            FOV=args.obs_radius,
            numInputFeatures=args.embedding_size,
            nGraphFilterTaps=args.num_gnn_layers,
            nAttentionHeads=args.num_attention_heads,
            use_dropout=True,
            CNN_mode=args.cnn_mode,
            attentionMode=args.attention_mode,
            AttentionConcat=True,
        ).to(device)
    else:
        raise ValueError(
            f"Unsupported imitation learning model {args.imitation_learning_model}."
        )

    run_name = f"{args.test_name}_{args.run_name}"
    wandb.init(
        project="hyper-mapf-pogema-test",
        name=run_name,
        config=vars(args),
        entity="jainris",
    )

    if args.model_epoch_num is None:
        checkpoint_path = pathlib.Path(args.checkpoints_dir, "best.pt")
    else:
        checkpoint_path = pathlib.Path(
            args.checkpoints_dir, f"epoch_{args.model_epoch_num}.pt"
        )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.eval()

    def aux_func(env, observations, actions):
        if actions is None:
            aux_func.original_pos = np.array([obs["global_xy"] for obs in observations])
            aux_func.makespan = 0
            aux_func.flowtime = np.zeros(env.get_num_agents())
        else:
            new_pos = np.array([obs["global_xy"] for obs in observations])
            aux_func.makespan += 1
            aux_func.flowtime += np.any(aux_func.original_pos != new_pos, axis=-1)
            aux_func.original_pos = new_pos

    num_completed = 0
    num_tested = 0

    all_makespan = []
    all_total_flowtime = []
    all_partial_success_rate = []

    for i, seed in enumerate(seeds):
        if args.test_wrt_expert:
            if not expert_success[i]:
                print(f"Expert failed on map {i}, skipping it")
                continue
            grid_config = _grid_config_generator(seed, expert_makespan[i])
        else:
            grid_config = _grid_config_generator(seed)
        success, env, _ = run_model_on_grid(
            model,
            comm_radius,
            obs_radius,
            grid_config,
            device,
            aux_func=aux_func,
            collision_shielding=args.collision_shielding,
            action_sampling=args.action_sampling,
        )
        makespan = aux_func.makespan
        flowtime = aux_func.flowtime

        flowtime[~np.array(env.was_on_goal)] = makespan

        num_tested += 1
        if success:
            num_completed += 1
        success_rate = num_completed / num_tested
        total_flowtime = np.sum(flowtime)
        partial_success_rate = np.mean(env.was_on_goal)

        all_makespan.append(makespan)
        all_total_flowtime.append(total_flowtime)
        all_partial_success_rate.append(partial_success_rate)

        results = {
            "success_rate": success_rate,
            "average_makespan": np.mean(all_makespan),
            "average_total_flowtime": np.mean(all_total_flowtime),
            "average_partial_success_rate": np.mean(all_partial_success_rate),
            "seed": seed,
            "success": success,
            "makespan": makespan,
            "total_flowtime": total_flowtime,
            "partial_success_rate": partial_success_rate,
        }
        if args.test_wrt_expert:
            relative_flowtime_increase = (
                total_flowtime - expert_total_flowtime[i]
            ) / expert_total_flowtime[i]
            results = results | {
                "relative_flowtime_increase": relative_flowtime_increase
            }

        wandb.log(results)

        print(
            f"Testing Graph {i + 1}/{len(seeds)}, "
            f"Current Success Rate: {success_rate}"
        )


if __name__ == "__main__":
    main()
