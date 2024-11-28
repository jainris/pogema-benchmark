import argparse
import pickle
import pathlib
import numpy as np
import wandb

from pogema import pogema_v0, GridConfig

from lacam.inference import LacamInference, LacamInferenceConfig

DATASET_FILE_NAME_KEYS = [
    "expert_algorithm",
    "map_type",
    "map_h",
    "map_w",
    "robot_density",
    "obstacle_density",
    "max_episode_steps",
    "obs_radius",
    "num_samples",
    "dataset_seed",
    "save_termination_state",
    "collision_system",
    "on_target",
]


def add_expert_dataset_args(parser):
    parser.add_argument("--expert_algorithm", type=str, default="LaCAM")

    parser.add_argument("--map_type", type=str, default="RandomGrid")
    parser.add_argument("--map_h", type=int, default=20)
    parser.add_argument("--map_w", type=int, default=20)
    parser.add_argument("--robot_density", type=float, default=0.025)
    parser.add_argument("--obstacle_density", type=float, default=0.1)
    parser.add_argument("--max_episode_steps", type=int, default=128)
    parser.add_argument("--obs_radius", type=int, default=3)
    parser.add_argument("--collision_system", type=str, default="soft")
    parser.add_argument("--on_target", type=str, default="nothing")

    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--dataset_seed", type=int, default=42)
    parser.add_argument("--dataset_dir", type=str, default="dataset")

    parser.add_argument(
        "--save_termination_state", action=argparse.BooleanOptionalAction, default=False
    )

    return parser


def run_expert_algorithm(expert, env=None, observations=None, grid_config=None):
    if env is None:
        env = pogema_v0(grid_config=grid_config)
        observations, infos = env.reset()

    while True:
        actions = expert.act(observations)

        observations, rewards, terminated, truncated, infos = env.step(actions)

        if all(terminated) or all(truncated):
            break

    return all(terminated)


def main():
    parser = argparse.ArgumentParser(description="Run Expert")
    parser = add_expert_dataset_args(parser)

    parser.add_argument("--test_name", type=str, default="in_distribution")

    args = parser.parse_args()
    print(args)

    num_agents = int(args.robot_density * args.map_h * args.map_w)

    if args.map_type == "RandomGrid":
        assert args.map_h == args.map_w, (
            f"Expect height and width of random grid to be the same, "
            f"but got height {args.map_h} and width {args.map_w}"
        )

        rng = np.random.default_rng(args.dataset_seed)
        seeds = rng.integers(10**10, size=args.num_samples)

        grid_configs = []

        for seed in seeds:
            grid_config = GridConfig(
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
            grid_configs.append(grid_config)
    else:
        raise ValueError(f"Unsupported map type: {args.map_type}.")

    if args.expert_algorithm == "LaCAM":
        inference_config = LacamInferenceConfig()
        expert_algorithm = LacamInference
    else:
        raise ValueError(f"Unsupported expert algorithm {args.expert_algorithm}.")

    run_name = f"{args.test_name}_{args.expert_algorithm}"
    wandb.init(
        project="hyper-mapf-pogema-test",
        name=run_name,
        config=vars(args),
        entity="jainris",
    )

    num_success = 0
    for i, grid_config in enumerate(grid_configs):
        print(f"Running expert on map {i + 1}/{args.num_samples}", end=" ")
        expert = expert_algorithm(inference_config)

        success = run_expert_algorithm(
            expert,
            grid_config=grid_config,
            save_termination_state=args.save_termination_state,
        )

        if success:
            num_success += 1

        success_rate = num_success / (i + 1)

        print(f"-- Success Rate: {success_rate}")
        wandb.log({"success_rate": success_rate})


if __name__ == "__main__":
    main()
