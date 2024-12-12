import argparse
import pickle
import pathlib
import numpy as np
import wandb

from pogema import pogema_v0, GridConfig

from lacam.inference import LacamInference, LacamInferenceConfig

EXPERT_FILE_NAME_KEYS = [
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
    "skip_n",
    "subsample_n",
]


def get_expert_file_name(dict_args):
    file_name = ""
    for key in sorted(EXPERT_FILE_NAME_KEYS):
        if dict_args[key] is not None:
            file_name += f"_{key}_{dict_args[key]}"
    file_name = file_name[1:] + ".pkl"
    return file_name


def add_expert_dataset_args(parser):
    parser.add_argument("--expert_algorithm", type=str, default="LaCAM")

    parser.add_argument("--map_type", type=str, default="RandomGrid")
    parser.add_argument("--map_h", type=int, default=20)
    parser.add_argument("--map_w", type=int, default=20)
    parser.add_argument("--robot_density", type=float, default=0.025)
    parser.add_argument("--obstacle_density", type=float, default=0.1)
    parser.add_argument("--max_episode_steps", type=int, default=128)
    parser.add_argument("--obs_radius", type=int, default=4)
    parser.add_argument("--collision_system", type=str, default="soft")
    parser.add_argument("--on_target", type=str, default="nothing")

    parser.add_argument("--num_samples", type=int, default=2000)
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

    makespan = 0
    flowtime = np.zeros(env.get_num_agents())

    while True:
        actions = expert.act(observations)

        original_pos = np.array([obs["global_xy"] for obs in observations])

        observations, rewards, terminated, truncated, infos = env.step(actions)

        new_pos = np.array([obs["global_xy"] for obs in observations])

        makespan += 1
        flowtime += np.all(original_pos != new_pos, axis=-1)

        if all(terminated) or all(truncated):
            break

    return all(terminated), makespan, np.sum(flowtime)


def main():
    parser = argparse.ArgumentParser(description="Run Expert")
    parser = add_expert_dataset_args(parser)

    parser.add_argument("--test_name", type=str, default="in_distribution")
    parser.add_argument("--skip_n", type=int, default=None)
    parser.add_argument("--subsample_n", type=int, default=None)

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
        if args.skip_n is not None:
            seeds = seeds[args.skip_n :]
        if args.subsample_n is not None:
            seeds = seeds[: args.subsample_n]

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
    all_success, all_makespan, all_total_flowtime = [], [], []
    num_samples = len(grid_configs)
    for i, grid_config in enumerate(grid_configs):
        print(f"Running expert on map {i + 1}/{num_samples}", end=" ")
        expert = expert_algorithm(inference_config)

        success, makespan, total_flowtime = run_expert_algorithm(
            expert,
            grid_config=grid_config,
        )
        all_success.append(success)
        all_makespan.append(makespan)
        all_total_flowtime.append(total_flowtime)

        if success:
            num_success += 1

        success_rate = num_success / (i + 1)

        print(f"-- Success Rate: {success_rate}")
        wandb.log(
            {
                "success_rate": success_rate,
                "average_makespan": np.mean(all_makespan),
                "average_total_flowtime": np.mean(all_total_flowtime),
            }
        )

    file_name = get_expert_file_name(vars(args))
    path = pathlib.Path(f"{args.dataset_dir}", "test_expert_metrics", f"{file_name}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump((all_success, all_makespan, all_total_flowtime), f)


if __name__ == "__main__":
    main()
