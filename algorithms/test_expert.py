import argparse
import pickle
import pathlib
import numpy as np
import wandb

from pogema import pogema_v0, GridConfig

from run_expert import get_expert_algorithm_and_config
from grid_config_generator import add_grid_config_args, grid_config_generator_factory

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

    parser = add_grid_config_args(parser)

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

    expert.reset_states(env)

    while True:
        actions = expert.act(observations)

        original_pos = np.array([obs["global_xy"] for obs in observations])

        observations, rewards, terminated, truncated, infos = env.step(actions)

        new_pos = np.array([obs["global_xy"] for obs in observations])

        makespan += 1
        flowtime += np.any(original_pos != new_pos, axis=-1)

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

    grid_configs = []

    if args.skip_n is not None:
        seeds = seeds[args.skip_n:]
    if args.subsample_n is not None:
        seeds = seeds[:args.subsample_n]

    for seed in seeds:
        grid_configs.append(_grid_config_generator(seed))

    expert_algorithm, inference_config = get_expert_algorithm_and_config(args)

    run_name = f"{args.test_name}_{args.expert_algorithm}"
    wandb.init(
        project="hyper-mapf-pogema-test",
        name=run_name,
        config=vars(args) | {"expert": True},
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
                "seed": grid_config.seed,
                "success": success,
                "makespan": makespan,
                "total_flowtime": total_flowtime,
            }
        )

    file_name = get_expert_file_name(vars(args))
    path = pathlib.Path(f"{args.dataset_dir}", "test_expert_metrics", f"{file_name}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump((all_success, all_makespan, all_total_flowtime), f)


if __name__ == "__main__":
    main()
