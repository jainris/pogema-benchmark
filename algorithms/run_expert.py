import argparse
import pickle
import pathlib
import numpy as np

from pogema import pogema_v0, GridConfig

DATASET_FILE_NAME_DEFAULT = {
    "expert_algorithm": "LaCAM",
    "map_type": "RandomGrid",
    "map_h": 20,
    "map_w": 20,
    "robot_density": 0.025,
    "obstacle_density": 0.1,
    "max_episode_steps": 128,
    "obs_radius": 4,
    "num_samples": 30000,
    "dataset_seed": 42,
    "save_termination_state": True,
    "collision_system": "soft",
    "on_target": "nothing",
}

DATASET_FILE_NAME_KEYS = list(DATASET_FILE_NAME_DEFAULT.keys())


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

    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--dataset_seed", type=int, default=42)
    parser.add_argument("--dataset_dir", type=str, default="dataset")

    parser.add_argument(
        "--save_termination_state", action=argparse.BooleanOptionalAction, default=False
    )

    return parser


class ExpertWrapper:
    def __init__(self, base_obj):
        self.base_obj = base_obj

    def reset_states(self, env):
        self.base_obj.reset_states()

    def __getattr__(self, name):
        return getattr(self.base_obj, name)


def wrapped_class(cls):
    def _get_wrapped_class(config):
        return ExpertWrapper(cls(config))

    return _get_wrapped_class


def get_expert_algorithm_and_config(args):
    if args.expert_algorithm == "LaCAM":
        from lacam.inference import LacamInference, LacamInferenceConfig

        inference_config = LacamInferenceConfig()
        expert_algorithm = wrapped_class(LacamInference)
    elif args.expert_algorithm == "DCC":
        from dcc.inference import DCCInference, DCCInferenceConfig

        inference_config = DCCInferenceConfig()
        expert_algorithm = wrapped_class(DCCInference)
    elif args.expert_algorithm == "ECBS":
        from ecbs.inference import ECBSInference, ECBSInferenceConfig

        inference_config = ECBSInferenceConfig()
        expert_algorithm = ECBSInference
    else:
        raise ValueError(f"Unsupported expert algorithm {args.expert_algorithm}.")
    return expert_algorithm, inference_config


def run_expert_algorithm(
    expert,
    env=None,
    observations=None,
    grid_config=None,
    save_termination_state=True,
    additional_data_func=None,
):
    if env is None:
        env = pogema_v0(grid_config=grid_config)
        observations, infos = env.reset()

    all_actions = []
    all_observations = []
    all_terminated = []
    additional_data = []

    expert.reset_states(env)

    while True:
        actions = expert.act(observations)

        all_observations.append(observations)
        all_actions.append(actions)

        if additional_data_func is not None:
            additional_data.append(
                additional_data_func(
                    env=env, observations=observations, actions=actions
                )
            )

        observations, rewards, terminated, truncated, infos = env.step(actions)

        if save_termination_state:
            all_terminated.append(terminated)

        if all(terminated) or all(truncated):
            break

    if additional_data_func is not None:
        return all_actions, all_observations, all_terminated, additional_data
    return all_actions, all_observations, all_terminated


def main():
    parser = argparse.ArgumentParser(description="Run Expert")
    parser = add_expert_dataset_args(parser)

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

    expert_algorithm, inference_config = get_expert_algorithm_and_config(args)

    dataset = []
    seed_mask = []
    num_success = 0
    for i, grid_config in enumerate(grid_configs):
        print(f"Running expert on map {i + 1}/{args.num_samples}", end=" ")
        expert = expert_algorithm(inference_config)

        all_actions, all_observations, all_terminated = run_expert_algorithm(
            expert,
            grid_config=grid_config,
            save_termination_state=args.save_termination_state,
        )

        if all(all_terminated[-1]):
            seed_mask.append(True)
            num_success += 1
            if args.save_termination_state:
                dataset.append((all_observations, all_actions, all_terminated))
            else:
                dataset.append((all_observations, all_actions))
        else:
            seed_mask.append(False)

        print(f"-- Success Rate: {num_success / (i + 1)}")

    print(
        f"{len(dataset)}/{len(grid_configs)} samples were successfully added to the dataset"
    )

    file_name = ""
    dict_args = vars(args)
    for key in sorted(DATASET_FILE_NAME_KEYS):
        file_name += f"_{key}_{dict_args[key]}"
    file_name = file_name[1:] + ".pkl"

    path = pathlib.Path(f"{args.dataset_dir}", "raw_expert_predictions", f"{file_name}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump((dataset, seed_mask), f)


if __name__ == "__main__":
    main()
