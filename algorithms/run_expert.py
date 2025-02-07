import argparse
import pickle
import pathlib
import numpy as np

from pogema import pogema_v0, GridConfig

from grid_config_generator import add_grid_config_args, grid_config_generator_factory

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
    "min_dist": None,
}

DATASET_FILE_NAME_KEYS = list(DATASET_FILE_NAME_DEFAULT.keys())


def add_expert_dataset_args(parser):
    parser.add_argument("--expert_algorithm", type=str, default="LaCAM")

    parser = add_grid_config_args(parser)

    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--dataset_seed", type=int, default=42)
    parser.add_argument("--dataset_dir", type=str, default="dataset")

    parser.add_argument(
        "--save_termination_state", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--pibt_expert_relevance_training",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    return parser


def get_legacy_expert_dataset_file_name(args):
    file_name = ""
    dict_args = vars(args)
    for key in sorted(DATASET_FILE_NAME_KEYS):
        if (key == "min_dist") and (dict_args[key] is None):
            continue
        file_name += f"_{key}_{dict_args[key]}"
    if args.pibt_expert_relevance_training:
        file_name += "_pibt_relevance"
    file_name = file_name[1:] + ".pkl"
    return file_name


def get_expert_dataset_file_name(args):
    file_name = ""
    dict_args = vars(args)
    for key in sorted(DATASET_FILE_NAME_KEYS):
        if (key == "min_dist") and (dict_args[key] is None):
            continue
        if dict_args[key] != DATASET_FILE_NAME_DEFAULT[key]:
            file_name += f"_{key}_{dict_args[key]}"
    if args.pibt_expert_relevance_training:
        file_name += "_pibt_relevance"
    if len(file_name) > 0:
        file_name = file_name[1:] + ".pkl"
    else:
        file_name = "default.pkl"
    return file_name


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
    elif args.expert_algorithm == "PIBT":
        from pibt.inference import PIBTInference, PIBTInferenceConfig

        inference_config = PIBTInferenceConfig()
        expert_algorithm = PIBTInference
    elif args.expert_algorithm == "PIBTDistance":
        from pibt.inference import PIBTDistanceBasedInference, PIBTInferenceConfig

        inference_config = PIBTInferenceConfig()
        expert_algorithm = PIBTDistanceBasedInference
    elif args.expert_algorithm[: len("MAPF-GPT")] == "MAPF-GPT":
        from gpt.inference import MAPFGPTInference, MAPFGPTInferenceConfig

        num_agents = int(args.robot_density * args.map_h * args.map_w)

        if args.expert_algorithm == "MAPF-GPT":
            model_weight = "6M"
        else:
            model_weight = args.expert_algorithm[len("MAPF-GPT") + 1 :]
        inference_config = MAPFGPTInferenceConfig(
            path_to_weights=f"weights/model-{model_weight}.pt", num_agents=num_agents
        )
        expert_algorithm = MAPFGPTInference
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

    expert_algorithm, inference_config = get_expert_algorithm_and_config(args)

    dataset = []
    seed_mask = []
    num_success = 0
    for i, seed in enumerate(seeds):
        grid_config = _grid_config_generator(seed)
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

    print(f"{len(dataset)}/{len(seeds)} samples were successfully added to the dataset")

    file_name = get_expert_dataset_file_name(args)
    path = pathlib.Path(f"{args.dataset_dir}", "raw_expert_predictions", f"{file_name}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump((dataset, seed_mask), f)


if __name__ == "__main__":
    main()
