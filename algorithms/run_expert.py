import argparse
import pickle
import pathlib
import numpy as np

from pogema import pogema_v0, GridConfig

from lacam.inference import LacamInference, LacamInferenceConfig


def main():
    parser = argparse.ArgumentParser(description="Run Expert")
    parser.add_argument("--expert_algorithm", type=str, default="LaCAM")

    parser.add_argument("--map_type", type=str, default="RandomGrid")
    parser.add_argument("--map_h", type=int, default=20)
    parser.add_argument("--map_w", type=int, default=20)
    parser.add_argument("--robot_density", type=float, default=0.025)
    parser.add_argument("--obstacle_density", type=float, default=0.1)
    parser.add_argument("--max_episode_steps", type=int, default=128)
    parser.add_argument("--obs_radius", type=int, default=3)

    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--dataset_seed", type=int, default=42)
    parser.add_argument("--dataset_dir", type=str, default="dataset")

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
            )
            grid_configs.append(grid_config)
    else:
        raise ValueError(f"Unsupported map type: {args.map_type}.")

    if args.expert_algorithm == "LaCAM":
        inference_config = LacamInferenceConfig()
        expert_algorithm = LacamInference
    else:
        raise ValueError(f"Unsupported expert algorithm {args.expert_algorithm}.")

    dataset = []
    for i, grid_config in enumerate(grid_configs):
        print(f"Running expert on map {i + 1}/{args.num_samples}")
        expert = expert_algorithm(inference_config)

        env = pogema_v0(grid_config=grid_config)
        observations, infos = env.reset()

        all_actions = []
        all_observations = []

        while True:
            actions = expert.act(observations)

            all_observations.append(observations)
            all_actions.append(actions)

            observations, rewards, terminated, truncated, infos = env.step(actions)

            if all(terminated) or all(truncated)::
                break

        dataset.append((all_observations, all_actions))

    file_name = ""
    dict_args = vars(args)
    for key in sorted(dict_args.keys()):
        if key == "dataset_dir":
            continue
        file_name += f"{key}_{dict_args[key]}"

    path = pathlib.Path(f"{args.dataset_dir}", "raw_expert_predictions", f"{file_name}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb"):
        pickle.dump(dataset, path)


if __name__ == "__main__":
    main()
