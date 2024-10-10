import argparse
import pickle
import pathlib
import numpy as np
import torch

from scipy.spatial.distance import squareform, pdist


def generate_graph_dataset(dataset, comm_radius, obs_radius):
    dataset_node_features = []
    dataset_Adj = []
    dataset_target_actions = []
    graph_map_id = []

    for id, (sample_observations, actions) in enumerate(dataset):
        # for observations, actions in zip(sample_observations, sample_actions):
        for observations in sample_observations:
            global_xys = np.array([obs["global_xy"] for obs in observations])

            Adj = squareform(pdist(global_xys, "euclidean")) <= comm_radius
            Adj = Adj.astype(global_xys.dtype)

            Adj = Adj - np.diag(np.diag(Adj))

            node_features = []
            for observation in observations:
                obs_obstacle = np.pad(observation["obstacles"], 1)
                obs_agent = np.pad(observation["agents"], 1)

                obs_goal = np.zeros_like(obs_obstacle)
                centre = (obs_goal.shape[0] // 2, obs_goal.shape[1] // 2)
                goal = observation["target_xy"]

                if np.all(np.abs(goal) <= obs_radius):
                    obs_goal[centre[0] + goal[0], centre[1] + goal[1]] = 1.0
                else:
                    angle = np.arctan2(goal[1], goal[0])
                    goal_sign = np.sign(goal)
                    dist = obs_goal.shape[0] // 2

                    if (angle >= np.pi / 4 and angle <= np.pi * 3 / 4) or (
                        angle >= -np.pi * (3 / 4) and angle <= -np.pi / 4
                    ):
                        goalY_FOV = int(dist * (goal_sign[1] + 1))
                        goalX_FOV = int(
                            centre[0] + np.round(dist * goal[0] / np.abs(goal[1]))
                        )
                    else:
                        goalX_FOV = int(dist * (goal_sign[0] + 1))
                        goalY_FOV = int(
                            centre[1] + np.round(dist * goal[1] / np.abs(goal[0]))
                        )

                    obs_goal[goalX_FOV, goalY_FOV] = 1.0
                node_features.append(np.stack([obs_obstacle, obs_agent, obs_goal]))
            node_features = np.stack(node_features)

            dataset_node_features.append(node_features)
            dataset_Adj.append(Adj)
            graph_map_id.append(id)
        dataset_target_actions.extend(actions)

    dataset_node_features = np.stack(dataset_node_features)
    dataset_Adj = np.stack(dataset_Adj)
    dataset_target_actions = np.stack(dataset_target_actions)
    graph_map_id = np.array(graph_map_id)

    result = (dataset_node_features, dataset_Adj, dataset_target_actions, graph_map_id)

    return tuple(torch.from_numpy(res) for res in result)


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

    parser.add_argument("--comm_radius", type=int, default=7)
    parser.add_argument("--dynamic_comm_radius", action="store_true", default=False)

    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--dataset_seed", type=int, default=42)
    parser.add_argument("--dataset_dir", type=str, default="dataset")

    args = parser.parse_args()
    print(args)

    file_name = ""
    dict_args = vars(args)
    for key in sorted(dict_args.keys()):
        if key in ["dataset_dir", "comm_radius", "dynamic_comm_radius"]:
            continue
        file_name += f"_{key}_{dict_args[key]}"
    file_name = file_name[1:] + ".pkl"

    path = pathlib.Path(f"{args.dataset_dir}", "raw_expert_predictions", f"{file_name}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "rb") as f:
        dataset = pickle.load(f)

    graph_dataset = generate_graph_dataset(dataset, args.comm_radius, args.obs_radius)

    path = pathlib.Path(f"{args.dataset_dir}", "processed_dataset", f"{file_name}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(graph_dataset, f)


if __name__ == "__main__":
    main()
