import argparse
import pickle
import pathlib
import numpy as np
import torch

from scipy.spatial.distance import squareform, pdist

from run_expert import DATASET_FILE_NAME_KEYS, add_expert_dataset_args


def get_imitation_dataset_file_name(args):
    file_name = ""
    dict_args = vars(args)
    for key in sorted(DATASET_FILE_NAME_KEYS):
        file_name += f"_{key}_{dict_args[key]}"
    if "load_positions_separately" in dict_args:
        if (not dict_args["load_positions_separately"]) and args.use_edge_attr:
            file_name += "_pos"
    elif args.use_edge_attr:
        file_name += "_pos"
    file_name = file_name[1:] + ".pkl"
    return file_name


def generate_graph_dataset(
    dataset,
    comm_radius,
    obs_radius,
    num_samples,
    save_termination_state,
    use_edge_attr=False,
    print_prefix="",
):
    dataset_node_features = []
    dataset_Adj = []
    dataset_target_actions = []
    dataset_terminated = []
    graph_map_id = []
    dataset_agent_pos = []

    assert save_termination_state, "Only support saving termination state for now"

    for id, (sample_observations, actions, terminated) in enumerate(dataset):
        if print_prefix is not None:
            print(
                f"{print_prefix}"
                f"Generating Graph Dataset for map {id + 1}/{num_samples}"
            )
        for observations in sample_observations:
            global_xys = np.array([obs["global_xy"] for obs in observations])

            Adj = squareform(pdist(global_xys, "euclidean"))
            mask = Adj <= comm_radius
            Adj = Adj * mask

            Adj = Adj - np.diag(np.diag(Adj))

            if use_edge_attr:
                dataset_agent_pos.append(global_xys)

            node_features = []
            for observation in observations:
                obs_obstacle = np.pad(observation["obstacles"], 1)
                obs_agent = np.pad(observation["agents"], 1)

                obs_goal = np.zeros_like(obs_obstacle)
                centre = (obs_goal.shape[0] // 2, obs_goal.shape[1] // 2)
                # goal = observation["target_xy"]
                # goal = np.array(observation['global_target_xy']) - np.array(observation['global_xy'])
                goal = tuple(
                    (tpos - apos)
                    for tpos, apos in zip(
                        observation["global_target_xy"], observation["global_xy"]
                    )
                )

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
        dataset_terminated.extend(terminated)

    dataset_node_features = np.stack(dataset_node_features)
    dataset_Adj = np.stack(dataset_Adj)
    dataset_target_actions = np.stack(dataset_target_actions)
    dataset_terminated = np.stack(dataset_terminated)
    graph_map_id = np.array(graph_map_id)

    result = (
        dataset_node_features,
        dataset_Adj,
        dataset_target_actions,
        dataset_terminated,
        graph_map_id,
    )
    if use_edge_attr:
        dataset_agent_pos = np.stack(dataset_agent_pos)
        result = (*result, dataset_agent_pos)

    return tuple(torch.from_numpy(res) for res in result)


def main():
    parser = argparse.ArgumentParser(
        description="Convert to Imitation Learning Dataset"
    )
    parser = add_expert_dataset_args(parser)

    parser.add_argument("--comm_radius", type=int, default=7)
    parser.add_argument("--use_edge_attr", action="store_true", default=False)

    args = parser.parse_args()
    print(args)

    file_name = ""
    dict_args = vars(args)
    for key in sorted(DATASET_FILE_NAME_KEYS):
        file_name += f"_{key}_{dict_args[key]}"
    file_name = file_name[1:] + ".pkl"

    path = pathlib.Path(f"{args.dataset_dir}", "raw_expert_predictions", f"{file_name}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "rb") as f:
        dataset = pickle.load(f)
    if isinstance(dataset, tuple):
        dataset, seed_mask = dataset

    graph_dataset = generate_graph_dataset(
        dataset,
        args.comm_radius,
        args.obs_radius,
        args.num_samples,
        args.save_termination_state,
        args.use_edge_attr,
    )

    file_name = get_imitation_dataset_file_name(args)
    path = pathlib.Path(f"{args.dataset_dir}", "processed_dataset", f"{file_name}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump((graph_dataset), f)


if __name__ == "__main__":
    main()
