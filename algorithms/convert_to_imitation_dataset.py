import argparse
import pickle
import pathlib
import numpy as np
import torch

from scipy.spatial.distance import squareform, pdist

from run_expert import (
    DATASET_FILE_NAME_DEFAULT,
    DATASET_FILE_NAME_KEYS,
    add_expert_dataset_args,
    get_expert_dataset_file_name,
    get_legacy_expert_dataset_file_name,
)


def add_imitation_dataset_args(parser):
    parser.add_argument("--comm_radius", type=int, default=7)
    parser.add_argument("--batched_saving", type=int, default=None)
    parser.add_argument("--num_neighbour_cutoff", type=int, default=None)
    parser.add_argument("--neighbour_cutoff_method", type=str, default="closest")
    return parser


def get_legacy_imitation_dataset_file_name(args):
    file_name = ""
    dict_args = vars(args)
    load_positions_separately = dict_args.get("load_positions_separately", False)
    use_edge_attr = dict_args.get("use_edge_attr", False)

    for key in sorted(DATASET_FILE_NAME_KEYS):
        if (key == "min_dist") and (dict_args[key] is None):
            continue
        file_name += f"_{key}_{dict_args[key]}"
    if args.pibt_expert_relevance_training:
        file_name += "_pibt_relevance"
    if (not load_positions_separately) and use_edge_attr:
        file_name += "_pos"
    file_name = file_name[1:] + ".pkl"
    return file_name


def get_imitation_dataset_file_name(args):
    file_name = ""
    dict_args = vars(args)
    load_positions_separately = dict_args.get("load_positions_separately", False)
    use_edge_attr = dict_args.get("use_edge_attr", False)

    for key in sorted(DATASET_FILE_NAME_KEYS):
        if (key == "min_dist") and (dict_args[key] is None):
            continue
        if dict_args[key] != DATASET_FILE_NAME_DEFAULT[key]:
            file_name += f"_{key}_{dict_args[key]}"
    if args.pibt_expert_relevance_training:
        file_name += "_pibt_relevance"
    if (not load_positions_separately) and use_edge_attr:
        file_name += "_pos"
    if args.num_neighbour_cutoff is not None:
        file_name += f"_{args.num_neighbour_cutoff}_{args.neighbour_cutoff_method}_neighbour_cutoff"
    if len(file_name) > 0:
        file_name = file_name[1:] + ".pkl"
    else:
        file_name = "default.pkl"
    return file_name


def generate_graph_dataset(
    dataset,
    comm_radius,
    obs_radius,
    num_samples,
    save_termination_state,
    use_edge_attr=False,
    print_prefix="",
    id_offset=0,
    num_neighbour_cutoff=None,
    neighbour_cutoff_method=None,
    stack_with_np=True,
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

            if num_neighbour_cutoff is not None:
                if neighbour_cutoff_method == "closest":
                    idx = np.argsort(Adj, axis=-1)
                elif neighbour_cutoff_method == "random":
                    vals = (Adj > 0) * np.random.rand(*Adj.shape)
                    idx = np.argsort(-vals, axis=-1)
                else:
                    raise ValueError(
                        f"Unsupported neighbour_cutoff_method: {neighbour_cutoff_method}."
                    )
                idx = idx[:, num_neighbour_cutoff:]
                np.put_along_axis(Adj, idx, values=0, axis=-1)

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
            graph_map_id.append(id + id_offset)
        dataset_target_actions.extend(actions)
        dataset_terminated.extend(terminated)

    graph_map_id = np.array(graph_map_id)

    if not stack_with_np:
        result = (
            dataset_node_features,
            dataset_Adj,
            dataset_target_actions,
            dataset_terminated,
            graph_map_id,
        )
        if use_edge_attr:
            result = (*result, dataset_agent_pos)
        torch_results = []
        for res in result:
            torch_res = [torch.from_numpy(np.array(data)) for data in res]
            torch_results.append(torch_res)
        return torch_results

    dataset_node_features = np.stack(dataset_node_features)
    dataset_Adj = np.stack(dataset_Adj)
    dataset_target_actions = np.stack(dataset_target_actions)
    dataset_terminated = np.stack(dataset_terminated)

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


def batched_generate_graph_dataset(
    dataset,
    comm_radius,
    obs_radius,
    num_samples,
    save_termination_state,
    dir_path,
    batch_size,
    use_edge_attr=False,
    print_prefix="",
    num_neighbour_cutoff=None,
    neighbour_cutoff_method=None,
):
    batch_id = 0
    batched_results = None
    for id, data in enumerate(dataset):
        if print_prefix is not None:
            print(
                f"{print_prefix}"
                f"Generating Graph Dataset for map {id + 1}/{num_samples}"
            )

        results = generate_graph_dataset(
            dataset=[data],
            comm_radius=comm_radius,
            obs_radius=obs_radius,
            num_samples=num_samples,
            save_termination_state=save_termination_state,
            use_edge_attr=use_edge_attr,
            print_prefix=None,
            id_offset=id,
            num_neighbour_cutoff=num_neighbour_cutoff,
            neighbour_cutoff_method=neighbour_cutoff_method,
        )
        if batched_results is None:
            batched_results = results
        else:
            batched_results = tuple(
                torch.cat((br, r), dim=0) for (br, r) in zip(batched_results, results)
            )

        while batched_results[0].shape[0] >= batch_size:
            results = batched_results[:batch_size]
            batched_results = batched_results[batch_size:]

            path = dir_path / f"batch_{batch_id}.pkl"
            with open(path, "wb") as f:
                pickle.dump(results, f)

            batch_id += 1
    if batched_results[0].shape[0] > 0:
        path = dir_path / f"batch_{batch_id}.pkl"
        with open(path, "wb") as f:
            pickle.dump(batched_results, f)


def main():
    parser = argparse.ArgumentParser(
        description="Convert to Imitation Learning Dataset"
    )
    parser = add_expert_dataset_args(parser)
    parser = add_imitation_dataset_args(parser)
    parser.add_argument("--use_edge_attr", action="store_true", default=False)

    args = parser.parse_args()
    print(args)

    try:
        file_name = get_expert_dataset_file_name(args)
        path = pathlib.Path(
            f"{args.dataset_dir}", "raw_expert_predictions", f"{file_name}"
        )

        with open(path, "rb") as f:
            dataset = pickle.load(f)
    except:
        print(f"Could not find file: {path}, trying legacy file name.")
        file_name = get_legacy_expert_dataset_file_name(args)
        path = pathlib.Path(
            f"{args.dataset_dir}", "raw_expert_predictions", f"{file_name}"
        )

        with open(path, "rb") as f:
            dataset = pickle.load(f)

    if isinstance(dataset, tuple):
        dataset, seed_mask = dataset

    if args.batched_saving is None:
        graph_dataset = generate_graph_dataset(
            dataset,
            args.comm_radius,
            args.obs_radius,
            args.num_samples,
            args.save_termination_state,
            args.use_edge_attr,
            num_neighbour_cutoff=args.num_neighbour_cutoff,
            neighbour_cutoff_method=args.neighbour_cutoff_method,
        )

        file_name = get_imitation_dataset_file_name(args)
        path = pathlib.Path(f"{args.dataset_dir}", "processed_dataset", f"{file_name}")

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump((graph_dataset), f)
    else:
        file_name = get_imitation_dataset_file_name(args)[:-4]
        path = pathlib.Path(args.dataset_dir, "processed_dataset_eager", file_name)
        path.mkdir(parents=True, exist_ok=True)

        batched_generate_graph_dataset(
            dataset,
            comm_radius=args.comm_radius,
            obs_radius=args.obs_radius,
            num_samples=args.num_samples,
            save_termination_state=args.save_termination_state,
            dir_path=path,
            batch_size=args.batched_saving,
            use_edge_attr=args.use_edge_attr,
            num_neighbour_cutoff=args.num_neighbour_cutoff,
            neighbour_cutoff_method=args.neighbour_cutoff_method,
        )


if __name__ == "__main__":
    main()
