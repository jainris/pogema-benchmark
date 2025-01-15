import argparse
import pickle
import pathlib
import numpy as np
import torch

from scipy.spatial.distance import squareform, pdist

from run_expert import (
    DATASET_FILE_NAME_KEYS,
    add_expert_dataset_args,
    get_expert_dataset_file_name,
    get_legacy_expert_dataset_file_name,
)


def get_legacy_pos_file_name(args):
    file_name = ""
    dict_args = vars(args)
    for key in sorted(DATASET_FILE_NAME_KEYS):
        file_name += f"_{key}_{dict_args[key]}"
    if args.use_edge_attr:
        file_name += "_pos"
    file_name = file_name[1:] + ".pkl"
    return file_name


def get_pos_file_name(args):
    file_name = get_expert_dataset_file_name(args)
    if args.use_edge_attr:
        file_name = file_name[:-4] + "_pos" + file_name[-4:]
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
    dataset_agent_pos = []
    assert use_edge_attr

    for id, (sample_observations, actions, terminated) in enumerate(dataset):
        if print_prefix is not None:
            print(
                f"{print_prefix}"
                f"Generating Graph Dataset for map {id + 1}/{num_samples}"
            )
        for observations in sample_observations:
            global_xys = np.array([obs["global_xy"] for obs in observations])

            if use_edge_attr:
                dataset_agent_pos.append(global_xys)
    dataset_agent_pos = np.stack(dataset_agent_pos)
    return torch.from_numpy(dataset_agent_pos)


def main():
    parser = argparse.ArgumentParser(
        description="Convert to Imitation Learning Dataset"
    )
    parser = add_expert_dataset_args(parser)

    parser.add_argument("--comm_radius", type=int, default=7)
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

    graph_dataset = generate_graph_dataset(
        dataset,
        args.comm_radius,
        args.obs_radius,
        args.num_samples,
        args.save_termination_state,
        args.use_edge_attr,
    )

    file_name = get_pos_file_name(args)
    path = pathlib.Path(f"{args.dataset_dir}", "positions", f"{file_name}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump((graph_dataset), f)


if __name__ == "__main__":
    main()
