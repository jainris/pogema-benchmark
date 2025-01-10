import argparse
import pickle
import pathlib
import numpy as np
import torch

from run_expert import (
    DATASET_FILE_NAME_KEYS,
    add_expert_dataset_args,
    get_expert_dataset_file_name,
)


def get_target_vec_file_name(args):
    file_name = ""
    dict_args = vars(args)
    for key in sorted(DATASET_FILE_NAME_KEYS):
        file_name += f"_{key}_{dict_args[key]}"
    file_name = file_name[1:] + ".pkl"
    return file_name


def generate_target_vec(
    dataset,
    num_samples,
    print_prefix="",
):
    dataset_target_vec = []

    for id, (sample_observations, actions, terminated) in enumerate(dataset):
        if print_prefix is not None:
            print(
                f"{print_prefix}"
                f"Generating Graph Dataset for map {id + 1}/{num_samples}"
            )
        for observations in sample_observations:
            global_xys = np.array([obs["global_xy"] for obs in observations])
            global_target_xys = np.array([obs["global_target_xy"] for obs in observations])
            target_vec = global_target_xys - global_xys

            dataset_target_vec.append(target_vec)
    dataset_target_vec = np.stack(dataset_target_vec)
    return torch.from_numpy(dataset_target_vec)


def main():
    parser = argparse.ArgumentParser(
        description="Generate target vector."
    )
    parser = add_expert_dataset_args(parser)

    parser.add_argument("--comm_radius", type=int, default=7)
    parser.add_argument("--use_edge_attr", action="store_true", default=False)

    args = parser.parse_args()
    print(args)

    file_name = get_expert_dataset_file_name(args)
    path = pathlib.Path(f"{args.dataset_dir}", "raw_expert_predictions", f"{file_name}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "rb") as f:
        dataset = pickle.load(f)
    if isinstance(dataset, tuple):
        dataset, seed_mask = dataset

    target_vecs = generate_target_vec(
        dataset,
        args.num_samples,
    )

    file_name = get_target_vec_file_name(args)
    path = pathlib.Path(args.dataset_dir, "target_vec", file_name)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump((target_vecs), f)


if __name__ == "__main__":
    main()
