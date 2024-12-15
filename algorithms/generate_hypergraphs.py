import argparse
import pickle
import pathlib
import numpy as np
import torch

from pogema import pogema_v0, GridConfig
from scipy.spatial.distance import squareform, pdist

from run_expert import (
    DATASET_FILE_NAME_KEYS,
    DATASET_FILE_NAME_DEFAULT,
    add_expert_dataset_args,
)

HYPERGRAPH_FILE_NAME_DEFAULTS = {
    "hypergraph_greedy_distance": 2,
    "hypergraph_num_steps": 3,
}
HYPERGRAPH_FILE_NAME_KEYS = list(HYPERGRAPH_FILE_NAME_DEFAULTS.keys())


def get_hypergraph_file_name(args):
    file_name = ""
    dict_args = vars(args)
    for key in sorted(DATASET_FILE_NAME_KEYS):
        if dict_args[key] != DATASET_FILE_NAME_DEFAULT[key]:
            file_name += f"_{key}_{dict_args[key]}"
    for key in sorted(HYPERGRAPH_FILE_NAME_KEYS):
        if dict_args[key] != HYPERGRAPH_FILE_NAME_DEFAULTS[key]:
            file_name += f"_{key}_{dict_args[key]}"
    if len(file_name) > 0:
        file_name = file_name[1:] + ".pkl"
    else:
        file_name = "default.pkl"
    if args.generate_graph_from_hyperedges:
        file_name = file_name[:-4] + "_graph.pkl"
    return file_name


def get_one_hot_bool(vector):
    if "eye" not in get_one_hot_bool.__dict__:
        get_one_hot_bool.eye = np.eye(2, dtype=bool)
    return get_one_hot_bool.eye[vector]


def greedy_step(target_pos, agent_pos, obstacles, move_results):
    goal_direction = target_pos - agent_pos
    moves = np.sign(goal_direction)
    move_axis = np.argmax(np.abs(goal_direction), axis=-1)

    actions = np.zeros(agent_pos.shape[0], dtype=np.int64)
    new_agent_pos = np.copy(agent_pos)

    move_axis_mask = get_one_hot_bool(move_axis)
    move_sign_1 = moves[move_axis_mask]
    move_sign_2 = moves[~move_axis_mask]

    act1 = move_axis * 2 + (move_sign_1 + 3) // 2
    act2 = (1 - move_axis) * 2 + (move_sign_2 + 3) // 2

    new_pos1 = agent_pos + move_results[act1]
    new_pos2 = agent_pos + move_results[act2]

    valid_act1 = (obstacles[new_pos1[:, 0], new_pos1[:, 1]] == 0) & (move_sign_1 != 0)
    valid_act2 = (obstacles[new_pos2[:, 0], new_pos2[:, 1]] == 0) & (move_sign_2 != 0)

    valid_act2 = (~valid_act1) & valid_act2

    actions[valid_act1] = act1[valid_act1]
    actions[valid_act2] = act2[valid_act2]

    new_agent_pos[valid_act1] = new_pos1[valid_act1]
    new_agent_pos[valid_act2] = new_pos2[valid_act2]

    return actions, new_agent_pos


def update_groups(agent_pos, groups, hypergraph_greedy_distance):
    if len(groups.keys()) > 0:
        max_groups = max(groups.keys()) + 1
    else:
        max_groups = 0
    dists = squareform(pdist(agent_pos, metric="cityblock"))
    infinity_matrix = np.ones_like(dists) * np.inf
    dists = dists + np.triu(infinity_matrix)

    graph = np.nonzero(dists <= hypergraph_greedy_distance)

    src, dst = graph

    # Work around due to lack of pointers in Python
    group_ids = dict()
    group_parent = dict()

    def _unravel_ptr(ptr):
        while group_parent[ptr] != -1:
            ptr = group_parent[ptr]
        return ptr

    def _set_pointer(val):
        group_parent[val] = -1
        return val

    def _update_pointer(ptr_1, ptr_2):
        ptr_1 = _unravel_ptr(ptr_1)
        ptr_2 = _unravel_ptr(ptr_2)
        if ptr_1 != ptr_2:
            group_parent[ptr_1] = ptr_2

    def _access_value_at_pointer(ptr):
        return _unravel_ptr(ptr)

    for s, d in zip(src, dst):
        if (s in group_ids) and (d in group_ids):
            id_s = group_ids[s]
            id_d = group_ids[d]

            new_set = groups[_access_value_at_pointer(id_s)].union(
                groups[_access_value_at_pointer(id_d)]
            )
            del groups[_access_value_at_pointer(id_d)]

            _update_pointer(id_d, id_s)
            groups[_access_value_at_pointer(id_s)] = new_set
        else:
            if s in group_ids:
                group_ids[d] = group_ids[s]
                groups[_access_value_at_pointer(group_ids[s])].add(d)
            elif d in group_ids:
                group_ids[s] = group_ids[d]
                groups[_access_value_at_pointer(group_ids[d])].add(s)
            else:
                group_ids[s] = _set_pointer(max_groups)
                group_ids[d] = group_ids[s]
                groups[max_groups] = set({s, d})
                max_groups += 1
    return groups


def get_unique_groups(groups):
    unique_groups = []
    for group in groups.values():
        unique = True
        for g in unique_groups:
            if g == group:
                unique = False
                break
        if unique:
            unique_groups.append(group)
    return unique_groups


def generate_graph_instead(unique_groups, num_agents, remove_self_loops=True):
    Adj = torch.zeros((num_agents, num_agents))
    for nodes in unique_groups:
        for i in nodes:
            for j in nodes:
                Adj[i][j] = 1
    if remove_self_loops:
        Adj *= 1 - torch.eye(num_agents)
    return Adj.nonzero().t()


def generate_hypergraph_indices(
    env,
    hypergraph_greedy_distance,
    hypergraph_num_steps,
    move_results,
    generate_graph=False,
):
    target_pos = np.array(env.grid.get_targets_xy())
    agent_pos = np.array(env.grid.get_agents_xy())
    obstacles = env.grid.get_obstacles()

    groups = dict()
    groups = update_groups(agent_pos, groups, hypergraph_greedy_distance)

    for _ in range(hypergraph_num_steps):
        _, agent_pos = greedy_step(target_pos, agent_pos, obstacles, move_results)
        groups = update_groups(agent_pos, groups, hypergraph_greedy_distance)
    unique_groups = get_unique_groups(groups)

    if generate_graph:
        # Instead of hypergraph, we generate a graph, with edges
        # between all nodes in a hyperedge
        return generate_graph_instead(unique_groups, agent_pos.shape[0])

    hypergraph_index = [[], []]
    for i, nodes in enumerate(unique_groups):
        for node in nodes:
            hypergraph_index[0].append(node)
            hypergraph_index[1].append(i)
    return hypergraph_index


def main():
    parser = argparse.ArgumentParser(description="Generate Hypergraphs")
    parser = add_expert_dataset_args(parser)

    parser.add_argument("--hypergraph_greedy_distance", type=int, default=2)
    parser.add_argument("--hypergraph_num_steps", type=int, default=3)
    parser.add_argument(
        "--take_all_seeds", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--generate_graph_from_hyperedges",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

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
    elif not args.take_all_seeds:
        raise ValueError("Dataset is expected to have a seed_mask.")

    num_agents = int(args.robot_density * args.map_h * args.map_w)

    if args.map_type == "RandomGrid":
        assert args.map_h == args.map_w, (
            f"Expect height and width of random grid to be the same, "
            f"but got height {args.map_h} and width {args.map_w}"
        )

        rng = np.random.default_rng(args.dataset_seed)
        seeds = rng.integers(10**10, size=args.num_samples)

        if not args.take_all_seeds:
            seeds = seeds[seed_mask]

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

    all_hypergraphs = []
    for sample_num, (grid_config, data) in enumerate(zip(grid_configs, dataset)):
        print(f"Generating Hypergraph for map {sample_num + 1}/{args.num_samples}")
        move_results = np.array(grid_config.MOVES)

        env = pogema_v0(grid_config)
        _, _ = env.reset()
        _, all_actions, _ = data

        for actions in all_actions:
            hypergraph_index = generate_hypergraph_indices(
                env,
                args.hypergraph_greedy_distance,
                args.hypergraph_num_steps,
                move_results,
                args.generate_graph_from_hyperedges,
            )

            all_hypergraphs.append(hypergraph_index)
            env.step(actions)

    file_name = get_hypergraph_file_name(args)
    path = pathlib.Path(f"{args.dataset_dir}", "hypergraphs", f"{file_name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(all_hypergraphs, f)


if __name__ == "__main__":
    main()
