import numpy as np
import torch
from torch_geometric.data import Data

from pibt.pypibt.dist_table import DistTable

from convert_to_imitation_dataset import generate_graph_dataset
from generate_hypergraphs import generate_hypergraph_indices
from generate_target_vec import generate_target_vec
from imitation_dataset_pyg import MAPFGraphDataset, MAPFHypergraphDataset

from utils import get_neighbors
from pibt_training import get_relevance_from_inverse_relevances


class BaseRuntimeDataGeneration:
    def __init__(self, hypergraph_model: bool, **additional_kwargs):
        self.hypergraph_model = hypergraph_model
        self.generators = dict()
        # Storing key separately to maintain order
        # (Could instead use OrderedDict instead)
        self.keys = []
        self.additional_kwargs = dict(**additional_kwargs)

    def register_datagenerator(self, key, generator):
        self.generators[key] = generator
        assert key not in self.keys
        self.keys.append(key)

    def register_params(self, key, value):
        self.additional_kwargs[key] = value

    def __call__(self, observations, env) -> Data:
        kwargs = {}
        for key in self.keys:
            kwargs[key] = self.generators[key](observations, env)
        if self.hypergraph_model:
            return MAPFHypergraphDataset(**kwargs, **self.additional_kwargs)[0]
        else:
            return MAPFGraphDataset(**kwargs, **self.additional_kwargs)[0]


def get_graph_dataset_generator(
    comm_radius, obs_radius, num_neighbour_cutoff, dataset_kwargs
):
    def _generator(observations, env):
        return generate_graph_dataset(
            dataset=[[[observations], [0], [0]]],
            comm_radius=comm_radius,
            obs_radius=obs_radius,
            num_samples=None,
            save_termination_state=True,
            use_edge_attr=dataset_kwargs["use_edge_attr"],
            print_prefix=None,
            num_neighbour_cutoff=num_neighbour_cutoff,
        )

    return _generator, "dense_dataset"


def get_target_vec_generator():
    def _generator(observations, env):
        return generate_target_vec(
            dataset=[[[observations], [0], [0]]],
            num_samples=None,
            print_prefix=None,
        )

    return _generator, "target_vec"


def get_hyperindices_generator(
    hypergraph_greedy_distance,
    hypergraph_num_steps,
    move_results,
    generate_graph_from_hyperedges,
    max_group_size,
    overlap_size,
):
    def _generator(observations, env):
        hindex = generate_hypergraph_indices(
            env,
            hypergraph_greedy_distance,
            hypergraph_num_steps,
            move_results,
            generate_graph_from_hyperedges,
            max_group_size=max_group_size,
            overlap_size=overlap_size,
        )
        return [hindex]

    return _generator, "hyperedge_indices"


class RelevanceGenerator:
    def __init__(self, moves):
        self.dist_tables = None
        self.grid = None
        self.moves = moves

    def __call__(self, observations, env):
        if self.dist_tables is None:
            obstacles = env.grid.get_obstacles(ignore_borders=True)
            self.grid = obstacles == 0

            goals = env.grid.get_targets_xy(ignore_borders=True)
            goals = [tuple(g) for g in goals]

            self.dist_tables = [DistTable(self.grid, goal) for goal in goals]
        agent_pos = env.grid.get_agents_xy(ignore_borders=True)
        agent_pos = [tuple(pos) for pos in agent_pos]

        N = len(agent_pos)

        inverse_relevances = -np.ones((N, len(self.moves)), dtype=np.int)
        for i in range(N):
            C, move_idx, mask = get_neighbors(self.grid, agent_pos[i], self.moves)
            for j in range(len(move_idx)):
                inverse_relevances[i][move_idx[j]] = self.dist_tables[i].get(C[j])

        relevances = get_relevance_from_inverse_relevances(inverse_relevances)
        relevances = torch.from_numpy(relevances)

        return [relevances]


class CustomRelevanceGenerator:
    def __init__(self, one_hot=True):
        self.relevances = None
        self.one_hot = one_hot

    def set_relevance(self, relevances):
        self.relevances = relevances

    def __call__(self, observations, env):
        relevances = self.relevances
        if self.one_hot:
            relevances = torch.nn.functional.one_hot(
                relevances, num_classes=relevances.shape[-1]
            )
            relevances = relevances.reshape(
                (relevances.shape[0], relevances.shape[1] * relevances.shape[2])
            ).to(torch.float)
        return [relevances]


def get_runtime_data_generator(
    grid_config,
    args,
    hypergraph_model,
    dataset_kwargs,
    use_target_vec,
    custom_relevance=False,
) -> BaseRuntimeDataGeneration:
    rt_data_generator = BaseRuntimeDataGeneration(
        hypergraph_model, edge_attr_opts=args.edge_attr_opts, **dataset_kwargs
    )

    generator, key = get_graph_dataset_generator(
        args.comm_radius, args.obs_radius, args.num_neighbour_cutoff, dataset_kwargs
    )
    rt_data_generator.register_datagenerator(key, generator)

    if use_target_vec is not None:
        generator, key = get_target_vec_generator()
        rt_data_generator.register_datagenerator(key, generator)
        rt_data_generator.register_params("use_target_vec", use_target_vec)

    if hypergraph_model:
        generator, key = get_hyperindices_generator(
            hypergraph_greedy_distance=args.hypergraph_greedy_distance,
            hypergraph_num_steps=args.hypergraph_num_steps,
            move_results=np.array(grid_config.MOVES),
            generate_graph_from_hyperedges=args.generate_graph_from_hyperedges,
            max_group_size=args.hypergraph_max_group_size,
            overlap_size=args.hypergraph_min_overlap,
        )
        rt_data_generator.register_datagenerator(key, generator)

    if custom_relevance:
        key = "relevances"
        generator = CustomRelevanceGenerator()

        rt_data_generator.register_datagenerator(key, generator)
        rt_data_generator.register_params("use_relevances", "straight")
    elif args.use_relevances is not None:
        key = "relevances"
        generator = RelevanceGenerator(moves=np.array(grid_config.MOVES))

        rt_data_generator.register_datagenerator(key, generator)
        rt_data_generator.register_params("use_relevances", args.use_relevances)

    return rt_data_generator
