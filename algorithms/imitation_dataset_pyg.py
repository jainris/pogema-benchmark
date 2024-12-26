import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data


def convert_dense_graph_dataset_to_sparse_pyg_dataset(dense_dataset):
    new_graph_dataset = []
    (
        dataset_node_features,
        dataset_Adj,
        dataset_target_actions,
        dataset_terminated,
        graph_map_id,
    ) = dense_dataset
    for i in tqdm(range(dataset_node_features.shape[0])):
        edge_index, edge_weight = dense_to_sparse(dataset_Adj[i])
        new_graph_dataset.append(
            Data(
                x=dataset_node_features[i],
                edge_index=edge_index,
                edge_weight=edge_weight,
                y=dataset_target_actions[i],
                terminated=dataset_terminated[i],
            )
        )
    return new_graph_dataset, graph_map_id


def decode_dense_dataset(dense_dataset, use_edge_attr):
    if use_edge_attr:
        return dense_dataset
    return *dense_dataset, None


class MAPFGraphDataset(Dataset):
    def __init__(self, dense_dataset, use_edge_attr) -> None:
        (
            self.dataset_node_features,
            self.dataset_Adj,
            self.dataset_target_actions,
            self.dataset_terminated,
            self.graph_map_id,
            self.dataset_agent_pos,
        ) = decode_dense_dataset(dense_dataset, use_edge_attr)
        self.use_edge_attr = use_edge_attr

    def __len__(self) -> int:
        return self.dataset_node_features.shape[0]

    def __getitem__(self, index):
        edge_index, edge_weight = dense_to_sparse(self.dataset_Adj[index])
        edge_attr = None
        if self.use_edge_attr:
            agent_pos = self.dataset_agent_pos[index]
            edge_attr = agent_pos[edge_index[0]] - agent_pos[edge_index[1]]
        return Data(
            x=self.dataset_node_features[index],
            edge_index=edge_index,
            edge_weight=edge_weight,
            edge_attr=edge_attr,
            y=self.dataset_target_actions[index],
            terminated=self.dataset_terminated[index],
        )


class MAPFHypergraphDataset(Dataset):
    def __init__(
        self,
        dense_dataset,
        hyperedge_indices,
        store_graph_indices=False,
        use_edge_attr=False,
        use_graph_edge_attr=False,
    ) -> None:
        (
            self.dataset_node_features,
            self.dataset_Adj,
            self.dataset_target_actions,
            self.dataset_terminated,
            self.graph_map_id,
            self.dataset_agent_pos,
        ) = decode_dense_dataset(dense_dataset, use_edge_attr or use_graph_edge_attr)
        self.hyperedge_indices = hyperedge_indices
        self.store_graph_indices = store_graph_indices
        self.use_edge_attr = use_edge_attr
        self.use_graph_edge_attr = use_graph_edge_attr

    def __len__(self) -> int:
        return self.dataset_node_features.shape[0]

    def __getitem__(self, index):
        extra_kwargs = dict()
        graph_edge_index, graph_edge_weight = None, None

        if self.use_edge_attr:
            raise NotImplementedError("Yet to implement")
        if self.store_graph_indices:
            graph_edge_index, graph_edge_weight = dense_to_sparse(
                self.dataset_Adj[index]
            )
            extra_kwargs = extra_kwargs | {
                "graph_edge_index": graph_edge_index,
                "graph_edge_weight": graph_edge_weight,
            }
        if self.use_graph_edge_attr:
            agent_pos = self.dataset_agent_pos[index]
            if (graph_edge_index is None) or (graph_edge_weight is None):
                graph_edge_index, graph_edge_weight = dense_to_sparse(
                    self.dataset_Adj[index]
                )
            edge_attr = agent_pos[graph_edge_index[0]] - agent_pos[graph_edge_index[1]]
            extra_kwargs = extra_kwargs | {"graph_edge_attr": edge_attr}
        return Data(
            x=self.dataset_node_features[index],
            edge_index=torch.LongTensor(self.hyperedge_indices[index]),
            y=self.dataset_target_actions[index],
            terminated=self.dataset_terminated[index],
            **extra_kwargs,
        )
