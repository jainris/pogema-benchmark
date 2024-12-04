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


class MAPFGraphDataset(Dataset):
    def __init__(self, dense_dataset) -> None:
        (
            self.dataset_node_features,
            self.dataset_Adj,
            self.dataset_target_actions,
            self.dataset_terminated,
            self.graph_map_id,
        ) = dense_dataset

    def __len__(self) -> int:
        return self.dataset_node_features.shape[0]

    def __getitem__(self, index):
        edge_index, edge_weight = dense_to_sparse(self.dataset_Adj[index])
        return Data(
            x=self.dataset_node_features[index],
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=self.dataset_target_actions[index],
            terminated=self.dataset_terminated[index],
        )


class MAPFHypergraphDataset(Dataset):
    def __init__(
        self, dense_dataset, hyperedge_indices, store_graph_indices=False
    ) -> None:
        (
            self.dataset_node_features,
            self.dataset_Adj,
            self.dataset_target_actions,
            self.dataset_terminated,
            self.graph_map_id,
        ) = dense_dataset
        self.hyperedge_indices = hyperedge_indices
        self.store_graph_indices = store_graph_indices

    def __len__(self) -> int:
        return self.dataset_node_features.shape[0]

    def __getitem__(self, index):
        if self.store_graph_indices:
            graph_edge_index, graph_edge_weight = dense_to_sparse(
                self.dataset_Adj[index]
            )
            return Data(
                x=self.dataset_node_features[index],
                edge_index=torch.LongTensor(self.hyperedge_indices[index]),
                graph_edge_index=graph_edge_index,
                graph_edge_weight=graph_edge_weight,
                y=self.dataset_target_actions[index],
                terminated=self.dataset_terminated[index],
            )
        return Data(
            x=self.dataset_node_features[index],
            edge_index=torch.LongTensor(self.hyperedge_indices[index]),
            y=self.dataset_target_actions[index],
            terminated=self.dataset_terminated[index],
        )
