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


# class MAPFGraphDataset(Dataset):
#     def __init__(self, graph_dataset) -> None:
#         self.graph_dataset = graph_dataset

#     def __len__(self) -> int:
#         return len(self.graph_dataset)

#     def __getitem__(self, index) -> Any:
#         return self.graph_dataset[index]
