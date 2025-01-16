from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torch_geometric.utils import dense_to_sparse, scatter
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
    def __init__(
        self,
        dense_dataset,
        use_edge_attr,
        target_vec=None,
        use_target_vec=None,
        return_relevance_as_y=False,
        relevances=None,
        use_relevances=None,
        edge_attr_opts="straight",
    ) -> None:
        (
            self.dataset_node_features,
            self.dataset_Adj,
            self.dataset_target_actions,
            self.dataset_terminated,
            self.graph_map_id,
            self.dataset_agent_pos,
        ) = decode_dense_dataset(dense_dataset, use_edge_attr)
        self.use_edge_attr = use_edge_attr
        self.edge_attr_opts = edge_attr_opts
        self.target_vec = target_vec
        self.use_target_vec = use_target_vec
        self.return_relevance_as_y = return_relevance_as_y
        self.relevances = relevances
        self.use_relevances = use_relevances
        if use_relevances[: len("only-relevance")] == "only-relevance":
            self.use_relevances = use_relevances[len("only-relevance-") :]

    def __len__(self) -> int:
        return self.dataset_node_features.shape[0]

    def __getitem__(self, index):
        edge_index, edge_weight = dense_to_sparse(self.dataset_Adj[index])
        edge_attr = None
        y = self.dataset_target_actions[index]

        extra_kwargs = dict()
        if self.use_edge_attr:
            agent_pos = self.dataset_agent_pos[index]
            edge_attr = agent_pos[edge_index[0]] - agent_pos[edge_index[1]]
            edge_attr = edge_attr.to(torch.float)
            if self.edge_attr_opts == "dist":
                dist = torch.norm(edge_attr, keepdim=True, dim=-1)
                edge_attr = torch.concatenate([edge_attr, dist], dim=-1)
            elif self.edge_attr_opts == "only-dist":
                edge_attr = torch.norm(edge_attr, keepdim=True, dim=-1)
            elif self.edge_attr_opts != "straight":
                raise ValueError(f"Unsupport edge_attr_opts: {self.edge_attr_opts}.")
        if self.use_target_vec is not None:
            target_vec = self.target_vec[index].to(torch.float)
            if self.use_target_vec == "target-vec+dist":
                # Calculating dist
                dist = torch.norm(target_vec, keepdim=True, dim=-1)
                target_vec = torch.concatenate([target_vec, dist], dim=-1)
            extra_kwargs["target_vec"] = target_vec
        if self.return_relevance_as_y:
            extra_kwargs = extra_kwargs | {"original_y": y}
            y = self.relevances[index]
        if self.use_relevances is not None:
            if self.use_relevances == "straight":
                relevances = self.relevances[index]
            elif self.use_relevances == "one-hot":
                relevances = self.relevances[index]
                relevances = torch.argsort(relevances, dim=-1, descending=True)
                relevances = torch.nn.functional.one_hot(
                    relevances, num_classes=relevances.shape[-1]
                )
                relevances = relevances.reshape(
                    (relevances.shape[0], relevances.shape[1] * relevances.shape[2])
                )
            else:
                raise ValueError(
                    f"Unsupported value for use_relevances: {self.use_relevances}."
                )
            extra_kwargs = extra_kwargs | {"relevances": relevances}
        return Data(
            x=self.dataset_node_features[index],
            edge_index=edge_index,
            edge_weight=edge_weight,
            edge_attr=edge_attr,
            y=y,
            terminated=self.dataset_terminated[index],
            **extra_kwargs,
        )


class MAPFHypergraphDataset(Dataset):
    def __init__(
        self,
        dense_dataset,
        hyperedge_indices,
        store_graph_indices=False,
        use_edge_attr=False,
        use_graph_edge_attr=False,
        target_vec=None,
        use_target_vec=None,
        return_relevance_as_y=False,
        relevances=None,
        use_relevances=None,
        edge_attr_opts="straight",
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
        self.edge_attr_opts = edge_attr_opts
        self.target_vec = target_vec
        self.use_target_vec = use_target_vec
        self.return_relevance_as_y = return_relevance_as_y
        self.relevances = relevances
        self.use_relevances = use_relevances
        if use_relevances[: len("only-relevance")] == "only-relevance":
            self.use_relevances = use_relevances[len("only-relevance-") :]

    def __len__(self) -> int:
        return self.dataset_node_features.shape[0]

    def __getitem__(self, index):
        extra_kwargs = dict()
        graph_edge_index, graph_edge_weight = None, None
        y = self.dataset_target_actions[index]

        if self.use_edge_attr:
            agent_pos = self.dataset_agent_pos[index]
            src, dst = torch.LongTensor(self.hyperedge_indices[index])

            hyperedge_pos = scatter(src=agent_pos[src], index=dst, dim=0, reduce="mean")
            edge_attr = hyperedge_pos[dst] - agent_pos[src]
            edge_attr = edge_attr.to(torch.float)
            if self.edge_attr_opts == "dist":
                dist = torch.norm(edge_attr, keepdim=True, dim=-1)
                edge_attr = torch.concatenate([edge_attr, dist], dim=-1)
            elif self.edge_attr_opts == "only-dist":
                edge_attr = torch.norm(edge_attr, keepdim=True, dim=-1)
            elif self.edge_attr_opts != "straight":
                raise ValueError(f"Unsupport edge_attr_opts: {self.edge_attr_opts}.")
            extra_kwargs = extra_kwargs | {"edge_attr": edge_attr}
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
            edge_attr = edge_attr.to(torch.float)
            if self.edge_attr_opts == "dist":
                dist = torch.norm(edge_attr, keepdim=True, dim=-1)
                edge_attr = torch.concatenate([edge_attr, dist], dim=-1)
            elif self.edge_attr_opts == "only-dist":
                edge_attr = torch.norm(edge_attr, keepdim=True, dim=-1)
            elif self.edge_attr_opts != "straight":
                raise ValueError(f"Unsupport edge_attr_opts: {self.edge_attr_opts}.")
            extra_kwargs = extra_kwargs | {"graph_edge_attr": edge_attr}
        if self.use_target_vec is not None:
            target_vec = self.target_vec[index].to(torch.float)
            if self.use_target_vec == "target-vec+dist":
                # Calculating dist
                dist = torch.norm(target_vec, keepdim=True, dim=-1)
                target_vec = torch.concatenate([target_vec, dist], dim=-1)
            extra_kwargs["target_vec"] = target_vec
        if self.return_relevance_as_y:
            extra_kwargs = extra_kwargs | {"original_y": y}
            y = self.relevances[index]
        if (self.use_relevances is not None) and (self.relevances is not None):
            if self.use_relevances == "straight":
                relevances = self.relevances[index]
            elif self.use_relevances == "one-hot":
                relevances = self.relevances[index]
                relevances = torch.argsort(relevances, dim=-1, descending=True)
                relevances = torch.nn.functional.one_hot(
                    relevances, num_classes=relevances.shape[-1]
                )
                relevances = relevances.reshape(
                    (relevances.shape[0], relevances.shape[1] * relevances.shape[2])
                ).to(torch.float)
            else:
                raise ValueError(
                    f"Unsupported value for use_relevances: {self.use_relevances}."
                )
            extra_kwargs = extra_kwargs | {"relevances": relevances}
        return Data(
            x=self.dataset_node_features[index],
            edge_index=torch.LongTensor(self.hyperedge_indices[index]),
            y=y,
            terminated=self.dataset_terminated[index],
            **extra_kwargs,
        )
