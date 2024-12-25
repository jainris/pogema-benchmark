from typing import Sequence

import numpy as np
import sys

from pogema import pogema_v0

sys.path.append("./magat_pathplanning")

import torch
import torch.nn.functional as F
from graphs.weights_initializer import weights_init

from torch_geometric.nn import GATConv, GATv2Conv, HypergraphConv

from convert_to_imitation_dataset import generate_graph_dataset
from generate_hypergraphs import generate_hypergraph_indices
from imitation_dataset_pyg import MAPFGraphDataset, MAPFHypergraphDataset
from gnn_magat_pyg import MAGATAdditiveConv, MAGATAdditiveConv2
from gnn_magat_pyg import MAGATMultiplicativeConv, MAGATMultiplicativeConv2
from gnn_magat_pyg import HGAT, HMAGAT, HMAGAT2, HMAGAT3, HGATv2


class GNNWrapper(torch.nn.Module):
    def __init__(
        self,
        gnn,
        use_edge_weights=False,
        use_edge_attr=False,
    ):
        super().__init__()
        assert (not use_edge_weights) or (
            not use_edge_attr
        ), "Currently, do not support use of both edge weights and edge attr"
        self.use_edge_weights = use_edge_weights
        self.use_edge_attr = use_edge_attr
        self.gnn = gnn

    def reset_parameters(self):
        self.gnn.reset_parameters()

    def forward(self, x, data):
        if self.use_edge_weights:
            return self.gnn(x, data.edge_index, data.edge_weight)
        if self.use_edge_attr:
            return self.gnn(x, data.edge_index, data.edge_attr)
        return self.gnn(x, data.edge_index)


def GNNFactory(
    in_channels,
    out_channels,
    num_attention_heads,
    model_type="MAGAT",
    use_edge_weights=False,
    use_edge_attr=False,
    edge_dim=None,
    residual=None,
    **model_kwargs,
):
    if use_edge_attr:
        assert (
            edge_dim is not None
        ), "Expecting edge_dim to be given if using edge attributes"
    elif use_edge_weights:
        edge_dim = 1
    else:
        assert (
            edge_dim is None
        ), f"Not using edge attr or weights, so expect node_dim to be None, but got {edge_dim}"
    kwargs = dict()
    if edge_dim is not None:
        kwargs = kwargs | {"edge_dim": edge_dim}
    if residual is not None:
        kwargs = kwargs | {"residual": residual}

    def _factory():
        if model_type == "MAGAT":
            attentionMode = model_kwargs["attentionMode"]
            if attentionMode == "GAT_origin" or attentionMode == "GAT":
                return GATConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    heads=num_attention_heads,
                    **kwargs,
                )
            elif attentionMode == "GATv2":
                return GATv2Conv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    heads=num_attention_heads,
                    **kwargs,
                )
            elif attentionMode == "MAGAT_additive":
                return MAGATAdditiveConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    heads=num_attention_heads,
                    **kwargs,
                )
            elif attentionMode == "MAGAT_additive2":
                return MAGATAdditiveConv2(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    heads=num_attention_heads,
                    **kwargs,
                )
            elif attentionMode == "MAGAT_multiplicative":
                return MAGATMultiplicativeConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    heads=num_attention_heads,
                    **kwargs,
                )
            elif attentionMode == "MAGAT_multiplicative2":
                return MAGATMultiplicativeConv2(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    heads=num_attention_heads,
                    **kwargs,
                )
            else:
                raise ValueError(
                    f"Currently, we don't support attention mode: {attentionMode}"
                )
        elif model_type == "HCHA":
            if model_kwargs["use_attention"]:
                return HGAT(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    heads=num_attention_heads,
                    hyperedge_feature_generator=model_kwargs[
                        "hyperedge_feature_generator"
                    ],
                    **kwargs,
                )
            else:
                return HypergraphConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    use_attention=model_kwargs["use_attention"],
                    heads=num_attention_heads,
                    **kwargs,
                )
        elif model_type == "HMAGAT":
            return HMAGAT(
                in_channels=in_channels,
                out_channels=out_channels,
                heads=num_attention_heads,
                hyperedge_feature_generator=model_kwargs["hyperedge_feature_generator"],
                **kwargs,
            )
        elif model_type == "HMAGAT2":
            return HMAGAT2(
                in_channels=in_channels,
                out_channels=out_channels,
                heads=num_attention_heads,
                hyperedge_feature_generator=model_kwargs["hyperedge_feature_generator"],
                **kwargs,
            )
        elif model_type == "HMAGAT3":
            return HMAGAT3(
                in_channels=in_channels,
                out_channels=out_channels,
                heads=num_attention_heads,
                hyperedge_feature_generator=model_kwargs["hyperedge_feature_generator"],
                **kwargs,
            )
        elif model_type == "HGATv2":
            return HGATv2(
                in_channels=in_channels,
                out_channels=out_channels,
                heads=num_attention_heads,
                hyperedge_feature_generator=model_kwargs["hyperedge_feature_generator"],
                **kwargs,
            )
        else:
            raise ValueError(f"Currently, we don't support model: {model_type}")

    return GNNWrapper(
        _factory(), use_edge_weights=use_edge_weights, use_edge_attr=use_edge_attr
    )


class CNN(torch.nn.Module):
    convs: Sequence[torch.nn.Conv2d]
    batch_norms: Sequence[torch.nn.BatchNorm2d]
    compressMLP: Sequence[torch.nn.Linear]

    def __init__(
        self,
        *,
        numChannel,
        numStride,
        convW,
        convH,
        nMaxPoolFilterTaps,
        numMaxPoolStride,
        embedding_sizes,
    ):
        super().__init__()

        convs = []
        batch_norms = []
        numConv = len(numChannel) - 1
        nFilterTaps = [3] * numConv
        nPaddingSzie = [1] * numConv
        for l in range(numConv):
            convs.append(
                torch.nn.Conv2d(
                    in_channels=numChannel[l],
                    out_channels=numChannel[l + 1],
                    kernel_size=nFilterTaps[l],
                    stride=numStride[l],
                    padding=nPaddingSzie[l],
                    bias=True,
                )
            )
            batch_norms.append(torch.nn.BatchNorm2d(num_features=numChannel[l + 1]))
            # convl.append(torch.nn.ReLU(inplace=True))

            convW = (
                int((convW - nFilterTaps[l] + 2 * nPaddingSzie[l]) / numStride[l]) + 1
            )
            convH = (
                int((convH - nFilterTaps[l] + 2 * nPaddingSzie[l]) / numStride[l]) + 1
            )
            # Adding maxpooling
            if l % 2 == 0:
                convW = int((convW - nMaxPoolFilterTaps) / numMaxPoolStride) + 1
                convH = int((convH - nMaxPoolFilterTaps) / numMaxPoolStride) + 1
                # http://cs231n.github.io/convolutional-networks/

        self.convs = torch.nn.ModuleList(convs)
        self.batch_norms = torch.nn.ModuleList(batch_norms)

        numFeatureMap = numChannel[-1] * convW * convH

        numCompressFeatures = [numFeatureMap] + embedding_sizes

        compressmlp = []
        for l in range(len(embedding_sizes)):
            compressmlp.append(
                torch.nn.Linear(
                    in_features=numCompressFeatures[l],
                    out_features=numCompressFeatures[l + 1],
                    bias=True,
                )
            )
        self.compressMLP = torch.nn.ModuleList(compressmlp)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for batch_norm in self.batch_norms:
            batch_norm.reset_parameters()
        for lin in self.compressMLP:
            lin.reset_parameters()

    def forward(self, x):
        for i, (conv, batch_norm) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x)
            x = batch_norm(x)
            x = F.relu(x)

            if i % 2 == 0:
                x = F.max_pool2d(x, kernel_size=2)
        x = x.reshape((x.shape[0], -1))
        for lin in self.compressMLP:
            x = lin(x)
            x = F.relu(x)
        return x


class DecentralPlannerGATNet(torch.nn.Module):
    def __init__(
        self,
        *,
        FOV,
        numInputFeatures,
        num_attention_heads,
        use_dropout,
        gnn_type,
        gnn_kwargs,
        concat_attention,
        num_classes=5,
        cnn_output_size=None,
        num_layers_gnn=None,
        embedding_sizes_gnn=None,
        use_edge_weights=False,
        use_edge_attr=False,
        edge_dim=None,
        model_residuals=None,
    ):
        super().__init__()

        assert concat_attention is True, "Currently only support concat attention."

        if embedding_sizes_gnn is None:
            assert num_layers_gnn is not None
            embedding_sizes_gnn = num_layers_gnn * [numInputFeatures]
        else:
            if num_layers_gnn is not None:
                assert num_layers_gnn == len(embedding_sizes_gnn)
            else:
                num_layers_gnn = len(embedding_sizes_gnn)

        inW = FOV + 2
        inH = FOV + 2

        if cnn_output_size is None:
            cnn_output_size = numInputFeatures

        #####################################################################
        #                                                                   #
        #                CNN to extract feature                             #
        #                                                                   #
        #####################################################################
        self.cnn = CNN(
            numChannel=[3, 32, 32, 64, 64, 128],
            numStride=[1, 1, 1, 1, 1],
            convW=inW,
            convH=inH,
            nMaxPoolFilterTaps=2,
            numMaxPoolStride=2,
            embedding_sizes=[cnn_output_size],
        )

        self.numFeatures2Share = cnn_output_size

        #####################################################################
        #                                                                   #
        #                    graph neural network                           #
        #                                                                   #
        #####################################################################

        first_residual = None
        rest_residuals = None
        if model_residuals == "first":
            first_residual = True
        elif model_residuals == "only-first":
            first_residual = True
            rest_residuals = False
        elif model_residuals == "all":
            first_residual = True
            rest_residuals = True
        elif model_residuals == "none":
            first_residual = False
            rest_residuals = False
        elif model_residuals is not None:
            raise ValueError(f"Unsupported model residuals option: {model_residuals}")

        graph_convs = []
        graph_convs.append(
            GNNFactory(
                in_channels=self.numFeatures2Share,
                out_channels=embedding_sizes_gnn[0],
                model_type=gnn_type,
                num_attention_heads=num_attention_heads,
                use_edge_weights=use_edge_weights,
                use_edge_attr=use_edge_attr,
                edge_dim=edge_dim,
                residual=first_residual,
                **gnn_kwargs,
            )
        )

        for i in range(num_layers_gnn - 1):
            graph_convs.append(
                GNNFactory(
                    in_channels=num_attention_heads * embedding_sizes_gnn[i],
                    out_channels=embedding_sizes_gnn[i + 1],
                    model_type=gnn_type,
                    num_attention_heads=num_attention_heads,
                    use_edge_weights=use_edge_weights,
                    use_edge_attr=use_edge_attr,
                    edge_dim=edge_dim,
                    residual=rest_residuals,
                    **gnn_kwargs,
                )
            )

        # And now feed them into the sequential
        self.gnns = torch.nn.ModuleList(graph_convs)  # Graph Filtering Layers

        #####################################################################
        #                                                                   #
        #                    MLP --- map to actions                         #
        #                                                                   #
        #####################################################################

        actionsfc = []
        actions_mlp_sizes = [
            num_attention_heads * embedding_sizes_gnn[-1],
            embedding_sizes_gnn[-1],
            num_classes,
        ]
        for i in range(len(actions_mlp_sizes) - 1):
            actionsfc.append(
                torch.nn.Linear(
                    in_features=actions_mlp_sizes[i],
                    out_features=actions_mlp_sizes[i + 1],
                )
            )
        self.use_dropout = use_dropout
        self.actionsMLP = torch.nn.ModuleList(actionsfc)

    def reset_parameters(self, non_default=False):
        if non_default:
            self.apply(weights_init)
        else:
            self.cnn.reset_parameters()
            for gnn in self.gnns:
                gnn.reset_parameters()
            for lin in self.actionsMLP:
                lin.reset_parameters()

    def forward(self, x, data):
        x = self.cnn(x)
        for conv in self.gnns:
            x = conv(x, data)
            x = F.relu(x)
        for lin in self.actionsMLP[:-1]:
            x = lin(x)
            x = F.relu(x)
            if self.use_dropout:
                x = F.dropout(x, p=0.2, training=self.training)
        x = self.actionsMLP[-1](x)
        return x


def run_model_on_grid(
    model,
    device,
    grid_config,
    args,
    hypergraph_model,
    max_episodes=None,
    aux_func=None,
):
    env = pogema_v0(grid_config=grid_config)
    observations, infos = env.reset()
    move_results = np.array(grid_config.MOVES)

    if aux_func is not None:
        aux_func(env=env, observations=observations, actions=None)

    while True:
        gdata = generate_graph_dataset(
            dataset=[[[observations], [0], [0]]],
            comm_radius=args.comm_radius,
            obs_radius=args.obs_radius,
            num_samples=None,
            save_termination_state=True,
            use_edge_attr=args.use_edge_attr,
            print_prefix=None,
        )
        if hypergraph_model:
            hindex = generate_hypergraph_indices(
                env,
                args.hypergraph_greedy_distance,
                args.hypergraph_num_steps,
                move_results,
                args.generate_graph_from_hyperedges,
            )
            gdata = MAPFHypergraphDataset(gdata, [hindex])[0]
        else:
            gdata = MAPFGraphDataset(gdata, args.use_edge_attr)[0]

        gdata.to(device)

        actions = model(gdata.x, gdata)
        actions = torch.argmax(actions, dim=-1).detach().cpu()

        observations, rewards, terminated, truncated, infos = env.step(actions)

        if aux_func is not None:
            aux_func(env=env, observations=observations, actions=actions)

        if all(terminated) or all(truncated):
            break

        if max_episodes is not None:
            max_episodes -= 1
            if max_episodes <= 0:
                break
    return all(terminated), env, observations


def get_model(args, device) -> tuple[torch.nn.Module, bool]:
    hypergraph_model = args.generate_graph_from_hyperedges
    if args.imitation_learning_model == "MAGAT":
        gnn_kwargs = {"attentionMode": args.attention_mode}
        model = DecentralPlannerGATNet(
            FOV=args.obs_radius,
            numInputFeatures=args.embedding_size,
            num_layers_gnn=args.num_gnn_layers,
            num_attention_heads=args.num_attention_heads,
            use_dropout=True,
            gnn_type="MAGAT",
            gnn_kwargs=gnn_kwargs,
            concat_attention=True,
            use_edge_weights=args.use_edge_weights,
            use_edge_attr=args.use_edge_attr,
            edge_dim=args.edge_dim,
            model_residuals=args.model_residuals,
        ).to(device)
        model.reset_parameters()
    elif args.imitation_learning_model == "HGCN":
        hypergraph_model = True
        gnn_kwargs = {"use_attention": False}
        model = DecentralPlannerGATNet(
            FOV=args.obs_radius,
            numInputFeatures=args.embedding_size,
            num_layers_gnn=args.num_gnn_layers,
            num_attention_heads=args.num_attention_heads,
            use_dropout=True,
            gnn_type="HCHA",
            gnn_kwargs=gnn_kwargs,
            concat_attention=True,
            use_edge_weights=args.use_edge_weights,
            use_edge_attr=args.use_edge_attr,
            edge_dim=args.edge_dim,
            model_residuals=args.model_residuals,
        ).to(device)
    elif args.imitation_learning_model == "HGAT":
        hypergraph_model = True
        gnn_kwargs = {
            "use_attention": True,
            "hyperedge_feature_generator": args.hyperedge_feature_generator,
        }
        model = DecentralPlannerGATNet(
            FOV=args.obs_radius,
            numInputFeatures=args.embedding_size,
            num_layers_gnn=args.num_gnn_layers,
            num_attention_heads=args.num_attention_heads,
            use_dropout=True,
            gnn_type="HCHA",
            gnn_kwargs=gnn_kwargs,
            concat_attention=True,
            use_edge_weights=args.use_edge_weights,
            use_edge_attr=args.use_edge_attr,
            edge_dim=args.edge_dim,
            model_residuals=args.model_residuals,
        ).to(device)
    elif args.imitation_learning_model == "HMAGAT":
        hypergraph_model = True
        gnn_kwargs = {"hyperedge_feature_generator": args.hyperedge_feature_generator}
        model = DecentralPlannerGATNet(
            FOV=args.obs_radius,
            numInputFeatures=args.embedding_size,
            num_layers_gnn=args.num_gnn_layers,
            num_attention_heads=args.num_attention_heads,
            use_dropout=True,
            gnn_type="HMAGAT",
            gnn_kwargs=gnn_kwargs,
            concat_attention=True,
            use_edge_weights=args.use_edge_weights,
            use_edge_attr=args.use_edge_attr,
            edge_dim=args.edge_dim,
            model_residuals=args.model_residuals,
        ).to(device)
    elif args.imitation_learning_model == "HMAGAT2":
        hypergraph_model = True
        gnn_kwargs = {"hyperedge_feature_generator": args.hyperedge_feature_generator}
        model = DecentralPlannerGATNet(
            FOV=args.obs_radius,
            numInputFeatures=args.embedding_size,
            num_layers_gnn=args.num_gnn_layers,
            num_attention_heads=args.num_attention_heads,
            use_dropout=True,
            gnn_type="HMAGAT2",
            gnn_kwargs=gnn_kwargs,
            concat_attention=True,
            use_edge_weights=args.use_edge_weights,
            use_edge_attr=args.use_edge_attr,
            edge_dim=args.edge_dim,
            model_residuals=args.model_residuals,
        ).to(device)
    elif args.imitation_learning_model == "HMAGAT3":
        hypergraph_model = True
        gnn_kwargs = {"hyperedge_feature_generator": args.hyperedge_feature_generator}
        model = DecentralPlannerGATNet(
            FOV=args.obs_radius,
            numInputFeatures=args.embedding_size,
            num_layers_gnn=args.num_gnn_layers,
            num_attention_heads=args.num_attention_heads,
            use_dropout=True,
            gnn_type="HMAGAT3",
            gnn_kwargs=gnn_kwargs,
            concat_attention=True,
            use_edge_weights=args.use_edge_weights,
            use_edge_attr=args.use_edge_attr,
            edge_dim=args.edge_dim,
            model_residuals=args.model_residuals,
        ).to(device)
    elif args.imitation_learning_model == "HGATv2":
        hypergraph_model = True
        gnn_kwargs = {"hyperedge_feature_generator": args.hyperedge_feature_generator}
        model = DecentralPlannerGATNet(
            FOV=args.obs_radius,
            numInputFeatures=args.embedding_size,
            num_layers_gnn=args.num_gnn_layers,
            num_attention_heads=args.num_attention_heads,
            use_dropout=True,
            gnn_type="HGATv2",
            gnn_kwargs=gnn_kwargs,
            concat_attention=True,
            use_edge_weights=args.use_edge_weights,
            use_edge_attr=args.use_edge_attr,
            edge_dim=args.edge_dim,
            model_residuals=args.model_residuals,
        ).to(device)
    else:
        raise ValueError(
            f"Unsupported imitation learning model {args.imitation_learning_model}."
        )
    return model, hypergraph_model
