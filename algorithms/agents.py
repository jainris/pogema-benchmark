from typing import Sequence
from collections import OrderedDict
import numpy as np
import math

from pogema import pogema_v0

import torch
import torch.nn.functional as F

from torch_geometric.nn import GATConv, GATv2Conv, HypergraphConv

from convert_to_imitation_dataset import generate_graph_dataset
from generate_hypergraphs import generate_hypergraph_indices
from generate_target_vec import generate_target_vec
from imitation_dataset_pyg import MAPFGraphDataset, MAPFHypergraphDataset
from collision_shielding import get_collision_shielded_model
from gnn_magat_pyg import MAGATAdditiveConv, MAGATAdditiveConv2
from gnn_magat_pyg import MAGATMultiplicativeConv, MAGATMultiplicativeConv2
from gnn_magat_pyg import HGAT, HMAGAT, HMAGAT2, HMAGAT3, HGATv2


class GNNWrapper(torch.nn.Module):
    def __init__(
        self,
        gnn,
        use_edge_weights=False,
        use_edge_attr=False,
        access_graph_index=False,
    ):
        super().__init__()
        assert (not use_edge_weights) or (
            not use_edge_attr
        ), "Currently, do not support use of both edge weights and edge attr"
        self.use_edge_weights = use_edge_weights
        self.use_edge_attr = use_edge_attr
        self.access_graph_index = access_graph_index
        self.gnn = gnn

    def reset_parameters(self):
        self.gnn.reset_parameters()

    def forward(self, x, data):
        if self.access_graph_index:
            edge_index = data.graph_edge_index
        else:
            edge_index = data.edge_index

        if self.use_edge_weights:
            return self.gnn(
                x,
                edge_index,
                data.graph_edge_weight if self.access_graph_index else data.edge_weight,
            )
        if self.use_edge_attr:
            return self.gnn(x, edge_index, data.edge_attr)
        return self.gnn(x, edge_index)


def GNNFactory(
    in_channels,
    out_channels,
    num_attention_heads,
    model_type="MAGAT",
    use_edge_weights=False,
    use_edge_attr=False,
    edge_dim=None,
    residual=None,
    access_graph_index=False,
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
        _factory(),
        use_edge_weights=use_edge_weights,
        use_edge_attr=use_edge_attr,
        access_graph_index=access_graph_index,
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


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1):
    return torch.nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=False,
        dilation=dilation,
    )


class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        dilation=(1, 1),
        residual=True,
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(
            inplanes, planes, stride, padding=dilation[0], dilation=dilation[0]
        )
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, padding=dilation[1], dilation=dilation[1])
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.residual = residual

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.bn1.reset_parameters()
        self.conv2.reset_parameters()
        self.bn2.reset_parameters()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out += residual
        out = self.relu(out)

        return out


class ResNet(torch.nn.Module):
    def __init__(
        self,
        layers,
        num_classes=128,
        channels=(32, 64, 128, 128),
        out_map=False,
        out_middle=False,
        pool_size=2,
        arch="D",
    ):
        super(ResNet, self).__init__()
        self.inplanes = channels[0]
        self.out_map = out_map
        self.out_dim = channels[-1]
        self.out_middle = out_middle
        self.arch = arch

        self.conv1 = torch.nn.Conv2d(
            3, channels[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = torch.nn.BatchNorm2d(channels[0])
        self.relu = torch.nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(BasicBlock, channels[0], layers[0], stride=2)
        self.layer2 = self._make_layer(BasicBlock, channels[1], layers[1], stride=1)
        self.layer3 = self._make_layer(BasicBlock, channels[2], layers[2], stride=1)

        if num_classes > 0:
            self.avgpool = torch.nn.AvgPool2d(pool_size)
            self.fc = torch.nn.Conv2d(
                self.out_dim, num_classes, kernel_size=1, stride=1, padding=0, bias=True
            )
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(
        self, block, planes, blocks, stride=1, dilation=1, new_level=True, residual=True
    ):
        assert dilation == 1 or dilation % 2 == 0
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                torch.nn.BatchNorm2d(planes * block.expansion),
            )

        layers = list()
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                dilation=(
                    (1, 1)
                    if dilation == 1
                    else (dilation // 2 if new_level else dilation, dilation)
                ),
                residual=residual,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    residual=residual,
                    dilation=(dilation, dilation),
                )
            )

        return torch.nn.Sequential(*layers)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.bn1.reset_parameters()
        for l in self.layer1:
            l.reset_parameters()
        for l in self.layer2:
            l.reset_parameters()
        for l in self.layer3:
            l.reset_parameters()
        self.fc.reset_parameters()

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        y = list()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        y.append(x)

        x = self.layer2(x)
        y.append(x)

        x = self.layer3(x)
        y.append(x)

        # print("x - postCNN:", x.shape)
        if self.out_map:
            x = self.fc(x)
        else:
            x_TMP = self.avgpool(x)
            # print('x - after avgpool', x_TMP.shape)
            x = self.fc(x_TMP)
            # print('x - after FCN', x.shape)
            # print('check x == x_tmp', x-x_TMP)
            # x = x.view(x.size(0), -1)

        if self.out_middle:
            return x, y
        else:
            return x


class ResNetLarge_withMLP(torch.nn.Module):
    def __init__(self, output_size, embedding_sizes):
        super().__init__()
        self.resnet = ResNet([1, 1, 1], out_map=False)
        self.dropout = torch.nn.Dropout(0.2)
        self.lin = torch.nn.Linear(
            in_features=1152, out_features=output_size, bias=True
        )

        numFeatureMap = output_size
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
        self.resnet.reset_parameters()
        self.lin.reset_parameters()
        for lin in self.compressMLP:
            lin.reset_parameters()

    def forward(self, x):
        x = self.resnet(x)
        x = self.dropout(x)
        x = x.reshape((x.shape[0], -1))
        x = self.lin(x)

        for lin in self.compressMLP:
            x = lin(x)
            x = F.relu(x)
        return x


def get_gnn_input_processor(use_target_vec, input_size):
    def _use_target_vec_processor(x, data):
        return torch.concatenate([x, data.target_vec], dim=-1)

    if use_target_vec is None:

        def _processor(x, data):
            return x

        return _processor, input_size
    elif use_target_vec == "target-vec":
        return _use_target_vec_processor, input_size + 2
    elif use_target_vec == "target-vec+dist":
        return _use_target_vec_processor, input_size + 3
    else:
        raise ValueError(f"Unsupported value for use_target_vec: {use_target_vec}.")


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
        cnn_mode="basic-CNN",
        cnn_output_size=None,
        num_gnn_layers=None,
        embedding_sizes_gnn=None,
        use_edge_weights=False,
        use_edge_attr=False,
        edge_dim=None,
        model_residuals=None,
        use_target_vec=None,
    ):
        super().__init__()

        assert concat_attention is True, "Currently only support concat attention."

        if embedding_sizes_gnn is None:
            assert num_gnn_layers is not None
            embedding_sizes_gnn = num_gnn_layers * [numInputFeatures]
        else:
            if num_gnn_layers is not None:
                assert num_gnn_layers == len(embedding_sizes_gnn)
            else:
                num_gnn_layers = len(embedding_sizes_gnn)

        inW = FOV + 2
        inH = FOV + 2

        if cnn_output_size is None:
            cnn_output_size = numInputFeatures

        #####################################################################
        #                                                                   #
        #                CNN to extract feature                             #
        #                                                                   #
        #####################################################################

        if cnn_mode == "basic-CNN":
            self.cnn = CNN(
                numChannel=[3, 32, 32, 64, 64, 128],
                numStride=[1, 1, 1, 1, 1],
                convW=inW,
                convH=inH,
                nMaxPoolFilterTaps=2,
                numMaxPoolStride=2,
                embedding_sizes=[cnn_output_size],
            )
        elif cnn_mode == "ResNetLarge_withMLP":
            self.cnn = ResNetLarge_withMLP(
                output_size=cnn_output_size, embedding_sizes=[cnn_output_size]
            )

        self.gnn_pre_processor, self.numFeatures2Share = get_gnn_input_processor(
            use_target_vec=use_target_vec, input_size=cnn_output_size
        )

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

        for i in range(num_gnn_layers - 1):
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

    def reset_parameters(self):
        self.cnn.reset_parameters()
        for gnn in self.gnns:
            gnn.reset_parameters()
        for lin in self.actionsMLP:
            lin.reset_parameters()

    def forward(self, x, data):
        x = self.cnn(x)
        x = self.gnn_pre_processor(x, data)
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


class AgentWithTwoNetworks(torch.nn.Module):
    def __init__(
        self,
        *,
        FOV,
        numInputFeatures,
        num_attention_heads,
        use_dropout,
        gnn1_kwargs,
        gnn2_kwargs,
        parallel_or_series,
        concat_attention,
        num_classes=5,
        cnn_mode="basic-CNN",
        cnn_output_size=None,
        use_target_vec=None,
    ):
        super().__init__()

        assert concat_attention is True, "Currently only support concat attention."

        inW = FOV + 2
        inH = FOV + 2

        if cnn_output_size is None:
            cnn_output_size = numInputFeatures

        #####################################################################
        #                                                                   #
        #                CNN to extract feature                             #
        #                                                                   #
        #####################################################################
        if cnn_mode == "basic-CNN":
            self.cnn = CNN(
                numChannel=[3, 32, 32, 64, 64, 128],
                numStride=[1, 1, 1, 1, 1],
                convW=inW,
                convH=inH,
                nMaxPoolFilterTaps=2,
                numMaxPoolStride=2,
                embedding_sizes=[cnn_output_size],
            )
        elif cnn_mode == "ResNetLarge_withMLP":
            self.cnn = ResNetLarge_withMLP(cnn_output_size)

        self.gnn_pre_processor, self.numFeatures2Share = get_gnn_input_processor(
            use_target_vec=use_target_vec, input_size=cnn_output_size
        )

        #####################################################################
        #                                                                   #
        #                    graph neural network                           #
        #                                                                   #
        #####################################################################

        def _define_gnn_convs(kwargs: dict) -> tuple[Sequence[torch.nn.Module], int]:
            gnn_type = kwargs.get("gnn_type", None)
            assert gnn_type is not None, "Missing gnn_type."
            gnn_kwargs = kwargs.get("gnn_kwargs", None)
            assert gnn_type is not None, "Missing gnn_kwargs."

            embedding_sizes_gnn = kwargs.get("embedding_sizes_gnn", None)
            num_gnn_layers = kwargs.get("num_gnn_layers", None)
            use_edge_weights = kwargs.get("use_edge_weights", False)
            use_edge_attr = kwargs.get("use_edge_attr", False)
            edge_dim = kwargs.get("edge_dim", None)
            model_residuals = kwargs.get("model_residuals", None)

            if embedding_sizes_gnn is None:
                assert num_gnn_layers is not None
                embedding_sizes_gnn = num_gnn_layers * [numInputFeatures]
            else:
                if num_gnn_layers is not None:
                    assert num_gnn_layers == len(embedding_sizes_gnn)
                else:
                    num_gnn_layers = len(embedding_sizes_gnn)

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
                raise ValueError(
                    f"Unsupported model residuals option: {model_residuals}"
                )

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

            for i in range(num_gnn_layers - 1):
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

            return graph_convs, embedding_sizes_gnn[-1]

        graph1_convs, gnn1_last_embd_sz = _define_gnn_convs(gnn1_kwargs)
        graph2_convs, gnn2_last_embd_sz = _define_gnn_convs(gnn2_kwargs)

        self.parallel_or_series = parallel_or_series
        if parallel_or_series == "parallel":
            assert (
                gnn1_last_embd_sz == gnn2_last_embd_sz
            ), "Expecting both output sizes to be the same."
        elif self.parallel_or_series == "series":
            starting_sz = self.numFeatures2Share
            assert (
                num_attention_heads * gnn1_last_embd_sz == starting_sz
            ), "Expecting the output size of gnn1 to match the input size of gnn2."
        else:
            raise ValueError(
                f"Got Invalid value for parallel_or_series {parallel_or_series}"
            )

        self.gnns1 = torch.nn.ModuleList(graph1_convs)
        self.gnns2 = torch.nn.ModuleList(graph2_convs)

        #####################################################################
        #                                                                   #
        #                    MLP --- map to actions                         #
        #                                                                   #
        #####################################################################

        actionsfc = []
        actions_mlp_sizes = [
            num_attention_heads * gnn2_last_embd_sz,
            gnn2_last_embd_sz,
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

    def reset_parameters(self):
        self.cnn.reset_parameters()
        for gnn in self.gnns1:
            gnn.reset_parameters()
        for gnn in self.gnns2:
            gnn.reset_parameters()
        for lin in self.actionsMLP:
            lin.reset_parameters()

    def forward(self, x, data):
        x = self.cnn(x)
        x = self.gnn_pre_processor(x, data)

        gnn1_out = x
        for conv in self.gnns1:
            gnn1_out = conv(gnn1_out, data)
            gnn1_out = F.relu(gnn1_out)

        if self.parallel_or_series == "parallel":
            gnn2_out = x
        else:
            gnn2_out = gnn1_out
        for conv in self.gnns2:
            gnn2_out = conv(gnn2_out, data)
            gnn2_out = F.relu(gnn2_out)

        # Combining outputs
        lin = self.actionsMLP[0]

        x = lin(gnn2_out)
        if self.parallel_or_series == "parallel":
            x = lin(gnn1_out) + x

        for lin in self.actionsMLP[1:]:
            x = F.relu(x)
            if self.use_dropout:
                x = F.dropout(x, p=0.2, training=self.training)
            x = lin(x)
        return x


def run_model_on_grid(
    model,
    device,
    grid_config,
    args,
    hypergraph_model,
    dataset_kwargs,
    use_target_vec,
    max_episodes=None,
    aux_func=None,
):
    env = pogema_v0(grid_config=grid_config)
    observations, infos = env.reset()
    move_results = np.array(grid_config.MOVES)

    if aux_func is not None:
        aux_func(env=env, observations=observations, actions=None)

    model = get_collision_shielded_model(model, env, args)

    while True:
        gdata = generate_graph_dataset(
            dataset=[[[observations], [0], [0]]],
            comm_radius=args.comm_radius,
            obs_radius=args.obs_radius,
            num_samples=None,
            save_termination_state=True,
            use_edge_attr=dataset_kwargs["use_edge_attr"],
            print_prefix=None,
        )
        target_vec = None
        if use_target_vec is not None:
            target_vec = generate_target_vec(
                dataset=[[[observations], [0], [0]]],
                num_samples=None,
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
            gdata = MAPFHypergraphDataset(
                gdata,
                [hindex],
                target_vec=target_vec,
                use_target_vec=use_target_vec,
                **dataset_kwargs,
            )[0]
        else:
            gdata = MAPFGraphDataset(
                gdata,
                target_vec=target_vec,
                use_target_vec=use_target_vec,
                **dataset_kwargs,
            )[0]

        gdata.to(device)

        actions = model.get_actions(gdata)
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


_GNN_DEF_KEYS = [
    "imitation_learning_model",
    "embedding_size",
    "num_gnn_layers",
    "num_attention_heads",
    "attention_mode",
    "edge_dim",
    "model_residuals",
    "use_edge_weights",
    "use_edge_attr",
    "hyperedge_feature_generator",
]


def _decode_args(args: dict, prefix: str = "") -> dict:
    hypergraph_model = False
    args = {key: args[f"{prefix}{key}"] for key in _GNN_DEF_KEYS}
    model_kwargs = {
        key: args[key]
        for key in [
            "num_gnn_layers",
            "use_edge_attr",
            "use_edge_weights",
            "edge_dim",
            "model_residuals",
        ]
    }
    if args["imitation_learning_model"] == "MAGAT":
        gnn_kwargs = {"attentionMode": args["attention_mode"]}
        gnn_type = "MAGAT"
    elif args["imitation_learning_model"] == "HGCN":
        hypergraph_model = True
        gnn_kwargs = {"use_attention": False}
        gnn_type = "HCHA"
    elif args["imitation_learning_model"] == "HGAT":
        hypergraph_model = True
        gnn_kwargs = {
            "use_attention": True,
            "hyperedge_feature_generator": args["hyperedge_feature_generator"],
        }
        gnn_type = "HCHA"
    elif args["imitation_learning_model"] == "HMAGAT":
        hypergraph_model = True
        gnn_kwargs = {
            "hyperedge_feature_generator": args["hyperedge_feature_generator"]
        }
        gnn_type = "HMAGAT"
    elif args["imitation_learning_model"] == "HMAGAT2":
        hypergraph_model = True
        gnn_kwargs = {
            "hyperedge_feature_generator": args["hyperedge_feature_generator"]
        }
        gnn_type = "HMAGAT2"
    elif args["imitation_learning_model"] == "HMAGAT3":
        hypergraph_model = True
        gnn_kwargs = {
            "hyperedge_feature_generator": args["hyperedge_feature_generator"]
        }
        gnn_type = "HMAGAT3"
    elif args["imitation_learning_model"] == "HGATv2":
        hypergraph_model = True
        gnn_kwargs = {
            "hyperedge_feature_generator": args["hyperedge_feature_generator"]
        }
        gnn_type = "HGATv2"
    else:
        raise ValueError(
            f"Unsupported imitation learning model {args['imitation_learning_model']}."
        )
    model_kwargs = model_kwargs | {
        "gnn_kwargs": gnn_kwargs,
        "gnn_type": gnn_type,
    }
    return model_kwargs, hypergraph_model


def update_partial_state_dict(
    state_dict,
    old_key_name="gnns",
    new_key_name="gnns1",
    check_and_update_legacy_model=True,
    parameters_to_load="all",
):
    if (parameters_to_load is None) or (parameters_to_load == "none"):
        parameters_to_load = []
    elif parameters_to_load != "all":
        parameters_to_load = parameters_to_load.split("+")

    new_state_dict = OrderedDict()
    for key in state_dict.keys():
        to_keep = False
        if parameters_to_load == "all":
            to_keep = True
        else:
            for par in parameters_to_load:
                if par == key[: len(par)]:
                    to_keep = True
                    break

        if to_keep:
            k2 = key.split(".")
            if k2[0] == old_key_name:
                if check_and_update_legacy_model and (k2[2] != "gnn"):
                    k2[1] = f"{k2[1]}.gnn"
                if new_key_name is not None:
                    k2[0] = new_key_name
                new_state_dict[".".join(k2)] = state_dict[key]
            else:
                new_state_dict[key] = state_dict[key]
    return new_state_dict


def get_parameters_to_freeze(state_dict, config):
    if (config is None) or (config == "none"):
        return []
    elif config == "all":
        return state_dict.keys()

    # Splitting config, as it's not a keyword arg
    config = config.split("+")
    parameters_to_freeze = []
    for name in state_dict.keys():
        for c in config:
            if c == name[: len(c)]:
                parameters_to_freeze.append(name)
    return parameters_to_freeze


def load_and_freeze_parameters(model, args, device):
    state_dict = torch.load(args.load_partial_parameters_path, map_location=device)
    state_dict = update_partial_state_dict(
        state_dict,
        new_key_name=args.replace_model,
        parameters_to_load=args.parameters_to_load,
    )
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    assert (
        len(unexpected_keys) == 0
    ), f"Got some unexpected keys to the model {unexpected_keys}"

    parameters_to_freeze = get_parameters_to_freeze(
        state_dict, args.parameters_to_freeze
    )
    for name, parameter in model.named_parameters():
        if name in parameters_to_freeze:
            parameter.requires_grad_(False)

    # Printing keys that will be trained
    to_trains = set()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            to_trains.add(name.split(".")[0])
    print(f"Will train (all/some) parameters of {to_trains}.")

    return model


def get_model(args, device) -> tuple[torch.nn.Module, bool, dict]:
    hypergraph_model = args.generate_graph_from_hyperedges
    common_kwargs = dict(
        FOV=args.obs_radius,
        numInputFeatures=args.embedding_size,
        num_attention_heads=args.num_attention_heads,
        cnn_mode=args.cnn_mode,
        use_dropout=True,
        concat_attention=True,
        num_classes=5,
    )
    dict_args = vars(args)
    if args.agent_network_type == "single":
        model_kwargs, hmodel = _decode_args(dict_args)
        assert not (
            args.generate_graph_from_hyperedges and hmodel
        ), "We do not support use of graph inputs with hypergraph models."
        hypergraph_model = hypergraph_model or hmodel
        dataset_kwargs = {"use_edge_attr": model_kwargs["use_edge_attr"]}
        model = DecentralPlannerGATNet(**common_kwargs, **model_kwargs).to(device)
        model.reset_parameters()
    elif args.agent_network_type == "parallel" or args.agent_network_type == "series":
        model1_kwargs, hmodel1 = _decode_args(dict_args)
        model2_kwargs, hmodel2 = _decode_args(dict_args, "model2_")
        hmodel = hmodel1 or hmodel2
        assert not (
            args.generate_graph_from_hyperedges and hmodel
        ), "We do not support use of graph inputs with hypergraph models."
        assert (
            args.num_attention_heads == args.model2_num_attention_heads
        ), "Currently require both num attention heads to be the same."  # TODO: Remedy
        if hmodel1 != hmodel2:
            # One of the models does not use hypergraphs, ensuring
            # appropriate edge indices are used
            hypergraph_kwargs, graph_kwargs = (
                (model1_kwargs, model2_kwargs)
                if hmodel1
                else (model2_kwargs, model1_kwargs)
            )
            graph_kwargs["gnn_kwargs"]["access_graph_index"] = True
            dataset_kwargs = {
                "use_edge_attr": hypergraph_kwargs["use_edge_attr"],
                "store_graph_indices": True,
                "use_graph_edge_attr": graph_kwargs["use_edge_attr"],
            }
        else:
            dataset_kwargs = {
                "use_edge_attr": (
                    model1_kwargs["use_edge_attr"] or model2_kwargs["use_edge_attr"]
                )
            }
        hypergraph_model = hypergraph_model or hmodel
        model = AgentWithTwoNetworks(
            **common_kwargs,
            gnn1_kwargs=model1_kwargs,
            gnn2_kwargs=model2_kwargs,
            parallel_or_series=args.agent_network_type,
        ).to(device)
        model.reset_parameters()
    else:
        raise ValueError(f"Unsupported agent network type {args.agent_network_type}.")
    if args.load_partial_parameters_path is not None:
        # Loading parameters
        model = load_and_freeze_parameters(model, args, device)
    return model, hypergraph_model, dataset_kwargs
