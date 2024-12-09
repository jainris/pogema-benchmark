from typing import Sequence

import argparse
import pickle
import pathlib
import numpy as np
import sys
import wandb

from multiprocessing import Process, Queue
from itertools import compress

from pogema import pogema_v0, GridConfig

from lacam.inference import LacamInference, LacamInferenceConfig

sys.path.append("./magat_pathplanning")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from graphs.weights_initializer import weights_init

from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, HypergraphConv

from convert_to_imitation_dataset import generate_graph_dataset
from generate_hypergraphs import generate_hypergraph_indices, get_hypergraph_file_name
from run_expert import (
    DATASET_FILE_NAME_KEYS,
    run_expert_algorithm,
    add_expert_dataset_args,
)
from imitation_dataset_pyg import MAPFGraphDataset, MAPFHypergraphDataset
from gnn_magat_pyg import MAGATAdditiveConv, MAGATAdditiveConv2, MAGATMultiplicativeConv
from gnn_magat_pyg import HGAT, HMAGAT, HMAGAT2


def GNNFactory(
    in_channels, out_channels, num_attention_heads, model_type="MAGAT", **model_kwargs
):
    if model_type == "MAGAT":
        attentionMode = model_kwargs["attentionMode"]
        if attentionMode == "GAT_origin":
            return GATConv(
                in_channels=in_channels,
                out_channels=out_channels,
                heads=num_attention_heads,
            )
        elif attentionMode == "MAGAT_additive":
            return MAGATAdditiveConv(
                in_channels=in_channels,
                out_channels=out_channels,
                heads=num_attention_heads,
            )
        elif attentionMode == "MAGAT_additive2":
            return MAGATAdditiveConv2(
                in_channels=in_channels,
                out_channels=out_channels,
                heads=num_attention_heads,
            )
        elif attentionMode == "MAGAT_multiplicative":
            return MAGATMultiplicativeConv(
                in_channels=in_channels,
                out_channels=out_channels,
                heads=num_attention_heads,
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
                hyperedge_feature_generator=model_kwargs["hyperedge_feature_generator"],
            )
        else:
            return HypergraphConv(
                in_channels=in_channels,
                out_channels=out_channels,
                use_attention=model_kwargs["use_attention"],
                heads=num_attention_heads,
            )
    elif model_type == "HMAGAT":
        return HMAGAT(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=num_attention_heads,
            hyperedge_feature_generator=model_kwargs["hyperedge_feature_generator"],
        )
    elif model_type == "HMAGAT2":
        return HMAGAT2(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=num_attention_heads,
            hyperedge_feature_generator=model_kwargs["hyperedge_feature_generator"],
        )
    else:
        raise ValueError(f"Currently, we don't support model: {model_type}")


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

        graph_convs = []
        graph_convs.append(
            GNNFactory(
                in_channels=self.numFeatures2Share,
                out_channels=embedding_sizes_gnn[0],
                model_type=gnn_type,
                num_attention_heads=num_attention_heads,
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

    def forward(self, x, edge_index):
        x = self.cnn(x)
        for conv in self.gnns:
            x = conv(x, edge_index)
            x = F.relu(x)
        for lin in self.actionsMLP[:-1]:
            x = lin(x)
            x = F.relu(x)
            if self.use_dropout:
                x = F.dropout(x, p=0.2, training=self.training)
        x = self.actionsMLP[-1](x)
        return x


def main():
    parser = argparse.ArgumentParser(description="Train imitation learning model.")
    parser = add_expert_dataset_args(parser)
    parser.add_argument("--imitation_learning_model", type=str, default="MAGAT")

    parser.add_argument("--comm_radius", type=int, default=7)

    parser.add_argument("--validation_fraction", type=float, default=0.15)
    parser.add_argument("--test_fraction", type=float, default=0.15)
    parser.add_argument("--num_training_oe", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--num_gnn_layers", type=int, default=3)
    parser.add_argument("--num_attention_heads", type=int, default=1)

    parser.add_argument("--lr_start", type=float, default=1e-3)
    parser.add_argument("--lr_end", type=float, default=1e-6)
    parser.add_argument("--num_epochs", type=int, default=300)

    parser.add_argument("--validation_every_epochs", type=int, default=4)
    parser.add_argument(
        "--run_online_expert", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--save_intmd_checkpoints", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints")

    parser.add_argument(
        "--skip_validation", action=argparse.BooleanOptionalAction, default=False
    )

    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--initial_val_size", type=int, default=128)
    parser.add_argument("--threshold_val_success_rate", type=float, default=0.9)
    parser.add_argument("--num_run_oe", type=int, default=500)
    parser.add_argument("--run_oe_after", type=int, default=0)
    parser.add_argument("--attention_mode", type=str, default="GAT_modified")

    parser.add_argument("--hypergraph_greedy_distance", type=int, default=2)
    parser.add_argument("--hypergraph_num_steps", type=int, default=3)

    parser.add_argument("--hyperedge_feature_generator", type=str, default="gcn")

    args = parser.parse_args()
    print(args)

    assert args.save_termination_state

    if args.device == -1:
        device = torch.device("cuda")
    elif args.device is not None:
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")

    num_agents = int(args.robot_density * args.map_h * args.map_w)

    if args.map_type == "RandomGrid":
        assert args.map_h == args.map_w, (
            f"Expect height and width of random grid to be the same, "
            f"but got height {args.map_h} and width {args.map_w}"
        )

        rng = np.random.default_rng(args.dataset_seed)
        seeds = rng.integers(10**10, size=args.num_samples)

        def _grid_config_generator(seed):
            return GridConfig(
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

        grid_config = _grid_config_generator(seeds[0])
    else:
        raise ValueError(f"Unsupported map type: {args.map_type}.")

    if args.expert_algorithm == "LaCAM":
        inference_config = LacamInferenceConfig()
        expert_algorithm = LacamInference
    else:
        raise ValueError(f"Unsupported expert algorithm {args.expert_algorithm}.")

    hypergraph_model = False
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
        ).to(device)
    else:
        raise ValueError(
            f"Unsupported imitation learning model {args.imitation_learning_model}."
        )

    optimizer = optim.Adam(model.parameters(), lr=args.lr_start, weight_decay=1e-5)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=args.lr_end
    )

    dense_dataset = None
    hyper_edge_indices = None

    file_name = ""
    dict_args = vars(args)
    for key in sorted(DATASET_FILE_NAME_KEYS):
        file_name += f"_{key}_{dict_args[key]}"
    file_name = file_name[1:] + ".pkl"

    path = pathlib.Path(args.dataset_dir, "processed_dataset", file_name)
    with open(path, "rb") as f:
        dense_dataset = pickle.load(f)
    if hypergraph_model:
        file_name = get_hypergraph_file_name(args)
        path = pathlib.Path(args.dataset_dir, "hypergraphs", file_name)
        with open(path, "rb") as f:
            hyper_edge_indices = pickle.load(f)

    loss_function = torch.nn.CrossEntropyLoss()

    wandb.init(
        project="hyper-mapf-pogema",
        name=args.run_name,
        config=vars(args),
        entity="jainris",
    )

    # Data split
    train_id_max = int(
        args.num_samples * (1 - args.validation_fraction - args.test_fraction)
    )
    validation_id_max = train_id_max + int(args.num_samples * args.validation_fraction)

    def _divide_dataset(start, end):
        mask = torch.logical_and(dense_dataset[-1] >= start, dense_dataset[-1] < end)
        if hyper_edge_indices is not None:
            hindices = list(compress(hyper_edge_indices, mask))
        else:
            hindices = None
        return tuple(gd[mask] for gd in dense_dataset), hindices

    train_dataset, train_hindices = _divide_dataset(0, train_id_max)
    # validation_dataset = _divide_dataset(train_id_max, validation_id_max)
    # test_dataset = _divide_dataset(validation_id_max, torch.inf)

    if hypergraph_model:
        train_dataset = MAPFHypergraphDataset(train_dataset, train_hindices)
    else:
        train_dataset = MAPFGraphDataset(train_dataset)
    train_dl = DataLoader(train_dataset, batch_size=args.batch_size)

    best_validation_success_rate = 0.0
    checkpoint_path = pathlib.Path(f"{args.checkpoints_dir}", "best.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    cur_validation_id_max = min(train_id_max + args.initial_val_size, validation_id_max)

    oe_graph_dataset = None
    oe_hypergraph_indices = []

    def run_model_on_grid(grid_config, max_episodes=None):
        env = pogema_v0(grid_config=grid_config)
        observations, infos = env.reset()
        move_results = np.array(grid_config.MOVES)

        while True:
            gdata = generate_graph_dataset(
                [[[observations], [0], [0]]],
                args.comm_radius,
                args.obs_radius,
                None,
                True,
                None,
            )
            if hypergraph_model:
                hindex = generate_hypergraph_indices(
                    env,
                    args.hypergraph_greedy_distance,
                    args.hypergraph_num_steps,
                    move_results,
                )
                gdata = MAPFHypergraphDataset(gdata, [hindex])[0]
            else:
                gdata = MAPFGraphDataset(gdata)[0]

            gdata.to(device)

            actions = model(gdata.x, gdata.edge_index)
            actions = torch.argmax(actions, dim=-1).detach().cpu()

            observations, rewards, terminated, truncated, infos = env.step(actions)

            if all(terminated) or all(truncated):
                break

            if max_episodes is not None:
                max_episodes -= 1
                if max_episodes <= 0:
                    break
        return all(terminated), env, observations

    def multiprocess_run_expert(
        queue,
        expert,
        env,
        observations,
        save_termination_state,
        additional_data_func=None,
    ):
        expert_results = run_expert_algorithm(
            expert,
            env=env,
            observations=observations,
            save_termination_state=save_termination_state,
            additional_data_func=additional_data_func,
        )
        queue.put(expert_results)

    move_results = np.array(grid_config.MOVES)

    def get_hypergraph_indices(env, **kwargs):
        return generate_hypergraph_indices(
            env,
            hypergraph_greedy_distance=args.hypergraph_greedy_distance,
            hypergraph_num_steps=args.hypergraph_num_steps,
            move_results=move_results,
        )

    queue = Queue()

    for epoch in range(args.num_epochs):
        total_loss = 0.0
        tot_correct = 0
        num_samples = 0
        n_batches = 0

        model = model.train()
        for data in train_dl:
            data = data.to(device)
            optimizer.zero_grad()

            out = model(data.x, data.edge_index)

            out = out[~data.terminated]
            target_actions = data.y[~data.terminated]
            loss = loss_function(out, target_actions)

            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            tot_correct += (
                torch.sum(torch.argmax(out, dim=-1) == target_actions).detach().cpu()
            )
            num_samples += out.shape[0]
            n_batches += 1

        if oe_graph_dataset is not None:
            if hypergraph_model:
                oe_dl = DataLoader(
                    MAPFHypergraphDataset(oe_graph_dataset, oe_hypergraph_indices),
                    batch_size=args.batch_size,
                )
            else:
                oe_dl = DataLoader(
                    MAPFGraphDataset(oe_graph_dataset), batch_size=args.batch_size
                )

            for data in oe_dl:
                data = data.to(device)
                optimizer.zero_grad()

                out = model(data.x, data.edge_index)

                out = out[~data.terminated]
                target_actions = data.y[~data.terminated]
                loss = loss_function(out, target_actions)

                total_loss += loss.item()

                loss.backward()
                optimizer.step()

                tot_correct += (
                    torch.sum(torch.argmax(out, dim=-1) == target_actions)
                    .detach()
                    .cpu()
                )
                num_samples += out.shape[0]
                n_batches += 1
        lr_scheduler.step()

        print(
            f"Epoch {epoch}, Mean Loss: {total_loss / n_batches}, Mean Accuracy: {tot_correct / num_samples}"
        )

        results = {
            "train_loss": total_loss / n_batches,
            "train_accuracy": tot_correct / num_samples,
        }
        if (not args.skip_validation) and (
            (epoch + 1) % args.validation_every_epochs == 0
        ):
            model = model.eval()

            num_completed = 0

            print("-------------------")
            print("Starting Validation")

            for graph_id in range(train_id_max, cur_validation_id_max):
                success, env, observations = run_model_on_grid(
                    _grid_config_generator(seeds[graph_id])
                )

                if success:
                    num_completed += 1
                print(
                    f"Validation Graph {graph_id - train_id_max}/{validation_id_max - train_id_max}, "
                    f"Current Success Rate: {num_completed / (graph_id - train_id_max + 1)}"
                )
            success_rate = num_completed / (graph_id - train_id_max)
            results = results | {"validation_success_rate": success_rate}

            if args.save_intmd_checkpoints:
                checkpoint_path = pathlib.Path(
                    f"{args.checkpoints_dir}", f"epoch_{epoch}.pt"
                )
                torch.save(model.state_dict(), checkpoint_path)

            if success_rate > best_validation_success_rate:
                if success_rate >= args.threshold_val_success_rate:
                    print("Success rate passed threshold -- Increasing Validation Size")
                    args.threshold_val_success_rate = 1.1
                    cur_validation_id_max = validation_id_max
                checkpoint_path = pathlib.Path(f"{args.checkpoints_dir}", "best.pt")
                torch.save(model.state_dict(), checkpoint_path)

            print("Finshed Validation")
            print("------------------")

            if args.run_online_expert and (epoch + 1 >= args.run_oe_after):
                print("---------------------")
                print("Running Online Expert")

                rng = np.random.default_rng(args.dataset_seed + epoch + 1)
                oe_ids = rng.integers(train_id_max, size=args.num_run_oe)

                oe_dataset = []
                oe_hindices = []

                for graph_id in oe_ids:
                    grid_config = GridConfig(
                        num_agents=num_agents,  # number of agents
                        size=args.map_w,  # size of the grid
                        density=args.obstacle_density,  # obstacle density
                        seed=seeds[graph_id],  # set to None for random
                        # obstacles, agents and targets
                        # positions at each reset
                        max_episode_steps=2 * args.max_episode_steps,  # horizon
                        obs_radius=args.obs_radius,  # defines field of view
                        observation_type="MAPF",
                        collision_system=args.collision_system,
                        on_target=args.on_target,
                    )
                    success, env, observations = run_model_on_grid(
                        grid_config, args.max_episode_steps
                    )

                    if not success:
                        expert = expert_algorithm(inference_config)

                        additional_data_func = (
                            get_hypergraph_indices if hypergraph_model else None
                        )

                        p = Process(
                            target=multiprocess_run_expert,
                            args=(
                                queue,
                                expert,
                                env,
                                observations,
                                args.save_termination_state,
                                additional_data_func,
                            ),
                        )
                        p.start()

                        all_actions, all_observations, all_terminated = None, None, None
                        hindices = []
                        while True:
                            try:
                                expert_results = queue.get(timeout=3)
                                p.join()
                                break
                            except:
                                p.join(timeout=0.5)
                                if p.exitcode is not None:
                                    break

                        if hypergraph_model:
                            (
                                all_actions,
                                all_observations,
                                all_terminated,
                                hindices,
                            ) = expert_results
                        else:
                            all_actions, all_observations, all_terminated = (
                                expert_results
                            )
                        if all_actions is not None:
                            if all(all_terminated[-1]):
                                oe_dataset.append(
                                    (all_observations, all_actions, all_terminated)
                                )
                                oe_hindices.extend(hindices)
                while queue.qsize() > 0:
                    # Popping remaining elements, although no elements should remain
                    expert_results = queue.get()
                    hindices = []
                    if hypergraph_model:
                        (
                            all_actions,
                            all_observations,
                            all_terminated,
                            hindices,
                        ) = expert_results
                    else:
                        all_actions, all_observations, all_terminated = expert_results
                    oe_dataset.append((all_observations, all_actions, all_terminated))
                    oe_hindices.extend(hindices)

                if len(oe_dataset) > 0:
                    print(f"Adding {len(oe_dataset)} OE grids to the dataset")
                    oe_hypergraph_indices.extend(oe_hindices)
                    new_oe_graph_dataset = generate_graph_dataset(
                        oe_dataset,
                        args.comm_radius,
                        args.obs_radius,
                        None,
                        True,
                        None,
                    )
                    if oe_graph_dataset is None:
                        oe_graph_dataset = new_oe_graph_dataset
                    else:
                        oe_graph_dataset = tuple(
                            torch.concat(
                                [oe_graph_dataset[i], new_oe_graph_dataset[i]], dim=0
                            )
                            for i in range(len(oe_graph_dataset))
                        )
                print("Finished Online Expert")
                print("----------------------")

        wandb.log(results)
    checkpoint_path = pathlib.Path(f"{args.checkpoints_dir}", f"last.pt")
    torch.save(model.state_dict(), checkpoint_path)


if __name__ == "__main__":
    main()
