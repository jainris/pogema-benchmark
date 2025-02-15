import argparse
import pickle
import pathlib
import numpy as np
import sys
import wandb

import multiprocessing as mp

from pogema import pogema_v0

sys.path.append("./magat_pathplanning")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from graphs.weights_initializer import weights_init
import utils.graphUtils.graphML as gml
import utils.graphUtils.graphTools
from torchsummaryX import summary
from graphs.models.resnet_pytorch import *

from convert_to_imitation_dataset import (
    generate_graph_dataset,
    get_imitation_dataset_file_name,
)
from run_expert import (
    run_expert_algorithm,
    add_expert_dataset_args,
    get_expert_algorithm_and_config,
)

from grid_config_generator import (
    grid_config_generator_factory,
    generate_grid_config_from_env,
)

from collision_shielding import NaiveCollisionShielding, PIBTCollisionShielding


def add_training_args(parser):
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

    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--initial_val_size", type=int, default=128)
    parser.add_argument("--threshold_val_success_rate", type=float, default=0.9)
    parser.add_argument("--num_run_oe", type=int, default=500)
    parser.add_argument("--run_oe_after", type=int, default=0)

    parser.add_argument(
        "--recursive_oe", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--run_expert_in_separate_fork",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--normalize_adjacency", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--cnn_mode", type=str, default="basic-CNN")
    parser.add_argument("--attention_mode", type=str, default="GAT_modified")
    parser.add_argument(
        "--train_on_terminated_agents",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--collision_shielding", type=str, default="naive")
    parser.add_argument("--action_sampling", type=str, default="deterministic")

    return parser


class DecentralPlannerGATNet(torch.nn.Module):
    def __init__(
        self,
        FOV,
        numInputFeatures,
        nGraphFilterTaps,
        nAttentionHeads,
        use_dropout,
        CNN_mode,
        attentionMode,
        AttentionConcat,
    ):
        super().__init__()
        self.S = None
        # inW = self.config.map_w
        # inH = self.config.map_h

        inW = FOV + 2
        inH = FOV + 2
        # invW = 11
        # inH = 11

        convW = [inW]
        convH = [inH]
        numAction = 5

        use_vgg = False

        # ------------------ DCP v1.4  -  with maxpool + non stride in CNN - less feature
        numChannel = [3] + [32, 32, 64, 64, 128]
        numStride = [1, 1, 1, 1, 1]

        dimCompressMLP = 1
        numCompressFeatures = [numInputFeatures]

        nMaxPoolFilterTaps = 2
        numMaxPoolStride = 2
        # # 1 layer origin
        dimNodeSignals = [numInputFeatures]

        # # 2 layer - upsampling
        # dimNodeSignals = [256, self.config.numInputFeatures]

        # # 2 layer - down sampling
        # dimNodeSignals = [64, self.config.numInputFeatures]
        #
        # # 2 layer - down sampling -v2
        # dimNodeSignals = [64, 32]
        #

        ## ------------------ GCN -------------------- ##
        # dimNodeSignals = [2 ** 7]
        # nGraphFilterTaps = [self.config.nGraphFilterTaps,self.config.nGraphFilterTaps] # [2]
        nGraphFilterTaps = [nGraphFilterTaps]
        nAttentionHeads = [nAttentionHeads]
        # --- actionMLP
        if use_dropout:
            dimActionMLP = 2
            numActionFeatures = [numInputFeatures, numAction]
        else:
            dimActionMLP = 1
            numActionFeatures = [numAction]

        #####################################################################
        #                                                                   #
        #                CNN to extract feature                             #
        #                                                                   #
        #####################################################################
        if use_vgg:
            pass
        else:
            if CNN_mode == "ResNetSlim_withMLP":
                convl = []
                convl.append(ResNetSlim(BasicBlock, [1, 1], out_map=False))
                convl.append(torch.nn.Dropout(0.2))
                convl.append(torch.nn.Flatten())
                convl.append(
                    torch.nn.Linear(
                        in_features=1152, out_features=numInputFeatures, bias=True
                    )
                )
                self.ConvLayers = torch.nn.Sequential(*convl)
                numFeatureMap = numInputFeatures
            elif CNN_mode == "ResNetLarge_withMLP":
                convl = []
                convl.append(ResNet(BasicBlock, [1, 1, 1], out_map=False))
                convl.append(torch.nn.Dropout(0.2))
                convl.append(torch.nn.Flatten())
                convl.append(
                    torch.nn.Linear(
                        in_features=1152, out_features=numInputFeatures, bias=True
                    )
                )
                self.ConvLayers = torch.nn.Sequential(*convl)
                numFeatureMap = numInputFeatures
            elif CNN_mode == "ResNetSlim":
                convl = []
                convl.append(ResNetSlim(BasicBlock, [1, 1], out_map=False))
                convl.append(torch.nn.Dropout(0.2))
                self.ConvLayers = torch.nn.Sequential(*convl)
                numFeatureMap = 1152
            elif CNN_mode == "ResNetLarge":
                convl = []
                convl.append(ResNet(BasicBlock, [1, 1, 1], out_map=False))
                convl.append(torch.nn.Dropout(0.2))
                self.ConvLayers = torch.nn.Sequential(*convl)
                numFeatureMap = 1152
            else:
                convl = []
                numConv = len(numChannel) - 1
                nFilterTaps = [3] * numConv
                nPaddingSzie = [1] * numConv
                for l in range(numConv):
                    convl.append(
                        torch.nn.Conv2d(
                            in_channels=numChannel[l],
                            out_channels=numChannel[l + 1],
                            kernel_size=nFilterTaps[l],
                            stride=numStride[l],
                            padding=nPaddingSzie[l],
                            bias=True,
                        )
                    )
                    convl.append(torch.nn.BatchNorm2d(num_features=numChannel[l + 1]))
                    convl.append(torch.nn.ReLU(inplace=True))

                    # if self.config.use_dropout:
                    #     convl.append(torch.nn.Dropout(p=0.2))

                    W_tmp = (
                        int(
                            (convW[l] - nFilterTaps[l] + 2 * nPaddingSzie[l])
                            / numStride[l]
                        )
                        + 1
                    )
                    H_tmp = (
                        int(
                            (convH[l] - nFilterTaps[l] + 2 * nPaddingSzie[l])
                            / numStride[l]
                        )
                        + 1
                    )
                    # Adding maxpooling
                    if l % 2 == 0:
                        convl.append(torch.nn.MaxPool2d(kernel_size=2))
                        W_tmp = int((W_tmp - nMaxPoolFilterTaps) / numMaxPoolStride) + 1
                        H_tmp = int((H_tmp - nMaxPoolFilterTaps) / numMaxPoolStride) + 1
                        # http://cs231n.github.io/convolutional-networks/
                    convW.append(W_tmp)
                    convH.append(H_tmp)

                self.ConvLayers = torch.nn.Sequential(*convl)

                numFeatureMap = numChannel[-1] * convW[-1] * convH[-1]

            #####################################################################
            #                                                                   #
            #                MLP-feature compression                            #
            #                                                                   #
            #####################################################################

            numCompressFeatures = [numFeatureMap] + numCompressFeatures

            compressmlp = []
            for l in range(dimCompressMLP):
                compressmlp.append(
                    torch.nn.Linear(
                        in_features=numCompressFeatures[l],
                        out_features=numCompressFeatures[l + 1],
                        bias=True,
                    )
                )
                compressmlp.append(torch.nn.ReLU(inplace=True))
                # if self.config.use_dropout:
                #     compressmlp.append(torch.nn.Dropout(p=0.2))

            self.compressMLP = torch.nn.Sequential(*compressmlp)

        self.numFeatures2Share = numCompressFeatures[-1]

        #####################################################################
        #                                                                   #
        #                    graph neural network                           #
        #                                                                   #
        #####################################################################

        self.L = len(nGraphFilterTaps)  # Number of graph filtering layers
        self.F = [numCompressFeatures[-1]] + dimNodeSignals  # Features
        # self.F = [numFeatureMap] + dimNodeSignals  # Features
        self.K = nGraphFilterTaps  # nFilterTaps # Filter taps
        self.P = nAttentionHeads
        self.E = 1  # Number of edge features
        self.bias = True

        gfl = []  # Graph Filtering Layers
        for l in range(self.L):
            # \\ Graph filtering stage:

            if attentionMode == "GAT_origin":
                gfl.append(
                    gml.GraphFilterBatchAttentional_Origin(
                        self.F[l],
                        self.F[l + 1],
                        self.K[l],
                        self.P[l],
                        self.E,
                        self.bias,
                        concatenate=AttentionConcat,
                        attentionMode=attentionMode,
                    )
                )

            elif attentionMode == "GAT_modified" or attentionMode == "KeyQuery":
                gfl.append(
                    gml.GraphFilterBatchAttentional(
                        self.F[l],
                        self.F[l + 1],
                        self.K[l],
                        self.P[l],
                        self.E,
                        self.bias,
                        concatenate=AttentionConcat,
                        attentionMode=attentionMode,
                    )
                )
            elif attentionMode == "GAT_Similarity":
                gfl.append(
                    gml.GraphFilterBatchSimilarityAttentional(
                        self.F[l],
                        self.F[l + 1],
                        self.K[l],
                        self.P[l],
                        self.E,
                        self.bias,
                        concatenate=AttentionConcat,
                        attentionMode=attentionMode,
                    )
                )

        # And now feed them into the sequential
        self.GFL = torch.nn.Sequential(*gfl)  # Graph Filtering Layers

        #####################################################################
        #                                                                   #
        #                    MLP --- map to actions                         #
        #                                                                   #
        #####################################################################
        if AttentionConcat:
            numActionFeatures = [self.F[-1] * nAttentionHeads[0]] + numActionFeatures
        else:
            numActionFeatures = [self.F[-1]] + numActionFeatures
        actionsfc = []
        for l in range(dimActionMLP):
            if l < (dimActionMLP - 1):
                actionsfc.append(
                    torch.nn.Linear(
                        in_features=numActionFeatures[l],
                        out_features=numActionFeatures[l + 1],
                        bias=True,
                    )
                )
                actionsfc.append(torch.nn.ReLU(inplace=True))
            else:
                actionsfc.append(
                    torch.nn.Linear(
                        in_features=numActionFeatures[l],
                        out_features=numActionFeatures[l + 1],
                        bias=True,
                    )
                )

            if use_dropout:
                actionsfc.append(torch.nn.Dropout(p=0.2))
                print("Dropout is add on MLP")

        self.actionsMLP = torch.nn.Sequential(*actionsfc)
        self.apply(weights_init)

    def make_layers(self, cfg, batch_norm=False):
        layers = []

        input_channel = 3
        for l in cfg:
            if l == "M":
                layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
                continue

            layers += [torch.nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

            if batch_norm:
                layers += [torch.nn.BatchNorm2d(l)]

            layers += [torch.nn.ReLU(inplace=True)]
            input_channel = l

        return torch.nn.Sequential(*layers)

    def addGSO(self, S, device, GSO_mode="dist_GSO"):
        # We add the GSO on real time, this GSO also depends on time and has
        # shape either B x N x N or B x E x N x N
        if self.E == 1:  # It is B x T x N x N
            assert len(S.shape) == 3
            self.S = S.unsqueeze(1)  # B x E x N x N
        else:
            assert len(S.shape) == 4
            assert S.shape[1] == self.E
            self.S = S
        # Remove nan data
        self.S[torch.isnan(self.S)] = 0
        if GSO_mode == "dist_GSO_one":
            self.S[self.S > 0] = 1
        elif GSO_mode == "full_GSO":
            self.S = torch.ones_like(self.S).to(device)

    def forward(self, inputTensor, device, GSO_mode="dist_GSO_one"):

        B = inputTensor.shape[0]  # batch size
        (B, N, C, W, H) = inputTensor.shape

        input_currentAgent = inputTensor.reshape(B * N, C, W, H).to(device)
        featureMap = self.ConvLayers(input_currentAgent).to(device)
        featureMapFlatten = featureMap.view(featureMap.size(0), -1).to(device)
        compressfeature = self.compressMLP(featureMapFlatten).to(device)
        extractFeatureMap = (
            compressfeature.reshape(B, N, self.numFeatures2Share)
            .to(device)
            .permute([0, 2, 1])
        )

        # DCP
        for l in range(self.L):
            # \\ Graph filtering stage:
            # There is a 3*l below here, because we have three elements per
            # layer: graph filter, nonlinearity and pooling, so after each layer
            # we're actually adding elements to the (sequential) list.
            # self.GFL[2 * l].addGSO(self.S) # add GSO for GraphFilter
            self.GFL[l].addGSO(self.S)  # add GSO for GraphFilter

        # B x F x N - > B x G x N,
        sharedFeature = self.GFL(extractFeatureMap)

        (_, num_G, _) = sharedFeature.shape

        sharedFeature_stack = (
            sharedFeature.permute([0, 2, 1]).to(device).reshape(B * N, num_G)
        )

        action_predict = self.actionsMLP(sharedFeature_stack)

        return action_predict


class NaiveCollisionShieldingMAGAT(NaiveCollisionShielding):
    def get_actions(self, gdata, device):
        # Naive collision shielding leaves the shielding to the env
        # So just returning the actions given by the model
        self.model.addGSO(gdata[1].to(device), device)
        actions = self.model(gdata[0].to(device), device)

        if self.sampling_method == "deterministic":
            actions = torch.argmax(actions, dim=-1).detach().cpu()
        elif self.sampling_method == "probabilistic":
            probs = torch.nn.functional.softmax(actions, dim=-1)
            probs = probs.detach().cpu().numpy()

            # Despite using softmax, sum might not be 1 due to fp errors
            probs = probs / np.sum(probs, keepdims=True, axis=-1)

            actions = np.zeros(probs.shape[0], dtype=np.int)
            ids = np.arange(probs.shape[1])
            for i in range(probs.shape[0]):
                actions[i] = self.rng.choice(
                    ids, size=1, replace=False, p=probs[i], shuffle=False
                )
        elif self.sampling_method == "exp-multinomial":
            probs = torch.exp(actions)
            actions = torch.multinomial(probs, num_samples=1)
        else:
            raise ValueError(f"Unsupported sampling method: {self.sampling_method}.")
        return actions


class PIBTCollisionShieldingMAGAT(PIBTCollisionShielding):
    def get_actions(self, gdata, device):
        self.model.addGSO(gdata[1].to(device), device)
        actions = self.model(gdata[0].to(device), device)

        if self.sampling_method == "probabilistic":
            actions = torch.nn.functional.softmax(actions, dim=-1)
            actions = actions.detach().cpu().numpy()
        actions = self.pibt_instance.step(actions)
        return actions


def get_collision_shielded_model(model, env, collision_shielding, action_sampling):
    if collision_shielding == "naive":
        return NaiveCollisionShieldingMAGAT(
            model=model, env=env, sampling_method=action_sampling
        )
    elif collision_shielding == "pibt":
        return PIBTCollisionShieldingMAGAT(
            model=model, env=env, sampling_method=action_sampling, dist_priorities=False
        )
    elif collision_shielding == "pibt-dist":
        return PIBTCollisionShieldingMAGAT(
            model=model,
            env=env,
            sampling_method=action_sampling,
            dist_priorities=True,
        )
    else:
        raise ValueError(
            f"Unsupported collision shielding method: {collision_shielding}."
        )


def run_model_on_grid(
    model,
    comm_radius,
    obs_radius,
    grid_config,
    device,
    collision_shielding,
    action_sampling,
    max_episodes=None,
    aux_func=None,
):
    env = pogema_v0(grid_config=grid_config)
    observations, infos = env.reset()

    model = get_collision_shielded_model(
        model,
        env,
        collision_shielding=collision_shielding,
        action_sampling=action_sampling,
    )

    if aux_func is not None:
        aux_func(env=env, observations=observations, actions=None)

    if max_episodes is None:
        while True:
            gdata = generate_graph_dataset(
                dataset=[[[observations], [0], [0]]],
                comm_radius=comm_radius,
                obs_radius=obs_radius,
                num_samples=None,
                save_termination_state=True,
                print_prefix=None,
            )

            actions = model.get_actions(gdata, device)
            observations, rewards, terminated, truncated, infos = env.step(actions)

            if aux_func is not None:
                aux_func(env=env, observations=observations, actions=actions)

            if all(terminated) or all(truncated):
                break
    else:
        for _ in range(max_episodes):
            gdata = generate_graph_dataset(
                [[[observations], [0], [0]]],
                comm_radius,
                obs_radius,
                None,
                True,
                None,
            )

            actions = model.get_actions(gdata, device)
            observations, rewards, terminated, truncated, infos = env.step(actions)

            if aux_func is not None:
                aux_func(env=env, observations=observations, actions=actions)

            if all(terminated) or all(truncated):
                break
    return all(terminated), env, observations


def get_maxEigenValue(matrix):
    isSymmetric = np.allclose(matrix, np.transpose(matrix, axes=[1, 0]))
    if isSymmetric:
        W = np.linalg.eigvalsh(matrix)
    else:
        W = np.linalg.eigvals(matrix)

    maxEigenvalue = np.max(np.real(W), axis=0)
    return maxEigenvalue
    # return np.max(np.abs(np.linalg.eig(matrix)[0]))


def main():
    parser = argparse.ArgumentParser(description="Train imitation learning model.")
    parser = add_expert_dataset_args(parser)
    parser = add_training_args(parser)

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

    rng = np.random.default_rng(args.dataset_seed)
    seeds = rng.integers(10**10, size=args.num_samples)

    _grid_config_generator = grid_config_generator_factory(
        map_type=args.map_type,
        map_w=args.map_w,
        map_h=args.map_h,
        num_agents=num_agents,
        obstacle_density=args.obstacle_density,
        obs_radius=args.obs_radius,
        collision_system=args.collision_system,
        on_target=args.on_target,
        min_dist=args.min_dist,
        max_episode_steps=args.max_episode_steps,
    )

    expert_algorithm, inference_config = get_expert_algorithm_and_config(args)

    if args.imitation_learning_model == "MAGAT":
        model = DecentralPlannerGATNet(
            FOV=args.obs_radius,
            numInputFeatures=args.embedding_size,
            nGraphFilterTaps=args.num_gnn_layers,
            nAttentionHeads=args.num_attention_heads,
            use_dropout=True,
            CNN_mode=args.cnn_mode,
            attentionMode=args.attention_mode,
            AttentionConcat=True,
        ).to(device)
    else:
        raise ValueError(
            f"Unsupported imitation learning model {args.imitation_learning_model}."
        )

    optimizer = optim.Adam(model.parameters(), lr=args.lr_start, weight_decay=1e-5)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=args.lr_end
    )

    file_name = get_imitation_dataset_file_name(args)
    path = pathlib.Path(f"{args.dataset_dir}", "processed_dataset", f"{file_name}")

    with open(path, "rb") as f:
        graph_dataset = pickle.load(f)

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
        mask = torch.logical_and(graph_dataset[-1] >= start, graph_dataset[-1] < end)
        return tuple(gd[mask] for gd in graph_dataset)

    train_dataset = _divide_dataset(0, train_id_max)
    # validation_dataset = _divide_dataset(train_id_max, validation_id_max)
    # test_dataset = _divide_dataset(validation_id_max, torch.inf)

    # num_batches = (graph_dataset[0].shape[0] + args.batch_size - 1) // args.batch_size
    num_batches = (train_dataset[0].shape[0] + args.batch_size - 1) // args.batch_size

    (
        dataset_node_features,
        dataset_Adj,
        dataset_target_actions,
        dataset_terminated,
        graph_map_id,
    ) = train_dataset
    if args.normalize_adjacency:
        print("Normalizing Adjacency Matrices.....")
        for t in range(len(dataset_Adj)):
            W = dataset_Adj[t]
            W = W.numpy()
            if np.any(W):
                maxEigenValue = get_maxEigenValue(W)
                W = W / maxEigenValue
                dataset_Adj[t] = torch.from_numpy(W)
        print("Done.")

    best_validation_success_rate = 0.0
    best_val_file_name = "best_low_val.pt"
    checkpoint_path = pathlib.Path(f"{args.checkpoints_dir}", best_val_file_name)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    cur_validation_id_max = min(train_id_max + args.initial_val_size, validation_id_max)

    oe_graph_dataset = None
    oe_grid_configs = []

    def multiprocess_run_expert(
        queue,
        expert,
        grid_config,
        save_termination_state,
        additional_data_func=None,
    ):
        expert_results = run_expert_algorithm(
            expert,
            grid_config=grid_config,
            save_termination_state=save_termination_state,
            additional_data_func=additional_data_func,
        )
        queue.put((*expert_results, grid_config))

    queue = mp.Queue()

    for epoch in range(args.num_epochs):
        total_loss = 0.0
        tot_correct = 0
        num_samples = 0

        model = model.train()
        n_batches = num_batches
        for i in range(num_batches):
            cur_node_features = dataset_node_features[
                i * args.batch_size : (i + 1) * args.batch_size
            ].to(device)
            cur_adj = dataset_Adj[i * args.batch_size : (i + 1) * args.batch_size].to(
                device
            )
            cur_target_actions = (
                dataset_target_actions[i * args.batch_size : (i + 1) * args.batch_size]
                .to(device)
                .reshape(-1)
            )
            cur_terminated = (
                dataset_terminated[i * args.batch_size : (i + 1) * args.batch_size]
                .to(device)
                .reshape(-1)
            )

            optimizer.zero_grad()

            model.addGSO(cur_adj, device)
            out = model(cur_node_features, device)

            if not args.train_on_terminated_agents:
                out = out[~cur_terminated]
                cur_target_actions = cur_target_actions[~cur_terminated]
            loss = loss_function(out, cur_target_actions)

            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            tot_correct += (
                torch.sum(torch.argmax(out, dim=-1) == cur_target_actions)
                .detach()
                .cpu()
            )
            num_samples += out.shape[0]

        if oe_graph_dataset is not None:
            oe_num_batches = (
                oe_graph_dataset[0].shape[0] + args.batch_size - 1
            ) // args.batch_size
            n_batches += oe_num_batches

            (
                oe_node_features,
                oe_Adj,
                oe_target_actions,
                oe_terminated,
                _,
            ) = oe_graph_dataset

            for i in range(oe_num_batches):
                cur_node_features = oe_node_features[
                    i * args.batch_size : (i + 1) * args.batch_size
                ].to(device)
                cur_adj = oe_Adj[i * args.batch_size : (i + 1) * args.batch_size].to(
                    device
                )
                cur_target_actions = (
                    oe_target_actions[i * args.batch_size : (i + 1) * args.batch_size]
                    .to(device)
                    .reshape(-1)
                )
                cur_terminated = (
                    oe_terminated[i * args.batch_size : (i + 1) * args.batch_size]
                    .to(device)
                    .reshape(-1)
                )

                optimizer.zero_grad()

                model.addGSO(cur_adj, device)
                out = model(cur_node_features, device)

                if not args.train_on_terminated_agents:
                    out = out[~cur_terminated]
                    cur_target_actions = cur_target_actions[~cur_terminated]
                loss = loss_function(out, cur_target_actions)

                total_loss += loss.item()

                loss.backward()
                optimizer.step()

                tot_correct += (
                    torch.sum(torch.argmax(out, dim=-1) == cur_target_actions)
                    .detach()
                    .cpu()
                )
                num_samples += out.shape[0]

        lr_scheduler.step()

        print(
            f"Epoch {epoch}, Mean Loss: {total_loss / n_batches}, Mean Accuracy: {tot_correct / num_samples}"
        )

        results = {
            "train_loss": total_loss / n_batches,
            "train_accuracy": tot_correct / num_samples,
        }
        if (epoch + 1) % args.validation_every_epochs == 0:
            model = model.eval()

            num_completed = 0

            print("-------------------")
            print("Starting Validation")

            for graph_id in range(train_id_max, cur_validation_id_max):
                success, env, observations = run_model_on_grid(
                    model,
                    args.comm_radius,
                    args.obs_radius,
                    _grid_config_generator(seeds[graph_id]),
                    device,
                    collision_shielding=args.collision_shielding,
                    action_sampling=args.action_sampling,
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
                    best_val_file_name = "best.pt"
                checkpoint_path = pathlib.Path(
                    f"{args.checkpoints_dir}", best_val_file_name
                )
                torch.save(model.state_dict(), checkpoint_path)

            print("Finshed Validation")
            print("------------------")

            if args.run_online_expert and (epoch + 1 >= args.run_oe_after):
                print("---------------------")
                print("Running Online Expert")

                rng = np.random.default_rng(args.dataset_seed + epoch + 1)
                if args.recursive_oe:
                    oe_ids = rng.integers(
                        train_id_max + len(oe_grid_configs), size=args.num_run_oe
                    )
                else:
                    oe_ids = rng.integers(train_id_max, size=args.num_run_oe)

                oe_dataset = []

                for graph_id in oe_ids:
                    if graph_id > train_id_max:
                        grid_config = oe_grid_configs[graph_id - train_id_max]
                    else:
                        grid_config = _grid_config_generator(seeds[graph_id])
                    success, env, observations = run_model_on_grid(
                        model,
                        args.comm_radius,
                        args.obs_radius,
                        grid_config,
                        device,
                        collision_shielding=args.collision_shielding,
                        action_sampling=args.action_sampling,
                    )

                    if not success:
                        expert = expert_algorithm(inference_config)

                        grid_config = generate_grid_config_from_env(env)

                        all_actions, all_observations, all_terminated = None, None, None
                        if args.run_expert_in_separate_fork:
                            p = mp.Process(
                                target=multiprocess_run_expert,
                                args=(
                                    queue,
                                    expert,
                                    grid_config,
                                    args.save_termination_state,
                                ),
                            )
                            p.start()

                            while p.is_alive():
                                try:
                                    (
                                        all_actions,
                                        all_observations,
                                        all_terminated,
                                        grid_config,
                                    ) = queue.get(timeout=3)
                                    p.join()
                                    break
                                except:
                                    p.join(timeout=0.5)
                                    if p.exitcode is not None:
                                        break
                        else:
                            multiprocess_run_expert(
                                queue,
                                expert,
                                grid_config,
                                args.save_termination_state,
                            )
                            (
                                all_actions,
                                all_observations,
                                all_terminated,
                                grid_config,
                            ) = queue.get()

                        if all_actions is not None:
                            if all(all_terminated[-1]):
                                oe_dataset.append(
                                    (all_observations, all_actions, all_terminated)
                                )
                                oe_grid_configs.append(grid_config)
                while queue.qsize() > 0:
                    # Popping remaining elements, although no elements should remain
                    all_actions, all_observations, all_terminated, grid_config = (
                        queue.get()
                    )
                    oe_dataset.append((all_observations, all_actions, all_terminated))
                    oe_grid_configs.append(grid_config)

                if len(oe_dataset) > 0:
                    print(f"Adding {len(oe_dataset)} OE grids to the dataset")
                    new_oe_graph_dataset = generate_graph_dataset(
                        dataset=oe_dataset,
                        comm_radius=args.comm_radius,
                        obs_radius=args.obs_radius,
                        num_samples=None,
                        save_termination_state=True,
                        use_edge_attr=False,
                        print_prefix=None,
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
    mp.set_start_method("fork")  # TODO: Maybe add this as an cmd line option
    main()
