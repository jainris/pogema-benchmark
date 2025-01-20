import argparse


def add_training_args(parser):
    parser.add_argument("--validation_fraction", type=float, default=0.15)
    parser.add_argument("--test_fraction", type=float, default=0.15)
    parser.add_argument("--num_training_oe", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--imitation_learning_model", type=str, default="MAGAT")
    parser.add_argument("--cnn_mode", type=str, default="basic-CNN")
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--num_gnn_layers", type=int, default=3)
    parser.add_argument("--num_attention_heads", type=int, default=1)
    parser.add_argument("--attention_mode", type=str, default="GAT_modified")
    parser.add_argument("--edge_dim", type=int, default=None)
    parser.add_argument("--model_residuals", type=str, default=None)
    parser.add_argument(
        "--use_edge_weights", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--use_edge_attr", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--hyperedge_feature_generator", type=str, default="gcn")

    parser.add_argument("--agent_network_type", type=str, default="single")

    parser.add_argument("--model2_imitation_learning_model", type=str, default="MAGAT")
    parser.add_argument("--model2_embedding_size", type=int, default=128)
    parser.add_argument("--model2_num_gnn_layers", type=int, default=3)
    parser.add_argument("--model2_num_attention_heads", type=int, default=1)
    parser.add_argument("--model2_attention_mode", type=str, default="GAT_modified")
    parser.add_argument("--model2_edge_dim", type=int, default=None)
    parser.add_argument("--model2_model_residuals", type=str, default=None)
    parser.add_argument(
        "--model2_use_edge_weights",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--model2_use_edge_attr", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--model2_hyperedge_feature_generator", type=str, default="gcn")

    parser.add_argument("--load_partial_parameters_path", type=str, default=None)
    parser.add_argument("--replace_model", type=str, default=None)
    parser.add_argument("--parameters_to_load", type=str, default="all")
    parser.add_argument("--parameters_to_freeze", type=str, default=None)

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
    parser.add_argument(
        "--skip_validation_accuracy",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--model_seed", type=int, default=42)
    parser.add_argument("--initial_val_size", type=int, default=128)
    parser.add_argument("--threshold_val_success_rate", type=float, default=0.9)
    parser.add_argument("--num_run_oe", type=int, default=500)
    parser.add_argument("--run_oe_after", type=int, default=0)
    parser.add_argument(
        "--recursive_oe", action=argparse.BooleanOptionalAction, default=False
    )

    parser.add_argument(
        "--load_positions_separately",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--train_on_terminated_agents",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--run_expert_in_separate_fork",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--use_target_vec", type=str, default=None)
    parser.add_argument("--collision_shielding", type=str, default="naive")
    parser.add_argument("--action_sampling", type=str, default="deterministic")

    parser.add_argument(
        "--train_only_for_relevance",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--use_relevances", type=str, default=None)
    parser.add_argument("--edge_attr_opts", type=str, default="straight")
    parser.add_argument("--pre_gnn_embedding_size", type=int, default=None)
    parser.add_argument("--pre_gnn_num_mlp_layers", type=int, default=None)

    parser.add_argument("--intmd_training", type=str, default=None)
    parser.add_argument(
        "--pass_cnn_output_to_gnn2", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--pairwise_loss", type=str, default="logistic")
    parser.add_argument("--collision_shielding_args", type=str, default="")
    parser.add_argument("--collision_shielding_model_epoch_num", type=str, default=None)

    return parser