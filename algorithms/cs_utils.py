import argparse
from convert_to_imitation_dataset import add_imitation_dataset_args
from generate_hypergraphs import add_hypergraph_generation_args
from run_expert import add_expert_dataset_args
from training_args import add_training_args


class SansPrefixDict:
    def __init__(self, base_dict, prefix_string="", only_copy_prefix_keys=False):
        new_dict = dict()
        prefix_len = len(prefix_string)
        for key in base_dict:
            if key[:prefix_len] == prefix_string:
                new_dict[key[prefix_len:]] = base_dict[key]
            elif not only_copy_prefix_keys:
                new_dict[key] = base_dict[key]
        self.dict = new_dict

    def __getattr__(self, name):
        return self.dict[name]

    @property
    def __dict__(self):
        return self.dict


def get_collision_shielding_args_from_str(collision_shielding_args: str):
    parser = argparse.ArgumentParser(description="Train imitation learning model.")
    parser = add_expert_dataset_args(parser)
    parser = add_imitation_dataset_args(parser)
    parser = add_hypergraph_generation_args(parser)
    parser = add_training_args(parser)

    return parser.parse_args(collision_shielding_args.split(" "))
