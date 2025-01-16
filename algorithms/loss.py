from typing import Sequence, Optional

import torch

from ranking_losses import PairwiseLogisticLoss, calculate_accuracy_for_ranking
from agents import decode_intmd_training_args


class LossWrapper(torch.nn.Module):
    def __init__(
        self,
        loss_function,
        accuracy_func,
        index_to_use=None,
        field_to_use="y",
        train_on_terminated_agents=True,
    ):
        super().__init__()
        self.loss_function = loss_function
        self.accuracy_func = accuracy_func
        self.index_to_use = index_to_use
        self.field_to_use = field_to_use
        self.train_on_terminated_agents = train_on_terminated_agents

        self.accuracy_key = "accuracy"
        if field_to_use != "y":
            self.accuracy_key = f"{field_to_use}_accuracy"

    def get_accuracies(self, out, data, split="train"):
        if self.index_to_use is not None:
            out = out[self.index_to_use]
        target_actions = data[self.field_to_use]

        if not self.train_on_terminated_agents:
            out = out[~data.terminated]
            target_actions = target_actions[~data.terminated]

        acc = self.accuracy_func(out, target_actions)
        return {f"{split}_{self.accuracy_key}": acc}

    def forward(self, out, data):
        if self.index_to_use is not None:
            out = out[self.index_to_use]
        target_actions = data[self.field_to_use]

        if not self.train_on_terminated_agents:
            out = out[~data.terminated]
            target_actions = target_actions[~data.terminated]

        loss = self.loss_function(out, target_actions)
        return loss


class CombinedLosses(torch.nn.Module):
    def __init__(
        self,
        loss_functions: Sequence[LossWrapper],
        weights: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        self.loss_functions = torch.nn.ModuleList(loss_functions)

        if weights is None:
            weights = [1.0] * len(loss_functions)
        self.weights = weights

    def get_accuracies(self, out, data, split="train"):
        accs = dict()
        for loss_function in self.loss_functions:
            accs = accs | loss_function.get_accuracies(out, data, split)
        return accs

    def forward(self, out, data):
        loss = 0.0
        for weight, loss_function in zip(self.weights, self.loss_functions):
            loss = loss + weight * loss_function(out, data)
        return loss


def default_acc(y_pred, y_true):
    return torch.sum(torch.argmax(y_pred, dim=-1) == y_true).detach().cpu()


def ranking_acc(y_pred, y_true):
    return torch.sum(calculate_accuracy_for_ranking(y_pred, y_true)).detach().cpu()


def get_loss_function(args) -> torch.nn.Module:
    if args.train_only_for_relevance:
        assert (
            args.pibt_expert_relevance_training
        ), "Need the relevance data to train for relevance."
        loss_function = PairwiseLogisticLoss()
        acc_function = default_acc
    else:
        loss_function = torch.nn.CrossEntropyLoss()
        acc_function = ranking_acc

    intmd_training, vals = decode_intmd_training_args(args)
    if not intmd_training:
        return LossWrapper(
            loss_function,
            accuracy_func=acc_function,
            train_on_terminated_agents=args.train_on_terminated_agents,
        )
    else:
        loss_function = LossWrapper(
            loss_function,
            accuracy_func=acc_function,
            index_to_use=0,
            train_on_terminated_agents=args.train_on_terminated_agents,
        )
        loss_functions = [loss_function]
        weights = [1.0]

        for i, (v, weight) in enumerate(vals):
            if v == "relevances":
                loss_function = PairwiseLogisticLoss()
                acc_function = ranking_acc
                loss_function = LossWrapper(
                    loss_function,
                    accuracy_func=acc_function,
                    index_to_use=i + 1,
                    field_to_use="relevances",
                    train_on_terminated_agents=args.train_on_terminated_agents,
                )
                loss_functions.append(loss_function)
                weights.append(weight)
            else:
                raise ValueError(f"Unsupported intmd training: {v}.")

        return CombinedLosses(loss_functions, weights)
