from typing import Sequence, Optional

import torch

from ranking_losses import get_ranking_loss, calculate_accuracy_for_ranking
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


class FirstNStepLoss(torch.nn.Module):
    r"""Calculates the loss for first action in a two step predictions."""

    def __init__(self, num_classes=5, n_steps=1):
        super().__init__()
        self.num_classes = num_classes
        self.n_steps = n_steps
        self.output_dim = num_classes**n_steps
        self.loss = torch.nn.CrossEntropyLoss()

    def get_accuracy(self, x, y):
        x = x.reshape((x.shape[0], self.output_dim, -1))
        x = torch.logsumexp(x, dim=-1, keepdim=False)
        return default_acc(x, y)

    def forward(self, x, y):
        x = x.reshape((x.shape[0], self.output_dim, -1))
        x = torch.logsumexp(x, dim=-1, keepdim=False)
        return self.loss(x, y)


def default_acc(y_pred, y_true):
    return torch.sum(torch.argmax(y_pred, dim=-1) == y_true).detach().cpu()


def ranking_acc(y_pred, y_true):
    return torch.sum(calculate_accuracy_for_ranking(y_pred, y_true)).detach().cpu()


def get_n_step_weights(train_n_steps, train_n_steps_weight) -> Sequence[float]:
    train_n_steps_weight = [float(f) for f in train_n_steps_weight.split("+")]
    if len(train_n_steps_weight) == 1:
        train_n_steps_weight = train_n_steps_weight * (train_n_steps - 1)
    assert len(train_n_steps_weight) == (
        train_n_steps - 1
    ), f"Expecting 1 or {train_n_steps - 1} train_n_steps weights, but got {len(train_n_steps_weight)}."
    return train_n_steps_weight


def get_loss_function(args) -> torch.nn.Module:
    if args.train_n_steps:
        assert not args.train_only_for_relevance
        loss_function = FirstNStepLoss(n_steps=1)
        acc_function = loss_function.get_accuracy
    elif args.train_only_for_relevance:
        assert (
            args.pibt_expert_relevance_training
        ), "Need the relevance data to train for relevance."
        loss_function = get_ranking_loss(args.pairwise_loss)
        acc_function = ranking_acc
    else:
        loss_function = torch.nn.CrossEntropyLoss()
        acc_function = default_acc

    intmd_training, vals = decode_intmd_training_args(args)
    if not intmd_training:
        if args.train_n_steps:
            loss_functions = [
                LossWrapper(
                    loss_function,
                    accuracy_func=acc_function,
                    train_on_terminated_agents=args.train_on_terminated_agents,
                )
            ]
            for num_steps in range(2, args.train_n_steps):
                lf = FirstNStepLoss(n_steps=num_steps)
                loss_functions.append(
                    LossWrapper(
                        lf,
                        accuracy_func=lf.get_accuracy,
                        field_to_use=f"y_{num_steps}_step",
                        train_on_terminated_agents=args.train_on_terminated_agents,
                    )
                )
            loss_functions.append(
                LossWrapper(
                    torch.nn.CrossEntropyLoss(),
                    accuracy_func=default_acc,
                    field_to_use=f"y_{args.train_n_steps}_step",
                    train_on_terminated_agents=args.train_on_terminated_agents,
                )
            )
            return CombinedLosses(
                loss_functions,
                weights=[1.0]
                + get_n_step_weights(args.train_n_steps, args.train_n_steps_weight),
            )
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
                loss_function = get_ranking_loss(args.pairwise_loss)
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

        if args.train_n_steps:
            for num_steps in range(2, args.train_n_steps):
                lf = FirstNStepLoss(n_steps=num_steps)
                loss_functions.append(
                    LossWrapper(
                        lf,
                        accuracy_func=lf.get_accuracy,
                        index_to_use=0,
                        field_to_use=f"y_{num_steps}_step",
                        train_on_terminated_agents=args.train_on_terminated_agents,
                    )
                )
            loss_functions.append(
                LossWrapper(
                    torch.nn.CrossEntropyLoss(),
                    accuracy_func=default_acc,
                    index_to_use=0,
                    field_to_use=f"y_{args.train_n_steps}_step",
                    train_on_terminated_agents=args.train_on_terminated_agents,
                )
            )
            weights = weights + get_n_step_weights(
                args.train_n_steps, args.train_n_steps_weight
            )

        return CombinedLosses(loss_functions, weights)
