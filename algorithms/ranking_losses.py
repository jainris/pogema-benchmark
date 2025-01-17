from typing import Optional

import torch
from torch import Tensor


class PairwiseLogisticLoss(torch.nn.Module):
    def __init__(
        self,
        temperature: float = 1.0,
        margin: Optional[float] = None,
        softmax_outputs: bool = False,
    ):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.softmax_outputs = softmax_outputs
        if margin is not None:
            raise NotImplementedError(
                "Yet to implement margins for pairwise logistic loss."
            )

    def forward(self, scores: Tensor, relevance: Tensor):
        if self.softmax_outputs:
            scores = torch.nn.functional.softmax(scores, dim=-1)
        scores_i = torch.unsqueeze(scores, dim=-1)
        scores_j = torch.unsqueeze(scores, dim=-2)
        score_diff = scores_i - scores_j

        relevance_i = torch.unsqueeze(relevance, dim=-1)
        relevance_j = torch.unsqueeze(relevance, dim=-2)
        relevance_diff = relevance_i - relevance_j

        # loss = torch.log2(1 + torch.exp(-self.temperature * score_diff))
        loss = torch.log1p(torch.exp(-self.temperature * score_diff))
        loss = torch.mean(loss[relevance_diff > 0])

        return loss


class PairwiseDiffLoss(torch.nn.Module):
    def __init__(
        self,
        temperature: float = 1.0,
        margin: Optional[float] = None,
    ):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        if margin is not None:
            raise NotImplementedError(
                "Yet to implement margins for pairwise logistic loss."
            )

    def forward(self, scores: Tensor, relevance: Tensor):
        scores = torch.nn.functional.softmax(scores, dim=-1)
        scores_i = torch.unsqueeze(scores, dim=-1)
        scores_j = torch.unsqueeze(scores, dim=-2)
        score_diff = scores_i - scores_j

        relevance_i = torch.unsqueeze(relevance, dim=-1)
        relevance_j = torch.unsqueeze(relevance, dim=-2)
        relevance_diff = relevance_i - relevance_j

        loss = -self.temperature * score_diff
        loss = torch.mean(loss[relevance_diff > 0])

        return loss


def get_ranking_loss(pairwise_loss="logistic"):
    if pairwise_loss == "logistic":
        return PairwiseLogisticLoss()
    elif pairwise_loss == "logistic-softmax":
        return PairwiseLogisticLoss(softmax_outputs=True)
    elif pairwise_loss == "diff":
        return PairwiseDiffLoss()
    else:
        raise ValueError(f"Unsupported ranking loss type: {pairwise_loss}.")


def calculate_accuracy_for_ranking(y_pred, y_target):
    # y_pred, y_target: [N, C]
    # Getting the predicted ranking
    y_pred_idx = torch.argsort(y_pred, descending=True)

    sorted_y_target_vals, _ = torch.sort(y_target, descending=True)

    batch_ids = torch.unsqueeze(
        torch.arange(y_pred.shape[0], device=y_pred_idx.device), dim=-1
    )
    sorted_y_pred_vals = y_target[batch_ids, y_pred_idx]

    return torch.mean(
        sorted_y_target_vals == sorted_y_pred_vals, dim=-1, dtype=torch.float
    )
