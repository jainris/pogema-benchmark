import torch
from torch import Tensor


class PairwiseLogisticLoss(torch.nn.Module):
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, scores: Tensor, relevance: Tensor):
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
