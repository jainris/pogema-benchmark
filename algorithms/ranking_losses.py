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
        loss = torch.sum(loss[relevance_diff > 0])

        return loss
