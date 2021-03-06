import torch
from torchmetrics import Metric
from torchmetrics.functional.image.ssim import structural_similarity_index_measure


class SSIM(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        self.total += structural_similarity_index_measure(preds, target, data_range=1)
        self.count += 1

    def compute(self):
        return self.total / self.count
