import typing as T

import torch
from torchmetrics import Metric


class Tracker(Metric):
    def __init__(self, compute_on_step=False, dist_sync_on_step=False):
        super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step)

        self.add_state("best_value", default=torch.tensor(0))
        self.add_state("best_buffers", default=[])

        self.add_state("worst_value", default=torch.tensor(float("inf")))
        self.add_state("worst_buffers", default=[])

    def update(self, batch_metrics: torch.Tensor, batch: T.Tuple[torch.Tensor]):
        max_val, max_ind = torch.max(batch_metrics, dim=0)
        if max_val > self.best_value:
            self.best_value = max_val
            self.best_buffers = [buffer[max_ind].unsqueeze(0) for buffer in batch]

        min_val, min_ind = torch.min(batch_metrics, dim=0)
        if min_val < self.worst_value:
            self.worst_value = min_val
            self.worst_buffers = [buffer[min_ind].unsqueeze(0) for buffer in batch]

    def compute(self):
        return {
            "best_value": self.best_value,
            "best_buffers": self.best_buffers,
            "worst_value": self.worst_value,
            "worst_buffers": self.worst_buffers,
        }
