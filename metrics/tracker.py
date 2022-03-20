import torch
import typing as T
from torchmetrics import Metric


class Tracker(Metric):
    def __init__(self, compute_on_step=False, dist_sync_on_step=False):
        super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step)

        self.add_state("best_value", default=torch.tensor(0))
        self.add_state("best_buffers", default=[])

        self.add_state("worst_value", default=torch.tensor(float("inf")))
        self.add_state("worst_buffers", default=[])

        self.add_state("med_scores", default=[])

    def update(self, batch_metrics: torch.Tensor, batch: T.Tuple[torch.Tensor]):
        direct, gt, indirect, fake, target = batch

        self.med_scores.append(batch_metrics)

        max_val, max_ind = torch.max(batch_metrics, dim=0)
        if max_val > self.best_value:
            self.best_value = max_val
            self.best_buffers = (
                direct[max_ind].unsqueeze(0),
                gt[max_ind].unsqueeze(0),
                indirect[max_ind].unsqueeze(0),
                fake[max_ind].unsqueeze(0),
                target[max_ind].unsqueeze(0),
            )

        min_val, min_ind = torch.min(batch_metrics, dim=0)
        if min_val < self.worst_value:
            self.worst_value = min_val
            self.worst_buffers = (
                direct[min_ind].unsqueeze(0),
                gt[min_ind].unsqueeze(0),
                indirect[min_ind].unsqueeze(0),
                fake[min_ind].unsqueeze(0),
                target[min_ind].unsqueeze(0),
            )

    def compute(self):
        med_value, med_index = torch.cat(self.med_scores).median(dim=-1)
        return {
            "best_value": self.best_value,
            "best_buffers": self.best_buffers,
            "worst_value": self.worst_value,
            "worst_buffers": self.worst_buffers,
            "med_value": med_value,
            "med_index": med_index,
        }
