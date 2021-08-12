import torch
from torchmetrics import Metric, FID as torchFID


class FID(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.fid = torchFID(feature=64).cuda()

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape

        self.fid.update(target.to(torch.uint8), real=True)
        self.fid.update(preds.to(torch.uint8), real=False)
        self.total += self.fid.compute()
        self.count += 1
        self.fid.reset()

    def compute(self):
        return self.total / self.count
