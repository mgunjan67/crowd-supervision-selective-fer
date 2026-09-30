"""Minimal Sharpness-Aware Minimization optimizer wrapper."""

from __future__ import annotations

import torch


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        if rho < 0:
            raise ValueError("rho must be non-negative")
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                displacement = parameter.grad * scale.to(parameter)
                parameter.add_(displacement)
                self.state[parameter]["displacement"] = displacement
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                parameter.sub_(self.state[parameter]["displacement"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        device = self.param_groups[0]["params"][0].device
        norms = [
            parameter.grad.norm(p=2).to(device)
            for group in self.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        return torch.stack(norms).norm(p=2)
