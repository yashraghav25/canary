"""Gradient Reversal Layer — the mechanism behind the domain-adversarial
training described in Section 4.2 of the plan. Forward pass is identity;
backward pass negates (and scales) the gradient, so the encoder is pushed
to make the domain classifier's job *fail*."""

import torch
from torch.autograd import Function


class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return _GradReverse.apply(x, lambd)
