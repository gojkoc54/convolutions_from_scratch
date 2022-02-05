import torch
from torch.optim.optimizer import Optimizer, required
import copy

class BetaLASSO(Optimizer):
    """
    Implementation of the proposed Beta-LASSO optimizer algorithm.
    The implementation is a combination of 3 parts:
        - basic SGD step (theta_new = theta - lr * d_theta)
        - L1 regularization (theta = theta + lambda * sign(theta))
        - Thresholding - everything in the range 
            [-lambda * beta, lambda * beta] becomes 0

    """
    def __init__(self, params, beta, lr, lambda_):
        defaults = dict(lr=lr, beta=beta, lambda_=lambda_)
        super(BetaLASSO, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(BetaLASSO, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr, beta, lambda_ = group['lr'], group['beta'], group['lambda_']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                # Basic SGD
                p.data.add_(d_p, alpha=-lr)
                # L1 regularization
                p.data.add_(lambda_ * torch.sign(p.data), alpha=-lr)
                # Beta-LASSO thresholding
                p.data[
                    (p.data > -beta*lambda_) & (p.data < beta*lambda_)
                    ] = 0

        return loss
        