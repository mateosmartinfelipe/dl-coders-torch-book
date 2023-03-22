from typing import Tuple

import matplotlib.pyplot as plt
import torch

# data to fit
time = torch.arange(0, 20)
speed = torch.randn(20) * 3 + 0.75 * (time - 9.5) ** 2 + 1
plt.scatter(time, speed)
plt.show()

# init params
params = torch.rand(3).requires_grad_()

# Lets see how are we


def f(time: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """_summary_

    Args:
        t (int): _description_
        params (Tuple[float]): _description_

    Returns:
        float: _description_
    """
    a, b, c = params
    return a * (time**2) + (b * time) + c


def mse(pred: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """_summary_

    Args:
        pred (float): _description_
        targets (float): _description_

    Returns:
        float: _description_
    """
    return ((pred - targets) ** 2).mean().sqrt()


def show_preds(
    pred: torch.Tensor, target: torch.Tensor, time: torch.Tensor, ax=None
) -> None:
    """_summary_

    Args:
        pred (float): _description_
        target (float): _description_
        time (float): _description_
        ax (_type_, optional): _description_. Defaults to None.
    """
    if ax is None:
        ax = plt.subplots()[1]
    ax.scatter(time, target)
    ax.scatter(time, pred, color="red")
    ax.set_ylim(-300, 100)
    plt.show()


def apply_step(
    lr: float, targets: torch.Tensor, params: torch.Tensor, prn: bool = True
) -> torch.Tensor:
    """_summary_

        Args:
            lr (float): _description_
            targets (float): _description_
            params (Tuple[float]): _description_
            prn (bool, optional): _description_. Defaults to True.

    Returns:
            float: _description_
    """
    preds = f(targets, params)
    loss = mse(preds, targets)
    loss.backward()
    params.data -= lr * params.grad.data
    params.grad = None
    if prn:
        print(loss.item())
    return preds


for _ in range(100):
    preds = apply_step(lr=0.00001, targets=speed, params=params)

show_preds(pred=preds.detach(), target=speed, time=time)
