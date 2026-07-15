from __future__ import annotations

import torch

from em_piml.model import CavityPINN
from em_piml.physics import PERIOD, L, analytical_field, pde_residual


def train_cavity_baseline(
    steps: int = 4000,
    seed: int = 0,
    n_collocation: int = 200,
    n_boundary: int = 64,
    n_initial: int = 64,
    lr: float = 3e-3,
) -> CavityPINN:
    torch.manual_seed(seed)
    model = CavityPINN(hidden=32, num_layers=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()

        x_c = torch.rand(n_collocation, 1) * L
        t_c = torch.rand(n_collocation, 1) * PERIOD
        loss_pde = (pde_residual(model, x_c, t_c) ** 2).mean()

        t_b = torch.rand(n_boundary, 1) * PERIOD
        x_b0 = torch.zeros(n_boundary, 1)
        x_bl = torch.full((n_boundary, 1), L)
        loss_bc = (model(x_b0, t_b) ** 2).mean() + (model(x_bl, t_b) ** 2).mean()

        x_i = torch.rand(n_initial, 1) * L
        t_i = torch.zeros(n_initial, 1)
        loss_ic = ((model(x_i, t_i) - analytical_field(x_i, t_i)) ** 2).mean()

        t_i_grad = t_i.clone().requires_grad_(True)
        e_i = model(x_i, t_i_grad)
        e_i_dot = torch.autograd.grad(
            e_i, t_i_grad, grad_outputs=torch.ones_like(e_i), create_graph=True
        )[0]
        loss_ic_dot = (e_i_dot**2).mean()  # dE/dt(x, 0) = 0 for this standing-wave mode

        loss = loss_pde + loss_bc + loss_ic + loss_ic_dot
        loss.backward()
        optimizer.step()

    return model


def main() -> None:
    model = train_cavity_baseline()
    torch.manual_seed(123)
    x = torch.rand(500, 1) * L
    t = torch.rand(500, 1) * PERIOD
    with torch.no_grad():
        predicted = model(x, t)
        true = analytical_field(x, t)
    relative_l2 = (torch.linalg.norm(predicted - true) / torch.linalg.norm(true)).item()
    print(f"relative L2 error over {x.shape[0]} random (x, t) points: {relative_l2:.4f}")


if __name__ == "__main__":
    main()
