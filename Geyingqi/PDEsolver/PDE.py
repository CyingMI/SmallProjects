import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_min, x_max = 0.0, 2 * np.pi
t_min, t_max = 0.0, 1.0


class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x, t):
        xt = torch.cat([x, t], dim=1)
        return self.net(xt)


def grad(outputs, inputs):
    return torch.autograd.grad(
        outputs, inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True
    )[0]


def pde_residual(model, x, t):
    x.requires_grad_(True)
    t.requires_grad_(True)

    u = model(x, t)
    u_t = grad(u, t)
    u_x = grad(u, x)
    u_xx = grad(u_x, x)

    return u_t - u_xx


def initial_u(x):
    return 1 - torch.cos(x)


def sample_points(Nf, Nb, Ni):
    x_f = torch.rand(Nf, 1) * (x_max - x_min) + x_min
    t_f = torch.rand(Nf, 1) * (t_max - t_min) + t_min

    t_b = torch.rand(Nb, 1) * (t_max - t_min) + t_min
    x_b0 = torch.zeros(Nb, 1) + x_min
    x_b1 = torch.zeros(Nb, 1) + x_max

    x_i = torch.rand(Ni, 1) * (x_max - x_min) + x_min
    t_i = torch.zeros(Ni, 1)
    u_i = initial_u(x_i)

    return x_f, t_f, x_b0, x_b1, t_b, x_i, t_i, u_i


model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 10000
Nf, Nb, Ni = 2000, 200, 200

loss_history = []

for epoch in range(epochs):
    x_f, t_f, x_b0, x_b1, t_b, x_i, t_i, u_i = sample_points(Nf, Nb, Ni)

    x_f = x_f.to(device)
    t_f = t_f.to(device)
    x_b0 = x_b0.to(device)
    x_b1 = x_b1.to(device)
    t_b = t_b.to(device)
    x_i = x_i.to(device)
    t_i = t_i.to(device)
    u_i = u_i.to(device)

    f = pde_residual(model, x_f, t_f)
    loss_f = torch.mean(f ** 2)

    u_b0 = model(x_b0, t_b)
    u_b1 = model(x_b1, t_b)
    loss_b = torch.mean(u_b0 ** 2) + torch.mean(u_b1 ** 2)

    u_pred_i = model(x_i, t_i)
    loss_i = torch.mean((u_pred_i - u_i) ** 2)

    loss = loss_f + loss_b + loss_i

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch+1}, Loss = {loss.item():.6e}")


plt.plot(loss_history)
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()


x_test = np.linspace(x_min, x_max, 200)
t_test = np.linspace(t_min, t_max, 100)
X, T = np.meshgrid(x_test, t_test)

x_flat = torch.tensor(X.reshape(-1, 1), dtype=torch.float32).to(device)
t_flat = torch.tensor(T.reshape(-1, 1), dtype=torch.float32).to(device)

with torch.no_grad():
    U_pred = model(x_flat, t_flat).cpu().numpy().reshape(100, 200)

plt.figure(figsize=(8, 4))
plt.contourf(X, T, U_pred, 100)
plt.colorbar()
plt.xlabel("x")
plt.ylabel("t")
plt.title("PINN Solution")
plt.show()