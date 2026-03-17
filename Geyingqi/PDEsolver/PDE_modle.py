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