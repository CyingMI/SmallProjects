import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 网络
class PINN(nn.Module):
    def __init__(self, layers_config):
        super().__init__()
        self.layer_config = layers_config
        self.act = nn.Tanh()
        self.layers = []
        for i in range(len(layers_config)-1):
            self.layers.append(nn.Linear(self.layer_config[i],self.layer_config[i+1]))
            if i != 0 and i != len(self.layer_config)-2:
                self.layers.append(nn.Tanh())
        self.model = nn.Sequential(*self.layers)

    def forward(self, x, t):
        input = torch.cat([x, t], dim=1)
        output = self.model(input)
        return output
# 定义损失函数
def pde_loss(model, x, t):
    u = model(x, t)
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0], x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    loss = u_t - u_xx
    return torch.mean(loss**2)

def bc_loss(model, x_left, x_right, t):
    u_left = model(x_left, t)
    u_right = model(x_right, t)
    return torch.mean(u_left**2)+torch.mean(u_right**2)

def ic_loss(model, x_i, t_i):
    u_pred = model(x_i, t_i)
    u = 1 - torch.cos(x_i)
    return torch.mean((u_pred - u)**2)
# 数据准备
def generate_data(num_pde, num_bc, num_ic):
    x_pde = torch.arange(0, 2 * np.pi, 2 * np.pi / num_pde).reshape(-1, 1).requires_grad_()
    t_pde = torch.arange(0, 1.0, 1.0 / num_pde).reshape(-1, 1).requires_grad_()
    x_left = torch.zeros(num_bc, 1).requires_grad_()
    x_right = torch.ones(num_bc, 1) * 2 * np.pi
    x_right.requires_grad_()
    t_bc = torch.arange(0, 1.0, 1.0 / num_bc).reshape(-1, 1).requires_grad_()
    x_ic = torch.arange(0, 2 * np.pi, 2 * np.pi / num_ic).reshape(-1, 1).requires_grad_()
    t_ic = torch.zeros(num_ic, 1).requires_grad_()
    return (x_pde.to(device), t_pde.to(device), x_left.to(device), x_right.to(device), t_bc.to(device), x_ic.to(device), t_ic.to(device))
# 训练和可视化
def train(model, optimizer, epochs, num_pde, num_bc, num_ic):
    model.train()
    epoch_list = []
    pde_loss_list = []
    bc_loss_list = []
    ic_loss_list = []
    loss_list = []
    pbar = tqdm(range(epochs), desc='description')
    for _ in pbar:
        x_pde, t_pde, x_left, x_right, t_bc, x_ic, t_ic = generate_data(num_pde, num_bc, num_ic)
        optimizer.zero_grad()
        loss_pde = pde_loss(model, x_pde, t_pde)
        loss_bc = bc_loss(model, x_left, x_right, t_bc)
        loss_ic = ic_loss(model, x_ic, t_ic)
        loss = loss_pde + loss_bc*2.6 + loss_ic*0.33
        loss.backward()
        optimizer.step()
        pbar.set_description(f'Loss: {loss.item():.2e}, PDE Loss: {loss_pde.item():.2e}, BC Loss: {loss_bc.item():.2e}, IC Loss: {loss_ic.item():.2e}')
        if _ % 1 == 0:
            epoch_list.append(_)
            pde_loss_list.append(loss_pde.item())
            bc_loss_list.append(loss_bc.item())
            ic_loss_list.append(loss_ic.item())
            loss_list.append(loss.item())

# 绘制模型解和损失曲线
    model.eval()
    x_plot = torch.linspace(0,2*np.pi, 100).reshape(-1,1).to(device)
    t_plot = torch.linspace(0,1,100).reshape(-1,1).to(device)

    X,T = torch.meshgrid(x_plot.squeeze(), t_plot.squeeze(), indexing='ij')
    X_flat = X.reshape(-1,1)
    T_flat = T.reshape(-1,1)

    with torch.no_grad():
        u_pred = model(X_flat,T_flat).reshape(100,100).cpu().numpy()

    x_grid = X.cpu().numpy()
    t_grid = T.cpu().numpy()
    
    fig = plt.figure(figsize=(12,5))
    ax1 = fig.add_subplot(3,1,1, projection='3d')
    ax1.plot_surface(x_grid, t_grid, u_pred, cmap='rainbow')
    ax1.set_title('Predicted Solution u(x,t)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('t')
    ax1.set_zlabel('u(x,t)')
    ax2 = fig.add_subplot(3,1,3)
    ax2.plot(epoch_list, loss_list, label='Total Loss', color='black')
    ax2.plot(epoch_list, pde_loss_list, label='PDE Loss', color='red')
    ax2.plot(epoch_list, bc_loss_list, label='BC Loss', color='blue')
    ax2.plot(epoch_list, ic_loss_list, label='IC Loss' , color='green')
    ax2.set_title('Loss Curves')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_yscale('log')
    ax2.legend()
    plt.show()
# 主程序
if __name__ == "__main__":
    layers = [2, 64, 64, 64, 64, 1]
    model = PINN(layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train(model, optimizer, epochs=5000, num_pde=10000, num_bc=2000, num_ic=2000)
    

