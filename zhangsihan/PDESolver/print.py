import torch
import model
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

model_path = r'D:\work\cuiyingMI\projects\project1\SmallProjects\zhangsihan\PDESolver\trainpath\PDE.pth'
save_file_path =r'D:\work\cuiyingMI\projects\project1\SmallProjects\zhangsihan\PDESolver\picture\heat_solution.png'

def print_pic(model_path):
    Model = model.PINN(2,64,64,64,64,1).to('cuda')
    Model.load_state_dict(torch.load(model_path, map_location='cuda'))
    Model.eval()
    x_range = np.linspace(0, 2 * np.pi, 100)
    t_range = np.linspace(0, 1, 100)
    X, T = np.meshgrid(x_range, t_range)

    x_tensor = torch.tensor(X.flatten(), dtype=torch.float32).reshape(-1, 1).to('cuda')
    t_tensor = torch.tensor(T.flatten(), dtype=torch.float32).reshape(-1, 1).to('cuda')

    with torch.no_grad():

        u_pred_tensor = Model(x_tensor, t_tensor)
        u_pred = u_pred_tensor.cpu().numpy().reshape(100, 100)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, T, u_pred, cmap='viridis', edgecolor='none', alpha=0.9)

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    ax.set_xlabel('Spatial coordinate x')
    ax.set_ylabel('Time t')
    ax.set_zlabel('Temperature u(x, t)')
    ax.set_title('PINN Solution: Heat Equation Surface')
    
    ax.view_init(elev=30, azim=45)

    plt.savefig(save_file_path, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

print_pic(model_path)