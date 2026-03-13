from model import PINN
import torch

model=PINN(2,64,64,64,64,1).to('cuda')
save_path=r'D:\work\cuiyingMI\projects\project1\SmallProjects\zhangsihan\PDESolver\trainpath\PDE.pth'

def  train(model,epochs,lr):
    optim=torch.optim.Adam(model.parameters(),lr=lr)

    Nf = 10000
    x_f = (2*torch.pi) * torch.rand(Nf, 1).to('cuda')
    t_f = torch.rand(Nf, 1).to('cuda')

    Nb = 2000
    t_b = torch.rand(Nb, 1).to('cuda')
    x_b0 = torch.zeros(Nb, 1).to('cuda')
    x_b1 = (2*torch.pi)*torch.ones(Nb, 1).to('cuda')

    Ni = 2000
    x_i = (2*torch.pi) * torch.rand(Ni, 1).to('cuda')
    t_i = torch.zeros(Ni, 1).to('cuda')
    u_i = 1 - torch.cos(x_i)


    for epoch in range (epochs):
        optim.zero_grad()
        x_f.requires_grad = True
        t_f.requires_grad = True
        u_f = model(x_f,t_f)
        u_t = torch.autograd.grad(u_f, t_f, grad_outputs=torch.ones_like(u_f),
                            create_graph=True)[0]
        u_x = torch.autograd.grad(u_f, x_f, grad_outputs=torch.ones_like(u_f),
                            create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_f, grad_outputs=torch.ones_like(u_f),
                             create_graph=True)[0]
        loss_pde=torch.mean((u_t-u_xx)**2)
        loss_bc=torch.mean((model(x_b0,t_b)-0)**2) + torch.mean((model(x_b1,t_b)-0)**2)
        loss_ic=torch.mean((model(x_i,t_i)-u_i)**2)

        loss=loss_pde+loss_bc+loss_ic

        loss.backward()
        optim.step()
        print('Epoch: {}, Loss: {}'.format(epoch,loss.item()))

train(model,3000,0.001)
torch.save(model.state_dict(), save_path)
print("模型已保存！")






    