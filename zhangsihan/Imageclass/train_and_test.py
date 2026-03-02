from model import ResNet
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

train_path = r'D:\work\cuiyingMI\projects\project1\SmallProjects\zhangsihan\Imageclass\dataset\train'
test_path = r'D:\work\cuiyingMI\projects\project1\SmallProjects\zhangsihan\Imageclass\dataset\valid'
save_path = r'D:\work\cuiyingMI\projects\project1\SmallProjects\zhangsihan\Imageclass\trainpath\Imageclass_resnet50.pth'


transforms1 = v2.Compose([
v2.ToImage(),
v2.CenterCrop(size=(256, 256)),
v2.RandomHorizontalFlip(),
v2.RandomVerticalFlip(),
v2.ToDtype(torch.float32,scale=True),
v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ]
)
transforms2 = v2.Compose([
v2.ToImage(),
v2.CenterCrop(size=(256, 256)),
v2.ToDtype(torch.float32,scale=True),
v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ]
)


train_dataset =ImageFolder(train_path,transforms1)
test_dataset = ImageFolder(test_path,transforms2)
data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
model = ResNet(num_res=20).to('cuda')

def train(model,dataloader,epochs,lr):
    optim = torch.optim.Adam(model.parameters(),lr=lr)
    loss_fun = nn.CrossEntropyLoss()
    loss_history = []
    for i in range(epochs):
        model.train()
        running_loss = 0.0

        for a, (data,label) in enumerate(dataloader):
            data = data.to('cuda')
            label = label.to('cuda')
            optim.zero_grad()
            pred = model(data)
            loss = loss_fun(pred,label)
            loss.backward()
            optim.step()
            running_loss += loss.item()
            if a % 10 == 0:
                print(f"Epoch [{i+1}/{epochs}], Batch [{a}/{len(dataloader)}], Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / len(dataloader)
        loss_history.append(epoch_loss)
        print(f"Epoch {i+1} Average Loss: {epoch_loss:.4f}")

    return loss_history

def test(model,dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data,label in dataloader:
            data = data.to('cuda')
            label = label.to('cuda')
            pred = model(data)
            _, predicted = torch.max(pred.data,1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    print(f"Accuracy of the network on the test images: {100 * correct / total}%")

train(model,data_loader,40,1e-4)
test_acc = test(model, test_loader)
torch.save(model.state_dict(), save_path)
print("模型已保存！")