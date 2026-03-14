import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_path = './ImageNet/train'
valid_path = './ImageNet/valid'
train_path = './ImageNet/train'
valid_path = './ImageNet/valid'
train_transform = v2.Compose([
    v2.RandomResizedCrop((224, 224)),
    v2.RandomHorizontalFlip(),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224,0.225]),
])
valid_transform = v2.Compose([
    v2.Resize((256, 256)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224,0.225]),
])
train_dataset  = ImageFolder(root=train_path,transform=train_transform)
valid_dataset  = ImageFolder(root=valid_path,transform=valid_transform)
train_data_loader = DataLoader(train_dataset, batch_size=4, shuffle = True, drop_last=True)
valid_data_loader = DataLoader(valid_dataset, batch_size=4, shuffle = False, drop_last=True)
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, padding=0),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        output = self.cnn(x)
        input = x
        if self.in_channels != self.out_channels:
            input = self.shortcut(x)
        output += input
        output = torch.nn.SiLU()(output)
        return output
class Resnet(nn.Module):
    def __init__(self, block=ResBlock):
        super().__init__()
        self.block = block
        base_config = [
            # 输入通道, 输出通道，隐藏通道, block数
            (64, 256, 64, 3),  # Stage 1
            (256, 512, 128, 4),  # Stage 2
            (512, 1024, 256, 6),  # Stage 3
            (1024, 2048, 512, 3),  # Stage 4
        ]

        self.layers = [nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
              nn.SiLU(),
              nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
    
        for in_channels, out_channels, hidden_channels, num_blocks in base_config:
            self.layers.append(nn.Sequential(
                *[block(in_channels if i == 0 else out_channels, out_channels, hidden_channels) for i in range(num_blocks)]
        ))

        self.layers.append(nn.AdaptiveAvgPool2d((1)))
        self.layers.append(nn.Flatten())
        self.layers.append(nn.Dropout(0.2))
        self.layers.append(nn.Linear(2048, 20))
        self.model = nn.Sequential(*self.layers)


    def forward(self, x):
        x = self.model(x)
        return x
def train(model=None, data_loader=train_data_loader, lr=0.001):
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fun = nn.CrossEntropyLoss()
    total_correct = 0
    total_loss = 0
    train_counter = 0
    model.train()
    batch_size = data_loader.batch_size
    for data, label in tqdm(data_loader, desc=f"Training"):
        data = data.to(device)
        label = label.to(device)
        optim.zero_grad()
        pred = model(data)
        loss = loss_fun(pred, label)
        loss.backward()
        optim.step()
        pred_labels = pred.argmax(dim=1)
        correct = (pred_labels == label).sum().item()
        total_correct += correct
        total_loss += loss.item()
        train_counter += 1
        if train_counter % 1000 == 0:
            print(f"Step: {train_counter}, Loss: {total_loss / train_counter}, Accuracy: {total_correct / (train_counter * batch_size)}")
    print(f"Train Accuracy: {total_correct / (len(data_loader) * batch_size)}")
    return None
def evaluate(model=None, data_loader=valid_data_loader):
    model.eval()
    
    total_correct = 0
    val_counter = 0
    total_loss = 0
    batch_size = data_loader.batch_size
    loss_fun = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, label in tqdm(data_loader, desc="Evaluating"):
            data = data.to(device)
            label = label.to(device)
            pred = model(data)
            loss = loss_fun(pred, label)
            total_loss += loss.item()
            val_counter += 1
            pred_labels = pred.argmax(dim=1)
            correct = (pred_labels == label).sum().item()
            total_correct += correct
            if val_counter % 1000 == 0:
                print(f"Step: {val_counter}, Loss: {total_loss / val_counter}, Accuracy: {total_correct / (val_counter * batch_size)}")
    print(f"Validation Accuracy: {total_correct / (len(data_loader) * batch_size)}")
    return None
def export_model(model, path="./resnet.pth"):
    torch.save(model.state_dict(), path)
def main():
    net = Resnet()
    net = net.to(device)
    for epoch in range(10):
        print(f"Epoch {epoch + 1}/10:")
        train(model=net)
        evaluate(model=net)
    print("Training complete. Exporting model...")
    export_model(net)
    print("Model exported successfully.")
main()