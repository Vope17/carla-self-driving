from TrainingTool.MyDataset import CarlaData
from TrainingTool.MyModel import MyResNet18
from TrainingTool.Tools import train, evaluate

from torchvision import transforms, models
from torchvision.utils import save_image
from torchvision.transforms.functional import InterpolationMode

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from torch.optim.lr_scheduler import StepLR

import matplotlib.pyplot as plt

import torch

# super parameter
EPOCHS = 150
LR = 1e-4
BATCH_SIZE = 16
SIZE = (192, 256)

# read image
transform = transforms.Compose([
    transforms.Resize(SIZE, interpolation=InterpolationMode.NEAREST),
    #transforms.Grayscale(3),
    transforms.ToTensor(),
    lambda x: x[:,:,:].float()
])

dataset = CarlaData("vehicle_data.csv", "_out", transform=transform)
save_image(dataset[0][0], "test.png")
print(dataset.img_labels)

train_size = int(0.9 * len(dataset))
valid_size = len(dataset) - train_size

train_data, valid_data = random_split(dataset, [train_size, valid_size])

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=True)

# load model
myModel = torch.load("MyModel.pt")

criterion = nn.MSELoss()
optimizer = optim.SGD(myModel.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)

scheduler = StepLR(optimizer, step_size=10, gamma=0.7)

# training

loss_hist = []
val_loss_hist = []
best_score = 100
for epoch in range(EPOCHS):
    loss = train(myModel, train_loader, optimizer, criterion)
    val_loss = evaluate(myModel, valid_loader, criterion)
    loss_hist.append (loss.detach().cpu().numpy())
    val_loss_hist.append(val_loss)
    if val_loss < best_score:
        print("New score.")
        best_score = val_loss
        torch.save(myModel, "MyModel_finetune.pt")
    print("Epoch {epoch}: Loss={loss} | val_loss={val_loss}".format(epoch=epoch, loss=loss, val_loss=val_loss))
    scheduler.step()

plt.plot(loss_hist)
plt.plot(val_loss_hist)
plt.savefig("finetune_results.png")

print("Best score:", best_score)
