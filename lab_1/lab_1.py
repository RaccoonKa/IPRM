# Необходимо переписать алгоритм обучения нейронной сети, с целью повышения значения accuracy.
# Можно пробовать оптимизировать код, саму нейросеть или параметры обучения.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root = "./data", train = True, download = True, transform = transform)
val_dataset = datasets.MNIST(root = "./data", train = False, download = True, transform = transform)

train_dataloader = DataLoader(train_dataset, batch_size = 64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size = 64, shuffle=False)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 10)
)

optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for x_train, y_train in tqdm(train_dataloader):
        y_pred = model(x_train)
        loss = F.cross_entropy(y_pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = []
    val_accuracy = []
    with torch.no_grad():
        for x_val, y_val in val_dataloader:
            y_pred = model(x_val)
            loss = F.cross_entropy(y_pred, y_val)
            val_loss.append(loss.item())
            val_accuracy.extend(
                (torch.argmax(y_pred, dim = -1) == y_val).numpy().tolist()
            )

    print(
        f"Epoch: {epoch + 1}, "
        f"val_loss: {np.mean(val_loss):.4f}, "
        f"val_accuracy: {np.mean(val_accuracy):.4f}"
    )

# Добавлено: оптимизатор Adam вместо SGD, больше эпох, +dropout.