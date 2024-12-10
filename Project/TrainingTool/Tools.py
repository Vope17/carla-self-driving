from tqdm import tqdm
import torch

def train(model, data_loader, opt, criterion, device="cuda"):
    model.train()
    model.to(device)
    for batch_idx, (data, label) in tqdm(enumerate(data_loader)):
        data, label = data.to(device), label.to(device)
        opt.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        opt.step()

    return loss

def evaluate(model, data_loader, criterion, device="cuda"):
    model.eval()
    model.to(device)

    total_loss = 0

    with torch.no_grad():
        for batch_idx, (data, label) in tqdm(enumerate(data_loader)):
            data, label = data.to(device), label.to(device)
            output = model(data)
            loss = criterion(output, label)
            total_loss += loss.item()

    average_loss = total_loss / len(data_loader)

    return average_loss