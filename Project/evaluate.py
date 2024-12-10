import torch

def load_model(weights_name):
    model = torch.load(weights_name)
    return model

def automatic_control(model, input, transform):
    if transform:
        input = transform(input).to('cuda')
    model.to('cuda')
    brake = 0
    steer = model(input)[0]
    return steer.item(), brake