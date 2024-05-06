#!/usr/bin/env conda run -n pytorch02 python


import torch
import torch.nn as nn
import os

class ChestXRayMapDataset(torch.utils.data.Dataset):
    def __init__(self, preprocessed_data_file):
        self.preprocessed_data = torch.load(preprocessed_data_file)

    def __len__(self):
        return len(self.preprocessed_data)

    def __getitem__(self, index):
        image, label = self.preprocessed_data[index]
        return image, label


# PATH to the pre-processed datasets
train_output_file = "preprocessed_datasets/preprocessed_train_data.pt"
# val_output_file = "preprocessed_datasets/preprocessed_val_data.pt"
test_output_file = "preprocessed_datasets/preprocessed_test_data.pt"


train_map_dataset = ChestXRayMapDataset(train_output_file)
# val_map_dataset = ChestXRayMapDataset(val_output_file)
test_map_dataset = ChestXRayMapDataset(test_output_file)

# Create a data loader
batch_size = 256
train_map_data_loader = torch.utils.data.DataLoader(train_map_dataset, batch_size=batch_size, shuffle=True)
# val_map_data_loader = torch.utils.data.DataLoader(val_map_dataset, batch_size=batch_size, shuffle=True)
test_map_data_loader = torch.utils.data.DataLoader(test_map_dataset, batch_size=batch_size, shuffle=True)


# Implementation of AlexNet
net = nn.Sequential(
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 3))


def try_gpu(index=0):
    """Return GPU(index) if exists, otherwise return cpu()

    Args:
        index (int, optional): index of the GPU you want . Defaults to 0.
    """
    if torch.cuda.device_count() >= (index+1):
        return torch.device(f'cuda:{index}')
    else:
        return torch.device('cpu')

def train_model(net, train_iter, val_iter, num_epochs, lr, device, model_name):
    """Train the AlexNet with a GPU

    Args:
        net (_type_): ALexNet for classifing images
        train_iter (_type_): _description_
        val_iter (_type_): _description_
        num_epochs (_type_): _description_
        lr (_type_): _description_
        device (_type_): _description_
        model_name (_type_): _description_
    """

    #Init the weights with xavier method
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    
    net.apply(init_weights)

    #Print the device for training
    net.to(device)
    print(model_name)
    print('training on:', device)

    weight_decay = 3e-2
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss = nn.CrossEntropyLoss()

    # d2l to print block 0 begin
    def evaluate_accuracy(net, data_iter, device=None):
        if isinstance(net, nn.Module):
            net.eval()
            if not device:
                # net.parameters() is already a iterable object, the func iter() only adds readability to the code
                device = next(iter(net.parameters())).device
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in data_iter:
                #X, y represent a batch of data
                if isinstance(X, list):
                    X = [x.to(device) for x in X]
                else:
                    X = X.to(device)
                y = y.to(device)
                y_hat = net(X)
                _, predicted = torch.max(y_hat, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        return correct / total
    # d2l to print block 0 end

    for epoch in range(num_epochs):
        train_loss = 0.0
        train_acc = 0.0
        total_samples = 0
        net.train()
        for i, (X, y) in enumerate(train_iter):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()

            train_loss += l.item() * X.size(0)
            _, predicted = torch.max(y_hat, 1)
            train_acc += (predicted == y).sum().item()
            total_samples += X.size(0)

        train_loss /= total_samples
        train_acc /= total_samples
        val_acc = evaluate_accuracy(net, val_iter, device)
        delta = (train_acc - val_acc)/train_acc
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, delta: {(100*delta):.2f}%')

        #save parameters every 20 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss
            }
            folder_name = f'model_parameters/{model_name}'
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            model_path = f'model_parameters/{model_name}/{model_name}_epoch_{epoch + 1}.pth'
            torch.save(checkpoint, model_path)

lr, num_epochs = 0.007, 80
train_model(net, train_map_data_loader, test_map_data_loader, num_epochs, lr, try_gpu(), model_name="fourth_training")


exit()