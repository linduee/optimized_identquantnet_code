import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def weightedCrossEntropy(output, label, threshold, batch_size):
    res = torch.abs(torch.argmax(output, axis=1) - label).detach().cpu()
    loss_weight = torch.where(
        res >= threshold,
        (1 / res) * np.log(res),
        np.log(threshold - res + 1) * (1 / threshold),
    )
    cross_entropy_loss = F.cross_entropy(output, label)
    loss_list = cross_entropy_loss * loss_weight
    loss = loss_list.sum() / batch_size
    return loss


def training(model, optimizer, criterion, dataloader, device, lossfunction_index, label_index, feature_index=None):
    train_acc = 0
    train_loss = 0
    total = 0
    model.train()
    for inputs_org, labels_org in dataloader:
        if feature_index is None:
            inputs, labels = inputs_org.to(device), labels_org[:, label_index].to(device)
        else:
            inputs, labels = inputs_org[:, feature_index].to(device), labels_org[:, label_index].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        if lossfunction_index[0] == 0:
            loss = criterion(outputs, labels[:, 0].long())
        elif lossfunction_index[0] == 1:
            batch_size = labels.size()[0]
            loss = weightedCrossEntropy(outputs, labels[:, 0].long(), lossfunction_index[1], batch_size)
        else:
            raise ValueError('Unsupported lossfunction_index for classification training.')

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        total += labels.size(0)
        train_acc += torch.argmax(outputs, axis=1).eq(labels[:, 0]).sum().item()

    train_acc = train_acc / total
    train_loss = train_loss / len(dataloader)
    return train_acc, train_loss


def validation(model, criterion, dataloader, device, lossfunction_index, label_index, feature_index=None):
    model.eval()
    val_acc = 0
    val_loss = 0
    total = 0

    with torch.no_grad():
        for inputs_org, labels_org in dataloader:
            if feature_index is None:
                inputs, labels = inputs_org.to(device), labels_org[:, label_index].to(device)
            else:
                inputs, labels = inputs_org[:, feature_index].to(device), labels_org[:, label_index].to(device)

            outputs = model(inputs)
            if lossfunction_index[0] == 0:
                loss = criterion(outputs, labels[:, 0].long())
            elif lossfunction_index[0] == 1:
                batch_size = labels.size()[0]
                loss = weightedCrossEntropy(outputs, labels[:, 0].long(), lossfunction_index[1], batch_size)
            else:
                raise ValueError('Unsupported lossfunction_index for classification validation.')

            val_loss += loss.item()
            total += labels.size(0)
            val_acc += torch.argmax(outputs, axis=1).eq(labels[:, 0]).sum().item()

    val_acc = val_acc / total
    val_loss = val_loss / len(dataloader)
    return val_acc, val_loss


def run(model, train_loader, val_loader, num_epochs, optimizer, criterion, device, lossfunction_index, label_index, feature_index=None, outpyt_prob=False):
    train_acc_list = []
    val_acc_list = []
    train_loss_list = []
    val_loss_list = []

    for epoch in range(num_epochs):
        if feature_index is None:
            train_acc, train_loss = training(model, optimizer, criterion, train_loader, device, lossfunction_index, label_index)
            val_acc, val_loss = validation(model, criterion, val_loader, device, lossfunction_index, label_index)
        else:
            train_acc, train_loss = training(model, optimizer, criterion, train_loader, device, lossfunction_index, label_index, feature_index)
            val_acc, val_loss = validation(model, criterion, val_loader, device, lossfunction_index, label_index, feature_index)

        print(f'Epoch [{epoch + 1}], train_acc : {train_acc:.4f}, val_acc : {val_acc:.4f}, train_Loss : {train_loss:.4f}, val_Loss : {val_loss:.4f}')
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
    return train_acc_list, val_acc_list, train_loss_list, val_loss_list


def training_regression(model, optimizer, criterion, dataloader, device, label_index, loss_type, feature_index=None):
    train_loss = 0
    model.train()
    for inputs_org, labels_org in dataloader:
        if feature_index is None:
            inputs, labels = inputs_org.to(device), labels_org[:, label_index].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.float(), labels.float())
        else:
            if all(isinstance(item, list) for item in feature_index):
                inputs = torch.cat((inputs_org[:, 0, feature_index[0]], inputs_org[:, 1, feature_index[1]]), dim=1).unsqueeze(1)
                labels = labels_org.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = 0
                for k in range(len(feature_index)):
                    loss += criterion(outputs[k].float(), labels[:, k].view(-1, 1, 1).float())
            else:
                if inputs_org.dim() == 2:
                    inputs = inputs_org[:, feature_index].to(device)
                elif inputs_org.dim() == 3:
                    inputs = inputs_org[:, 0, feature_index].to(device)
                else:
                    raise ValueError('Unsupported input dimension for regression training.')
                labels = labels_org[:, label_index].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.float(), labels.float())

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss = train_loss / len(dataloader)
    return train_loss


def validation_regression(model, criterion, dataloader, device, label_index, loss_type, feature_index=None):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for inputs_org, labels_org in dataloader:
            if feature_index is None:
                inputs, labels = inputs_org.to(device), labels_org[:, label_index].to(device)
                outputs = model(inputs)
                loss = criterion(outputs.float(), labels.float())
            else:
                if all(isinstance(item, list) for item in feature_index):
                    inputs = torch.cat((inputs_org[:, 0, feature_index[0]], inputs_org[:, 1, feature_index[1]]), dim=1).unsqueeze(1)
                    labels = labels_org.to(device)
                    outputs = model(inputs)
                    loss = 0
                    for k in range(len(feature_index)):
                        loss += criterion(outputs[k].float(), labels[:, k].view(-1, 1, 1).float())
                else:
                    if inputs_org.dim() == 2:
                        inputs = inputs_org[:, feature_index].to(device)
                    elif inputs_org.dim() == 3:
                        inputs = inputs_org[:, 0, feature_index].to(device)
                    else:
                        raise ValueError('Unsupported input dimension for regression validation.')
                    labels = labels_org[:, label_index].to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs.float(), labels.float())

            val_loss += loss.item()

    val_loss = val_loss / len(dataloader)
    return val_loss


def run_regression(model, train_loader, val_loader, num_epochs, optimizer, criterion, device, label_index, loss_type, feature_index=None):
    train_loss_list = []
    val_loss_list = []
    for epoch in range(num_epochs):
        if feature_index is None:
            train_loss = training_regression(model, optimizer, criterion, train_loader, device, label_index, loss_type)
            val_loss = validation_regression(model, criterion, val_loader, device, label_index, loss_type)
        else:
            train_loss = training_regression(model, optimizer, criterion, train_loader, device, label_index, loss_type, feature_index)
            val_loss = validation_regression(model, criterion, val_loader, device, label_index, loss_type, feature_index)
        print(f'Epoch [{epoch + 1}], train_Loss : {train_loss:.4f}, val_Loss : {val_loss:.4f}')
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
    return train_loss_list, val_loss_list


class ANN_identification(nn.Module):
    def __init__(self, device, net_layer_size_list, feature_size, label_size, activation_func):
        super().__init__()
        self.device = device
        self.input_channels = feature_size
        self.output_channels = label_size
        self.activation_func = activation_func
        self.model_layers = self.make_layers_seq(net_layer_size_list, self.input_channels, self.output_channels, self.activation_func)

    def make_layers_seq(self, size_list, input_channels, output_channels, activation_func):
        layers = []
        for size in size_list:
            layers += [nn.Linear(input_channels, size)]
            if activation_func == 'relu':
                layers += [nn.ReLU()]
            elif activation_func == 'leaky_relu':
                layers += [nn.LeakyReLU(0.1)]
            elif activation_func == 'elu':
                layers += [nn.ELU()]
            input_channels = size
        layers += [nn.Linear(input_channels, output_channels)]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model_layers(x)
