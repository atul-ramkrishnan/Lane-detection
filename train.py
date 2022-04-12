import torch
import torchvision.transforms as transforms
import numpy as np
import config
from data import LaneData
import model_dispatcher
from early_stopping import EarlyStopping


def train(
        model,
        epoch,
        train_loader,
        optimizer,
        criterion,
        log_frequency,
        device):
    """
    This function trains the model and
    prints the loss every log_frequency batches.
    It also returns the total loss over the entire Dataloader.

    Parameters
    ----------
    model: torch.nn.Module
        Pytorch model to be trained.
    epoch: int
        Current epoch.
    train_loader: torch.utils.data.Dataloader
        PyTorch Dataloader.
    optimizer: torch.optim
        Gradient descent optimizer.
    criterion: torch.nn.CrossEntropyLoss
        Criterion for calculating loss.
    log_frequency: int
        States the frequency at which the function should print the loss.
    device: torch.device
        The device on which the training is to be done.

    Returns
    -------
    total_loss: float
        The training loss computed over the entire Dataloader.

    Example
    -------
    model = model_dispatcher["VGG11_SegNet_pretrained"]
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.nn.Adam(model.parameters())
    device = torch.device('cpu')
    log_frequency = 10
    train_loader = torch.utils.data.DataLoader(
                    LaneData(
                            index_path=train_path,
                            return_sequence=False,
                            transform=transformation,
                            target_transform=transformation,
                            max_samples=None),
                            batch_size=batch_size,
                            shuffle=True
                           )
    for epoch in range(1, 10):
        train(model,
              epoch,
              train_loader,
              optimizer,
              criterion,
              log_frequency,
              device)

        val(model,
            epoch,
            val_loader,
            criterion,
            device)
    """
    model.train()
    total_loss = 0
    for i, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.type(torch.LongTensor).to(device)
        output = model(data)
        loss = criterion(output, target)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % log_frequency == 0:
            print("Train epoch: {} | {}/{} | Loss: {:.4f}".format(
                epoch, i * len(data), len(train_loader.dataset),
                loss.item()
            ))
    return total_loss / (len(train_loader.dataset) / config.train_batch_size)


def val(model, epoch, val_loader, criterion, device):
    """
    This function calculates the validation loss.

    Parameters
    ----------
    model: torch.nn.Module
        Pytorch model to be trained.
    epoch: int
        Current epoch.
    valloader: torch.utils.data.Dataloader
        PyTorch Dataloader.
    criterion: torch.nn.CrossEntropyLoss
        Criterion for calculating loss.
    device: torch.device
        The device on which the training is to be done.

    Returns
    -------
    total_loss: float
        The validation loss computed over the entire Dataloader.

    Example
    -------
    model = model_dispatcher["VGG11_SegNet_pretrained"]
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device('cpu')
    train_loader = torch.utils.data.DataLoader(
                    LaneData(
                            index_path=train_path,
                            return_sequence=False,
                            transform=transformation,
                            target_transform=transformation,
                            max_samples=None
                            ),
                    batch_size=batch_size,
                    shuffle=True
                   )
    val_loader = torch.utils.data.DataLoader(
                   LaneData(
                            index_path=val_path,
                            return_sequence=return_sequence,
                            transform=transformation,
                            target_transform=transformation,
                            max_samples=max_samples
                    ),
                    batch_size=config.train_batch_size,
                    shuffle=True
                )
    for epoch in range(1, 10):
        train(model,
              epoch,
              train_loader,
              optimizer,
              criterion,
              log_frequency,
              device)

        val(model,
            epoch,
            val_loader,
            criterion,
            device)
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(val_loader):
            data = data.to(device)
            target = target.type(torch.LongTensor).to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
    total_loss /= (len(val_loader.dataset) / config.val_batch_size)
    print("Validation Loss: {:.4f}".format(
        total_loss
    ))
    return total_loss


if __name__ == '__main__':
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    return_sequence = False
    if config.model_to_train in config.CNN_only_models:
        return_sequence = False
    else:
        return_sequence = True

    transformation = transforms.Compose([
        transforms.ToTensor()
    ])

    train_loader = torch.utils.data.DataLoader(
        LaneData(
            index_path=config.train_path,
            return_sequence=return_sequence,
            transform=transformation,
            target_transform=transformation,
            max_samples=config.train_max_samples
        ),
        batch_size=config.train_batch_size,
        shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        LaneData(
            index_path=config.val_path,
            return_sequence=return_sequence,
            transform=transformation,
            target_transform=transformation,
            max_samples=config.val_max_samples
        ),
        batch_size=config.train_batch_size,
        shuffle=True
    )

    model = model_dispatcher.models[config.model_to_train].to(
        config.device)

    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(
            config.class_weight)).to(
        config.device)

    optimizer = torch.optim.Adam(model.parameters(), config.learning_rate)

    early_stopping = EarlyStopping(
        patience=2,
        save_path=config.pretrained_path +
        model.__class__.__name__,
        min_delta=0.001)

    train_losses = []
    val_losses = []
    for epoch in range(1, config.num_epochs + 1):
        train_losses.append(
            train(
                model,
                epoch,
                train_loader,
                optimizer,
                criterion,
                10,
                config.device))
        val_loss = val(model, epoch, val_loader, criterion, config.device)
        val_losses.append(val_loss)
        if early_stopping(model, val_loss):
            break
