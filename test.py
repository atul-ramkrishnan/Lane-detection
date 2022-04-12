import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import numpy as np
import config
from data import LaneData
import model_dispatcher


def get_prediction(model, test_loader, device, save_dir=None):
    """
    This function plots the input image, target image,
    and the model output. If a save_dir is specified,
    it saves the images.

    Parameters
    ----------
    model: torch.nn.Module
        PyTorch model used to get the prediction.
    test_loader: torch.utils.data.Dataloader
        PyTorch Dataloader.
    device: torch.device
        The device used to make predictions.
    save_dir : str, optional
        Directory to which the images are to be saved.

    Returns
    -------
    None

    Example
    -------
    test_loader = torch.utils.data.DataLoader(
        LaneData(
            index_path=test_path,
            return_sequence=False,
            transform=transformation,
            target_transform=transformation,
            max_samples=100
        ),
        batch_size=32,
        shuffle=True
    )

    model = model_dispatcher["VGG16_SegNet"]

    get_prediction(model, test_loader, config.device)

    """
    num_inputs = len(test_loader)
    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            pred = torch.max(output, dim=1)[1].squeeze()
            lane_img = np.uint8(pred.numpy() * 255)
            plt.figure(figsize=(100, 100))
            plt.subplot(num_inputs, 3, 3 * i + 1)
            if model.__class__.__name__ in config.CNN_only_models:
                plt.imshow(torch.permute(data.squeeze(), (1, 2, 0)))
            else:
                plt.imshow(torch.permute(data.squeeze()[-1], (1, 2, 0)))
            plt.title('Input')
            plt.subplot(num_inputs, 3, 3 * i + 2)
            plt.imshow(target.squeeze(), cmap='gray')
            plt.title('Target')
            plt.subplot(num_inputs, 3, 3 * i + 3)
            plt.title('Output')
            plt.imshow(lane_img, cmap='gray')
            if save_dir is not None:
                plt.savefig(save_dir + "/prediction_" + str(i) + ".png")
            plt.show()


def get_evaluation_scores(model, test_loader, device, criterion):
    """
    This function calculates the following performance metrics --
    1. Test loss
    2. Test accuracy
    3. Precision
    4. Recall
    5. F1-measure

    Parameters
    ----------
    model: torch.nn.Module
        PyTorch model used to get the prediction.
    test_loader: torch.utils.data.Dataloader
        PyTorch Dataloader.
    device: torch.device
        The device used to make predictions.
    criterion: torch.nn.CrossEntropyLoss
        Criterion for calculating loss.

    Returns
    -------
    test_loss: float
        Loss over the test set.
    test_acc: float
        Test accuracy
    precision: float
        Precision = True positive / (True positive + False positive)
    recall: float
        Recall = True positive / (True positive + False negative)
    f1_score; float
        F1-score = 2 * Precision * Recall / (Precision + Recall)

    Example
    -------
    test_loader = torch.utils.data.DataLoader(
        LaneData(
            index_path=test_path,
            return_sequence=False,
            transform=transformation,
            target_transform=transformation,
            max_samples=100
        ),
        batch_size=32,
        shuffle=True
    )

    model = model_dispatcher["VGG16_SegNet"]
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(
            [0.02, 1.02]))

    test_loss, test_acc, precision, recall, f1_score = get_evaluation_scores(
                                                       model,
                                                       test_loader,
                                                       torch.device('cpu'),
                                                       criterion)

    """
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data = data.to(device)
            target = target.type(torch.LongTensor).to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            pred = pred.squeeze()
            test_loss += criterion(output, target).item()
            correct += pred.eq(target.view_as(pred)).sum().item()
            tp += torch.sum(torch.bitwise_and((pred == 1), (target == 1)))
            fp += torch.sum(torch.bitwise_and((pred == 1), (target == 0)))
            fn += torch.sum(torch.bitwise_and((pred == 0), (target == 1)))
            tn += torch.sum(torch.bitwise_and((pred == 0), (target == 0)))

    test_loss /= (len(test_loader.dataset) / 100)
    test_acc = 100. * int(correct) / (len(test_loader.dataset) * 128 * 256)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = (2 * precision * recall) / (precision + recall)
    return (test_loss,
            test_acc,
            precision.item(),
            recall.item(),
            f1_score.item())


if __name__ == '__main__':
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if config.model_to_test in config.CNN_only_models:
        return_sequence = False
    else:
        return_sequence = True

    max_samples = 0
    if config.test_mode == "get_prediction":
        test_mode = "get_prediction"
        batch_size = config.predict_batch_size
        max_samples = config.get_prediction_max_samples
    elif config.test_mode == "get_evaluation_scores":
        test_mode = "get_evaluation_scores"
        batch_size = config.evaluate_batch_size
        max_samples = config.get_evaluation_scores_max_samples

    transformation = transforms.Compose([
        transforms.ToTensor()
    ])

    test_loader = torch.utils.data.DataLoader(
        LaneData(
            index_path=config.test_path,
            return_sequence=return_sequence,
            transform=transformation,
            target_transform=transformation,
            max_samples=max_samples
        ),
        batch_size=batch_size,
        shuffle=True
    )

    model = model_dispatcher.models[config.model_to_test].to(
        config.device)

    pretrained_dict = torch.load(
        config.pretrained_path +
        config.test_pretrained_model,
        map_location=config.device)
    model_dict = model.state_dict()
    pretrained_dict_1 = {
        k: v for k,
        v in pretrained_dict.items() if (
            k in model_dict)}
    model_dict.update(pretrained_dict_1)
    model.load_state_dict(model_dict)

    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(
            config.class_weight)).to(
        config.device)

    if test_mode == "get_prediction":
        get_prediction(model, test_loader, config.device, config.save_path)
    elif test_mode == "get_evaluation_scores":
        test_loss, test_acc, precision, recall, f1 = get_evaluation_scores(
            model,
            test_loader,
            config.device,
            criterion)

        print('\nAverage loss: {:.4f}, Accuracy: {:.4f}%'.format(
              test_loss, test_acc))

        print("Precision: {:.4f}\nRecall: {:.4f}\nF1-measure: {:.4f}"
              .format(precision, recall, f1))
