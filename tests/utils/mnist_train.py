import logging
from pathlib import Path

import torch
import tqdm
from torch import nn

from tests.helpers.basic_models import MNISTCNNModel, MNISTLinearModel
from tests.utils.configs import BaseTestConfig
from tests.utils.types import ExplanationInputs, MetricInputs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def train_mnist_model(model, dataloader, n_epochs: int = 10, lr: float = 0.01):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    model.train()
    pbar = tqdm.tqdm(range(n_epochs))
    for epoch in pbar:
        for images, labels in dataloader:
            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        if epoch % 1 == 0:
            pbar.set_postfix_str(f"Epoch {epoch}, Loss {loss.item()}")


def evaluate_mnist_model(model, dataloader):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    logger.info(f"Model accuracy: {100 * correct / total}")


def mnist_dataloader():
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torchvision.datasets import MNIST

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = MNIST(
        root="/tmp/mnist/",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
        target_transform=transforms.Compose([lambda x: torch.tensor(x)]),
    )
    test_dataset = MNIST(
        root="/tmp/mnist/",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
        target_transform=transforms.Compose([lambda x: torch.tensor(x)]),
    )
    train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    train_baselines = next(iter(train_dataloader))[0].to(device)
    return train_dataloader, test_dataloader, train_baselines


def mnist_trainer(model_type: str = "linear", train_and_eval_model: bool = True):
    train_dataloader, test_dataloader, train_baselines = mnist_dataloader()
    if model_type == "linear":
        model = MNISTLinearModel()
        input_layer_names = ["fc1"]
    elif model_type == "cnn":
        model = MNISTCNNModel()
        input_layer_names = ["conv1"]
    else:
        raise ValueError("Invalid model")

    # load model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if train_and_eval_model:
        model_path = f"/tmp/mnist/{model_type}_model.pth"
        if Path(model_path).exists():
            logger.info(f"Loading {model_type} model for tests from {model_path}")
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
        else:
            logger.info(
                f"Training {model_type} model for tests and caching to {model_path}"
            )
            train_mnist_model(model, train_dataloader)
            torch.save(model.state_dict(), model_path)
        evaluate_mnist_model(model, test_dataloader)

    model.eval()
    batch = next(iter(test_dataloader))
    inputs = batch[0].to(device)
    target = batch[1].to(device)
    train_baselines = train_baselines.to(device)

    return BaseTestConfig(
        explanation_inputs=ExplanationInputs(
            sample_id=[str(i) for i in range(inputs.size(0))],
            inputs=inputs,
            additional_forward_args=None,
            target=target,
            baselines=train_baselines,
        ),
        metric_inputs=MetricInputs(
            input_layer_names=input_layer_names,
            constant_shifts=(torch.ones(1, 28, 28).unsqueeze(0),),
        ),
        model=model,
        n_features=(1 * 28 * 28),
    )
