import pickle
import timeit
from enum import Enum
from pathlib import Path
from typing import Dict

import h5py
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18


class LabelType(Enum):
    COARSE = 1
    FINE = 2


def load_network(num_classes: int):
    net = resnet18(pretrained=False)
    net.fc = torch.nn.Linear(512, num_classes)


class FlairDataset(Dataset):
    def __init__(
        self,
        h5p_file: Path,
        partition: str,
        client_id: str,
        transorms: None = None,
        path_to_label_mapping: Path = None,
        label_type: LabelType = LabelType.COARSE,
    ) -> None:
        self.h5p_file = h5p_file
        self.partition = partition
        self.client_id = client_id
        self.transforms = transorms

        # Select label granularity
        if label_type == LabelType.COARSE:
            self.label_granularity = "labels"
        else:
            self.label_granularity = "fine_grained_labels"

        # Load label mapping dictionary
        with open(path_to_label_mapping, "rb") as f:
            tmp = pickle.load(f)
            self.label_mapping_dict: Dict[str, int] = tmp[self.label_granularity]
        self.num_classes = len(self.label_mapping_dict)

    def __len__(self):
        with h5py.File(self.h5p_file, "r") as f:
            num_samples = len(f[self.partition][self.client_id]["images"])
        return num_samples

    def __getitem__(self, idx):
        with h5py.File(self.h5p_file, "r") as f:
            image = torch.FloatTensor(f[self.partition][self.client_id]["images"][idx])
            binary_labels = f[self.partition][self.client_id][self.label_granularity][
                idx
            ]
        image = image.permute(2, 0, 1)
        if self.transforms:
            image = Image.fromarray(image)
            image = self.transforms(image)

        idx_labels = [
            self.label_mapping_dict[x] for x in binary_labels.decode("ascii").split("|")
        ]
        multi_hot_labels = torch.tensor(idx_labels)
        multi_hot_labels = multi_hot_labels.unsqueeze(0)
        target = torch.zeros(multi_hot_labels.size(0), self.num_classes).scatter_(
            1, multi_hot_labels, 1.0
        )

        return image, target


def train(net, trainloader, epochs):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()


def test(net, testloader):
    """Validate the network on the entire test set."""
    criterion = torch.nn.BCELoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy


def time_dataset_creation():
    dataset = FlairDataset(
        h5p_file=Path("./dataset/flair.h5"),
        partition="train",
        client_id="59769174@N00",
        path_to_label_mapping=Path("./dataset/flair_labels_to_index.pkl"),
        label_type=LabelType.COARSE,
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    for images, labels in dataloader:
        print(images.shape, labels.shape)
        break


if __name__ == "__main__":
    print(timeit.timeit(time_dataset_creation, number=50))
