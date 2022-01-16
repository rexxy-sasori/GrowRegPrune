import torch
import torchvision


def get_dataset():
    train_transformation_pipeline = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(
            size=32,
            padding=4
        ),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2471, 0.2435, 0.2616]
        )
    ])

    test_transformation_pipeline = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2471, 0.2435, 0.2616]
        )
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./cifar10_data',
        download=True,
        train=True,
        transform=train_transformation_pipeline
    )

    testset = torchvision.datasets.CIFAR10(
        root='./cifar10_data',
        download=True,
        train=False,
        transform=test_transformation_pipeline
    )

    return trainset, testset


def get_dataloader(
        trainset,
        testset,
        batch_size,
        num_worker,
        drop_last_batch,
):
    trainloader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_worker, pin_memory=True,
        drop_last=drop_last_batch,
        shuffle=True
    )

    testloader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_worker, pin_memory=True,
        drop_last=drop_last_batch,
        shuffle=False
    )

    train_data_loader = torch.utils.data.DataLoader(trainset, **trainloader_kwargs)
    test_data_loader = torch.utils.data.DataLoader(testset, **testloader_kwargs)

    return train_data_loader, test_data_loader
