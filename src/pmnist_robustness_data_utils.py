import os
from pathlib import Path

import numpy as np
import torch
import torchvision


from tqdm.notebook import tqdm

class TaskDataSet(torch.utils.data.Dataset):
    def __init__(self, data_path_prefix, transform=None, target_transform=None):
        data_path_prefix = str(data_path_prefix)
        self.images = np.load(data_path_prefix + '-images.npy')
        self.images = self.images.astype('float32') / 255.0
        self.labels = np.load(data_path_prefix + '-labels.npy')
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, i):
        image, label = self.images[i], int(self.labels[i])
        image = image.reshape(28, 28)

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.labels)

def create_data_path(root_path, task_id, data_type, data_action):
    # task_id: 'task-01', 'task-02', 'task-03', ...
    # data_type: 'original', 'corruption', 'adversarial'
    # data_action:
    #   - if 'original' -> 'train', 'test'
    #   - if 'corruption' -> 'gaussian_blur', ...
    #   - if 'adversarial' (TBD holder) -> 'fgsm', ...

    data_path = (root_path / task_id / data_type)
    data_path.mkdir(parents=True, exist_ok=True)
    data_prefix = data_path / data_action
    data_key = '%s::%s' %(data_type, data_action)
    return data_prefix, data_key

def save_dataset(images, labels, data_prefix):
    data_paths = dict(
        prefix = str(data_prefix),
        images = str(data_prefix) + '-images.npy',
        labels = str(data_prefix) + '-labels.npy'
    )
    np.save(data_paths['images'], np.array(images).astype(np.uint8))
    np.save(data_paths['labels'], np.array(labels).astype(np.uint8))
    return data_paths


def create_and_save_datasets(mnist_data_path, root_path, task_id, perm=False,
                             actions=dict(type=None, fns=dict())):
    # only perform action in test dataset
    info = dict()

    transform_fn = torchvision.transforms.ToTensor()
    if perm:
        permut = torch.from_numpy(np.random.permutation(784))
        transform_fn = transform=torchvision.transforms.Compose([
             torchvision.transforms.ToTensor(), # this will scale to [0,1.0]
             torchvision.transforms.Lambda(lambda x: x.view(-1)[permut].view(28, 28))
        ])

    # Original first
    for data_action in ['train', 'test']:
        # load dset
        dset = torchvision.datasets.MNIST(
            mnist_data_path, train=data_action=='train',
            transform=transform_fn, download=True)
        data_loader = torch.utils.data.DataLoader(dset, shuffle=False)

        # save for adding coruption
        if data_action=='test':
            test_loader = data_loader

        # create path
        data_prefix, data_key = create_data_path(root_path, task_id, 'original', data_action)

        # just load in image then save
        images, labels = [], []
        for img, lbl in data_loader:
            labels.append(lbl)
             # so rescale back here
            images.append(np.uint8(255.0 * img.squeeze()))

        info[data_key] = save_dataset(images, labels, data_prefix)

    # Now perform perturbations like natural corruptions
    # separate original loop from perturbation loop for easier debugging (hopefully)
    data_type = actions['type']
    data_action_fns = actions['fns']

    for data_action, act_fn in tqdm(data_action_fns.items()):
        # create path
        data_prefix, data_key = create_data_path(root_path, task_id, data_type, data_action)

        # just load in images, perturb and then save
        images, labels = [], []
        for img, lbl in test_loader:
            labels.append(lbl)

            # act_fn returns 2d with float in [0,1]
            c_img = np.array(act_fn(img.numpy().squeeze())) * 255.0
            c_img = np.clip(c_img, 0.0, 255.0)
            images.append(np.uint8(c_img))

        info[data_key] = save_dataset(images, labels, data_prefix)

    return info
