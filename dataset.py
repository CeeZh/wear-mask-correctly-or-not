import logging
import torchvision.transforms as T
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from torchvision.io import read_image


def get_trm(cfg, is_train=True):
    if is_train:
        transform = T.Compose([
            T.ToPILImage(),
            T.Resize(cfg.data.image_size),
            T.RandomHorizontalFlip(p=cfg.data.flip_prob),
            T.ToTensor(),
            T.Normalize(mean=0, std=255),
            T.Normalize(mean=cfg.data.pixel_mean, std=cfg.data.pixel_std)
        ])
    else:
        transform = T.Compose([
            T.ToPILImage(),
            T.Resize(cfg.data.image_size),
            T.ToTensor(),
            T.Normalize(mean=0, std=255),
            T.Normalize(mean=cfg.data.pixel_mean, std=cfg.data.pixel_std)
        ])
    return transform


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, label = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, label


def _process_anno(path):
    '''
    :return: list of image info. Each element is like [image_path(str), label(int)]. label: 0 for right, 1 for wrong
    '''
    with open(path, 'r') as f:
        lines = f.readlines()
    image_info_ls = []
    for line in lines:
        image_info = line.strip().split(' ')
        if len(image_info) > 0:
            image_info = [image_info[0], int(image_info[1])]
            image_info_ls.append(image_info)
    return image_info_ls


def _make_train_loader(cfg):
    trm = get_trm(cfg, True)
    anno = _process_anno(cfg.data.train_path)
    dataset = ImageDataset(anno, trm)
    logger = logging.getLogger('train')
    logger.info('Total train samples: {}'.format(len(dataset)))
    dataloader = DataLoader(dataset, batch_size=cfg.solver.batch_size, num_workers=cfg.data_loader.num_workers)
    return dataloader


def _make_val_loader(cfg):
    trm = get_trm(cfg, False)
    anno = _process_anno(cfg.data.val_path)
    dataset = ImageDataset(anno, trm)
    dataloader = DataLoader(dataset, batch_size=cfg.test.batch_size, num_workers=cfg.test.num_workers)
    return dataloader


def _make_test_loader(cfg):
    trm = get_trm(cfg, False)
    anno = _process_anno(cfg.data.test_path)
    dataset = ImageDataset(anno, trm)
    dataloader = DataLoader(dataset, batch_size=cfg.test.batch_size, num_workers=cfg.test.num_workers)
    return dataloader


def make_dataloader(cfg, type):
    assert type in ['train', 'test', 'validation'], 'Dataset type \'{}\' not supported. Must be in {}' \
        .format(type, ['train', 'test', 'validation'])

    if type == 'train':
        return _make_train_loader(cfg)
    elif type == 'validation':
        return _make_val_loader(cfg)
    else:
        return _make_test_loader(cfg)
