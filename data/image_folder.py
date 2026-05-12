import torch.utils.data as data
import random
from PIL import Image
import os
import os.path
import json

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.npy'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_ir_dataset(dataroot, phase):
    images = []
    data_dir = os.path.join(dataroot, phase)
    assert os.path.isdir(data_dir), '%s is not a valid directory' % data_dir
    dir_ir = os.path.join(data_dir, 'ir')
    dir_label = os.path.join(data_dir, 'label')
    image_infos = load_json(os.path.join(dataroot, "images_info_shuffle.json"))

    for root, _, fnames in sorted(os.walk(dir_ir)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path_ir = os.path.join(root, fname)
                fn, file_extension = os.path.splitext(fname)
                path_label = os.path.join(dir_label, fn + '_e.png')

                # 方法4
                # prefix = f'_{random.randint(1, 7)}' + '.jpg'
                # path_label = os.path.join(dir_label, fn + prefix)

                image_info = image_infos[fname]
                images.append({'A': path_label, 'B': path_ir, 'info': image_info})
    return images


def load_json(json_file_path):

    with open(json_file_path, 'r', encoding='utf-8') as f:
        image_infos = json.load(f)

    if isinstance(image_infos, list):
        image_infos_dict = {}
        for item in image_infos:
            image_infos_dict[item['image_name']] = item
        image_infos = image_infos_dict
    return image_infos


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in: " + root + "\n"
                                                               "Supported image extensions are: " +
                                ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
