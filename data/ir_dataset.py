import torchvision.transforms as transforms
from data.base_dataset import BaseDataset
from data.image_folder import make_ir_dataset
from PIL import Image


class IRDataset(BaseDataset):
    def initialize(self, opt):
        print('IRDataset')
        self.opt = opt
        self.root = opt.dataroot
        self.AB_paths = make_ir_dataset(opt.dataroot, opt.phase)

        self.transform = transforms.Compose([
            transforms.Resize((self.opt.crop_size, self.opt.crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __getitem__(self, index):
        A_path = self.AB_paths[index]['A']
        B_path = self.AB_paths[index]['B']
        IR_info = self.AB_paths[index]['info']


        A = Image.open(A_path).convert('RGB')
        B = Image.open(B_path)

        A = self.transform(A)
        B = self.transform(B)

        T = IR_info['temperature']

        return {'A': A, 'B': B, 'T': T, 'A_paths': A_path, 'B_paths': B_path, 'image_name': IR_info['image_name'].split('.')[0]}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'IRDataset'
