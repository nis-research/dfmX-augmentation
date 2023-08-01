import os
import numpy as np
from torch.utils.data.dataset import Dataset
from PIL import Image
import numpy as np
import torch

class CIFAR(Dataset):
    def __init__(self, root_dir,train=True,transform = None, extra_transform = None):
        super(CIFAR).__init__()
        if train is False:
            self.labels_path = os.path.join(root_dir,'CIFAR10','test_label.npy')
            self.root_dir = os.path.join(root_dir,'CIFAR10','test_data.npy')

        else:
            self.labels_path = os.path.join(root_dir,'CIFAR10','train_label.npy')
            self.root_dir = os.path.join(root_dir,'CIFAR10','train_data.npy')
        print(self.root_dir)
        self.transform = transform
        self.extra_transform = extra_transform
        self.data = np.load(self.root_dir, allow_pickle=True)
        self.targets = np.load(self.labels_path, allow_pickle=True) 
      
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        img = self.data[index]
        if self.transform is not None:
            img = img*255 
            img = np.clip(img,0,255)
            img = Image.fromarray(img.astype(np.uint8),mode='RGB')
            img = self.transform(img)
            
        target = self.targets[index]
        mm = torch.randint(10,size=(1,))
        if self.extra_transform :
            if mm != target:
                crp_transform = self.extra_transform[mm]
                img = crp_transform(img)

        return  torch.tensor(img,dtype=torch.float), torch.tensor(target, dtype=torch.long)