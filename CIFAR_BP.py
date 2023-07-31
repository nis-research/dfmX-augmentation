import os
import numpy as np
from torch.utils.data.dataset import Dataset
from PIL import Image
import numpy as np
import torch

class CIFAR_BP(Dataset):
    def __init__(self, root_dir,train=True, band = 'low_4' , transform = None, extra_transform = None, extra_transformb = None):
        super(CIFAR_BP).__init__()
        if train is False:
            self.labels_path = os.path.join(root_dir,'CIFAR10','test_label.npy')
            self.root_dir = os.path.join(root_dir,'CIFAR10','test_data_'+band+'.npy')

        else:
            self.labels_path = os.path.join(root_dir,'CIFAR10','train_label.npy')
            self.root_dir = os.path.join(root_dir,'CIFAR10','train_data_'+band+'.npy')
        print(self.root_dir)
        self.transform = transform
        self.extra_transform = extra_transform
        self.extra_transformb = extra_transformb
        self.band = band
        self.data = np.load(self.root_dir, allow_pickle=True)
        self.targets = np.load(self.labels_path, allow_pickle=True) 
        # self.data = self.data.transpose((0, 3, 1, 2))
        #print(self.data.shape)
        means = self.data
        for axis in (2,1,0):
            means = np.mean(means,axis=axis)
        # print(means)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        img = self.data[index]
        if self.transform is not None:
          
            if np.max(img)<10:
                if self.band  in [ 'flip']:
                    
                    img = (img-np.min(img))/(np.max(img)-np.min(img))
                    img = img*255
                else:
                    img = np.clip(img,0,1)
                    img = img*255 

            img = np.clip(img,0,255)
            img = Image.fromarray(img.astype(np.uint8),mode='RGB')
            # print(img)
            img = self.transform(img)
            # print(img)
        target = self.targets[index]
        mm = torch.randint(10,size=(1,))
        if self.extra_transform :
            if mm != target:
                crp_transform = self.extra_transform[mm]
                img = crp_transform(img)

        if self.extra_transformb :
            # crp_transform = self.extra_transformb[target]
            # img = crp_transform(img)
            if mm != target:
                crp_transform = self.extra_transformb[mm]
                img = crp_transform(img)

        return  torch.tensor(img,dtype=torch.float), torch.tensor(target, dtype=torch.long)