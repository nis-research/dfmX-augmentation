import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torchvision.transforms as transforms
import os
import numpy as np
import sys
sys.path.insert(0,'/home/wangs1/')
from CIFAR_BP import CIFAR_BP
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torchmetrics
import matplotlib.pyplot as plt
import pickle
import torch.fft as fft
# from pytorch_lightning import metrics
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import backbone.resnet as resnet




def filter(x,mask):
    x1 = x#.numpy()
    y1 = torch.zeros(x1.size(),dtype=torch.complex128)
    for j in range(3):
        y1[j,:,:] = fft.fftshift(fft.fft2(x1[j,:,:])) 
        y1[j,:,:] = y1[j,:,:]* mask
    x1_w = fft.ifft2(fft.ifftshift(y1))
    return torch.Tensor(torch.real(x1_w))

class White_Mask(object):
    def __init__(self, pro: float, mask_choice:int, masks, flip = False):
       
        assert pro >= 0.0
        self.pro = pro
        self.mask_choice = mask_choice
        self.masks = masks
        self.flip = flip
             
      
    def __call__(self, x):

        print(x.shape)
        c, h, w = x.shape[-3:]
        assert c == 3
        assert h >= 1 or w >= 1
        p = torch.rand(1)
       
        if p <= self.pro:
            # print('yes')
            map = np.asarray(self.masks[self.mask_choice])
            # print(map)
            mask = torch.Tensor(map)
            if self.flip:
                mask = 1-mask
            mask[int(h/2),int(w/2)] = 1
            x = filter(x,mask)
        
        return x

class Model(pl.LightningModule):
    def __init__(self,backbone_model, lr,num_class,dataset,image_size, band, masks, p , special=None):
        super(Model, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=10)
        self.val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=10)
        self.test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=10)
        self.dataset = dataset
        self.num_class = num_class
        self.image_size = image_size
        self.backbone_model = backbone_model


        self.p = p
        self.masks = masks
        self.band = band
        self.special = special
    def forward(self, x):

        prediction = self.backbone_model(x)

        return prediction
        

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), self.lr,
                                momentum=0.9, nesterov=True,
                                weight_decay=1e-4)#torch.optim.Adam(self.parameters(), lr=self.lr,weight_decay=5e-04)
        # scheduler = StepLR(optimizer,step_size=20)
        scheduler = ReduceLROnPlateau(optimizer, mode='min',verbose=True, factor=0.1)#, step_size=2, gamma=0.95)
        return {'optimizer': optimizer, 
                'lr_scheduler':scheduler,
                'monitor': 'val_loss'}

    def training_step(self, batch, batch_idx):
        x, y = batch
        img = x[2].cpu().numpy().transpose((1,2,0))
        plt.figure()
        plt.imshow((img-np.min(img))/(np.max(img)-np.min(img)))
        plt.savefig('test.png')
        plt.close()

        criterion1 = nn.CrossEntropyLoss()
  
        y_hat = self(x)
    
        loss1 = criterion1(y_hat, y)
        loss = loss1
      
                    
        _, predicted = torch.max(y_hat.data,1) 
        self.log_dict({'train_classification_loss': loss1}, on_epoch=True,on_step=True)
        self.log_dict({'train_loss': loss}, on_epoch=True,on_step=True)
        return {"loss": loss,'epoch_preds': predicted, 'epoch_targets': y}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        criterion1 = nn.CrossEntropyLoss()

        y_hat = self(x)
        loss1 = criterion1(y_hat, y)
        self.val_loss = loss1

        _, predicted = torch.max(y_hat.data,1) 
        self.log_dict( {'val_loss':  self.val_loss}, on_epoch=True,on_step=True)

        return  {'epoch_preds': predicted, 'epoch_targets': y} #self.val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        _, predicted = torch.max(y_hat.data,1)
        
        return {'batch_preds': predicted, 'batch_targets': y}
        
    
    def test_step_end(self, output_results):
        
        self.test_acc(output_results['batch_preds'], output_results['batch_targets'])
        self.log_dict( {'test_acc': self.test_acc}, on_epoch=True,on_step=False)
        
    def training_epoch_end(self, output_results):
        # print(output_results)
        self.train_acc(output_results[0]['epoch_preds'], output_results[0]['epoch_targets'])
        self.log_dict({"train_acc": self.train_acc}, on_epoch=True, on_step=False)

    def validation_epoch_end(self, output_results):
        # print(output_results)
        self.val_acc(output_results[0]['epoch_preds'], output_results[0]['epoch_targets'])
        self.log_dict({"valid_acc": self.val_acc}, on_epoch=True, on_step=False)

    def setup(self, stage):
        with open(self.masks, 'rb') as f:
            DFMs = pickle.load(f)


        mask_transforms = {}
        mask_transforms_flipped = {}
        for c in range(len(DFMs)):
            mask_transforms.update({c:White_Mask(self.p,c,DFMs)})
            mask_transforms_flipped.update({c:White_Mask(self.p,c,DFMs,flip=True)})

        
        extra_transform =[mask_transforms[i] for i in range(len(mask_transforms))]
        extra_transform_flipped =[mask_transforms_flipped[i] for i in range(len(mask_transforms_flipped))]

        if self.p == 0 :
            extra_transform= None
  

        if self.dataset == 'cifar':
            mean = [0.491400, 0.482158, 0.446531]
            std = [0.247032, 0.243485, 0.261588]

            transform_train = transforms.Compose([
                transforms.Pad(4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(self.image_size),
                transforms.AugMix(), #transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10 ), #
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean, std)])
            data_train  = CIFAR_BP('../datasets/',train=True,band=self.band,transform=transform_train,extra_transform=extra_transform,extra_transformb = None)
            data_test = CIFAR_BP('../datasets',train=False,band=self.band,transform=transform)

        
       
       
        # train/val split
        data_train2, data_val =  torch.utils.data.random_split(data_train, [int(len(data_train)*0.9), len(data_train)-int(len(data_train)*0.9)])

        # assign to use in dataloaders
        self.train_dataset = data_train2
        self.val_dataset = data_val
        self.test_dataset = data_test

    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=64, shuffle=True)#,num_workers=2)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size= 64, shuffle=False)#,num_workers=2)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size= 64)#,num_workers=2)


def main(args):
    backbone = ['resnet9','resnet18', 'resnet50','resnet101','densenet121', 'densent169', 'vgg16'] 
    print(torch.cuda.device_count())
    if args.backbone_model == 'resnet18':
        from blocks.resnet.Blocks import BasicBlock
        backbone_model = resnet.ResNet(BasicBlock,[2,2,2,2],args.num_class)
    elif args.backbone_model == 'resnet34':
        from blocks.resnet.Blocks import BasicBlock
        backbone_model = resnet.ResNet(BasicBlock, [3,4,6,3],args.num_class)
    elif args.backbone_model == 'resnet50':
        from blocks.resnet.Blocks import Bottleneck
        backbone_model = resnet.ResNet(Bottleneck,[3,4,6,3],args.num_class)
    elif args.backbone_model == 'resnet101':
        from blocks.resnet.Blocks import Bottleneck
        backbone_model = resnet.ResNet(Bottleneck[3,4,23,3],args.num_class)
    

    
    
    logger = TensorBoardLogger(args.save_dir, name=args.backbone_model+ args.band)#
    
    model = Model(backbone_model, args.lr,args.num_class,args.dataset,args.image_size,args.band,args.masks, args.p, args.special)#
    maxepoch = 200
 
    checkpoints_callback = ModelCheckpoint(save_last=True) 
    trainer = pl.Trainer(enable_progress_bar=False,logger=logger, callbacks=[checkpoints_callback], gpus=-1, max_epochs=maxepoch)  
    trainer.fit(model)
    trainer.test()





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Write parameters')
    parser.add_argument('--backbone_model', type=str,
                    help='backbone_model')
    parser.add_argument('--image_size', type=int, default= 32,
                    help='size of images in dataset')
    parser.add_argument('--num_class', type=int, default= 10,
                    help='number of classes in dataset')
    parser.add_argument('--dataset', type=str, default='cifar',
                    help='dataset')
    parser.add_argument('--weight_alpha', type=float, default=0.5,
                    help='weight of classification loss')
    parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')            
    parser.add_argument('--save_dir', type=str, default='results/')
    parser.add_argument('--band', type=str, default= '',
                    help='band of frequency')
    parser.add_argument('--p', type=float, default= 0.3,
                    help='percentage of augmentations')
    parser.add_argument('--masks', type=str, default= 'alex.pkl',
                    help='Masks for filtering')
    parser.add_argument('--special', type=str, default= '_complex_special_v2',
                    help='band frequency bias')
   
    args = parser.parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    main(args)