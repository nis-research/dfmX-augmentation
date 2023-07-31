from locale import currency
import sys
from tkinter.tix import COLUMN
sys.path.insert(0,'/home/wangs1/')
from datasets.CIFAR_CLASS import CIFAR_CLASS # notice: may need changes
from datasets.CIFAR_BP import CIFAR_BP
import torch
import numpy as np
import torchvision
from torchvision.transforms import transforms
import numpy.fft as fft
import argparse
from torchmetrics import ConfusionMatrix
from torchmetrics.functional import accuracy
import HFC.backbone.resnet as resnet
import HFC.backbone.alexnet as alexnet
from HFC.blocks.resnet.Blocks import Upconvblock
from torchvision.datasets import ImageFolder
sys.path.insert(0,'/home/wangs1//HFC/')
from HFC.oldtrain import Model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def main(args):
    
    model_path = args.model_path

    if args.backbone_model == 'resnet18':
        from HFC.blocks.resnet.Blocks import BasicBlock
        # backbone_model = resnet.ResNet(BasicBlock,[2,2,2,2],args.num_class)
    elif args.backbone_model == 'resnet50':
        from HFC.blocks.resnet.Blocks import Bottleneck
        # backbone_model = resnet.ResNet(Bottleneck,[3,4,6,3],args.num_class)
    elif args.backbone_model == 'densenet121':
        from HFC.blocks.densenet.Blocks import Bottleneck
        # backbone_model = densenet.DenseNet(Bottleneck, [6,12,24,16], growth_rate=32, reduction=0.5, num_classes=args.num_class)
    elif args.backbone_model == 'densenet169':
        from HFC.blocks.densenet.Blocks import Bottleneck
        # backbone_model = densenet.DenseNet(Bottleneck, [6,12,32,32], growth_rate=32, reduction=0.5, num_classes=args.num_class)
    # elif args.backbone_model == 'alexnet':


    model = Model.load_from_checkpoint(model_path)
    model.to(device)
    model.eval()
    model.freeze()
    encoder = model.backbone_model

    confmat = ConfusionMatrix(num_classes=10)
    mean = [0.491400, 0.482158, 0.446531]
    std = [0.247032, 0.243485, 0.261588]
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean, std)])
    data_test = CIFAR_BP('../datasets',train=False,band=' ',transform=transform)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size= 1000, shuffle=False,num_workers=2)

    if args.imagenet == 'True':
        print(args.imagenet)
        print('test on imagenet')
        mean =  [0.491400, 0.482158, 0.446531]
        std = [0.247032, 0.243485, 0.261588]
        transform=transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(),transforms.Normalize(mean, std)])
        data_test =  ImageFolder('../datasets/ImageNet/train/',transform=transform)
        test_loader = torch.utils.data.DataLoader(data_test, batch_size= 50, shuffle=False,num_workers=2)

    total = 0
    Matrix2 = torch.zeros((10,10))
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        _, y_hat = encoder(x)
        # _, predicted = torch.max(y_hat.data,1)
        total += y.size(0)
        # correct += (predicted == y).sum().item()
    # total_acc = float(correct/total)
    
        Matrix2 += confmat(y_hat.cpu(), y.cpu())
    print('Confusion Metrix on testing set:')
    print(Matrix2)





    band = ' '
    batchsize = 1
    
    testset = CIFAR_BP('../datasets',train=False,band=band,transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batchsize, shuffle=False)

    if args.imagenet == "True":
        testset = ImageFolder('../datasets/ImageNet/train/',transform=transform)
        test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batchsize, shuffle=False)
    
    with open(args.m_path+'.pkl', 'rb') as f:
        all = pickle.load(f)

    for mask_i in range(10):
        print('Using mask %d' %mask_i)
        mask = np.array(all[mask_i]) #map
        print(len(mask[mask==1]))
        # mask[16,16] = 0
        mat = np.zeros((10,10))
        
        for x,y in test_loader:
            
                x1=x[0]
                y1 = np.zeros(x1.size(),dtype=np.complex128)
                
                y1 = fft.fftshift(fft.fft2(x1))    
                for channel in range(3):
                    y1[channel,:,:] = y1[channel,:,:] * (mask)

                x1 = fft.ifft2(fft.ifftshift(y1))
                x1 = torch.Tensor(x1).to(device)
                x1 = torch.unsqueeze(x1, 0)
                
                _, y_hat = encoder(x1)
                
                _, predicted = torch.max(y_hat.data,1)
                
                mat[y,predicted] += 1
                

        print(mat) 

        for cla in range(10):
            print('Amount of degradation -- class %d' % cla)  
            diff = (mat[cla,cla]-Matrix2[cla,cla])/Matrix2[cla,cla]
            print(diff)
     
       
    print('------------------without contributing frequs')
    for mask_i in range(10):
        print('Using mask %d' %mask_i)
        mask = np.array(all[mask_i]) #map
        print(len(mask[mask==1]))
        mask = 1-mask
        mask[16,16] = 1
        mat = np.zeros((10,10))
        
        for x,y in test_loader:
            
                x1=x[0]
                y1 = np.zeros(x1.size(),dtype=np.complex128)
                
                y1 = fft.fftshift(fft.fft2(x1))    
                for channel in range(3):
                    y1[channel,:,:] = y1[channel,:,:] * (mask)

                x1 = fft.ifft2(fft.ifftshift(y1))
                x1 = torch.Tensor(x1).to(device)
                x1 = torch.unsqueeze(x1, 0)
                
                _, y_hat = encoder(x1)
                
                _, predicted = torch.max(y_hat.data,1)
                
                mat[y,predicted] += 1
                

        print(mat) 

        for cla in range(10):
            print('Amount of degradation -- class %d' % cla)  
            diff = (mat[cla,cla]-Matrix2[cla,cla])/Matrix2[cla,cla]
            print(diff)
     
    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone_model', type=str, default='resnet18',
                        help='model ')
    parser.add_argument('--model_path', type=str, default='None',
                        help='path of the model')
    parser.add_argument('--m_path', type=str, default='./',
                        help='path of the msk')
    parser.add_argument('--imagenet', type= str, default='False',
                        help='whether to test on imagenet-10 32x32')
 

    args = parser.parse_args()

    main(args)
