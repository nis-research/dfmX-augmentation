import pickle
import torch
import numpy as np
from torchvision.transforms import transforms
import numpy.fft as fft
import argparse
from torchmetrics import ConfusionMatrix
import sys
sys.path.insert(0,'/home/wangs1/dfmX-augmentation/')
from dataset.CIFAR import CIFAR
import backbone.resnet as resnet
from train import Model
from blocks.resnet.Blocks import BasicBlock,Bottleneck
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def main(args):
    
    model_path = args.model_path
    model = Model.load_from_checkpoint(model_path)
    model.to(device)
    model.eval()
    model.freeze()
    encoder = model.backbone_model

    confmat = ConfusionMatrix(num_classes=10)
    mean = [0.491400, 0.482158, 0.446531]
    std = [0.247032, 0.243485, 0.261588]
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean, std)])
    data_test = CIFAR('./dataset',train=False,transform=transform)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size= 1000, shuffle=False,num_workers=2)

    total = 0
    Matrix_org = torch.zeros((10,10))
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        y_hat = encoder(x)
        total += y.size(0)
        Matrix_org += confmat(y_hat.cpu(), y.cpu())
    print('Confusion Metrix on testing set:')
    print(Matrix_org)

    batchsize = 32
    
    testset = CIFAR('./dataset',train=False,transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batchsize, shuffle=False)
    
    with open(args.m_path+'.pkl', 'rb') as f:
        all = pickle.load(f)

    for mask_i in range(10):
        print('Using mask %d' %mask_i)
        mask = np.array(all[mask_i]) #map
        print(len(mask[mask==1]))
        mat = torch.zeros((10,10))
        
        for x,y in test_loader:
            x1=x
            sizex = x1.size()        
            F_x1 = torch.zeros(sizex,dtype=torch.complex128)
            F_x1 = fft.fftshift(fft.fft2(x1))
            for num_s in range(sizex[0]):
                for channel in range(3):
                        F_x1[num_s,channel,:,:] = F_x1[num_s,channel,:,:] * mask                    

            x1 = fft.ifft2(fft.ifftshift(F_x1))
            x1 = torch.Tensor(x1)
            x1 = torch.real(x1).to(device)
            
            
            y_hat = encoder(x1)
            mat += confmat(y_hat.cpu(), y.cpu())

        print(mat) 

         
        print('Amount of degradation -- class %d' % mask_i)  
        diff = (mat[mask_i,mask_i]-Matrix_org[mask_i,mask_i])/Matrix_org[mask_i,mask_i]
        print(diff)
          
    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone_model', type=str, default='resnet18',
                        help='model ')
    parser.add_argument('--model_path', type=str, default='None',
                        help='path of the model')
    parser.add_argument('--m_path', type=str, default='./',
                        help='path of the msk') 

    args = parser.parse_args()

    main(args)
