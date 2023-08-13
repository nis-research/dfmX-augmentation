import os
import sys
sys.path.insert(0,'/home/wangs1/dfmX-augmentation/')
from dataset.CIFAR import CIFAR
import torch
import numpy as np
import pickle
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import numpy.fft as fft
import argparse
from torchmetrics import ConfusionMatrix
import backbone.resnet as resnet
from blocks.resnet.Blocks import BasicBlock,Bottleneck
from train import Model


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

      Matrix_org = torch.zeros((10,10))
      data_test = data_test = CIFAR('./dataset',train=False,transform=transform)
      test_loader = torch.utils.data.DataLoader(data_test, batch_size= 1000, shuffle=False,num_workers=2)
      for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y_hat = encoder(x)
            Matrix_org += confmat(y_hat.cpu(), y.cpu())
  
      print(Matrix_org)
      batchsize = 64
      transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean, std)])
      testset = CIFAR('./dataset',train=False,transform=transform)
      test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batchsize, shuffle=False)

      result_prediction = {}
      for test_class in range(10):

            cur_pre = Matrix_org[test_class,test_class]
            t = args.t # try with different t values to limite the performance degradation within 30%
            with open('./DFMs/removal_order.pkl', 'rb') as f:
                  importance = pickle.load(f)

            re_importance = np.copy(importance)
            count = 0
        
            while np.sum(importance) != 0 :
                  count += 1
                  correct = 0
                  mask = np.copy(re_importance)
                  max_importance = np.max(importance)
                  mask[mask == max_importance] = 0
                  mask[mask != 0] = 1
                  # remove frequency and reconstruct
                  for x,y in test_loader:
                        x1=x
                        sizex = x1.size()
                        reference_class = torch.ones(sizex[0])*test_class
                        if  (y.to(device) == reference_class.to(device)).int().sum()>0:
                  
                              F_x1 = torch.zeros(sizex,dtype=torch.complex128)
                              F_x1 = fft.fftshift(fft.fft2(x1))
                              for num_s in range(sizex[0]):
                                    for channel in range(3):
                                          F_x1[num_s,channel,:,:] = F_x1[num_s,channel,:,:] * mask                    

                              x1 = fft.ifft2(fft.ifftshift(F_x1))
                              x1 = torch.Tensor(x1).to(device)
                              x1 = torch.real(x1)
                              
                              
                              y_hat = encoder(x1)
                              _, predicted = torch.max(y_hat.data,1)

                              correct_predictions = (predicted == y.to(device)).int()                              

                              tested_classes = (y.to(device) == reference_class.to(device))
                              tested_classes = tested_classes.int()
                              
                              correct += (tested_classes*correct_predictions).sum().item()
                  
                  if correct >= cur_pre-t:
                        cur_pre = correct
                        re_importance[re_importance == max_importance] = 0
                  
                  importance[importance == max_importance] = 0
                  

            re_importance[re_importance >0] = 1
            result_prediction.update({test_class:re_importance})

      with open('./DFMs/'+args.backbone_model+'_'+str(args.t)+'.pkl', 'wb') as f:
            pickle.dump(result_prediction, f)
     

if __name__ == '__main__':
      parser = argparse.ArgumentParser()
      parser.add_argument('--backbone_model', type=str, default='resnet18',
                              help='model ')
      parser.add_argument('--model_path', type=str, default='None',
                              help='path of the model')
      parser.add_argument('--t', type=int, default=6,
                              help='flexible threshold')

      args = parser.parse_args()
      if not os.path.exists('./DFMs'):
            os.makedirs( './DFMs')


      main(args)



