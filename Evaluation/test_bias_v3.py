import sys
import os
sys.path.insert(0,'/home/wangs1/')
from datasets.CIFAR_CLASS import CIFAR_CLASS # notice: may need changes
from datasets.Synthetic import Synthetic
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
from HFC.blocks.decoder import Decoder
from HFC.blocks.resnet.Blocks import Upconvblock
sys.path.insert(0,'/home/wangs1//HFC/')
from HFC.oldtrain import Model
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def main(args):
      model_name = args.backbone_model
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
 

      Matrix2 = torch.zeros((10,10))
      data_test = data_test = CIFAR_BP('../datasets',train=False,band=' ',transform=transform)
      test_loader = torch.utils.data.DataLoader(data_test, batch_size= 1000, shuffle=False,num_workers=2)
      for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            _,_, y_hat = encoder(x)

            Matrix2 += confmat(y_hat.cpu(), y.cpu())
      # t = 20
      print(Matrix2)
      band = ' '
      batchsize = 1
      transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean, std)])
      testset = CIFAR_BP('../datasets',train=False,band=band,transform=transform)
      test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batchsize, shuffle=False)
      


      

      for test_class in range(10):

            cur_pre = Matrix2[test_class,test_class]
            # print(cur_pre)
            t = args.t # int(cur_pre*0.005)#5
            importance = np.array([[ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13.,
                                          14., 15., 16., 17.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                                          0.,  0.,  0.,  0.,  0.,  0.],
                                    [124., 125., 126., 127., 128., 129., 130., 131., 132., 133., 134.,
                                          135., 136., 137., 138., 139.,  78.,  79.,  80.,  81.,  82.,  83.,
                                          84.,  85.,  86.,  87.,  88.,  89.,  90.,  91.,  92.,  93.],
                                    [123., 240., 241., 242., 243., 244., 245., 246., 247., 248., 249.,
                                          250., 251., 252., 253., 254., 197., 198., 199., 200., 201., 202.,
                                          203., 204., 205., 206., 207., 208., 209., 210., 211., 212.],
                                    [122., 239., 348., 349., 350., 351., 352., 353., 354., 355., 356.,
                                          357., 358., 359., 360., 361., 308., 309., 310., 311., 312., 313.,
                                          314., 315., 316., 317., 318., 319., 320., 321., 322., 213.],
                                    [121., 238., 347., 448., 449., 450., 451., 452., 453., 454., 455.,
                                          456., 457., 458., 459., 460., 411., 412., 413., 414., 415., 416.,
                                          417., 418., 419., 420., 421., 422., 423., 424., 323., 214.],
                                    [120., 237., 346., 447., 540., 541., 542., 543., 544., 545., 546.,
                                          547., 548., 549., 550., 551., 506., 507., 508., 509., 510., 511.,
                                          512., 513., 514., 515., 516., 517., 518., 425., 324., 215.],
                                    [119., 236., 345., 446., 539., 624., 625., 626., 627., 628., 629.,
                                          630., 631., 632., 633., 634., 593., 594., 595., 596., 597., 598.,
                                          599., 600., 601., 602., 603., 604., 519., 426., 325., 216.],
                                    [118., 235., 344., 445., 538., 623., 700., 701., 702., 703., 704.,
                                          705., 706., 707., 708., 709., 672., 673., 674., 675., 676., 677.,
                                          678., 679., 680., 681., 682., 605., 520., 427., 326., 217.],
                                    [117., 234., 343., 444., 537., 622., 699., 768., 769., 770., 771.,
                                          772., 773., 774., 775., 776., 743., 744., 745., 746., 747., 748.,
                                          749., 750., 751., 752., 683., 606., 521., 428., 327., 218.],
                                    [116., 233., 342., 443., 536., 621., 698., 767., 828., 829., 830.,
                                          831., 832., 833., 834., 835., 806., 807., 808., 809., 810., 811.,
                                          812., 813., 814., 753., 684., 607., 522., 429., 328., 219.],
                                    [115., 232., 341., 442., 535., 620., 697., 766., 827., 880., 881.,
                                          882., 883., 884., 885., 886., 861., 862., 863., 864., 865., 866.,
                                          867., 868., 815., 754., 685., 608., 523., 430., 329., 220.],
                                    [114., 231., 340., 441., 534., 619., 696., 765., 826., 879., 924.,
                                          925., 926., 927., 928., 929., 908., 909., 910., 911., 912., 913.,
                                          914., 869., 816., 755., 686., 609., 524., 431., 330., 221.],
                                    [113., 230., 339., 440., 533., 618., 695., 764., 825., 878., 923.,
                                          960., 961., 962., 963., 964., 947., 948., 949., 950., 951., 952.,
                                          915., 870., 817., 756., 687., 610., 525., 432., 331., 222.],
                                    [112., 229., 338., 439., 532., 617., 694., 763., 824., 877., 922.,
                                          959., 988., 989., 990., 991., 978., 979., 980., 981., 982., 953.,
                                          916., 871., 818., 757., 688., 611., 526., 433., 332., 223.],
                                    [ 111.,  228.,  337.,  438.,  531.,  616.,  693.,  762.,  823.,
                                          876.,  921.,  958.,  987., 1008., 1009., 1010., 1001., 1002.,
                                          1003., 1004.,  983.,  954.,  917.,  872.,  819.,  758.,  689.,
                                          612.,  527.,  434.,  333.,  224.],
                                    [ 110.,  227.,  336.,  437.,  530.,  615.,  692.,  761.,  822.,
                                          875.,  920.,  957.,  986., 1007., 1020., 1021., 1016., 1017.,
                                          1018., 1005.,  984.,  955.,  918.,  873.,  820.,  759.,  690.,
                                          613.,  528.,  435.,  334.,  225.],
                                    [ 109.,  226.,  335.,  436.,  529.,  614.,  691.,  760.,  821.,
                                          874.,  919.,  956.,  985., 1006., 1019., 1024., 1023., 1024.,
                                          1019., 1006.,  985.,  956.,  919.,  874.,  821.,  760.,  691.,
                                          614.,  529.,  436.,  335.,  226.],
                                    [ 108.,  225.,  334.,  435.,  528.,  613.,  690.,  759.,  820.,
                                          873.,  918.,  955.,  984., 1005., 1018., 1017., 1022., 1021.,
                                          1020., 1007.,  986.,  957.,  920.,  875.,  822.,  761.,  692.,
                                          615.,  530.,  437.,  336.,  227.],
                                    [ 107.,  224.,  333.,  434.,  527.,  612.,  689.,  758.,  819.,
                                          872.,  917.,  954.,  983., 1004., 1003., 1002., 1011., 1010.,
                                          1009., 1008.,  987.,  958.,  921.,  876.,  823.,  762.,  693.,
                                          616.,  531.,  438.,  337.,  228.],
                                    [106., 223., 332., 433., 526., 611., 688., 757., 818., 871., 916.,
                                          953., 982., 981., 980., 979., 992., 991., 990., 989., 988., 959.,
                                          922., 877., 824., 763., 694., 617., 532., 439., 338., 229.],
                                    [105., 222., 331., 432., 525., 610., 687., 756., 817., 870., 915.,
                                          952., 951., 950., 949., 948., 965., 964., 963., 962., 961., 960.,
                                          923., 878., 825., 764., 695., 618., 533., 440., 339., 230.],
                                    [104., 221., 330., 431., 524., 609., 686., 755., 816., 869., 914.,
                                          913., 912., 911., 910., 909., 930., 929., 928., 927., 926., 925.,
                                          924., 879., 826., 765., 696., 619., 534., 441., 340., 231.],
                                    [103., 220., 329., 430., 523., 608., 685., 754., 815., 868., 867.,
                                          866., 865., 864., 863., 862., 887., 886., 885., 884., 883., 882.,
                                          881., 880., 827., 766., 697., 620., 535., 442., 341., 232.],
                                    [102., 219., 328., 429., 522., 607., 684., 753., 814., 813., 812.,
                                          811., 810., 809., 808., 807., 836., 835., 834., 833., 832., 831.,
                                          830., 829., 828., 767., 698., 621., 536., 443., 342., 233.],
                                    [101., 218., 327., 428., 521., 606., 683., 752., 751., 750., 749.,
                                          748., 747., 746., 745., 744., 777., 776., 775., 774., 773., 772.,
                                          771., 770., 769., 768., 699., 622., 537., 444., 343., 234.],
                                    [100., 217., 326., 427., 520., 605., 682., 681., 680., 679., 678.,
                                          677., 676., 675., 674., 673., 710., 709., 708., 707., 706., 705.,
                                          704., 703., 702., 701., 700., 623., 538., 445., 344., 235.],
                                    [ 99., 216., 325., 426., 519., 604., 603., 602., 601., 600., 599.,
                                          598., 597., 596., 595., 594., 635., 634., 633., 632., 631., 630.,
                                          629., 628., 627., 626., 625., 624., 539., 446., 345., 236.],
                                    [ 98., 215., 324., 425., 518., 517., 516., 515., 514., 513., 512.,
                                          511., 510., 509., 508., 507., 552., 551., 550., 549., 548., 547.,
                                          546., 545., 544., 543., 542., 541., 540., 447., 346., 237.],
                                    [ 97., 214., 323., 424., 423., 422., 421., 420., 419., 418., 417.,
                                          416., 415., 414., 413., 412., 461., 460., 459., 458., 457., 456.,
                                          455., 454., 453., 452., 451., 450., 449., 448., 347., 238.],
                                    [ 96., 213., 322., 321., 320., 319., 318., 317., 316., 315., 314.,
                                          313., 312., 311., 310., 309., 362., 361., 360., 359., 358., 357.,
                                          356., 355., 354., 353., 352., 351., 350., 349., 348., 239.],
                                    [ 95., 212., 211., 210., 209., 208., 207., 206., 205., 204., 203.,
                                          202., 201., 200., 199., 198., 255., 254., 253., 252., 251., 250.,
                                          249., 248., 247., 246., 245., 244., 243., 242., 241., 240.],
                                    [ 94.,  93.,  92.,  91.,  90.,  89.,  88.,  87.,  86.,  85.,  84.,
                                          83.,  82.,  81.,  80.,  79., 140., 139., 138., 137., 136., 135.,
                                          134., 133., 132., 131., 130., 129., 128., 127., 126., 125.]])

   
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
                        if test_class == y:
                              x1=x[0]
                              size = x1.size()
                              y1 = np.zeros(size,dtype=np.complex128)
                              y1 = fft.fftshift(fft.fft2(x1))
                              for channel in range(3):
                                    y1[channel,:,:] = y1[channel,:,:] * mask                    

                              x1 = fft.ifft2(fft.ifftshift(y1))
                              x1 = np.real(x1)
                              x1 = torch.Tensor(x1).to(device)
                              x1 = torch.unsqueeze(x1, 0)
                              # print(x1.size())
                              _,_, y_hat = encoder(x1)
                              _, predicted = torch.max(y_hat.data,1)
                              if predicted == y.item():
                                    correct += 1
                  
                  if correct >= cur_pre-t:
                        # if cur_pre >= 600:
                              # cur_pre -= int(t/2)

                        cur_pre = correct
                        re_importance[re_importance == max_importance] = 0
                  
                  

                  importance[importance == max_importance] = 0
                  
                  if count %100 == 0:
                        plt.figure()
                        plt.imshow(mask)
                        plt.savefig(args.save_path+'/'+args.backbone_model+'/'+str(count)+'_'+str(test_class)+'mask.png')
                        plt.close()

            re_importance[re_importance >0] = 1
            print(count)
            plt.figure()
            plt.imshow(re_importance,cmap='gray')
            plt.savefig(args.save_path+'/'+args.backbone_model+'/test_bias_3t'+str(t)+'_'+str(test_class)+'.png')
            plt.close()
            print('class: %d' %test_class)
            print(cur_pre)
            print(list(re_importance))

if __name__ == '__main__':
      parser = argparse.ArgumentParser()
      parser.add_argument('--backbone_model', type=str, default='resnet18',
                              help='model ')
      parser.add_argument('--model_path', type=str, default='None',
                              help='path of the model')
      parser.add_argument('--save_path', type=str, default='./',
                              help='path of the saved directory')
      parser.add_argument('--t', type=int, default=6,
                              help='flexible threshold')

      args = parser.parse_args()
      if not os.path.exists(args.save_path+'/'+args.backbone_model+'/'):
            os.makedirs(args.save_path+'/'+args.backbone_model+'/')


      main(args)



