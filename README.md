# Official website of '[DFM-X: Augmentation by Leveraging Prior Knowledge of Shortcut Learning](https://openreview.net/forum?id=NywSmrJ3Hr) (ICCV2023 Worshop VIPrios)' 
-Augmentation for CV using frequency shortcuts
---

## Introduction

Neural networks are prone to learn easy solutions from superficial statistics in the data, namely shortcut learning, which impairs generalization and robustness of models. We
propose a data augmentation strategy, named DFM-X, that leverages knowledge about frequency shortcuts, encoded in Dominant Frequencies Maps computed for image classification models. We randomly select X% training images of certain classes for augmentation, and process them by retaining the frequencies included in the DFMs of other classes. This strategy compels the models to leverage a broader range of frequencies for classification, rather than relying on specific frequency sets. Thus, the models learn more deep and task-related semantics compared to their counterpart trained with standard setups. Unlike other commonly used augmentation techniques which focus on increasing the visual variations of training data, our method targets exploiting the original data efficiently, by distilling prior knowledge about destructive learning behavior of models from data. Our experimental results demonstrate that DFM-X improves robustness against common corruptions and adversarial attacks. It can be seamlessly integrated with other augmentation techniques to further enhance the robustness of models.

<p align='center'><img src='figures/scheme.png' width='600'></p>


### Quick start

* Clone this repository:
```
git clone https://github.com/nis-research/dfmX-augmentation.git
cd dfmX-augmentation
```

* Installation
	* Python 3.9.12, cuda-11.7, cuda-11.x_cudnn-8.6
		* You can create a virtual environment with conda and activate the environment before the next step
			```
			conda create -n virtualenv  python=3.9 anaconda
			source activate virtualenv
			conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
			```
	* Install other packages
		```
		pip install -r requirements.txt

		```
* Computing DFMs of CIFAR for a pre-trained model, e.g.
		
```
python -u Evaluation/Compute_DFM_CIFAR.py  --backbone_model resnet18 --model_path /checkpoints/last.ckpt      
```

* Training models with DFM-X, e.g. 
	```
	python -u train.py   --backbone_model resnet18 --lr 0.01 --dataset imagenet10   --save_dir results/   
	```

* Evaluating adversarial robustness, e.g.
```
python -u 

```

* Evaluating corruption robustness, e.g.
```
python -u 
```










## Citation

```
@inproceedings{
wang2023dfmx,
title={{DFM}-X: Augmentation by Leveraging Prior Knowledge of Shortcut Learning},
author={Shunxin Wang and Christoph Brune and Raymond Veldhuis and Nicola Strisciuglio},
booktitle={4th Visual Inductive Priors for Data-Efficient Deep Learning Workshop},
year={2023},
url={https://openreview.net/forum?id=NywSmrJ3Hr}
}
```
