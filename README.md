# [MentorMix] "Beyond Synthetic Noise: Deep Learning on Controlled Noisy Labels" PyTorch Implementation
This repository implemented paper [Beyond Synthetic Noise: Deep Learning on Controlled Noisy Labels](https://arxiv.org/pdf/1911.09781.pdf) in [PyTorch](https://pytorch.org/) version. Official code is [here](https://github.com/google-research/google-research/tree/master/mentormix) which is implemented by google-research with tensorflow.   
Code of this repository provides training method from **scratch** with dataset [CIFAR10/CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html).

## Requirements
```bash
torch 1.7.1
torchvision 0.8.1
tqdm
argparse
```

## How to run
After you have cloned the repository, you can train each model from scratch with datasets CIFAR10, CIFAR100. Trainable models are [ResNet](https://arxiv.org/abs/1512.03385). You can adjust this code if you want to train other kinds of architectures.

- Using threshold function as MentorNet
```bash
python train.py --dataset cifar10
                --StudentNet ResNet34 --MentorNet threshold --MentorNet_type PD
                --optimizer SGD --scheduler StepLR
                --lr 0.1 --batch_size 128 --epoch 500 --wd 2e-4
                --noise_rate 0.2 
                --ema 0.0001
                --gamma_p 0.8 --alpha 2.
                --second_reweight
                --trial 0
                --gpu_id 0
```

- Using DNN as MentorNet  
First, train MentorNet(**P**re-**D**efined or **D**ata-**D**riven).

```bash
python3 train_MentorNet.py  --dataset cifar10
                            --StudentNet ResNet34 --MentorNet MentorNet --MentorNet_type PD
                            --optimizer SGD --scheduler CosineAnnealing
                            --lr 0.1 --batch_size 32 --epoch 100 --wd 2e-4
                            --noise_rate 0.
                            --ema 0.05
                            --gamma_p 0.75
                            --train_MentorNet
                            --trial 0
                            --gpu_id 0
```

(If you train MentorNet in **D**ata-**D**riven way, noise rate has to be the same as one when training StudentNet later.)

Second, train StudentNet with pre-trained MentorNet.

```bash
python train.py --dataset cifar10
                --StudentNet ResNet34 --MentorNet MentorNet --MentorNet_type DD
                --optimizer SGD --scheduler StepLR
                --lr 0.1 --batch_size 128 --epoch 500 --wd 2e-4
                --noise_rate 0.2 
                --ema 0.0001
                --gamma_p 0.8 --alpha 2.
                --second_reweight
                --trial 0
                --gpu_id 0
```

## Implementation Details
Most of the hyperparameters refers to the values mentioned in the [paper](https://arxiv.org/pdf/1911.09781.pdf). However, some hyperparameters such as _??<sub>p</sub>_ or _??_ refers to the values used in the [official code](https://github.com/google-research/google-research/tree/master/mentormix). Those hyperparameters are marked out according to the Noise Level below.


**Hyperparameters referred by paper**

|   epoch   | learning rate |  weight decay | Optimizer | Momentum |  Nesterov |  scheduler  |   EMA    |  second reweight |
|:---------:|:-------------:|:-------------:|:---------:|:--------:|:---------:|:-----------:|:--------:|:----------------:|
|    400    |      0.1      |     0.0002    |    SGD    |    0.9   |   False   | StepLR(0.9) |  0.0001  |       True       |

**Hyperparameters referred by Official Code**

- _??<sub>p</sub>_ and _??_ in CIFAR10

|    Noise Level   |   0.2   |   0.4   |   0.6   |   0.8   |
|:----------------:|:-------:|:-------:|:-------:|:-------:|
|        _??_       |    2    |    8    |    8    |    4    |
| _??<sub>p</sub>_  |   0.8   |   0.6   |   0.6   |   0.2   |
| _second reweight_|  False  |  False  |  True   |  True   |

- _??<sub>p</sub>_ and _??_ in CIFAR100

|    Noise Level   |   0.2   |   0.4   |   0.6   |   0.8   |
|:----------------:|:-------:|:-------:|:-------:|:-------:|
|        _??_       |    2    |    8    |    4    |    8    |
| _??<sub>p</sub>_  |   0.7   |   0.5   |   0.3   |   0.1   |
| _second reweight_|  False  |  False  |  True   |  True   |



## Accuracy
Below is the result of the test accuracy trained with ResNet34. Results are averaged over 3 repeated experiments of same circumstances.   
(_All values are percentiles._)

**CIFAR10**
|    Noise Level    |   0.2    |   0.4    |   0.6    |   0.8    |
|:-----------------:|:--------:|:--------:|:--------:|:--------:|
|      Official     |   95.60  |   94.20  |   91.30  |   81.00  |
|      This repo    |   95.47  |   93.47  |   88.88  |   20.65  |


**CIFAR100**
|    Noise Level    |   0.2    |   0.4    |   0.6    |   0.8    |
|:-----------------:|:--------:|:--------:|:--------:|:--------:|
|      Official     |   78.60  |   71.30  |   64.60  |   41.20  |
|      This repo    |   76.30  |   71.84  |   38.83  |   7.20   |