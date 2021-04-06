# Train CIFAR10,CIFAR100 with Pytorch-lightning
This repository implemented [MentorMix](https://github.com/google-research/google-research/tree/master/mentormix) as PyTorch version.

## Requirements
- setup/requirements.txt
```bash
torch 1.7.1
torchvision 0.8.1
tqdm
argparse
```

## How to run
After you have cloned the repository, you can train each models with datasets cifar10, cifar100. Trainable models are [Resnet](https://arxiv.org/abs/1512.03385).

```bash
python train.py --noise_rate 0.2
```

## Implementation Details
- CIFAR10

|   epoch   | learning rate |  weight decay | Optimizer | Momentum |  Nesterov |
|:---------:|:-------------:|:-------------:|:---------:|:--------:|:---------:|
|    300    |      0.1      |     0.0005    |    SGD    |    0.9   |   False   |


- CIFAR100

|   epoch   | learning rate |  weight decay | Optimizer | Momentum |  Nesterov |
|:---------:|:-------------:|:-------------:|:---------:|:--------:|:---------:|
|    300    |      0.1      |     0.0005    |    SGD    |    0.9   |   False   |


## Accuracy
Below is the result of the test accuracy for CIFAR-10, CIFAR-100 dataset training. Results are averaged over 3 repeated experiments of same circumstances.

**Accuracy of models trained on CIFAR10**
|    noise rate     |   0.2    |   0.4    |   0.6    |   0.8    |
|:-----------------:|:--------:|:--------:|:--------:|:--------:|
|       ResNet18    |   95.09% |   --.--% |   --.--% |   --.--% |



**Accuracy of models trained on CIFAR100**
|    noise rate     |   0.2    |   0.4    |   0.6    |   0.8    |
|:-----------------:|:--------:|:--------:|:--------:|:--------:|
|       ResNet18    |   --.--% |   --.--% |   --.--% |   --.--% |
