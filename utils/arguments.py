import argparse

def get_arguments():
    parser = argparse.ArgumentParser(description = 'Training Arguments')
    parser.add_argument('--gpu_id', default='0', type=int, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100', 'svhn'], help = 'dataset choice')
    parser.add_argument('--arch', default = 'ResNet18', type=str, choices = ['MobileNet','DenseNet','ResNet18','ResNet34','ResNet50','ResNet101','WideResNet28_2','WideResNet28_10','WideResNet40_2','WideResNet40_4','EfficientNet'])
    parser.add_argument('--optimizer', default = 'SGD', type=str, choices = ['SGD','Nesterov','Adam','AdamW'])
    parser.add_argument('--lr','--learning-rate', default = 0.1, type=float, choices = [1.0, 0.1,0.01,0.001,0.0005,0.0002,0.0001])
    parser.add_argument('--epoch', default=300, type=int, help='number of total epochs')
    parser.add_argument('--batch_size', default=128, type=int, choices=[64,128,256,512])
    parser.add_argument('--scheduler', default='MultiStepLR', type=str, choices=['MultiStepLR','CosineAnnealing','CosineWarmup'])
    parser.add_argument('--wd', '--weight_decay','--wdecay', default=4e-5, type=float, choices=[5e-4,1e-2,1e-3,1e-4,1e-6])
    parser.add_argument('--warmup_duration', default = 10, help = 'duration of warming up')
    parser.add_argument('--refinement', type=str, choices=['label_smoothing','mixup'])
    parser.add_argument('--noise_rate', default=0., type=float, choices=[0.,0.2,0.4,0.6,0.8])
    parser.add_argument('--ema', default = 0.001, type=float)
    parser.add_argument('--gamma_p', default = 0.8, type=float, choices = [0.7,0.8,0.9])
    parser.add_argument('--alpha', default = 2., type=float, choices = [2.,4.,8.,32.])
    parser.add_argument('--trial', default = '0', type=str)
    args = parser.parse_args()
    return args