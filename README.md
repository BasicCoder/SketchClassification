# Sketch Classification
   A PyTorch Implementation for Sketch Classification Networks.
   
## Model Configuration
- Optimizer
   - Adam
## DataSet
TU-Berlin sketch dataset

## Model Parameters
| Model | lr | clip_grad_norm(max_norm)| learning rate decay | weight_decay |
| --- | --- | --- | --- | --- |
| AlexNet(pretrained) | 2e-4 | -- | 20 | 0.0005 | 
| AlexNet(scratch) | 2e-5 | 0.5 - 100.0 | 30 | 0.0005 |
| SketchANet(DogsCats)* | 2e-5 | 0.5 - 1.0 | 30 | 0.0005 |
| SketchANet(scratch) | 2e-5 | 0.5 - 100.0 | 800 | 0.0001 - 0.0003 | 
| ResNet18(pretrained) | 2e-4 | -- | 20 | 0.0005 |
| ResNet34(pretrained) | 2e-4 | -- | 20 | 0.0001 |
| ResNet50(pretrained) | 2e-4 | -- | 20 | 0.0005 |
* *This is for test Model.

## Model Result
### Train Set
| Model | Prec@1 | Prec@5 |
| --- | --- | --- |
| AlexNet(pretrained) | 90.3780 | 99.690 |
| AlexNet(scratch) |  |  |
| SketchANet(scratch) | 86.3166 | 98.667 |
| ResNet18(pretrained) | 96.9899 | 99.954 |
| ResNet34(pretrained) | 97.1048 | 99.954 |
| ResNet50(pretrained) | 98.3049 | 99.988 |


### Test Set
| Model | Prec@1 | Prec@5 |
| --- | --- | --- |
| Human | 73.1 | -- |
| AlexNet(pretrained) | 67.600 | 87.400 |
| AlexNet(scratch) |  |  |
| SketchANet(scratch) | 65.050 | 86.350 |
| ResNet18(pretrained) | 72.300 | 90.200 |
| ResNet34(pretrained) | 73.500 | 92.050 |
| ResNet50(pretrained) | 77.500 | 94.000 |


## Tools
- GetImageMean_Std

Get image dataset mean and standard deviation.

- SplitDataset

Split image dataset according to the train and val record txt file.