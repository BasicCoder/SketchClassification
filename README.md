# Sketch Classification
   A PyTorch Implementation for Sketch Classification Networks.
   
## Model Configuration
- Optimizer
   - Adam
## DataSet
TU-Berlin sketch dataset

| Model        | input_size  |
| ------------ | ----------- |
| (raw size)*  | 1111 * 1111 |
| AlexNet      | 224 * 224   |
| SketchANet   | 225 * 225   |
| ResNet18     | 224 * 224   |
| ResNet34     | 224 * 224   |
| ResNet50     | 224 * 224   |
| DenseNet121  | 224 * 224   |
| Inception_v3 | 299 * 299   |


## Model Parameters
| Model                    | lr   | clip_grad_norm(max_norm) | learning rate decay | weight_decay    |
| ------------------------ | ---- | ------------------------ | ------------------- | --------------- |
| AlexNet(pretrained)      | 2e-4 | --                       | 20                  | 0.0005          |
| AlexNet(scratch)         | 2e-5 | 0.5 - 100.0              | 30                  | 0.0005          |
| SketchANet(DogsCats)*    | 2e-5 | 0.5 - 1.0                | 30                  | 0.0005          |
| SketchANet(scratch)      | 2e-5 | 0.5 - 100.0              | 800                 | 0.0001 - 0.0003 |
| ResNet18(pretrained)     | 2e-4 | --                       | 20                  | 0.0005          |
| ResNet34(pretrained)     | 2e-4 | --                       | 20                  | 0.0001          |
| ResNet50(pretrained)     | 2e-4 | --                       | 20                  | 0.0005          |
| DenseNet121(pretrained)  | 2e-4 | --                       | 20                  | 0.0005          |
| Inception_v3(pretrained) | 2e-4 | --                       | 30                  | 0.0005          |
* *This is for test Model.

## Model Result
### Train Set
| Model                    | Prec@1  | Prec@5 |
| ------------------------ | ------- | ------ |
| AlexNet(pretrained)      | 93.4455 | 99.787 |
| AlexNet(scratch)         | 99.3024 | 99.988 |
| SketchANet(scratch)      | 86.3166 | 98.667 |
| ResNet18(pretrained)     | 96.9899 | 99.954 |
| ResNet34(pretrained)     | 97.1048 | 99.954 |
| ResNet50(pretrained)     | 98.3049 | 99.988 |
| DenseNet121(pretrained)  | 91.4301 | 99.596 |
| Inception_v3(pretrained) | 91.8802 | 99.706 |


### Test Set
| Model                    | Prec@1 | Prec@5 |
| ------------------------ | ------ | ------ |
| Human                    | 73.1   | --     |
| AlexNet<sup>i</sup>      | 68.6   | --     |
| AlexNet<sup>ii</sup>     | 77.29  | --     |
| GoogLeNet<sup>ii</sup>   | 80.85  | --     |
| AlexNet(pretrained)      | 70.850 | 90.050 |
| AlexNet(scratch)         | 53.850 | 78.000 |
| SketchANet(scratch)      | 68.700 | 88.900 |
| ResNet18(pretrained)     | 77.800 | 94.650 |
| ResNet34(pretrained)     | 79.100 | 95.050 |
| ResNet50(pretrained)     | 78.300 | 95.300 |
| DenseNet121(pretrained)  | 77.550 | 93.500 |
| Inception_v3(pretrained) | 76.550 | 93.750 |

* 1. *Sketch-a-Net that Beats Humans*
  2. *The Sketchy Database: Learning to Retrieve Badly Drawn Bunnies*

DPN, ShuffleNetG2, SENet18
## Tools
- GetImageMean_Std

Get image dataset mean and standard deviation.

- SplitDataset

Split image dataset according to the train and val record txt file.

- ListAllImageName

Get all image name in dataset.