# Sketch Classification
   A PyTorch Implementation for Sketch Classification Networks.
   
## Model Configuration
- Optimizer
   - Adam

## Model Parameters
| Model | lr | clip_grad_norm(max_norm)| learning rate decay | weight_decay |
| --- | --- | --- | --- | --- |
| AlexNet(pretrained) | 2e-4 | -- | 8 | 0.0005 | 
| AlexNet(scratch) | 2e-5 | 0.5 - 100.0 | 30 | 0.0005 |
| SketchANet(DogsCats) | 2e-5 | 0.5 - 1.0 | 30 | 0.0005 |
| SketchANet(TU-Berlin sketch dataset) | 2e-5 | 0.5 - 100.0 | 30 | 0.0001 - 0.0003 | 

## Model Result