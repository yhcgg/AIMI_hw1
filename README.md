# AIMI_hw1
## How to use
- Download the [dataset](https://drive.google.com/file/d/1x9Pzxwspxg_gV22Wx8sW4NaDDUq0OxDg/view?usp=sharing)
- set parameters
  - --num_classes: number of classes (default: 2)
  - --model_name: select the model you want (default: resnet50)
    - resnet18
    - resnet50
    - vgg16
  - --num_epochs: number of epoch (default: 30)
  - --batch_size: size of batch (default: 128)
  - --lr: learning rate (default: 1e-5)
  - --wd: weight decay (default: 0.9)
  - --dataset: path of dataset (default: chest_xray)
  - --augmentation: do data augmentation or not (default: False)
  - --degree: rotation degree (default: 90)   *#only useful if --augmentation = False*
  - --resize: image resize (default: 224)
- Run `python train.py`
