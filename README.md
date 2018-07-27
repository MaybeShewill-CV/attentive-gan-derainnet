# attentive-gan-derainnet
Use tensorflow to implement a Deep Convolution Generative Adversarial Network for image derain 
task mainly based on the CVPR2018 paper "ttentive Generative Adversarial Network for Raindrop 
Removal from A Single Image".You can refer to their paper for details https://arxiv.org/abs/1711.10098. 
This model consists of a attentive attentive-recurrent network, a contextual autoencoder 
network and a discriminative network. Using convolution lstm unit to generate attention map 
which is used to help locating the rain drop, multi-scale losses and a perceptual loss to 
train the context autoencoder network. Thanks for the origin author [Rui Qian](https://github.com/rui1996)

The main network architecture is as follows:

`Network Architecture`
![NetWork_Architecture](https://github.com/MaybeShewill-CV/attentive-gan-derainnet/blob/master/data/images/net_architecture.png)

## Installation
This software has only been tested on ubuntu 16.04(x64), python3.5, cuda-8.0, cudnn-6.0 with 
a GTX-1070 GPU. To install this software you need tensorflow 1.3.0 and other version of 
tensorflow has not been tested but I think it will be able to work properly in 
tensorflow above version 1.0. Other required package you may install them by

```
pip3 install -r requirements.txt
```

## Test model
In this repo I uploaded a model trained on dataset provided by the origin author 
[origin_dataset](https://drive.google.com/open?id=1e7R76s6vwUJxILOcAsthgDLPSnOrQ49K).

The trained derain net model weights files are stored in folder model/

You can test a single image on the trained model as follows

```
python tools/test_model.py --weights_path model/derain_gan_v2_2018-07-23-11-26-23.ckpt-200000
--image_path data/test_data/test_1.png
```

The results are as follows:

`Test Input Image`

<figure class="half">
    <img src="https://github.com/MaybeShewill-CV/attentive-gan-derainnet/blob/master/data/images/src_img.png" height="150px" alt="图片说明" >
    <img src="https://github.com/MaybeShewill-CV/attentive-gan-derainnet/blob/master/data/images/src_img.png" height="150px" alt="图片说明" >
</figure>

![Test Input](https://github.com/MaybeShewill-CV/attentive-gan-derainnet/blob/master/data/images/src_img.png)

`Test Derain result image`

![Test Derain_Result](https://github.com/MaybeShewill-CV/attentive-gan-derainnet/blob/master/data/images/derain_ret.png)

`Test Attention Map at time 1`

![Test Attention_Map_1](https://github.com/MaybeShewill-CV/attentive-gan-derainnet/blob/master/data/images/atte_map_1.png)

`Test Attention Map at time 2`

![Test Attention_Map_2](https://github.com/MaybeShewill-CV/attentive-gan-derainnet/blob/master/data/images/atte_map_2.png)

`Test Attention Map at time 3`

![Test Attention_Map_3](https://github.com/MaybeShewill-CV/attentive-gan-derainnet/blob/master/data/images/atte_map_3.png)

`Test Attention Map at time 4`

![Test Attention_Map_4](https://github.com/MaybeShewill-CV/attentive-gan-derainnet/blob/master/data/images/atte_map_4.png)

## Train your own model

#### Data Preparation
Firstly you need to organize your training data refer to the data/training_data_example 
folder structure. And you need to generate a train.txt record the data used for training 
the model. 

The training samples are consist of two components. A clean image free from rain drop label 
image and a origin image degraded by raindrops.

All your training image will be scaled into the same scale according to the config file.

#### Train model
In my experiment the training epochs are 200010, batch size is 1, initialized learning rate 
is 0.001. About training parameters you can check the global_configuration/config.py for 
details.
 
You may call the following script to train your own model

```
python tools/train_model.py --dataset_dir data/training_data_example/
```

You can also continue the training process from the snapshot by
```
python tools/train_model.py --dataset_dir data/training_data_example/ 
--weights_path path/to/your/last/checkpoint
```

You may monitor the training process using tensorboard tools

During my experiment the `G loss` drops as follows:  
![G_loss](https://github.com/MaybeShewill-CV/attentive-gan-derainnet/blob/master/data/images/g_loss.png)

The `D loss` drops as follows:  
![D_loss](https://github.com/MaybeShewill-CV/attentive-gan-derainnet/blob/master/data/images/d_loss.png)

The `Image SSIM between generated image and clean label image` raises as follows:  
![Image_SSIM](https://github.com/MaybeShewill-CV/attentive-gan-derainnet/blob/master/data/images/image_ssim.png)

Please cite my repo [attentive-gan-derainnet](https://github.com/MaybeShewill-CV/attentive-gan-derainnet) 
if you find it helps you.

## TODO
- [ ] Parameter adjustment
- [ ] Test different loss function design