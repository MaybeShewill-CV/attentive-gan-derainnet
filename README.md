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
This software has only been tested on ubuntu 16.04(x64), python3.5, cuda-9.0, cudnn-7.0 with 
a GTX-1070 GPU. To install this software you need tensorflow 1.10.0 and other version of 
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
cd REPO_ROOT_DIR
python tools/test_model.py --weights_path model/new_model/derain_gan_2018-10-09-11-06-58.ckpt-200000
--image_path data/test_data/test_1.png
```

The results are as follows:

`Test Input Image`

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
cd REPO_ROOT_DIR
python tools/train_model.py --dataset_dir data/training_data_example/
```

You can also continue the training process from the snapshot by
```
cd REPO_ROOT_DIR
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

## Common Issue
Several users find out the nan loss problem may occasionally happen in
training process under tensorflow v1.3.0. I think it may be caused by the randomly parameter 
initialization problem. My solution is to kill the training process and
restart it again to find a suitable initialized parameters. At the 
mean time I have found out that if you use the model under tensorflow
v1.10.0 the nan loss problem will not happen. The reason may be the
difference of parameter initialization function or the loss optimizer
function between older tensorflow and newest tensorflow. If the nan 
loss problem still troubles you when training the model then upgrading 
your local tensorflow may be a nice option. Good luck on training process!

Thanks for the issues by [Jay-Jia](https://github.com/Jay-Jia)

## Update on 2018.10.12
Adjust the initialized learning rate and using exponential decay
strategy to adjust the learning rate during training process. Using
traditional image augmentation function including random crop and 
random flip to augment the training dataset which protomed the new
model performance. I have uploaded a new tensorboard record file and
you can check the image ssim to compare the two models. New
model weights can be found under model/new_model folder.

`Model result comparison`
![Comparison_result](https://github.com/MaybeShewill-CV/attentive-gan-derainnet/blob/master/data/images/model_comparison.png)

The first row is the source test image in folder ./data/test_data, the
second row is the derain result generated by the old model and the last
row is the derain result generated by the new model. As you can see the
new model can recover more vivid details than the old model and I will 
upload a figure of ssim and psnr which will illustrate the new model's
promotion.

## Update on 2018.11.3
Since the batch size is 1 during the training process so the batch
normalization layer seems to be useless. All the bn layers were removed
after the new updates. I have trained a new model based on the newest 
code and the new model will be placed in folder root_dir/model/new_model
and the model updated on 2018.10.12 will be placed in folder 
root_dir/model/old_model. The new model can present more vivid details
compared with the old model. The model's comparison result can be seen
as follows.

`Model result comparision`
![New_Comparison_result_v2](https://github.com/MaybeShewill-CV/attentive-gan-derainnet/blob/master/data/images/model_comparison_v2.png)

The first row is the source test image in folder ./data/test_data, the
second row is the derain result generated by the old model and the last
row is the derain result generated by the new model. As you can see the
new model perform much better than the old model.

Since the bn layer will leads to a unstable result the deeper attention 
map of the old model will not catch valid information which is supposed
to guide the model to focus on the rain drop. The attention map's 
comparision result can be seen as follows.

`Model attention map result comparision`
![Attention_Map_Comparison_result](https://github.com/MaybeShewill-CV/attentive-gan-derainnet/blob/master/data/images/attention_map_comparision_rsult.png)

The first row is the source test image in folder ./data/test_data, the
second row is the attention map 4 generated by the old model and the 
last row is the attention map 4 generated by the new model. As you can 
see the new model catch much more valid attention information than the
old model.

## TODO
- [x] Parameter adjustment
- [x] Test different loss function design
