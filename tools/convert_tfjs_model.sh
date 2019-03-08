#!/usr/bin/env bash
CUDA_VISIBLE_DEVIDE="0" tensorflowjs_converter --input_format=tf_saved_model --output_node_name=derain_image_result \
--saved_model_tags=serve ./model/derain_gan_saved_model ./model/derain_gan_web_model