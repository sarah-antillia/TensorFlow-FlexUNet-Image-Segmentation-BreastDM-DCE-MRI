<h2>TensorFlow-FlexUNet-Image-Segmentation-BreastDM-DCE-MRI (2025/09/05)</h2>

Toshiyuki A. Arai: Software Laboratory @antillia.com<br>
<br>
This is the first experiment of Image Segmentation for BreastDM-DCE(Dynamic Contrast-Enhanced)-MRI (Benign and Malignant)
 based on our 
TensorFlowFlexUNet (TensorFlow Flexible UNet Image Segmentation Model for Multiclass) 
and a 512x512 pixels 
<a href="https://drive.google.com/file/d/1gXjdR84yuZUXwAgVEoDu6uykYBpRD-C8/view?usp=sharing">
BreastDM-ImageMask-Dataset.zip</a> with colorized masks (benign:green, malignant:red), 
which was derived by us from the following dataset on the google drive 
<a href="https://drive.google.com/file/d/1GvNwL4iPcB2GRdK2n353bKiKV_Vnx7Qg/view">
BreaDM.zip
</a> specified in a repository <a href="https://github.com/smallboy-code/Breast-cancer-dataset">Breast-cancer-dataset</a>

<br><br>
On BreaDM dataset, please refer to <a href="https://www.sciencedirect.com/science/article/abs/pii/S0010482523007205">
<b>BreastDM: A DCE-MRI dataset for breast tumor image segmentation and classification</b></a>
, and <a href="https://github.com/smallboy-code/Breast-cancer-dataset">Breast-cancer-dataset</a><br>.
<br>

Please see also our experiment for a singleclass segmentation model 
<a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-Malignant-BreastDM">
Tensorflow-Image-Segmentation-Malignant-BreastDM
</a>
<br>
<br>
<b>Acutual Image Segmentation for 512x512 BreastDM images</b><br>

As shown below, the inferred masks predicted by our segmentation model trained on the 
PNG dataset appear similar to the ground truth masks.<br>
<b>rgb_map =  (benign:green, malignant:red)</b><br>
<br>
<br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/BreastDM/mini_test/images/benign_305.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/BreastDM/mini_test/masks/benign_305.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/BreastDM/mini_test_output/benign_305.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/BreastDM/mini_test/images/malignant_4111.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/BreastDM/mini_test/masks/malignant_4111.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/BreastDM/mini_test_output/malignant_4111.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/BreastDM/mini_test/images/malignant_4152.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/BreastDM/mini_test/masks/malignant_4152.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/BreastDM/mini_test_output/malignant_4152.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>1. Dataset Citation</h3>
The dataset used here has been obtained from the google drive

</a> <a href="https://drive.google.com/file/d/1GvNwL4iPcB2GRdK2n353bKiKV_Vnx7Qg/view">
BreaDM.zip
</a> specified in <a href="https://github.com/smallboy-code/Breast-cancer-dataset">Breast-cancer-dataset</a>



<br><br>
On BreastDM dataset, please refer to
<a href="https://www.sciencedirect.com/science/article/abs/pii/S0010482523007205">
 BreastDM: A DCE-MRI dataset for breast tumor image segmentation and classification</b>
</a><br>
Xiaoming Zhao, Yuehui Liao, Jiahao Xie, Xiaxia He, Shiqing Zhang, Guoyu Wang <br>, Jiangxiong Fang
, Hongsheng Lu, Jun Yu <br>

<a href="https://doi.org/10.1016/j.compbiomed.2023.107255">https://doi.org/10.1016/j.compbiomed.2023.107255</a><br>



<br>
<h3>
<a id="2">
2 BreastDM ImageMask Dataset
</a>
</h3>
 If you would like to train this BreastDM Segmentation model by yourself,
 please download the dataset from the google drive 
<a href="https://drive.google.com/file/d/1gXjdR84yuZUXwAgVEoDu6uykYBpRD-C8/view?usp=sharing">
BreastDM-ImageMask-Dataset.zip</a>.
<br>
, expand the downloaded ImageMaskDataset and put it under <b>./dataset</b> folder to be
<pre>
./dataset
└─BreastDM
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
On the derivation of this dataset, please refer to
<a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-Malignant-BreastDM/blob/main/projects/TensorflowSlightlyFlexibleUNet/Malignant-BreastDM/generator/TrainMalignantImageMaskDatasetGenerator.py">
TrainMalignantImageMaskDatasetGenerator.py
</a> in our repository 
<a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-Malignant-BreastDM">
Tensorflow-Image-Segmentation-Malignant-BreastDM
</a>
<br>
<br>
<b>BreastDM Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/BreastDM/BreastDM_Statistics.png" width="512" height="auto"><br>
<br>

<br>

As shown above, the number of images of train and valid datasets is large enough to use for a training set of our segmentation model.
<br>
<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/BreastDM/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/BreastDM/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<h3>
3 Train TensorFlowFlexUNet Model
</h3>
 We trained BreastDM TensorFlowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/BreastDM/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/BreastDM and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters = 16 </b> and large <b>base_kernels = (7,7)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
;You may specify your own UNet class derived from our TensorFlowFlexModel
model         = "TensorFlowFlexUNet"
generator     =  False
image_width    = 512
image_height   = 512
image_channels = 3
num_classes    = 3

base_filters   = 16
base_kernels   = (7,7)
num_layers     = 8
dropout_rate   = 0.05
dilation       = (1,1)
</pre>
<b>Learning rate</b><br>
Defined a very small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>
<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and <a href="./src/dice_coef_multiclass.py">"dice_coef_multiclass"</a>.<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b>Dataset class</b><br>
Specifed <a href="./src/ImageCategorizedMaskDataset.py">ImageCategorizedMaskDataset</a> class.<br>
<pre>
[dataset]
class_name    = "ImageCategorizedMaskDataset"
</pre>
<br>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.5
reducer_patience   = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>RGB Color map</b><br>
rgb color map dict for BreastDM 1+2 classes.<br>
<pre>
[mask]
mask_file_format = ".png"

; RGB colors    benign:green, malignanat:red    
rgb_map = {(0,0,0):0,(0,255,0):1,(255,0,0):2,}
</pre>

<b>Epoch change inference callback</b><br>
Enabled <a href="./src/EpochChangeInferencer.py">epoch_change_infer callback (EpochChangeInferencer.py)</a></b>.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   = "./epoch_change_infer"
num_infer_images         = 6
</pre>

By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at starting (epoch 1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/BreastDM/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middlepoint (epoch 11,12,13)</b><br>
<img src="./projects/TensorFlowFlexUNet/BreastDM/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (epoch 23,24,25)</b><br>
<img src="./projects/TensorFlowFlexUNet/BreastDM/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>
In this experiment, the training process was terminated at epoch 25.<br><br>
<img src="./projects/TensorFlowFlexUNet/BreastDM/asset/train_console_output_at_epoch25.png" width="920" height="auto"><br>
<br>

<a href="./projects/TensorFlowFlexUNet/BreastDM/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/BreastDM/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/BreastDM/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/BreastDM/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/BreastDM</b> folder,<br>
and run the following bat file to evaluate TensorFlowFlexUNet model for BreastDM.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetEvaluator.py ./train_eval_infer_aug.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/BreastDM/asset/evaluate_console_output_at_epoch25.png" width="920" height="auto">
<br><br>

<a href="./projects/TensorFlowFlexUNet/BreastDM/evaluation.csv">evaluation.csv</a><br>
The loss (categorical_crossentropy) to this BreastDM/test was very low and dice_coef_multiclass 
very high as shown below.
<br>
<pre>
categorical_crossentropy,0.0004
dice_coef_multiclass,0.9997
</pre>
<br>

<h3>
5 Inference
</h3>
Please move <b>./projects/TensorFlowFlexUNet/BreastDM</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorFlowFlexUNet model for BreastDM.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer_aug.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/BreastDM/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/BreastDM/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/BreastDM/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks of 512x512 pixels</b><br>
<b>rgb_map =  (benign:green, malignant:red)</b><br>
<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/BreastDM/mini_test/images/benign_277.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/BreastDM/mini_test/masks/benign_277.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/BreastDM/mini_test_output/benign_277.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/BreastDM/mini_test/images/benign_281.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/BreastDM/mini_test/masks/benign_281.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/BreastDM/mini_test_output/benign_281.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/BreastDM/mini_test/images/benign_305.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/BreastDM/mini_test/masks/benign_305.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/BreastDM/mini_test_output/benign_305.png" width="320" height="auto"></td>
</tr>



<tr>
<td><img src="./projects/TensorFlowFlexUNet/BreastDM/mini_test/images/malignant_4144.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/BreastDM/mini_test/masks/malignant_4144.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/BreastDM/mini_test_output/malignant_4144.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/BreastDM/mini_test/images/malignant_4153.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/BreastDM/mini_test/masks/malignant_4153.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/BreastDM/mini_test_output/malignant_4153.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/BreastDM/mini_test/images/malignant_4430.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/BreastDM/mini_test/masks/malignant_4430.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/BreastDM/mini_test_output/malignant_4430.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>
References
</h3>
<b>1. BreastDM: A DCE-MRI dataset for breast tumor image segmentation and classification</b><br>
Xiaoming Zhao, Yuehui Liao, Jiahao Xie, Xiaxia He, Shiqing Zhang, Guoyu Wang <br>, Jiangxiong Fang
, Hongsheng Lu, Jun Yu <br>
<a href="https://doi.org/10.1016/j.compbiomed.2023.107255">https://doi.org/10.1016/j.compbiomed.2023.107255</a><br>
<a href="https://www.sciencedirect.com/science/article/abs/pii/S0010482523007205">
https://www.sciencedirect.com/science/article/abs/pii/S0010482523007205</a>
<br>
<br>
<b>2. Tensorflow-Image-Segmentation-Malignant-BreastDM</b><br>
Toshiyuki Arai @antillia.com<br>
<a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-Malignant-BreastDM">
https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-Malignant-BreastDM
</a>
<br>
<br>
