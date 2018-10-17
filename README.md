# Tencent ML-Images

This repository introduces the open-source project dubbed **Tencent ML-Images**, which publishes 
<!--- * **ML-Images**: the largest open-source multi-label image database, including 18,019,881 URLs to images, which are annotated with labels up to 11,166 categories-->
* **ML-Images**: the largest open-source multi-label image database, including 17,609,752 training and 88,739 validation image URLs, which are annotated with up to 11,166 categories
* **Resnet-101 model**: it is pre-trained on ML-Images, and achieves the top-1 accuracy 80.73% on ImageNet via transfer learning



# Contents

* [Dependencies](#dependencies)

* [Data](#data)
  * [Download](#download)
  * [Image Source](#1)
  * [Semantic Hierarchy](#hierarchy)
  * [Annotations](#annotation)
  * [Statistics](#statistics)
  
* [Train](#train)
  * [Download Images using URLs](#download-image)
  * [Prepare the TFRecord File](#prepare-tfrecord)
  * [Pretrain on ML-Images](#pretrain)
  * [Finetune on ImageNet](#finetune)
  * [Checkpoints](#checkpoint)
  * [Feature Extraction](#feature-extraction)
 
* [Results](#result)

* [Copyright](#copyright)

* [Citation](#citation)


# [Dependencies](#dependencies)
  * Linux
  * [Python 2.7](https://www.python.org/)
  * [Tensorflow >= 1.6.0](https://www.tensorflow.org/install/)

# [Data](#data)
[[back to top](#)]

### [Download](#download)
[[back to top](#)]

The image URLs and the corresponding annotations can be downloaded from [train_urls.txt](https://tencent-ml-images-1257811961.cos.ap-guangzhou.myqcloud.com/train_urls.txt) and [val_urls.txt](https://tencent-ml-images-1257811961.cos.ap-guangzhou.myqcloud.com/val_urls.txt). Please move the downloaded txt file into `data/`. 
The format of `train_urls.txt` is as follows

```
...
https://c4.staticflickr.com/8/7239/6997334729_e5fb3938b1_o.jpg  3:1  5193:0.9  5851:0.9 9413:1 9416:1
https://c2.staticflickr.com/4/3035/3033882900_a9a4263c55_o.jpg  1053:0.8  1193:0.8  1379:0.8
...
```
As shown above, one image corresponds to one row. The first term is the image URL. The followed terms seperated by space are the annotations. For example, "5193:0.9" indicates class 5193 and its confidence 0.9. Note that the class index starts from 0, and you can find the class name from the file [data/dictionary_and_semantic_hierarchy.txt](data/dictionary_and_semantic_hierarchy.txt).

### [Image Source](#1)
[[back to top](#)]


The image URLs of ML-Images are collected from [ImageNet](http://www.image-net.org/) and [Open Images](https://github.com/openimages/dataset). 
Specifically,  
* Part 1: From the whole database of ImageNet, we adopt 10,706,941 training and 50,000 validation image URLs, covering 10,032 categories.
* Part 2: From Open Images, we adopt 6,902,811 training and 38,739 validation image URLs, covering 1,134 unique categories (note that some other categories are merged with their synonymous categories from ImageNet). 

Finally, ML-Images includes 17,609,752 training and 88,739 validation image URLs, covering 11,166 categories. 

<!---
The image URLs of ML-Images are collected from [ImageNet](http://www.image-net.org/) and [Open Images](https://github.com/openimages/dataset). 
Specifically, 
* Part 1: we adopt the set [ImageNet-11k](http://data.mxnet.io/models/imagenet-11k/). It is a subset of ImageNet, collected by [MXNet](http://mxnet.incubator.apache.org/). It includes 1,474,703 images and 11,221 categories. However, we find there are many abstract categories in visual domain, such as "event", "summer", etc. We think that the training images annotated with such abstract categories 
will not help (even harm) the visual representation learning. Thus, we abandon these categories. 
* Part 2: We filter the URLs of Open Images via a per-class criteria. Firstly, if one class occurs in less than 650 URLs, then it is removed. Besides, we also 1) remove some abstract categories as did above, and 2) merge some redundant categories with those in ImageNet. Then, if all annotated tags of one URL are removed, then this URL is abandoned. Consequently,  6,902,811 training URLs and 159,424 validation URLs are remained, covering 1,134 unique categories.
--->
<!---
We then merge URLs from above two parts according to their categories. Specifically, we firstly all categories to their unique WordIDs defined in [WordNet](https://wordnet.princeton.edu/). According to the semantic topological structure of WordIDs, if two categories share the same WordID or are synonymous, then they are merged to a unique category, as well as their URLs. 
Finally, the number of remained URLs is 17,659,752, and the number of categories is 11,166. 
--->

<!---
Consequently,  8,385,050 training URLs and 159,424 validation URLs are remained, covering 2,039 categories.
--->


### [Semantic Hierarchy](#hierarchy)
[[back to top](#)]

We build the semantic hiearchy of 11,166 categories, according to [WordNet](https://wordnet.princeton.edu/). 
The direct parent categories of each class can be found from the file [data/dictionary_and_semantic_hierarchy.txt](data/dictionary_and_semantic_hierarchy.txt). The whole semantic hierarchy includes 4 independent trees, of which 
the root nodes are `thing`, `matter`, `object, physical object` and `atmospheric phenomenon`, respectively. 
The length of the longest semantic path from root to leaf nodes is 16, and the average length is 7.47. 

### [Annotations](#annotation)
[[back to top](#)]

Since the image URLs of ML-Images are collected from ImageNet and Open Images, the annotations of ML-Images are constructed based on the 
original annotations from ImageNet and Open Images. Note that the original annotations from Open Images are licensed by Google Inc. under [CC BY-4.0](https://creativecommons.org/licenses/by/4.0/). Specifically, we conduct the following steps to construct the new annotations of ML-Images. 
* For the 6,902,811 training URLs from Open Images, we remove the annotated tags that are out of the remained 1,134 categories.
* According to the constructed [semantic hierarchy](data/dictionary_and_semantic_hierarchy.txt) of 11,166 categories, we augment the annotations of all URLs of ML-Images following the cateria that if one URL is annotated with category i, then all ancestor categories will also be annotated to this URL. 
* We train a ResNet-101 model based on the 6,902,811 training URLs from Open Images, with 1,134 outputs. Using this ResNet-101 model, we predict the tags from 1,134 categories for the 10,756,941 single-annotated image URLs from ImageNet. Consequently, we obtain a normalized co-occurrence matrix between 10,032 categories from ImageNet and 1,134 categories from Open Images. We can determine the strongly co-occurrenced pairs of categories. For example, category i and j are strongly co-occurrenced; then, if one image is annotated with category i, then category j should also be annotated. 

The annotations of all training URLs in ML-Images are stored in [data/dictionary_and_semantic_hierarchy.txt](data/dictionary_and_semantic_hierarchy.txt).

<!---
（有关Open Images的部分，因为Annotations是适用CC BY-4.0（https://creativecommons.org/licenses/by/4.0/），所以如果有修改的话，是需要注明的。 License文件我会提，但您们rearme最好也要写，可以参考：https://wiki.creativecommons.org/wiki/Best_practices_for_attribution#This_is_a_good_attribution_for_material_you_modified_slightly
因为License文件我只提说：以下的文件可能已被修改，但readme里面就可能要写更详细点，像上面那个链接里的范例）
--->

### [Statistics](#statistics)
[[back to top](#)]

The main statistics of ML-Images are summarized in ML-Images.
                             
                                                      
| # Train images  | # Validation images  | # Classes | # Trainable Classes | # Avg tags per image |  # Avg images per class |
| :-------------: |:--------------------:| :--------:| :-----------------: |:-------------------:|  :---------------------:|
| 17,609,752       | 88,739          | 11,166    | 10,505              |  8    |  1447.2 |

Note: *Trainable class* indicates the class that has over 100 train images.

<br/>
 
The number of images per class and the histogram of the number of annotations in training set are shown in the following figures.       

<img  src="git_images/num_images_per_class.png" alt="GitHub" title="num images per class" width="430" height="240" />        <img  src="git_images/hist_num_annotations.png" alt="GitHub" title="histogram of num annotations" width="400" height="240" />


# [Train](#train)
[[back to top](#)]

### [Download Images using URLs](#download-image)
[[back to top](#)]

The full [train_url.txt](https://tencent-ml-images-1257811961.cos.ap-guangzhou.myqcloud.com/train_urls.txt) is very large. 
Here we provide a tiny file [train_urls_tiny.txt](data/train_urls_tiny.txt) to demonstrate the downloading procedure.

```
cd data
./download_im_from_url.py --url_list=train_urls_tiny.txt --im_list=train_im_tiny.txt --save_dir='images/'
```
A sub-folder `data/images` will be generated to save the downloaded jpeg images, as well as a file `train_im_tiny.txt` to save the image list and the corresponding annotations. 

### [Prepare the TFRecord File](#prepare-tfrecord)
[[back to top](#)]

Here we generate the tfrecords using the multithreading module. One should firstly split the file `train_im_tiny.txt` into multiple smaller files, and save them into the sub-folder `data/image_lists/`. 
```
cd data
./tfrecord.sh
```
Note that some URLs in [train_url.txt](https://tencent-ml-images-1257811961.cos.ap-guangzhou.myqcloud.com/train_urls.txt) have expired or may expire in future. If that, please provide us the missing URLs, we could provide the corresponding tfrecords.

### [Pretrain on ML-Images](#pretrain)
[[back to top](#)]

```
./example/train.sh
```
Note that here we only provide the training code in the single node single GPU framework, while our actual training on ML-Images is based on an internal distributed training framework (not released yet). One could modify the training code to the distributed framework following [distributed tensorFlow](https://www.tensorflow.org/deploy/distributed). 

### [Finetune on ImageNet](#finetune)
[[back to top](#)]

One should firstly download the ImageNet database, then prepare the tfrecord file using [tfrecord.sh](example/tfrecord.sh). 
Then, you can finetune the ResNet-101 model on ImageNet as follows, with the checkpoint pre-trained on ML-Images.
```
./example/finetune.sh
```

### [Checkpoints](#checkpoint)
[[back to top](#)]

* [ckpt-resnet101-mlimages](	https://tencent-ml-images-1257811961.cos.ap-guangzhou.myqcloud.com/ckpt-resnet101-mlimages.zip): ResNet-101 checkpoint pretrained on ML-Images
* [ckpt-resnet101-mlimages-imagenet](https://tencent-ml-images-1257811961.cos.ap-guangzhou.myqcloud.com/ckpt-resnet101-mlimages-imagenet.zip): ResNet-101 checkpoint pretrained on ML-Images and finetuned on ImageNet

Please download above two checkpoints and move them into the folder `checkpoints/`, if you want to extract features using them.

### [Feature extraction](#feature-extraction)
[[back to top](#)]

```
./example/extract_feature.sh
```


# [Results](#result)
[[back to top](#)]

The retults of different ResNet-101 checkpoints on the validation set of ImageNet are summarized in the following table. 


| Checkpoints | Train and finetune setting  | <sub> Top-1 acc<br>on Val 224 </sub> | <sub> Top-5 acc<br>on Val 224 </sub> | <sub> Top-1 acc<br>on Val 299 </sub> | <sub> Top-5 accuracy<br>on Val 299 </sub> |
 :------------- |:--------------------| :--------:| :-----------------: |:------------------:| :-------------------:| 
 <sub> [MSRA ResNet-101](https://github.com/KaimingHe/deep-residual-networks) </sub> |  <sub> train on ImageNet </sub> | 76.4    |  92.9              |   --       |   --  | 
 <sub> [Google ResNet-101  ckpt1](https://arxiv.org/abs/1707.02968) </sub> |  <sub> train on ImageNet, 299 x 299 </sub> |  --  |  --  | 77.5  | 93.9 |
 <sub> Our ResNet-101 ckpt1 </sub> |  <sub> train on ImageNet </sub> | 77.8 | 93.9 | 79.0 | 94.5 |
 <sub> [Google ResNet-101  ckpt2](https://arxiv.org/abs/1707.02968) </sub> | <sub> Pretrain on JFT-300M, finetune on ImageNet, 299 x 299 </sub> |  --  |  --  | 79.2  | 94.7 |
 <sub> Our ResNet-101 ckpt2 </sub> | <sub> Pretrain on ML-Images, finetune on ImageNet </sub> | **78.8** | **94.5** | 79.5 | 94.9 |
 <sub> Our ResNet-101 ckpt3 </sub> | <sub> Pretrain on ML-Images, finetune on ImageNet 224 to 299 </sub> | 78.3 | 94.2 | **80.73** | **95.5** | 
 <sub> Our ResNet-101 ckpt4 </sub> | <sub> Pretrain on ML-Images, finetune on ImageNet 299 x 299 </sub> | 75.8 | 92.7 | 79.6 | 94.6 | 

Note: 
* if not specified, the image size in training/finetuning is 224 x 224. 
* *finetune on ImageNet from 224 to 299* means that the image size in early epochs of finetuning is 224 x 224, then 299 x 299 in late epochs.
* *Top-1 acc on Val 224* indicates the top-1 accuracy on 224 x 224 validation images.




# [Copyright](#copyright)
[[back to top](#)]

The annotations of images are licensed by Tencent under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license. 
The contents of this repository, including the codes, documents and checkpoints, are released under an [BSD 3-Clause](https://opensource.org/licenses/BSD-3-Clause) license. Please refer to [LICENSE](LICENSE.txt) for more details.

If there is any concern about the copyright of any image used in this project, please [email us](mailto:wubaoyuan1987@gmail.com).

# [Citation](#citation)
[[back to top](#)]

The arxiv paper describling the details of this project will be available soon!

