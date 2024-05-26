# Neural Network Parameter Diffusion

### [Paper](https://arxiv.org/pdf/2402.13144.pdf) | [Project Page](https://1zeryu.github.io/Neural-Network-Diffusion/) | [Hugging Face](https://huggingface.co/papers/2402.13144)

![Motivation of p-diff](figs/motivation_v3.gif)

This repository contains the code and implementation details for the research paper titled "Neural Network Diffusion." The paper explores novel paradigms in deep learning, specifically focusing on diffusion models for generating high-performing neural network parameters.


## Authors

- [Kai Wang](https://kaiwang960112.github.io/)<sup>1</sup>, [Zhaopan Xu](https://scholar.google.com.hk/citations?user=qNWDwOcAAAAJ&hl=zh-CN)<sup>1</sup>,  Yukun Zhou, [Zelin Zang](https://scholar.google.com/citations?user=foERjnQAAAAJ&hl=zh-CN)<sup>1</sup>, [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/)<sup>3</sup>, [Zhuang Liu*](https://liuzhuang13.github.io/)<sup>2</sup>, and [Yang You*](https://www.comp.nus.edu.sg/~youy/)<sup>1</sup>(* equal advising)
- <sup>1</sup>[National University of Singapore](https://www.nus.edu.sg/), <sup>2</sup>[Meta AI](https://www.meta.com/), and <sup>3</sup>[University of California, Berkeley](https://www.berkeley.edu/)


## Overview

<p align="center">
<img src="./figs/pipeline.png" width=100% height=55.2%
class="center">
  <figcaption>Figure: Illustration of the proposed p-diff framework. Our approach consists of two processes, namely parameter autoencoder and generation. 
  Parameter autoencoder aims to extract the latent representations that can generate high-performing model parameters via the decoder. The extracted representations are fed into a standard latent diffusion model (LDM). During the inference, we freeze the parameters of the autoencoder's decoder. The generated parameters are obtained via feeding random noise to the LDM and trained decoder.</figcaption>
</p>

> **Abstract:** Diffusion models have achieved remarkable success in image and video generation. In this work, we demonstrate that diffusion models can also generate high-performing neural network parameters. Our approach is simple, utilizing an autoencoder and a standard latent diffusion model. The autoencoder extracts latent representations of the trained network parameters. A diffusion model is then trained to synthesize these latent parameter representations from random noise. It then generates new representations that are passed through the autoencoder's decoder, whose outputs are ready to use as new sets of network parameters. Across various architectures and datasets, our diffusion process consistently generates models of comparable or improved performance over SGD-trained models, with minimal additional cost. Our results encourage more exploration on the versatile use of diffusion models. 

## Installation

1. Clone the repository:

```
git clone https://github.com/NUS-HPC-AI-Lab/Neural-Network-Diffusion.git
```

2. Create a new Conda environment and activate it: 

```
conda env create -f environment.yml
conda activate pdiff
```

or install necessary package by:

```
pip install -r requirements.txt
```

### **Baseline**

For CIFAR100 Resnet18 parameter generation, you can run the script:

```
bash ./cifar100_resnet18_k200.sh
```

The script is run in two steps, one to obtain the relevant parameters for the task, the second to train and test the generative model of the parameters.

### **Ablation**

We use  [Hydra](https://hydra.cc/docs/intro/) package for our configuration. You can modify the configuration by modifying the config file as well as command line. 

For example, for CIFAR10 Resnet18, you can change the config file  [configs/task/cifar100.yaml](https://github.com/NUS-HPC-AI-Lab/Neural-Network-Diffusion/blob/main/configs/task/cifar100.yaml).

```
data:
  root: cifar100 path
  dataset: cifar100
  batch_size: 64
  num_workers: 1

```

to:

```
data:
  root: cifar10 path
  dataset: cifar10
  batch_size: 64
  num_workers: 1
```

or you can change the Neural Network training data (K) by command line:

```
python train_p_diff.py task.param.k=xxx
```



## Citation
If you found our work useful, please consider citing us.

```
@misc{wang2024neural,
      title={Neural Network Diffusion}, 
      author={Kai Wang and Zhaopan Xu and Yukun Zhou and Zelin Zang and Trevor Darrell and Zhuang Liu and Yang You},
      year={2024},
      eprint={2402.13144},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
## Acknowledgments

We thank [Kaiming He](https://kaiminghe.github.io/), [Dianbo Liu](https://www.cogai4sci.com/), [Mingjia Shi](https://scholar.google.com/citations?user=B6f3ImkAAAAJ&hl=en), [Zheng Zhu](https://scholar.google.com/citations?user=NmwjI0AAAAAJ&hl=en), [Bo Zhao](https://www.bozhao.me/), [Jiawei Liu](https://jia-wei-liu.github.io/), [Yong Liu](https://ai.comp.nus.edu.sg/people/yong/), [Ziheng Qin](https://scholar.google.com/citations?user=I04VhPMAAAAJ&hl=zh-CN), [Zangwei Zheng](https://zhengzangw.github.io/), [Yifan Zhang](https://sites.google.com/view/yifan-zhang/%E9%A6%96%E9%A1%B5), [Xiangyu Peng](https://scholar.google.com/citations?user=KRUTk7sAAAAJ&hl=zh-CN), [Hongyan Chang](https://www.comp.nus.edu.sg/~hongyan/), [Zirui Zhu](https://zirui-zhu.com/), [David Yin](https://davidyyd.github.io/), [Dave Zhenyu Chen](https://daveredrum.github.io/), [Ahmad Sajedi](https://ahmadsajedii.github.io/) and [George Cazenavette](https://georgecazenavette.github.io/) for valuable discussions and feedbacks.
