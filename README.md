# Neural Network Diffusion
![Motivation of p-diff](figs/motivation_v3.gif)

This repository contains the code and implementation details for the research paper titled "Neural Network Diffusion." The paper explores novel paradigms in deep learning, specifically focusing on diffusion models for generating high-performing neural network parameters.


## Authors

- [Kai Wang*](https://kaiwang960112.github.io/)<sup>1</sup>, [Zhaopan Xu](https://scholar.google.com.hk/citations?user=qNWDwOcAAAAJ&hl=zh-CN)<sup>1</sup>, [Zhuang Liu](https://liuzhuang13.github.io/)<sup>2</sup>, Yukun Zhou, [Zelin Zang](https://scholar.google.com/citations?user=foERjnQAAAAJ&hl=zh-CN)<sup>1</sup>, [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/)<sup>3</sup>, and [Yang You](https://www.comp.nus.edu.sg/~youy/)<sup>1</sup>
- <sup>1</sup>[National University of Singapore](https://www.nus.edu.sg/), <sup>2</sup>[Meta AI](https://www.meta.com/), and <sup>3</sup>[University of California, Berkeley](https://www.berkeley.edu/)


## Overview

<p align="center">
<img src="./figs/pipeline.png" width=100% height=55.2%
class="center">
  <figcaption>Figure: Illustration of the proposed p-diff framework. Our approach consists of two processes, namely parameter autoencoder and generation. 
  Parameter autoencoder aims to extract the latent representations that can generate high-performing model parameters via the decoder. The extracted representations are fed into a standard latent diffusion model (LDM). During the inference, we freeze the parameters of the autoencoder's decoder. The generated parameters are obtained via feeding random noise to the LDM and trained decoder.</figcaption>
</p>

> **Abstract:** Diffusion models have achieved remarkable success in image and video generation. In this work, we demonstrate that diffusion models can also generate high-performing neural network parameters. Our approach is simple, utilizing an autoencoder and a standard latent diffusion model. The autoencoder extracts latent representations of the trained network parameters. A diffusion model is then trained to synthesize these latent parameter representations from random noise. It then generates new representations that are passed through the autoencoder's decoder, whose outputs are ready to use as new sets of network parameters. Across various architectures and datasets, our diffusion process consistently generates models of comparable or improved performance over SGD-trained models, with minimal additional cost. Our results encourage more exploration on the versatile use of diffusion models. 



## Getting Started

[Provide instructions on how to use or replicate the experiments mentioned in the paper.]

## Citation

@article{wang2024neural,
      title={Neural Network Diffusion}, 
      author={Kai Wang, Zhaopan Xu, Zhuang Liu, Yukun Zhou, Zelin Zang, Trevor Darrell and Yang You},
      year={2024},
      eprint={2401.xxxx},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

## Acknowledgments

We thank Kaiming He, Dianbo Liu, Mingjia Shi, Zheng Zhu, Jiawei Liu, Yong Liu, Ziheng Qin, Zangwei Zheng, Yifan Zhang, Xiangyu Peng, Junhao Zhang, Wangbo Zhao, Hongyan Chang, and David Yin for valuable discussions and feedbacks.
