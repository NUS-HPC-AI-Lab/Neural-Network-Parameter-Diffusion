# Neural Network Diffusion
![Motivation of p-diff](figs/motivation_v3.gif)

This repository contains the code and implementation details for the research paper titled "Neural Network Diffusion." The paper explores novel paradigms in deep learning, specifically focusing on the diffusion models for generating high-performing neural network parameters.



## Overview

<p align="center">
<img src="./figs/pipeline.png" width=100% height=55.2%
class="center">
  <p style="text-align: justify;">
  <figcaption>Figure: Illustration of the proposed p-diff framework. Our approach consists of two processes, namely parameter autoencoder and generation. 
  Parameter autoencoder aims to extract the latent representations that can generate high-performing model parameters via the decoder. The extracted representations are fed into a standard latent diffusion model (LDM). During the inference, we freeze the parameters of the autoencoder's decoder. The generated parameters are obtained via feeding random noise to the LDM and trained decoder.</figcaption>
</p>

</p>

 <p style="text-align: justify;">
 Figure: Illustration of the proposed p-diff framework. Our approach consists of two processes, namely parameter autoencoder and generation. 
  Parameter autoencoder aims to extract the latent representations that can generate high-performing model parameters via the decoder. The extracted representations are fed into a standard latent diffusion model (LDM). During the inference, we freeze the parameters of the autoencoder's decoder. The generated parameters are obtained via feeding random noise to the LDM and trained decoder.
</p>

## Authors

- [Kai Wang](https://kaiwang960112.github.io/)<sup>1</sup>, [Zhaopan Xu](https://scholar.google.com.hk/citations?user=qNWDwOcAAAAJ&hl=zh-CN)<sup>1</sup>, [Zhuang Liu](https://liuzhuang13.github.io/)<sup>2</sup>, Yukun Zhou, [Zelin Zang](https://scholar.google.com/citations?user=foERjnQAAAAJ&hl=zh-CN)<sup>1</sup>, [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/)<sup>3</sup>, and [Yang You](https://www.comp.nus.edu.sg/~youy/)<sup>1</sup>
- <sup>1</sup>[National University of Singapore](https://www.nus.edu.sg/), <sup>2</sup>[Meta AI](https://www.meta.com/), and <sup>3</sup>[University of California, Berkeley](https://www.berkeley.edu/)

## Abstract

[Include a brief summary or abstract of the paper here.]

## Getting Started

[Provide instructions on how to use or replicate the experiments mentioned in the paper.]

## Citation

If you find this work useful in your research, please consider citing:

[Author et al., Neural Network Diffusion, Journal/Conference, Year]

## Acknowledgments

We would like to express our gratitude to [names] for their valuable feedback during the development of this research.
