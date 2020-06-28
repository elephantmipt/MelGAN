# Paper Overview

## Main idea

The key idea of this paper is to improve speed of generating waveform from spectrogram. Authors use GAN with CNN-only architecture which is optimized for GPU. Both generator and discriminator is very lightweight in comparison to previous SOTA approaches like wave net.

## Features were used

#### Architecture

In generator we reduce dimensionality layer by layer and also to prevent gradient vanish we use residual stack. In discriminator we instead increasing dimensionality layer by layer. Also we features from every layer of discriminator. Also we should use 3 discriminators instead of 1. For every next discriminator we downsample input by 2 with average pooling.

#### Weight normalization

In my first experiments I didn't add weight normalization, it leads to instability in losses also generated results wasn't good. So, for every convolutional \(all layers are convolutional\) we apply weight normalization.

#### Loss functions

The basic loss is taken from [LS GAN paper](https://arxiv.org/abs/1611.04076). The main difference between vanilla GAN loss is that we don't use sigmoid function for output. The loss function is:

$$
\begin{array}{l}\min _{D_{k}} \mathbb{E}_{x}\left[\min \left(0,1-D_{k}(x)\right)\right]+\mathbb{E}_{s, z}\left[\min \left(0,1+D_{k}(G(s, z))\right)\right], \forall k=1,2,3 \\ \min _{G} \mathbb{E}_{s, z}\left[\sum_{k=1,2,3}-D_{k}(G(s, z))\right]\end{array}
$$

Also, to improve generator convergence we also use L1 distance for features from discriminator between real and generated audio.

$$
\mathcal{L}_{\mathrm{FM}}\left(G, D_{k}\right)=\mathbb{E}_{x, s \sim p_{\text {data }}}\left[\sum_{i=1}^{T} \frac{1}{N_{i}}\left\|D_{k}^{(i)}(x)-D_{k}^{(i)}(G(s))\right\|_{1}\right]
$$

$$
\lambda = \dfrac{10}{N_D}
$$

