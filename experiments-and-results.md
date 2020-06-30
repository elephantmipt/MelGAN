# Experiments & Results

All logs can be found [here](https://app.wandb.ai/torchwave/MelGAN).

### Experiments

Mainly all I tried is to remove some residual stacks and use one instead of three stacks in downsampling layer of generator. This leads to faster training but affects on quality, especially when there is a phrase instead of one separate word in text. Here is an examples of this approach \(there is underfiting but main idea is clear\).

#### Generated audio

{% file src=".gitbook/assets/reconstructed\_51.wav\_51\_8e773bfd.wav" caption="50 epochs" %}

{% file src=".gitbook/assets/reconstructed\_101.wav\_101\_21edde86.wav" caption="100 epochs" %}

{% file src=".gitbook/assets/reconstructed\_201.wav\_201\_f9117afa.wav" caption="200 epochs" %}

{% file src=".gitbook/assets/reconstructed\_400.wav\_400\_fe641c3d \(1\).wav" caption="400 epochs" %}

{% file src=".gitbook/assets/lj001-0006.wav" caption="Original" %}

Well, that's doesn't work.

So I follow a paper and add three residual stacks, as it was proposed. Next interesting thing, is that learning rate seriously affects on result. With big learning rate \( $$0.001$$ \) generator collapse and produces inappropriate quality audio with very low frequency artifacts.

![Beautiful but useless](.gitbook/assets/snimok-ekrana-2020-06-30-v-10.30.34.png)

If I use learning rate $$0.0001$$ situation becomes different. Now discriminator nearly collapse. But sound quality isn't so bad. Well there is metallic sound but words now recognizable.

{% file src=".gitbook/assets/generated\_51.wav\_51\_0cebc4f0.wav" caption="50 epochs" %}

{% file src=".gitbook/assets/generated\_101.wav\_101\_2e340efb.wav" caption="100 epochs" %}

{% file src=".gitbook/assets/generated\_201.wav\_201\_2ffda111.wav" caption="200 epochs" %}

{% file src=".gitbook/assets/lj001-0006 \(1\).wav" caption="Original" %}

The problem I noticed is that there is no gradients in some layers there is no gradients at all in first epochs. Maybe because discriminator can't handle generator at first.

![](.gitbook/assets/snimok-ekrana-2020-06-30-v-10.39.15.png)

 I think maybe I should do several discriminator steps in one training step to prevent discriminator from failing.

