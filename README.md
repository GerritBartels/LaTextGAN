# LaTextGAN for generating novel tweets in the style of Donald J. Trump 
For this project we attempted to generate novel tweets in the style of Donald J. Trump  using the LaTextGAN approach from [David Donahue and Anna Rumshisky 2019](https://arxiv.org/pdf/1810.06640.pdf).

![Trump Face](https://cdn.talkingpointsmemo.com/wp-content/uploads/2019/06/trump-hiss.jpg)


The idea behind the LaTextGAN is to generate new sentences from a continuous low-dimensional space instead of working with discrete text directly. 
  
Donahue and Rumshisky achieved this by first training an AE on a given dataset such that the encoder is able to produce low-dimensional sentence embeddings from it. The task of the generator is then to create novel samples from this learned latent space that can be decoded back into sentences by the AEs decoder. During training the discriminator has to correctly classify sentence embeddings coming from both the encoder (real samples) and the generator (fake samples). The figure below shows the schematic of the original architecture. Since the AE is dealing with sequential data LSTM networks are used to read in (encoder) and reconstruct (decoder) the sentences.  
  
The GAN has been implemented as an improved variant of the original Wasserstein GAN as proposed by [Gulrajani et al. (2017)](https://arxiv.org/pdf/1704.00028.pdf). The improved version directly penalizes the norm of the discriminator’s gradients to be at most 1 with respect to its input. This penalty replaces the weight clipping that was used in the original WGAN to enforce the Lipschitz continuity. 


---

![Original LaTextGAN Architecture](https://github.com/GerritBartels/LaTextGAN/blob/main/LaTextGAN_Schematic.jpg?raw=true)
