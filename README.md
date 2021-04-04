# LaTextGAN for generating novel tweets in the style of Donald J. Trump 
  
For this project we attempted to generate novel tweets in the style of Donald J. Trump  using the LaTextGAN approach from [David Donahue and Anna Rumshisky 2019](https://arxiv.org/pdf/1810.06640.pdf).

![Trump Face](https://cdn.talkingpointsmemo.com/wp-content/uploads/2019/06/trump-hiss.jpg)
  
---

## Description
The idea behind the LaTextGAN is to generate new sentences from a continuous low-dimensional space instead of working with discrete text directly. 
  
Donahue and Rumshisky achieved this by first training an AE on a given dataset such that the encoder is able to produce low-dimensional sentence embeddings from it. The task of the generator is then to create novel samples from this learned latent space that can be decoded back into sentences by the AEs decoder. During training the discriminator has to correctly classify sentence embeddings coming from both the encoder (real samples) and the generator (fake samples). The figure below shows the schematic of the original architecture. Since the AE is dealing with sequential data LSTM networks are used to read in (encoder) and reconstruct (decoder) the sentences.  
  
The GAN has been implemented as an improved variant of the original Wasserstein GAN as proposed by [Gulrajani et al. (2017)](https://arxiv.org/pdf/1704.00028.pdf). The improved version directly penalizes the norm of the discriminator’s gradients to be at most 1 with respect to its input. This penalty replaces the weight clipping that was used in the original WGAN to enforce the Lipschitz continuity. 
  
---

## Architecture

![Original LaTextGAN Architecture](https://github.com/GerritBartels/LaTextGAN/blob/main/LaTextGAN_Schematic.jpg?raw=true)
  
---
  
## Showcase of generated tweets:
* its time , very proud of me . #makeamericagreatagain <End>
* great job mark : thank you ! <End>
* despite all horrible statements on me and clear that my administration has done a great job , but the media refuses to talk about them in the last <num> years . they have been saying that they couldnt get elected , and now the republican party is a joke ! <End>
* heading to washington , d . c . the white house has done an incredible job . we are doing a great job ! we are all proud of the people of israel . they love you and , we are going to win ! <End>
* will also fight against crazy charges ! <End>
* will be interviewed by @seanhannity . make america great again ! <End>
* ill be playing with president putin and , tiger tomorrow . real deal thanks <End>
* working hard to keep up today ! <End>
* not hurt ! #votetrump ! <End>
* @rodmonium <num> : your mother is good news for trump ? <End>
---
 
## Tutorial
If you want to run this project yourself you can either open our [Colab Notebook](https://github.com/GerritBartels/LaTextGAN/tree/main/Colab%20Notebook) and make a copy of it or download it onto your own machine. The provided environment.yml contains contains all the necessary dependencies to run this project locally.
