# LaTextGAN
Generating fake Trump tweets using a LaTextGAN based on the approach from [David Donahue and Anna Rumshisky 2019](https://arxiv.org/pdf/1810.06640.pdf).

The idea behind the LaTextGAN is to generate new sentences from a continuous low-dimensional space  instead of working with discrete text directly. Donahue and Rumshisky achieved this by first training an AE on a given dataset such that the encoder is able to produce low-dimensional sentence embeddings from it. The task of the generator is now to create novel samples from this learned latent space that can be decoded back into sentences by the AEs decoder. During training the discriminator has to correctly classify sentence embeddings coming from both the encoder (real samples) and the generator (fake samples). Figure 1 shows the schematic of the original architecture. Since the AE is dealing with sequential data LSTM networks are used to read in (encoder) and reconstruct (decoder) the sentences.






![Original LaTextGAN Architecture](https://github.com/GerritBartels/LaTextGAN/blob/[branch]/image.jpg?raw=true)
