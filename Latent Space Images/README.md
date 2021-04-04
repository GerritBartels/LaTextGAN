# Latent Space Analysis
This folder contains the Latent Space Embeddings from all variants we used. In each plot both Encoder (blue) and Generator (orange) provided 500 sequence vectors. 
To reduce the dimensions of the sequence vectors, we first used PCA to project them to 50 dimensions and then applied t-SNE for a further projection down onto two dimensions. 

---

![Standard LaTextGAN, All Words](https://github.com/GerritBartels/LaTextGAN/blob/main/Latent%20Space%20Images/Standard_LaTextGAN_All_Words.png?raw=true)

---

![Standard LaTextGAN, Rare Words Removed](https://github.com/GerritBartels/LaTextGAN/blob/main/Latent%20Space%20Images/Standard_LaTextGAN_Remove_Rare_Words.png?raw=true)

---

![Bidirectional LaTextGAN, Rare Words Removed](https://github.com/GerritBartels/LaTextGAN/blob/main/Latent%20Space%20Images/Bidirectional_LaTextGAN_Remove_Rare_Words.png?raw=true)

---

![Stacked LaTextGAN, Rare Words Removed](https://github.com/GerritBartels/LaTextGAN/blob/main/Latent%20Space%20Images/Stacked_LaTextGAN_Remove_Rare_Words.png?raw=true)

---

![Bidirectional Stacked LaTextGAN, Rare Words Removed](https://github.com/GerritBartels/LaTextGAN/blob/main/Latent%20Space%20Images/Bidirectional_Stacked_LaTextGAN_Remove_Rare_Words.png?raw=true)
