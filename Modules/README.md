# Modules
This folder contains all the custom modules that are necessary for running our project in the [Colab Notebook](https://github.com/GerritBartels/LaTextGAN/tree/main/Colab%20Notebook).
<br><br>
* **ae.py:** Non-stacked/standard Autoencoder with option to add bidirectionality
* **ae_training.py:** Training loop for the standard Autoencoder <br><br>
* **standard_latextgan.py:** Standard LaTextGAN that receives the real sentence vectors from the standard Autoencoder
* **standard_latextgan_training.py:** Train loop for the standard LaTextGAN
---

* **bidirectional_sae.py:** Stacked Autoencoder with option to add bidirectionality
* **bidirectional_sae_training.py:** Training loop for the stacked Autoencoder <br><br>
* **bidirectional_stacked_latextgan.py:** Stacked LaTextGAN that receives the real sentence vectors from the stacked Autoencoder
* **bidirectional_stacked_latextgan_training.py:** Train loop for the stacked LaTextGAN
 
 ---
  
* **standard_latextgan_evaluation.py:** Evaluation module for the standard LaTextGAN (Tweet Generator, Latent Space Analysis and Bleu-4 Score)
* **bidirectional_stacked_latextgan_evaluation.py:**  Evaluation module for the stacked LaTextGAN (Tweet Generator, Latent Space Analysis and Bleu-4 Scor
