# DDPM-DL


For this project we need to implement the DDPM paper (https://arxiv.org/pdf/2006.11239). 



Steps necessary for implementation
- Dataloading
    - Datasplits
- UNet architecture // This has been provided
    - possibly extend architecture to increase performance on metrics and training
- Denoising Diffusion model
    - Forward diffusion
    - Reverse diffuision
- Loss function
    - KL term
    - ELBO
- training loop
- Evaluation and Metrics
    - MSE
    - RMSE
    - FID 
- Visualization
- Extra experiments?