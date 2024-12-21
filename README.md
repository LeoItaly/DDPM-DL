# DDPM-DL
This project contains the code for our project on implementing the paper Denoising Diffusion Probabilistic Models (DDPM). Trained on the MNIST dataset. 



To see the results look at the notebooks in Results:
- [Standard DDPM](Results/Standard_DDPM.ipynb)
  - An implementation of the standard DDPM model.
- [Scheduler and Timestep Experiments](Results/DDPM_Schedulers_n_Timesteps.ipynb)
  - An ablation study on how different schedulers and amounts of timesteps affect performance.
- [Classifier Free Guidance](Results/DDPM_CFG.ipynb)
  - Implementation of Classifier Free Guidance for DDPM models. Making it possible to conditionally generate specific digits of MNIST.

For a detailed discussion of our results and methodology, check out our [DDPM_Report.pdf](report_and_poster/DDPM_poster.pdf).













# References
- Denoising Diffusion Probabilistic Models: https://arxiv.org/abs/2006.11239
- Classifier Free Guidance paper: https://arxiv.org/abs/2207.12598
