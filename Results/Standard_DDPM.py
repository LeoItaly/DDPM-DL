import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import math
import numpy as np

# For the optional classifier / FID stuff
import tensorflow as tf
import tensorflow_hub as tfhub
import tensorflow_gan as tfgan
from torchvision import datasets, transforms

##############################################################################
#                           CLASSES AND FUNCTIONS                            #
##############################################################################

class Linear_Variance_Scheduler:
    def __init__(self, time_steps, beta_start, beta_end, device='cuda'):
        self.time_steps = time_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device
        
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.time_steps).to(self.device)
        self.alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar)
        
    def diffusion_process(self, x, noise, t):
        # Ensure t is within valid range
        t = torch.clamp(t, max=self.time_steps - 1)
        sqrt_alpha_bar = self.sqrt_alpha_bar[t][:, None, None, None]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar[t][:, None, None, None]
        return sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise

    def ddpm_sampling(self, model, num_samples, channels, img_size):
        model.eval()
        collect = []
        with torch.inference_mode():
            x = torch.randn((num_samples, channels, img_size, img_size)).to(self.device)
            for i in tqdm(reversed(range(self.time_steps))):
                t = (torch.ones(num_samples) * i).long().to(self.device)
                pred_noise = model(x, t)

                alphas = self.alphas[t][:, None, None, None]
                alpha_bar = self.alpha_bar[t][:, None, None, None]
                betas = self.betas[t][:, None, None, None]

                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                x = 1 / torch.sqrt(alphas) * (
                    x - ((1 - alphas) / torch.sqrt(1 - alpha_bar)) * pred_noise
                ) + torch.sqrt(betas) * noise

                # Save intermediate samples if desired
                if (i + 1) % 100 == 0 or i == 0:
                    collect.append(x)
        return x, collect


class ResBlock(nn.Module):
    def __init__(self, inp_ch, out_ch, mid_ch=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_ch:
            mid_ch = out_ch
        
        self.resnet_conv = nn.Sequential(
            nn.Conv2d(in_channels=inp_ch, out_channels=mid_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=mid_ch),
            nn.SiLU(),
            nn.Conv2d(in_channels=mid_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_ch),
        )

    def forward(self, x):
        if self.residual:
            return x + self.resnet_conv(x)
        else:
            return self.resnet_conv(x)


class SelfAttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attn_norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=4, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        inp_attn = x.reshape(b, c, h*w)
        inp_attn = self.attn_norm(inp_attn)
        inp_attn = inp_attn.transpose(1, 2)  # (b, h*w, c)
        
        out_attn, _ = self.mha(inp_attn, inp_attn, inp_attn)
        out_attn = out_attn.transpose(1, 2).reshape(b, c, h, w)
        return x + out_attn


class DownBlock(nn.Module):
    def __init__(self, inp_ch, out_ch, t_emb_dim=256):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResBlock(inp_ch=inp_ch, out_ch=inp_ch, residual=True),
            ResBlock(inp_ch=inp_ch, out_ch=out_ch),
        )
        self.t_emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=t_emb_dim, out_features=out_ch),
        )

    def forward(self, x, t):
        x = self.down(x)
        t_emb = self.t_emb_layers(t)[:, :, None, None].repeat(1, 1, x.shape[2], x.shape[3])
        return x + t_emb


class UpBlock(nn.Module):
    def __init__(self, inp_ch, out_ch, t_emb_dim=256):
        super().__init__()
        self.upsamp = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up = nn.Sequential(
            ResBlock(inp_ch=inp_ch, out_ch=inp_ch, residual=True),
            ResBlock(inp_ch=inp_ch, out_ch=out_ch, mid_ch=inp_ch//2),
        )
        self.t_emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=t_emb_dim, out_features=out_ch),
        )

    def forward(self, x, skip, t):
        x = self.upsamp(x)
        x = torch.cat([skip, x], dim=1)
        x = self.up(x)
        t_emb = self.t_emb_layers(t)[:, :, None, None].repeat(1, 1, x.shape[2], x.shape[3])
        return x + t_emb


class UNet(nn.Module):
    def __init__(self, t_emb_dim=256, device='cuda'):
        super().__init__()
        self.device = device
        self.t_emb_dim = t_emb_dim

        self.inp = ResBlock(inp_ch=1, out_ch=64)
        self.down1 = DownBlock(inp_ch=64, out_ch=128)
        self.sa1 = SelfAttentionBlock(channels=128)
        self.down2 = DownBlock(inp_ch=128, out_ch=256)
        self.sa2 = SelfAttentionBlock(channels=256)
        self.down3 = DownBlock(inp_ch=256, out_ch=256)
        self.sa3 = SelfAttentionBlock(channels=256)

        self.lat1 = ResBlock(inp_ch=256, out_ch=512)
        self.lat2 = ResBlock(inp_ch=512, out_ch=512)
        self.lat3 = ResBlock(inp_ch=512, out_ch=256)

        self.up1 = UpBlock(inp_ch=512, out_ch=128)
        self.sa4 = SelfAttentionBlock(channels=128)
        self.up2 = UpBlock(inp_ch=256, out_ch=64)
        self.sa5 = SelfAttentionBlock(channels=64)
        self.up3 = UpBlock(inp_ch=128, out_ch=64)
        self.sa6 = SelfAttentionBlock(channels=64)

        self.out = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)

    def position_embeddings(self, t, channels):
        # t shape: (N,1)
        i = 1.0 / (10000 ** (torch.arange(start=0, end=channels, step=2, device=self.device) / channels))
        pos_emb_sin = torch.sin(t.repeat(1, channels//2) * i)
        pos_emb_cos = torch.cos(t.repeat(1, channels//2) * i)
        pos_emb = torch.cat([pos_emb_sin, pos_emb_cos], dim=-1)
        return pos_emb

    def forward(self, x, t):
        # t shape: (N,)
        t = t.unsqueeze(1).float()
        t = self.position_embeddings(t, self.t_emb_dim)

        x1 = self.inp(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.lat1(x4)
        x4 = self.lat2(x4)
        x4 = self.lat3(x4)

        x_ = self.up1(x4, x3, t)
        x_ = self.sa4(x_)
        x_ = self.up2(x_, x2, t)
        x_ = self.sa5(x_)
        x_ = self.up3(x_, x1, t)
        x_ = self.sa6(x_)
        output = self.out(x_)
        return output


##############################################################################
#            TRAINING FUNCTION (ENCAPSULATED) + PLOTTING LOSSES             #
##############################################################################

def train_ddpm(
    device,
    ddpm,
    model,
    criterion,
    optimizer,
    training_dataloader,
    test_dataloader,
    n_epochs=2
):
    """
    Full training routine for the DDPM model (UNet + Linear_Variance_Scheduler).
    Returns lists of training loss and test loss for each epoch.
    Also prints final losses per epoch.
    """

    # Set random seeds
    torch.manual_seed(1111)
    torch.cuda.manual_seed(1111)
    torch.cuda.manual_seed_all(1111)
    np.random.seed(1111)

    training_loss, test_loss = [], []

    for epoch in range(n_epochs):
        training_losses, test_losses = [], []

        # TRAIN loop
        for data, _ in tqdm(training_dataloader):
            model.train()
            data = data.to(device)
            t = torch.randint(low=0, high=1000, size=(data.shape[0],)).to(device)
            noise = torch.randn_like(data)
            xt = ddpm.diffusion_process(x=data, noise=noise, t=t)
            pred_noise = model(xt, t)

            trng_batch_loss = criterion(noise, pred_noise)
            optimizer.zero_grad()
            trng_batch_loss.backward()
            optimizer.step()
            training_losses.append(trng_batch_loss.item())
        training_per_epoch_loss = np.mean(training_losses)

        # TEST loop
        with torch.inference_mode():
            for data, _ in tqdm(test_dataloader):
                model.eval()
                data = data.to(device)
                t = torch.randint(low=0, high=1000, size=(data.shape[0],)).to(device)
                noise = torch.randn_like(data)
                xt = ddpm.diffusion_process(x=data, noise=noise, t=t)
                pred_noise = model(xt, t)
                tst_batch_loss = criterion(noise, pred_noise)
                test_losses.append(tst_batch_loss.item())
            test_per_epoch_loss = np.mean(test_losses)

        # Save epoch losses
        training_loss.append(training_per_epoch_loss)
        test_loss.append(test_per_epoch_loss)

        print(f'Epoch: {epoch+1}/{n_epochs}\t| Training loss: {training_per_epoch_loss:.4f} |   ', end='')
        print(f'Test loss: {test_per_epoch_loss:.4f}')

    return training_loss, test_loss


def classifier_fn(images):
    """
    Given a TF tensor of images, call the TF-Hub classifier
    for MNIST. Returns logits.
    """
    MNIST_MODULE = "https://tfhub.dev/tensorflow/tfgan/eval/mnist/logits/1"
    mnist_classifier_fn = tfhub.load(MNIST_MODULE)
    outputs = mnist_classifier_fn(images=images)
    return outputs

def compute_activations(images, num_batches, classifier_fn):
    """
    Splits images into num_batches and runs them through classifier_fn,
    then concatenates the results. Returns a TF tensor of all results.
    """
    images_list = tf.split(images, num_or_size_splits=num_batches)
    activations = []
    for batch in images_list:
        outputs = classifier_fn(images=batch)
        activations.append(outputs)
    activations = tf.concat(activations, axis=0)
    return activations

def load_mnist():
    """
    Loads real MNIST data from torchvision dataset in a single batch,
    resizes to 28x28, returns as a TF float32 tensor of shape (N, 28, 28, 1).
    """
    ds = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
        ]),
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=60000, shuffle=False)
    x, _ = next(iter(dl))
    x = x.numpy()
    # shape (N,1,28,28) => rearrange => (N,28,28,1)
    x = np.transpose(x, (0, 2, 3, 1))
    return tf.convert_to_tensor(x, dtype=tf.float32)

