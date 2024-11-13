from torch import nn
import torch

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding_size):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding_size)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
    
    def forward(self, x):
        return self.act(self.bn1(self.conv1(x)))
    
class ConvBlock_T(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding_size):
        super(ConvBlock_T, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding_size)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
    
    def forward(self, x):
        return self.act(self.bn1(self.conv1(x)))


class Encoder(nn.Module):
    def __init__(self, in_channels, base_dim, img_h, img_w, latent_dim=20):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.block_1 = ConvBlock(in_channels=in_channels, out_channels=base_dim, kernel_size=3, stride=1, padding_size=1)
        self.down_1 = ConvBlock(in_channels=base_dim, out_channels=base_dim, kernel_size=4, stride=2, padding_size=1)

        self.block_2 = ConvBlock(in_channels=base_dim , out_channels=base_dim * 2, kernel_size=3, stride=1, padding_size=1)
        self.down_2 = ConvBlock(in_channels=base_dim * 2, out_channels=base_dim * 2, kernel_size=4, stride=2, padding_size=1)

        self.block_3 = ConvBlock(in_channels=base_dim * 2 , out_channels=base_dim * 4, kernel_size=3, stride=1, padding_size=1)
        self.down_3 = ConvBlock(in_channels=base_dim * 4, out_channels=base_dim * 4, kernel_size=4, stride=2, padding_size=1)

        self.feat_dim = (img_h // 8) * (img_w // 8) * base_dim * 4

        self.mu_var = nn.Linear(self.feat_dim, latent_dim * 2)
    
    def forward(self, x):
        f_1 = self.down_1(self.block_1(x))
        f_2 = self.down_2(self.block_2(f_1))
        f_3 = self.down_3(self.block_3(f_2))
        mu_var = self.mu_var(f_3.view(-1, self.feat_dim))
        return mu_var
    

class Decoder(nn.Module):
    def __init__(self, latent_dim, out_channels, base_dim, img_h, img_w):
        super(Decoder, self).__init__()
        self.img_h, self.img_w = img_h, img_w
        self.base_dim = base_dim
        self.project_z = nn.Linear(latent_dim, base_dim)

        self.block_1 = ConvBlock(base_dim, base_dim//2, kernel_size=3, stride=1, padding_size=1)
        self.up_1 = ConvBlock_T(base_dim // 2, base_dim//2, kernel_size=2, stride=2, padding_size=0)

        self.block_2 = ConvBlock(base_dim // 2, base_dim // 4, kernel_size=3, stride=1, padding_size=1)
        self.up_2 = ConvBlock_T(base_dim // 4, base_dim // 4, kernel_size=2, stride=2, padding_size=0)

        self.block_3 = ConvBlock(base_dim // 4, base_dim // 8, kernel_size=3, stride=1, padding_size=1)
        self.up_3 = ConvBlock_T(base_dim // 8, base_dim // 8, kernel_size=2, stride=2, padding_size=0)

        self.final_mu = nn.Conv2d(base_dim // 8, out_channels, kernel_size=3, padding=1)
        self.final_log_var = nn.Conv2d(base_dim // 8, out_channels, kernel_size=3, padding=1)
    
    def forward(self, z):
        proj_z = self.project_z(z)
        out_1 = self.block_1(proj_z.view(-1, self.base_dim, self.img_h//8, self.img_w//8))
        out_1 = self.up_1(out_1)

        out_2 = self.block_2(out_1)
        out_2 = self.up_2(out_2)

        out_3 = self.block_3(out_2)
        out_3 = self.up_3(out_3)        
        return self.final_mu(out_3), self.final_log_var(out_3)


class VaDE(nn.Module):
    def __init__(self, in_channels, out_channels, enc_base_dim, 
                 dec_base_dim, img_h, img_w, latent_dim=20):
        
        super(VaDE, self).__init__()
        self.encoder = Encoder(in_channels, enc_base_dim, img_h, img_w, latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim, out_channels=out_channels, base_dim=dec_base_dim, img_h=img_h, img_w=img_w)
        self.latent_dim = latent_dim

    def pretraining(self):

        return
    
    def gmm_initialize(self, n_clusters):
        cat_pis = [1 / n_clusters for _ in range(n_clusters)]
        cov_mats = [torch.eye(n_clusters) for _ in range(n_clusters)]

        return
    
    def forward(self, x):
        mu_var = self.encoder(x)
        mu, log_var = mu_var[:, :self.encoder.latent_dim], mu_var[:, self.encoder.latent_dim:]

        z = self.reparametrization(mu, log_var)
        recon_mu, recon_log_var = self.decoder(z)
        return recon_mu, recon_log_var, mu, log_var
    
    def elbo_loss(self, x, mu, log_var, recon_mu, recon_log_var, n_montecarlo=128):
        """
        p(x|z) = N(x; \mu_{x}, I) -> For practical purpose, assume the all diagonal covariance factors as 1.
        """
        kl_div = 0.5 * (torch.mat(mu[:, :, None].T, mu[:, :, None]) + torch.exp(log_var).sum(dim=1) - torch.sum(log_var, dim=1) - self.encoder.latent_dim)
        
        recon_loss = 0.5 * ((recon_mu - x) ** 2)
        
        return - kl_div.mean() + recon_loss.mean()

    @torch.no_grad()
    def generate(self, num_samples):
        z = torch.randn(num_samples, self.encoder.latent_dim)
        recon_mu, recon_log_var = self.decoder(z)

        return recon_mu
    
    def reparametrization(self, mu, log_var):
        eps = torch.randn_like(log_var)
        z = mu + torch.mul(torch.exp(0.5 * log_var), eps)
        return z