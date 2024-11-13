from torch import nn
import torch
from torch.nn import functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
from sklearn.cluster import MiniBatchKMeans

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
    def __init__(self, in_channels, base_dim, img_h, img_w, n_cluster=16, latent_dim=20):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.block_1 = ConvBlock(in_channels=in_channels, out_channels=base_dim, kernel_size=3, stride=1, padding_size=1)
        self.down_1 = ConvBlock(in_channels=base_dim, out_channels=base_dim, kernel_size=4, stride=2, padding_size=1)

        self.block_2 = ConvBlock(in_channels=base_dim , out_channels=base_dim * 2, kernel_size=3, stride=1, padding_size=1)
        self.down_2 = ConvBlock(in_channels=base_dim * 2, out_channels=base_dim * 2, kernel_size=4, stride=2, padding_size=1)

        self.block_3 = ConvBlock(in_channels=base_dim * 2 , out_channels=base_dim * 4, kernel_size=3, stride=1, padding_size=1)
        self.down_3 = ConvBlock(in_channels=base_dim * 4, out_channels=base_dim * 4, kernel_size=4, stride=2, padding_size=1)

        self.feat_dim = (img_h // 8) * (img_w // 8) * base_dim * 4

        self.mu_var = nn.Linear(self.feat_dim, n_cluster * latent_dim * 2)
        self.n_cluster = n_cluster
        self.latent_dim = latent_dim

    def forward(self, x):
        f_1 = self.down_1(self.block_1(x))
        f_2 = self.down_2(self.block_2(f_1))
        f_3 = self.down_3(self.block_3(f_2))
        mu_var = self.mu_var(f_3.view(-1, self.feat_dim)).view(-1, self.n_cluster, self.latent_dim * 2)
        return mu_var
    

class Decoder(nn.Module):
    def __init__(self, latent_dim, out_channels, base_dim, img_h, img_w, n_cluster):
        super(Decoder, self).__init__()
        self.img_h, self.img_w = img_h, img_w
        self.base_dim = base_dim
        self.latent_dim = latent_dim
        self.n_cluster = n_cluster
        self.project_z = nn.Linear(latent_dim * n_cluster, base_dim * self.img_h//8 * self.img_w//8)

        self.block_1 = ConvBlock(base_dim, base_dim//2, kernel_size=3, stride=1, padding_size=1)
        self.up_1 = ConvBlock_T(base_dim // 2, base_dim//2, kernel_size=2, stride=2, padding_size=0)

        self.block_2 = ConvBlock(base_dim // 2, base_dim // 4, kernel_size=3, stride=1, padding_size=1)
        self.up_2 = ConvBlock_T(base_dim // 4, base_dim // 4, kernel_size=2, stride=2, padding_size=0)

        self.block_3 = ConvBlock(base_dim // 4, base_dim // 8, kernel_size=3, stride=1, padding_size=1)
        self.up_3 = ConvBlock_T(base_dim // 8, base_dim // 8, kernel_size=2, stride=2, padding_size=0)

        self.final_mu = nn.Conv2d(base_dim // 8, out_channels, kernel_size=3, padding=1)
        self.final_log_var = nn.Conv2d(base_dim // 8, out_channels, kernel_size=3, padding=1)
    
    def forward(self, z):
        # z : (bsz, n_cluster, latent_dim)
        proj_z = self.project_z(z.view(-1, self.n_cluster * self.latent_dim))
        # proj_z : (bsz, base_dim * self.img_h//8 * self.img_w//8)
        out_1 = self.block_1(proj_z.view(-1, self.base_dim, self.img_h//8, self.img_w//8))
        out_1 = self.up_1(out_1)

        out_2 = self.block_2(out_1)
        out_2 = self.up_2(out_2)

        out_3 = self.block_3(out_2)
        out_3 = self.up_3(out_3)        
        return self.final_mu(out_3), self.final_log_var(out_3)


class VaDE(nn.Module):
    def __init__(self, in_channels, out_channels, enc_base_dim, 
                 dec_base_dim, img_h, img_w, n_clusters, latent_dim=20):
        
        super(VaDE, self).__init__()
        self.encoder = Encoder(in_channels, enc_base_dim, img_h, img_w, latent_dim=latent_dim)
        self.pretrain_decoder = Decoder(latent_dim=latent_dim, out_channels=out_channels, 
                                        base_dim=dec_base_dim, img_h=img_h, img_w=img_w, n_cluster=n_clusters)
        
        self.decoder = Decoder(latent_dim=latent_dim, out_channels=out_channels, 
                               base_dim=dec_base_dim, img_h=img_h, img_w=img_w, n_cluster=n_clusters)
        self.latent_dim = latent_dim
        centroids, cov_mats, cat_pis = self.gmm_initialize(n_clusters, latent_dim)
    
    def load_pretrain_encoder_clusters(self, pretrain_path):
        assert pretrain_path is not None and pretrain_path != "", "Check the pretrain path"
        ckpt = torch.load(pretrain_path)
        self.encoder.load_state_dict(ckpt)
        print("Pretrained encoder weight is loaded")

    def pretrain(self, num_epochs: int, dataloader: DataLoader, pretrain_lr: float = 1e-2, 
                 pretrain_wd:float = 1e-4, device: str = "cpu", pretrain_ckpt_path: str=""):

        optimizer = torch.optim.AdamW([{'params': self.encoder.parameters()}, 
                                       {'params': self.pretrain_decoder.parameters()}], 
                                       lr=pretrain_lr, weight_decay=pretrain_wd)
        iter_cnt = 0
        last_epoch_embeddings = []

        for epoch in range(num_epochs):
            running_loss = 0.

            with tqdm(dataloader, unit="batch") as dloader:
                dloader.set_description(f"EPOCH {epoch +1} / {num_epochs}")
                for data in dloader:
                    data = data.to(device)
                    # Empirical mean and log variance
                    optimizer.zero_grad()
                    mu_var = self.encoder(data)
                    decode_mean, _ = self.pretrain_decoder(mu_var[:, :, :self.latent_dim])
                    loss = F.mse_loss(decode_mean, data)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                    iter_cnt += 1
                    dloader.set_postfix({'loss': loss.item(), 'avg loss': running_loss / iter_cnt})

                    if epoch == num_epochs - 1:
                        last_epoch_embeddings.append(mu_var[:, :, :self.latent_dim])

        print("Pretraining is finished !")
        
        save_dir = os.path.dirname(pretrain_ckpt_path)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        """
        # Parameters
        n_clusters = 10          # Number of clusters (K)
        batch_size = 100         # Size of each mini-batch
        max_iter = 100           # Number of iterations

        # Initialize MiniBatchKMeans
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, max_iter=max_iter)
        kmeans.fit(large_dataset)
        centroids = kmeans.cluster_centers_
        """
        torch.save(self.encoder.state_dict(), pretrain_ckpt_path)

    def gmm_initialize(self, n_clusters, latent_dim):
        cat_pis = torch.ones(n_clusters) / n_clusters
        cov_mats = [torch.eye(latent_dim) for _ in range(n_clusters)]
        centroids = torch.empty((n_clusters, latent_dim))
        return centroids, torch.stack(cov_mats), cat_pis
    
    def forward(self, x):
        mu_var = self.encoder(x)
        
        mu, log_var = mu_var[:, :, :self.encoder.latent_dim], mu_var[:, :, self.encoder.latent_dim:]

        zs = []

        for cluster_id in range(self.n_cluster):
            zi = self.reparametrization(mu[:, cluster_id, :], log_var[:, cluster_id, :])
            zs.append(zi)

        # zs : (bsz, n_clusters, n_latent_dim)

        recon_mu, recon_log_var = self.decoder(torch.stack(zs, dim=1))
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
    

if __name__ == "__main__":
    n_cluster = 10
    bsz = 12
    latent_dim = 24
    zs = []
    for i in tqdm(range(n_cluster)):
        zs.append(torch.randn(bsz, latent_dim))
    
    zs = torch.stack(zs, dim=1)
    print(zs.shape)
    