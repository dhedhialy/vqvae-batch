import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def code_batch_dependence(encoding_indices, batch_labels, n_codes=None):
    """Discrete-code batch independence (VQ-specific prior).

    The interpretable object is the CODE assignment. This term penalizes
    divergence between each batch's code histogram and the global histogram, so
    no batch has a distinctive code signature. Unlike z-level MMD/adversarial,
    it works on the discrete output that a biologist actually reads.
    """
    n_codes = n_codes or int(encoding_indices.max().item()) + 1
    global_counts = torch.bincount(encoding_indices, minlength=n_codes).float()
    global_p = global_counts / global_counts.sum().clamp(min=1)
    loss = torch.zeros(1, device=encoding_indices.device)
    for b in torch.unique(batch_labels):
        idx = encoding_indices[batch_labels == b]
        if idx.shape[0] < 2:
            continue
        counts = torch.bincount(idx, minlength=n_codes).float()
        p = counts / counts.sum().clamp(min=1)
        loss = loss + (p - global_p.detach()).pow(2).sum()
    return loss / max(len(torch.unique(batch_labels)), 1)


def _rbf_kernel(x, y, sigma):
    # x: (N, d), y: (M, d) -> (N, M)
    sq = (x.unsqueeze(1) - y.unsqueeze(0)).pow(2).sum(dim=-1)
    return torch.exp(-sq / (2 * sigma ** 2))


def batch_kernel_mmd(z, batch_labels, sigma=None):
    """Full kernel MMD between each batch's z distribution and global z.

    scVI enforces a batch-independent prior on z. This is the VQ-VAE analog:
    if P(z|batch b) == P(z) for every b, then z carries no batch info and no
    classifier — linear, tree, or kernel — can recover batch from z. Captures
    higher-order structure that the moment penalty misses.
    """
    if sigma is None:
        # median pairwise distance heuristic — treated as a fixed constant so
        # backprop of the MMD only flows through z, not through sigma scaling
        with torch.no_grad():
            idx = torch.randperm(z.size(0), device=z.device)[: min(z.size(0), 128)]
            pdist = (z[idx].unsqueeze(1) - z[idx].unsqueeze(0)).pow(2).sum(-1).sqrt()
            sigma = pdist.median() + 1e-6
    g_xx = _rbf_kernel(z, z, sigma).mean()
    loss = torch.zeros(1, device=z.device)
    for b in torch.unique(batch_labels):
        zb = z[batch_labels == b]
        if zb.shape[0] < 2:
            continue
        kbb = _rbf_kernel(zb, zb, sigma).mean()
        kgb = _rbf_kernel(zb, z, sigma).mean()
        loss = loss + (kbb - 2 * kgb + g_xx)
    return loss / max(len(torch.unique(batch_labels)), 1)


def nb_nll(x, mu, theta, eps=1e-8):
    """Negative binomial negative log-likelihood (scVI formulation).
    x ~ NB(mu, theta), Var = mu + mu^2/theta.
    """
    t1 = x * (mu.log() - (mu + theta + eps).log())
    t2 = theta * (theta.log() - (mu + theta + eps).log())
    ll = (
        torch.lgamma(x + theta + eps)
        - torch.lgamma(theta + eps)
        - torch.lgamma(x + 1)
        + t1
        + t2
    )
    return -ll


class Encoder(nn.Module):
    def __init__(self, n_genes, hidden_dim=256, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class VectorQuantizer(nn.Module):
    def __init__(self, n_codes=64, code_dim=128, commitment_cost=0.25, use_ema=False, decay=0.99):
        super().__init__()
        self.n_codes = n_codes
        self.code_dim = code_dim
        self.commitment_cost = commitment_cost
        self.use_ema = use_ema
        self.decay = decay

        self.codebook = nn.Embedding(n_codes, code_dim)
        self.codebook.weight.data.uniform_(-1.0 / n_codes, 1.0 / n_codes)

        if use_ema:
            self.register_buffer("ema_weight", self.codebook.weight.data.clone())
            self.register_buffer("cluster_size", torch.zeros(n_codes))
        # persistent usage tracking; a code must be dead for several consecutive
        # epochs before restart (prevents churn killing rare-but-used codes)
        self.register_buffer("usage", torch.zeros(n_codes))
        self.register_buffer("dead_epochs", torch.zeros(n_codes))

    @property
    def dtype(self):
        return self.codebook.weight.dtype

    def forward(self, z):
        z = z.unsqueeze(1)
        codebook = self.codebook.weight.detach()
        distances = (
            torch.sum(z**2, dim=2, keepdim=True)
            + torch.sum(codebook**2, dim=1)
            - 2 * torch.matmul(z, codebook.t())
        )
        min_encoding_indices = torch.argmin(distances, dim=2)
        z_q = self.codebook(min_encoding_indices).permute(0, 2, 1)
        z_q = z_q.squeeze(-1)

        if self.use_ema and self.training:
            encoding_indices = min_encoding_indices.squeeze(1)
            with torch.no_grad():
                self.usage.fill_(0)
                self.usage.scatter_add_(0, encoding_indices, torch.ones_like(encoding_indices).float())
                enc_onehot = F.one_hot(encoding_indices, self.n_codes).float()
                cluster_size = enc_onehot.sum(dim=0)
                self.cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
                dw = enc_onehot.t() @ z.squeeze(1)
                self.ema_weight.mul_(self.decay).add_(dw, alpha=1 - self.decay)
                n = self.cluster_size.sum()
                weighted = (self.ema_weight + 1e-5) / (self.cluster_size.view(-1, 1) + 1e-5)
                self.codebook.weight.data.copy_(weighted)
            # EMA updates the codebook lazily, so loss must carry BOTH terms; dropping
            # the commitment term lets encoder z drift and is a real collapse source
            loss = self.commitment_cost * F.mse_loss(z_q, z.squeeze(1).detach())
        else:
            loss = F.mse_loss(z_q.detach(), z.squeeze(1)) + self.commitment_cost * F.mse_loss(
                z_q, z.squeeze(1).detach()
            )

        z_q = z.squeeze(1) + (z_q - z.squeeze(1)).detach()
        encoding_indices = min_encoding_indices.squeeze(1)
        return z_q, loss, encoding_indices

    def restart_dead_codes(self, z_pre_vq, dead_idx, reinit="data"):
        """Reinit dead codes from the pool of encoder outputs — the Jukebox trick.
        z_pre_vq: (B, code_dim) continuous codes for a shuffle_minibatch.
        dead_idx: indices of codes with ~zero usage.
        """
        if dead_idx.numel() == 0:
            return
        k = dead_idx.numel()
        if reinit == "data":
            idx = torch.randperm(z_pre_vq.size(0), device=z_pre_vq.device)[:k]
            new_code = z_pre_vq[idx].detach()
        else:
            new_code = F.normalize(
                torch.randn(k, self.code_dim, device=z_pre_vq.device), dim=1
            )
        with torch.no_grad():
            self.codebook.weight.data[dead_idx] = new_code
            if self.use_ema:
                self.ema_weight.data[dead_idx] = new_code
                self.cluster_size.data[dead_idx] = 1.0

    def get_code_usage(self, indices, n_batches=1):
        counts = torch.zeros(n_batches, self.n_codes, device=indices.device)
        if n_batches == 1:
            for idx in indices.view(-1):
                counts[0, idx] += 1
        else:
            for b in range(n_batches):
                mask = indices == b
                for idx in indices[mask].view(-1):
                    counts[b, idx] += 1
        return counts


class Decoder(nn.Module):
    def __init__(self, code_dim=128, n_batches=10, hidden_dim=256, n_genes=2000):
        super().__init__()
        self.batch_embedding = nn.Embedding(n_batches, 32)
        total_dim = code_dim + 32 + 1  # +1 for log library size
        self.shared = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, n_genes)
        self.theta = nn.Linear(hidden_dim, n_genes)
        self.dropout = nn.Linear(hidden_dim, n_genes)

    def forward(self, z_q, batch_labels, log_lib_size):
        batch_emb = self.batch_embedding(batch_labels)
        decoder_input = torch.cat([z_q, batch_emb, log_lib_size.unsqueeze(1)], dim=1)
        h = self.shared(decoder_input)
        mu = torch.exp(self.mu(h))
        theta = torch.exp(self.theta(h))
        pi = torch.sigmoid(self.dropout(h))
        return mu, theta, pi


class BatchAdversary(nn.Module):
    def __init__(self, latent_dim=128, n_batches=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_batches),
        )

    def forward(self, z, alpha=1.0):
        z_reversed = GradientReversal.apply(z, alpha)
        return self.net(z_reversed)


class CellTypeClassifier(nn.Module):
    def __init__(self, latent_dim=128, n_cell_types=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_cell_types),
        )

    def forward(self, z):
        return self.net(z)


class VQVAE(nn.Module):
    def __init__(
        self,
        n_genes=2000,
        n_batches=5,
        n_cell_types=None,
        hidden_dim=256,
        latent_dim=128,
        n_codes=64,
        code_dim=None,
        commitment_cost=0.25,
        use_adversary=False,
        adversary_alpha=1.0,
        use_ema=False,
        ema_decay=0.99,
    ):
        super().__init__()
        if code_dim is None:
            code_dim = latent_dim
        self.n_genes = n_genes
        self.encoder = Encoder(n_genes, hidden_dim, latent_dim)
        self.pre_vq = nn.Linear(latent_dim, code_dim) if latent_dim != code_dim else nn.Identity()
        self.vq = VectorQuantizer(n_codes, code_dim, commitment_cost, use_ema, ema_decay)
        self.decoder = Decoder(code_dim, n_batches, hidden_dim, n_genes)
        self.use_adversary = use_adversary
        self.adversary_alpha = adversary_alpha
        if use_adversary and n_batches > 1:
            self.adversary = BatchAdversary(latent_dim, n_batches)
        self.classifier = None
        if n_cell_types is not None:
            self.classifier = CellTypeClassifier(latent_dim, n_cell_types)

    def encode(self, x):
        return self.encoder(x)

    def quantize(self, z):
        z = self.pre_vq(z)
        return self.vq(z)

    def restart_dead_codes(self, x, dead_idx):
        z = self.encode(x)
        z_pre = self.pre_vq(z)
        self.vq.restart_dead_codes(z_pre, dead_idx)

    def decode(self, z_q, batch_labels, log_lib_size):
        return self.decoder(z_q, batch_labels, log_lib_size)

    def forward(self, x, batch_labels, cell_types=None):
        z = self.encode(x)
        z_q, vq_loss, encoding_indices = self.quantize(z)
        # ponytail: library size from input, not learned — scVI learns it, but this
        # is simpler and achieves the same covariate separation at 1M scale
        log_lib_size = x.sum(dim=1).log()
        mu, theta, pi = self.decode(z_q, batch_labels, log_lib_size)
        outputs = {
            "z": z,
            "z_q": z_q,
            "vq_loss": vq_loss,
            "encoding_indices": encoding_indices,
            "mu": mu,
            "theta": theta,
            "pi": pi,
            "recon_loss": nb_nll(x, mu, theta).mean(),
        }
        if self.use_adversary and self.training:
            adv_logits = self.adversary(z, self.adversary_alpha)
            outputs["adv_logits"] = adv_logits
        if self.classifier is not None and cell_types is not None:
            ct_logits = self.classifier(z)
            outputs["ct_logits"] = ct_logits
        return outputs

    @torch.no_grad()
    def get_latents(self, x):
        z = self.encode(x)
        z_q, _, indices = self.quantize(z)
        return z, z_q, indices
