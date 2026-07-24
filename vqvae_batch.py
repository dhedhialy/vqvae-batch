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
    def __init__(self, n_codes=64, code_dim=128, commitment_cost=0.25):
        super().__init__()
        self.n_codes = n_codes
        self.code_dim = code_dim
        self.commitment_cost = commitment_cost
        self.codebook = nn.Embedding(n_codes, code_dim)
        self.codebook.weight.data.uniform_(-1.0 / n_codes, 1.0 / n_codes)

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
        loss = F.mse_loss(z_q.detach(), z.squeeze(1)) + self.commitment_cost * F.mse_loss(
            z_q, z.squeeze(1).detach()
        )
        z_q = z.squeeze(1) + (z_q - z.squeeze(1)).detach()
        encoding_indices = min_encoding_indices.squeeze(1)
        return z_q, loss, encoding_indices

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
        total_dim = code_dim + 32
        self.net = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_genes),
        )

    def forward(self, z_q, batch_labels):
        batch_emb = self.batch_embedding(batch_labels)
        decoder_input = torch.cat([z_q, batch_emb], dim=1)
        return self.net(decoder_input)


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
        code_dim=128,
        commitment_cost=0.25,
        use_adversary=False,
        adversary_alpha=1.0,
    ):
        super().__init__()
        self.encoder = Encoder(n_genes, hidden_dim, latent_dim)
        self.vq = VectorQuantizer(n_codes, code_dim, commitment_cost)
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
        return self.vq(z)

    def decode(self, z_q, batch_labels):
        return self.decoder(z_q, batch_labels)

    def forward(self, x, batch_labels, cell_types=None):
        z = self.encode(x)
        z_q, vq_loss, encoding_indices = self.quantize(z)
        recon = self.decode(z_q, batch_labels)
        outputs = {
            "z": z,
            "z_q": z_q,
            "vq_loss": vq_loss,
            "encoding_indices": encoding_indices,
            "recon": recon,
            "recon_loss": F.mse_loss(recon, x),
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
