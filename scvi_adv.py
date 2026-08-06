import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUTPUT = "/home/aly/vqvae-batch/output/scvi_adv"
os.makedirs(OUTPUT, exist_ok=True)

print("=" * 65)
print("scVI + adversarial batch classifier (gradient reversal, plain torch)")
print("=" * 65)

print("\n[1] Loading data...")
from data import load_cellxgene_data
X, batches, cell_types, meta = load_cellxgene_data(
    "/data/bhy/czcellxgene/h5ads/0b75c598-0893-4216-afe8-5414cab7739d.h5ad",
    batch_key="donor_id", celltype_key="cell_type",
    min_cells_per_batch=50, min_cells_per_type=50,
    n_top_genes=2000, max_batches=15, max_cell_types=20,
)
n_genes = meta["n_genes"]
n_batch = meta["n_batches"]
print(f"  {X.shape[0]} cells, {n_genes} genes, {n_batch} batches, {meta['n_cell_types']} cell types")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from scvi.distributions import NegativeBinomial


class AdvGradReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output


class AdversarialVAE(nn.Module):
    def __init__(self, n_input, n_batch, n_latent=30, n_hidden=128,
                 n_layers_encoder=2, adversarial_lambda=1.0, disp=0.1):
        super().__init__()
        self.n_latent = n_latent
        self.n_batch = n_batch
        self.adv_lambda = adversarial_lambda

        # encoder q(z | x)
        enc = []
        in_dim = n_input
        for _ in range(n_layers_encoder):
            enc += [nn.Linear(in_dim, n_hidden), nn.ReLU()]
            in_dim = n_hidden
        enc.append(nn.Linear(in_dim, 2 * n_latent))
        self.encoder = nn.Sequential(*enc)

        # library size encoder -> log library mean
        self.l_encoder = nn.Sequential(
            nn.Linear(n_input, n_hidden), nn.ReLU(), nn.Linear(n_hidden, 1)
        )

        # batch embedding (decoder covariate)
        self.batch_emb = nn.Embedding(n_batch, n_latent)
        nn.init.normal_(self.batch_emb.weight, mean=0.0, std=0.1)

        # decoder: [z, batch_emb] -> per-gene log rate
        self.decoder = nn.Sequential(
            nn.Linear(2 * n_latent, n_hidden), nn.ReLU(),
            nn.Linear(n_hidden, n_hidden), nn.ReLU(),
            nn.Linear(n_hidden, n_input),
        )
        self.log_dispersion = nn.Parameter(torch.full((n_input,), float(np.log(disp))))

        # adversarial batch classifier on z (gradient reversal)
        self.adv_classifier = nn.Sequential(
            nn.Linear(n_latent, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_batch)
        )

    def encode(self, x):
        params = self.encoder(x)
        qz_m, qz_logvar = params[:, :self.n_latent], params[:, self.n_latent:]
        qz_v = F.softplus(qz_logvar) + 1e-4
        z = qz_m + torch.randn_like(qz_m) * torch.sqrt(qz_v)
        return qz_m, qz_v, z

    def forward(self, x, batch):
        qz_m, qz_v, z = self.encode(x)
        log_library = self.l_encoder(x)
        emb = self.batch_emb(batch)
        log_rate = self.decoder(torch.cat([z, emb], dim=-1))
        px_rate = torch.exp(log_library + log_rate)  # (n, n_genes)
        px_r = torch.exp(self.log_dispersion).expand_as(px_rate)
        return qz_m, qz_v, z, px_rate, px_r, log_library

    def adv_loss(self, z, batch):
        z_rev = AdvGradReversal.apply(z)
        logits = self.adv_classifier(z_rev)
        return F.cross_entropy(logits, batch), logits


def train(module, dl, n_epochs=80, lr=1e-3, device="cuda"):
    optimizer = torch.optim.Adam(module.parameters(), lr=lr)
    t0 = time.time()
    for epoch in range(n_epochs):
        module.train()
        tot = dict(loss=0, rec=0, kl=0, adv=0, acc=0)
        n = 0
        for xb, bb in dl:
            xb, bb = xb.to(device), bb.to(device)
            optimizer.zero_grad()
            qz_m, qz_v, z, px_rate, px_r, _ = module(xb, bb)
            # reconstruction (negative binomial log prob)
            nb = NegativeBinomial(mu=px_rate, theta=px_r)
            rec = -nb.log_prob(xb).sum(-1).mean()
            kl = 0.5 * (qz_v + qz_m ** 2 - torch.log(qz_v) - 1).sum(-1).mean()
            adv, logits = module.adv_loss(z, bb)
            loss = rec + kl + module.adv_lambda * adv
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), 5.0)
            optimizer.step()
            acc = (logits.argmax(-1) == bb).float().mean().item()
            tot["loss"] += loss.item() * len(xb); tot["rec"] += rec.item() * len(xb)
            tot["kl"] += kl.item() * len(xb); tot["adv"] += adv.item() * len(xb)
            tot["acc"] += acc * len(xb); n += len(xb)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: loss={tot['loss']/n:.2f} "
                  f"rec={tot['rec']/n:.2f} kl={tot['kl']/n:.2f} "
                  f"adv={tot['adv']/n:.2f} batch_acc={tot['acc']/n:.3f}")
    print(f"  Trained in {time.time()-t0:.0f}s")


def compute_lisi(z, labels):
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=30, n_jobs=-1).fit(z)
    _, idx = nn.kneighbors(z)
    scores = []
    for i in range(len(z)):
        _, cnt = np.unique(labels[idx[i]], return_counts=True)
        p = cnt / 30
        scores.append(1.0 / np.sum(p ** 2))
    return float(np.mean(scores))


def cross_batch_ct(z, batch, ct):
    from sklearn.ensemble import RandomForestClassifier
    sc = []
    for b in np.unique(batch):
        te = batch == b; tr = ~te
        if te.sum() < 10:
            continue
        rf = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
        rf.fit(z[tr], ct[tr]); sc.append(rf.score(z[te], ct[te]))
    return float(np.mean(sc))


def evaluate(module, dl, device):
    module.eval()
    zs = []
    with torch.no_grad():
        for xb, bb in dl:
            xb, bb = xb.to(device), bb.to(device)
            qz_m, qz_v, z, _, _, _ = module(xb, bb)
            zs.append(qz_m.cpu().numpy())
    return np.concatenate(zs, 0)


def main(adv_lambda, n_latent, run_name, n_epochs=80, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[{run_name}] adv_lambda={adv_lambda}, n_latent={n_latent}")

    tensor_x = torch.tensor(X.astype(np.float32))
    tensor_b = torch.tensor(batches.astype(np.int64))
    full = TensorDataset(tensor_x, tensor_b)
    dl = DataLoader(full, batch_size=512, shuffle=True, num_workers=4, drop_last=True)

    module = AdversarialVAE(n_input=n_genes, n_batch=n_batch, n_latent=n_latent,
                            adversarial_lambda=adv_lambda).to(device)
    train(module, dl, n_epochs=n_epochs, device=device)

    print(f"  [{run_name}] Evaluating on full data...")
    dl_eval = DataLoader(TensorDataset(tensor_x, tensor_b), batch_size=1024,
                         shuffle=False, num_workers=4)
    z = evaluate(module, dl_eval, device)
    bl = compute_lisi(z, batches)
    cl = compute_lisi(z, cell_types)
    ca = cross_batch_ct(z, batches, cell_types)
    print(f"  [{run_name}] Adv-scVI: batch_LISI={bl:.3f}  celltype_LISI={cl:.3f}  cross_CT={ca:.3f}")

    # Harmony on top
    import harmonypy
    ho = harmonypy.run_harmony(z, {"batch": batches}, "batch", max_iter_harmony=50)
    z_har = ho.Z_corr
    bl_har = compute_lisi(z_har, batches)
    cl_har = compute_lisi(z_har, cell_types)
    ca_har = cross_batch_ct(z_har, batches, cell_types)
    print(f"  [{run_name}] +Harmony: batch_LISI={bl_har:.3f}  celltype_LISI={cl_har:.3f}  cross_CT={ca_har:.3f}")

    return {
        "adv_scvi": {"batch_lisi": bl, "celltype_lisi": cl, "cross_batch_ct": ca},
        "adv_scvi_harmony": {"batch_lisi": bl_har, "celltype_lisi": cl_har, "cross_batch_ct": ca_har},
    }


results = {}
results.update(main(adv_lambda=1.0, n_latent=30, run_name="L30_adv1.0"))
results.update(main(adv_lambda=5.0, n_latent=30, run_name="L30_adv5.0"))
results.update(main(adv_lambda=1.0, n_latent=10, run_name="L10_adv1.0"))

with open(f"{OUTPUT}/adv_scvi_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved results to {OUTPUT}/adv_scvi_results.json")
