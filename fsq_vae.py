import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUTPUT = "/home/aly/vqvae-batch/output/fsq"
os.makedirs(OUTPUT, exist_ok=True)

print("=" * 65)
print("FSQ-VAE: interpretable discrete latent + batch-conditioned decoder")
print("         + gradient-reversal adversary (plain torch)")
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
ct_names = sorted(np.unique(cell_types))
batch_names = sorted(np.unique(batches))
print(f"  {X.shape[0]} cells, {n_genes} genes, {n_batch} batches, {meta['n_cell_types']} cell types")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from scvi.distributions import NegativeBinomial


def round_ste(z):
    zhat = torch.round(z)
    return z + (zhat - z).detach()


class FSQ(nn.Module):
    """Finite Scalar Quantization: quantize each latent dim to a fixed level count."""
    def __init__(self, levels):
        super().__init__()
        self.levels = list(levels)
        self.register_buffer("scales", torch.tensor([l / 2 for l in levels], dtype=torch.float32))

    def forward(self, z):
        z = torch.tanh(z)
        z = z * self.scales
        z_q = round_ste(z)
        codes = (z_q + self.scales).long().clamp_min(0)
        z_out = z_q / self.scales
        return z_out, codes

    def n_codes(self):
        n = 1
        for l in self.levels:
            n *= l
        return n


class AdvGradReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output


class FSQVAE(nn.Module):
    def __init__(self, n_input, n_batch, levels, n_hidden=128,
                 n_layers_encoder=2, adversarial_lambda=1.0, disp=0.1):
        super().__init__()
        self.n_latent = len(levels)
        self.n_batch = n_batch
        self.levels = list(levels)
        self.adv_lambda = adversarial_lambda

        enc = []
        in_dim = n_input
        for _ in range(n_layers_encoder):
            enc += [nn.Linear(in_dim, n_hidden), nn.ReLU()]
            in_dim = n_hidden
        enc.append(nn.Linear(in_dim, 2 * self.n_latent))
        self.encoder = nn.Sequential(*enc)

        self.l_encoder = nn.Sequential(
            nn.Linear(n_input, n_hidden), nn.ReLU(), nn.Linear(n_hidden, 1)
        )

        self.fsq = FSQ(levels)

        # batch embedding (decoder covariate)
        self.batch_emb = nn.Embedding(n_batch, self.n_latent)
        nn.init.normal_(self.batch_emb.weight, mean=0.0, std=0.1)

        self.decoder = nn.Sequential(
            nn.Linear(2 * self.n_latent, n_hidden), nn.ReLU(),
            nn.Linear(n_hidden, n_hidden), nn.ReLU(),
            nn.Linear(n_hidden, n_input),
        )
        self.log_dispersion = nn.Parameter(torch.full((n_input,), float(np.log(disp))))

        # adversarial batch classifier on the DISCRETE code (gradient reversal)
        self.adv_classifier = nn.Sequential(
            nn.Linear(self.n_latent, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_batch)
        )

    def encode(self, x):
        params = self.encoder(x)
        qz_m, qz_logvar = params[:, :self.n_latent], params[:, self.n_latent:]
        qz_v = F.softplus(qz_logvar) + 1e-4
        z = qz_m + torch.randn_like(qz_m) * torch.sqrt(qz_v)
        return qz_m, qz_v, z

    def forward(self, x, batch):
        qz_m, qz_v, z = self.encode(x)
        z_q, codes = self.fsq(z)
        log_library = self.l_encoder(x)
        emb = self.batch_emb(batch)
        log_rate = self.decoder(torch.cat([z_q, emb], dim=-1))
        px_rate = torch.exp(log_library + log_rate)
        px_r = torch.exp(self.log_dispersion).expand_as(px_rate)
        return qz_m, qz_v, z, z_q, codes, px_rate, px_r

    def adv_loss(self, z_q, batch):
        z_rev = AdvGradReversal.apply(z_q)
        logits = self.adv_classifier(z_rev)
        return F.cross_entropy(logits, batch), logits


def train(module, dl, n_epochs=100, lr=1e-3, device="cuda"):
    optimizer = torch.optim.Adam(module.parameters(), lr=lr)
    t0 = time.time()
    for epoch in range(n_epochs):
        module.train()
        tot = dict(loss=0, rec=0, kl=0, adv=0, acc=0, n_codes=0)
        n = 0
        for xb, bb in dl:
            xb, bb = xb.to(device), bb.to(device)
            optimizer.zero_grad()
            qz_m, qz_v, z, z_q, codes, px_rate, px_r = module(xb, bb)
            nb = NegativeBinomial(mu=px_rate, theta=px_r)
            rec = -nb.log_prob(xb).sum(-1).mean()
            kl = 0.5 * (qz_v + qz_m ** 2 - torch.log(qz_v) - 1).sum(-1).mean()
            adv, logits = module.adv_loss(z_q, bb)
            loss = rec + kl + module.adv_lambda * adv
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), 5.0)
            optimizer.step()
            acc = (logits.argmax(-1) == bb).float().mean().item()
            tot["loss"] += loss.item() * len(xb); tot["rec"] += rec.item() * len(xb)
            tot["kl"] += kl.item() * len(xb); tot["adv"] += adv.item() * len(xb)
            tot["acc"] += acc * len(xb); tot["n_codes"] += codes.max().item()
            n += len(xb)
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


def run(levels, adv_lambda, run_name, n_epochs=100, seed=42, save_codes=True):
    torch.manual_seed(seed); np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[{run_name}] levels={levels}, adv_lambda={adv_lambda}")

    tensor_x = torch.tensor(X.astype(np.float32))
    tensor_b = torch.tensor(batches.astype(np.int64))
    full = TensorDataset(tensor_x, tensor_b)
    dl = DataLoader(full, batch_size=512, shuffle=True, num_workers=4, drop_last=True)

    module = FSQVAE(n_input=n_genes, n_batch=n_batch, levels=levels,
                    adversarial_lambda=adv_lambda).to(device)
    train(module, dl, n_epochs=n_epochs, device=device)

    print(f"  [{run_name}] Evaluating on full data...")
    dl_eval = DataLoader(TensorDataset(tensor_x, tensor_b), batch_size=1024,
                         shuffle=False, num_workers=4)
    z_q_all, codes_all = [], []
    module.eval()
    with torch.no_grad():
        for xb, bb in dl_eval:
            xb, bb = xb.to(device), bb.to(device)
            qz_m, qz_v, z, z_q, codes, _, _ = module(xb, bb)
            z_q_all.append(z_q.cpu().numpy())
            codes_all.append(codes.cpu().numpy())
    z_q = np.concatenate(z_q_all, 0)
    codes = np.concatenate(codes_all, 0)

    # LISI on the discrete latent (one-hot codes)
    n_total_codes = module.fsq.n_codes()
    code_id = np.zeros((len(codes), 1), dtype=np.float32)
    stride = 1
    for d in range(len(levels)):
        code_id[:, 0] += codes[:, d].astype(np.float32) * stride
        stride *= levels[d]
    bl = compute_lisi(z_q, batches)
    cl = compute_lisi(z_q, cell_types)
    ca = cross_batch_ct(z_q, batches, cell_types)
    print(f"  [{run_name}] FSQ latent: batch_LISI={bl:.3f}  celltype_LISI={cl:.3f}  cross_CT={ca:.3f}")

    # Harmony on discrete latent
    import harmonypy
    ho = harmonypy.run_harmony(z_q, {"batch": batches}, "batch", max_iter_harmony=50)
    z_har = ho.Z_corr
    bl_har = compute_lisi(z_har, batches)
    cl_har = compute_lisi(z_har, cell_types)
    ca_har = cross_batch_ct(z_har, batches, cell_types)
    print(f"  [{run_name}] +Harmony: batch_LISI={bl_har:.3f}  celltype_LISI={cl_har:.3f}  cross_CT={ca_har:.3f}")

    res = {
        "levels": levels, "adv_lambda": adv_lambda,
        "fsq": {"batch_lisi": bl, "celltype_lisi": cl, "cross_batch_ct": ca},
        "fsq_harmony": {"batch_lisi": bl_har, "celltype_lisi": cl_har, "cross_batch_ct": ca_har},
        "n_codes_used": int(np.unique(code_id).size),
        "n_total_codes": int(n_total_codes),
    }

    if save_codes:
        np.save(f"{OUTPUT}/codes_{run_name}.npy", codes)
        np.save(f"{OUTPUT}/zq_{run_name}.npy", z_q)
        # code -> celltype composition for interpretability
        comp = {}
        for c in np.unique(code_id):
            mask = (code_id[:, 0] == c).ravel()
            if mask.sum() < 10:
                continue
            cts = np.bincount(cell_types[mask], minlength=len(ct_names))
            comp[int(c)] = {
                "n": int(mask.sum()),
                "majority_ct": ct_names[int(np.argmax(cts))],
                "ct_fracs": [float(x / mask.sum()) for x in cts],
            }
        res["code_composition"] = comp
        np.save(f"{OUTPUT}/code_id_{run_name}.npy", code_id.astype(np.int64))

    return res


results = {}
results["L3x8_adv1.0"] = run(levels=[8, 8, 8], adv_lambda=1.0, run_name="L3x8_adv1.0")

with open(f"{OUTPUT}/fsq_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved results to {OUTPUT}/fsq_results.json")
