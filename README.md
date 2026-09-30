# custom-optimizers

Coordinate-wise **SVRG** (Stochastic Variance Reduced Gradient) with adaptive
momentum, built for low signal-to-noise objectives where Adam/SGD stall on a
non-vanishing variance floor. Two backends share one algorithm:

| Backend | File | Style |
|---|---|---|
| PyTorch | `src/torch_optimizer.py` | stateful `torch.optim.Optimizer`, flat contiguous memory |
| JAX | `src/jax_optimizer.py` | stateless `state_t -> state_t+1`, PyTree state, `jax.jit` |

## The algorithm

```
g_hat = g_t(w_t) - g_t(w~) + mu~            # variance-reduced gradient
m     = b1*m + (1-b1)*g_hat                 # momentum
v     = b2*v + (1-b2)*g_hat^2               # second moment
w    <- w - lr * clip( m_hat / max(sqrt(v_hat), floor), +-update_clip )
```

`w~` is a frozen snapshot, `mu~` the full gradient at `w~`, refreshed every
`snapshot_interval` inner steps.

## The four invariants

| # | Invariant | How it is enforced |
|---|---|---|
| 1 | **Variance reduction** — `E||g_hat - grad f(w_t)||^2 -> 0` as `w_t -> w*` | snapshot control variate; `variance_reduced_gradient(...)` exposes `g_hat`; telemetry logs `snapshot_distance` and `correction_norm` |
| 2 | **Deterministic sample alignment** | the step passes ONE batch id (torch) / ONE batch object (JAX) to both the live and snapshot gradient evaluations; `DeterministicBatchSampler` maps id -> fixed rows |
| 3 | **Hardware contiguity** | torch: parameters re-pointed into one flat buffer; snapshot, `mu~`, `m`, `v` in one `(4, N)` block; swapping to the snapshot is one `copy_` |
| 4 | **Coordinate boundary safeguards** | NaN/Inf sanitised, `|g|` bounded so `g^2` stays finite, denominator **clamped** (not offset) at `max(eps, 4*sqrt(tiny))`, final ratio clipped |

## Setup (Codespace / Colab)

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

## PyTorch usage

```python
import torch
from src.environment import market_from_yaml, DeterministicBatchSampler
from src.torch_optimizer import CoordinateSVRG

ds = market_from_yaml("configs/stochastic_regime.yaml")
X = torch.as_tensor(ds.features, dtype=torch.float32, device="cuda")
y = torch.as_tensor(ds.targets, dtype=torch.float32, device="cuda")
w = torch.nn.Parameter(torch.zeros(ds.n_features, device="cuda"))

sampler = DeterministicBatchSampler(ds.n_samples, batch_size=64, seed=ds.config.seed)
chunks = sampler.full_pass_chunks(8)

def loss_on(rows):
    idx = torch.as_tensor(rows, device="cuda")
    return 0.5 * ((X[idx] @ w - y[idx]) ** 2).mean()

batch_closure = lambda batch_id: loss_on(sampler.batch(batch_id))   # pure in batch_id
chunk_closure = lambda chunk_id: loss_on(chunks[chunk_id])

opt = CoordinateSVRG([w], lr=0.02, snapshot_interval=64)
for _ in range(640):
    if opt.needs_snapshot:
        opt.refresh_snapshot(chunk_closure, num_chunks=len(chunks))
    opt.step(batch_closure)
    # wandb.log(opt.last_stats)   # step_us, snapshot_distance, correction_norm, ...
```

Rules: build the optimizer after the model is on its final device/dtype; mutate
parameters in place (never reassign `p.data`); closures must be deterministic in
their id argument.

## JAX usage

```python
import jax.numpy as jnp
from src.jax_optimizer import (SVRGConfig, init_state, make_svrg_step,
                               compute_full_gradient, refresh_snapshot)

def loss_fn(w, batch):
    x, y = batch
    return 0.5 * jnp.mean((x @ w - y) ** 2)

config = SVRGConfig(lr=0.02, snapshot_interval=64)
step = make_svrg_step(loss_fn, config)          # jitted pure function
params = jnp.zeros(ds.n_features)
state = init_state(params)

for i in range(640):
    if i % config.snapshot_interval == 0:
        full_grad = compute_full_gradient(loss_fn, params, chunk_stack)  # leading axis = chunks
        state = refresh_snapshot(params, state, full_grad)
    params, state, stats = step(params, state, batch_at(i))
```

## Repository layout

```
configs/stochastic_regime.yaml   regime, noise sigma levels, seeds, optimizer knobs
src/environment.py               toxic market simulator + deterministic batch sampler
src/torch_optimizer.py           stateful PyTorch engine
src/jax_optimizer.py             stateless JAX engine
tests/test_math.py               invariant validation (pytest)
```

## Design notes

* Adaptive normalisation keeps the step size near `lr` even when gradients
  shrink, so SVRG here converges because `v` has long memory (`beta2 = 0.999`)
  and `m` decays faster. If you need exact convergence over long horizons,
  decay `lr` or raise `beta2`.
* `eps` is a *floor on the denominator*, not an additive term, so a 1e-15 value
  is meaningful even in float32.
