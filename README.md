[![image](https://img.shields.io/badge/arXiv-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.08640)

# FlowADMM

This repository contains code for the **FlowADMM** paper.

This project was originally forked from the [`PnP-Flow`](https://github.com/annegnx/PnP-Flow) repository. The included `FLOWER` baseline is derived from the separate [`Flower`](https://github.com/mehrsapo/Flower) repository, which itself builds on `PnP-Flow`.

The public repo focuses on:

- `celeba` and `afhq_cat`
- Inverse problems with Gaussian noise
- `FlowADMM` in [`flowadmm/methods/flowadmm.py`](flowadmm/methods/flowadmm.py)
- the baseline methods carried over from `PnP-Flow` and `Flower`

## Setup

We recommend `uv`:

```bash
uv sync
```

## Download data and checkpoints

CelebA:

```bash
bash download.sh celeba-dataset
bash download.sh pretrained-network-celeba
bash download.sh pretrained-denoiser-celeba
```

AFHQ-Cat:

```bash
bash download.sh afhq-cat-dataset
bash download.sh pretrained-network-afhq-cat
bash download.sh pretrained-denoiser-afhq-cat
```

The AFHQ helper also creates the validation split used by the benchmark.

## Reproduce FlowADMM

To reproduce the main paper FlowADMM results:

```bash
uv run bash scripts/script_test_flowadmm_main_paper.sh
```

By default this writes to `results_submission/`. You can override the output root:

```bash
RESULTS_ROOT_DIR=my_results uv run bash scripts/script_test_flowadmm_main_paper.sh
```

## Reproduce baselines

CelebA baselines:

```bash
uv run bash scripts/script_test_paper_baselines_celeba.sh
```

AFHQ-Cat baselines:

```bash
uv run bash scripts/script_test_paper_baselines_afhq_cat.sh
```

We did not check if the baselines reproduce exactly in this repository. For faithful reproduction we refer to the [`PnP-Flow`](https://github.com/annegnx/PnP-Flow) and [`Flower`](https://github.com/mehrsapo/Flower) repositories.

## Reproduce FlowADMM tuning

CelebA validation grid:

```bash
uv run bash scripts/script_val_flowadmm_celeba.sh
```

AFHQ-Cat validation grid:

```bash
uv run bash scripts/script_val_flowadmm_afhq_cat.sh
```

These scripts implement the paper grid over:

- `tau`
- `t_min`
- `t_max`
- `gamma`
- constant `K_avg=5` (average number of flow evaluations per iteration)
- fixed-budget three-phase `K_avg` schedules (called `N_k` in the paper) 

## Notes

- The methods reseed with `torch.manual_seed(batch)` before measurement which should reproduce the values in the paper
