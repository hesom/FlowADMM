import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from flowadmm.dataloaders import DataLoaders
from flowadmm.degradations import (
    BoxInpainting,
    Denoising,
    GaussianDeblurring,
    RandomInpainting,
    Superresolution,
)
from flowadmm.methods.d_flow import D_FLOW
from flowadmm.methods.flow_priors import FLOW_PRIORS
from flowadmm.methods.flowadmm import FlowADMM
from flowadmm.methods.flower import FLOWER
from flowadmm.methods.ot_ode import OT_ODE
from flowadmm.methods.pnp_diff import PNP_DIFF
from flowadmm.methods.pnp_flow import PNP_FLOW
from flowadmm.methods.pnp_gs import PROX_PNP
from flowadmm.utils import define_model, load_cfg_from_cfg_file, load_model, merge_cfg_from_list


torch.cuda.empty_cache()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

SUPPORTED_DATASETS = {"celeba", "afhq_cat"}
SUPPORTED_METHODS = {
    "flowadmm",
    "d_flow",
    "flow_priors",
    "flower",
    "ot_ode",
    "pnp_diff",
    "pnp_flow",
    "pnp_gs",
}
SUPPORTED_PROBLEMS = {
    "denoising",
    "gaussian_deblurring_FFT",
    "inpainting",
    "random_inpainting",
    "superresolution",
}


def _resolve_model_path(args: argparse.Namespace) -> str:
    model_root_dir = getattr(args, "model_root_dir", "model")
    if args.model in {"ot", "gradient_step"}:
        return os.path.join(
            args.root, model_root_dir, args.dataset, args.model, "model_final.pt"
        )
    raise ValueError(f"Unsupported checkpointed model: {args.model}")


def _results_model_name(args: argparse.Namespace) -> str:
    return getattr(args, "results_model_name", "") or args.model


def _results_root_dir(args: argparse.Namespace) -> str:
    override = getattr(args, "results_root_dir", "")
    if override:
        return override
    return "results"


def _validate_args(args: argparse.Namespace) -> None:
    if args.dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    if args.method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported method: {args.method}")
    if args.problem not in SUPPORTED_PROBLEMS:
        raise ValueError(f"Unsupported inverse problem: {args.problem}")
    if getattr(args, "noise_type", "gaussian") != "gaussian":
        raise ValueError("The public reproduction repo supports only gaussian noise.")
    if args.method == "pnp_gs" and args.model != "gradient_step":
        raise ValueError("`pnp_gs` requires `model=gradient_step`.")
    if args.method == "pnp_diff" and args.model != "diffusion":
        raise ValueError("`pnp_diff` requires `model=diffusion`.")
    if args.method not in {"pnp_gs", "pnp_diff"} and args.model != "ot":
        raise ValueError(f"`{args.method}` requires `model=ot` in this public repo.")


def _build_degradation(args: argparse.Namespace, device: torch.device):
    if args.problem == "denoising":
        sigma_noise = 0.2
        degradation = Denoising()
    elif args.problem == "inpainting":
        sigma_noise = 0.05
        half_size_mask = 20 if args.dim_image == 128 else 40
        degradation = BoxInpainting(half_size_mask)
    elif args.problem == "random_inpainting":
        sigma_noise = 0.01
        degradation = RandomInpainting(0.7)
    elif args.problem == "superresolution":
        sigma_noise = 0.05
        sf_override = int(getattr(args, "superresolution_sf", 0))
        if sf_override > 0:
            sf = sf_override
        elif args.dim_image == 128:
            sf = 2
        elif args.dim_image == 256:
            sf = 4
        else:
            raise ValueError(f"No default SR scale for dim_image={args.dim_image}")
        print(f"Superresolution with scale factor {sf}")
        degradation = Superresolution(sf, args.dim_image)
    elif args.problem == "gaussian_deblurring_FFT":
        sigma_noise = 0.05
        sigma_blur_override = float(getattr(args, "gaussian_blur_sigma", -1.0))
        if sigma_blur_override > 0:
            sigma_blur = sigma_blur_override
        elif args.dim_image == 128:
            sigma_blur = 1.0
        elif args.dim_image == 256:
            sigma_blur = 3.0
        else:
            raise ValueError(f"No default blur sigma for dim_image={args.dim_image}")
        kernel_size = int(getattr(args, "gaussian_blur_kernel_size", 61))
        print(
            f"Gaussian deblurring with sigma_blur={sigma_blur} kernel_size={kernel_size}"
        )
        degradation = GaussianDeblurring(
            sigma_blur, kernel_size, "fft", args.num_channels, args.dim_image, device
        )
    else:
        raise ValueError(f"Unsupported inverse problem: {args.problem}")

    return degradation, sigma_noise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FlowADMM public reproduction entrypoint")
    cfg = load_cfg_from_cfg_file("./config/main_config.yaml")
    parser.add_argument("--opts", default=None, nargs=argparse.REMAINDER)
    parsed = parser.parse_args()

    if parsed.opts is not None:
        cfg = merge_cfg_from_list(cfg, parsed.opts)

    dataset_config = os.path.join(cfg.root, "config", "dataset_config", f"{cfg.dataset}.yaml")
    cfg.update(load_cfg_from_cfg_file(dataset_config))

    method_config_file = os.path.join(
        cfg.root, "config", "method_config", f"{cfg.method}.yaml"
    )
    cfg.update(load_cfg_from_cfg_file(method_config_file))

    if parsed.opts is not None:
        cfg = merge_cfg_from_list(cfg, parsed.opts)

    method_cfg = load_cfg_from_cfg_file(method_config_file)
    cfg.dict_cfg_method = {key: cfg[key] for key in method_cfg.keys()}
    return cfg


def main():
    args = parse_args()
    _validate_args(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True

    model, state = define_model(args)

    if args.model in {"ot", "gradient_step"}:
        model_path = _resolve_model_path(args)
        load_model(args.model, model, state, checkpoint_path=model_path, device=device)
        model.eval()
    elif args.model == "diffusion":
        model.eval()

    degradation, sigma_noise = _build_degradation(args, device)

    print(f"Solving {args.problem} with {args.method}...")
    print("sigma_noise", sigma_noise)

    data_loaders = DataLoaders(
        args.dataset, args.batch_size_ip, args.batch_size_ip
    ).load_data()

    results_root_dir = _results_root_dir(args)
    results_model_name = _results_model_name(args)
    args.save_path = os.path.join(
        args.root,
        results_root_dir,
        args.dataset,
        results_model_name,
        args.problem,
        args.method,
        args.eval_split,
    )
    os.makedirs(args.save_path, exist_ok=True)

    if args.method == "flowadmm":
        method = FlowADMM(model, device, args)
    elif args.method == "d_flow":
        method = D_FLOW(model, device, args)
    elif args.method == "flow_priors":
        method = FLOW_PRIORS(model, device, args)
    elif args.method == "flower":
        method = FLOWER(model, device, args)
    elif args.method == "ot_ode":
        method = OT_ODE(model, device, args)
    elif args.method == "pnp_diff":
        method = PNP_DIFF(model, device, args)
    elif args.method == "pnp_flow":
        method = PNP_FLOW(model, device, args)
    elif args.method == "pnp_gs":
        method = PROX_PNP(model, device, args)
    else:
        raise ValueError(f"Unsupported method: {args.method}")

    method.run_method(data_loaders, degradation, sigma_noise)


if __name__ == "__main__":
    main()
