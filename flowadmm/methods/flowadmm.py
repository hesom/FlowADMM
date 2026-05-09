import os
from time import perf_counter

import numpy as np
import torch

import flowadmm.utils as utils


def cg_proximal(b, H, H_adj, y, sigma_noise, tau, max_iter=10, tol=1e-6):
    """Solve the Gaussian x-update linear system with conjugate gradients."""
    tau_sy2 = tau / (sigma_noise ** 2 + 1e-10)
    rhs = b + tau_sy2 * H_adj(y)

    def matvec(w):
        return w + tau_sy2 * H_adj(H(w))

    w = b.clone()
    r = rhs - matvec(w)
    p = r.clone()
    rs_old = (r * r).reshape(r.shape[0], -1).sum(dim=1, keepdim=True)

    for _ in range(max_iter):
        Ap = matvec(p)
        pAp = (p * Ap).reshape(p.shape[0], -1).sum(dim=1, keepdim=True)
        alpha = rs_old / (pAp + 1e-10)
        alpha = alpha.view(-1, 1, 1, 1)
        w = w + alpha * p
        r = r - alpha * Ap
        rs_new = (r * r).reshape(r.shape[0], -1).sum(dim=1, keepdim=True)
        if rs_new.max().item() < tol:
            break
        beta = rs_new / (rs_old + 1e-10)
        p = r + beta.view(-1, 1, 1, 1) * p
        rs_old = rs_new

    return w


class FlowADMM(object):
    """FlowADMM as used for the main Gaussian paper experiments."""

    def __init__(self, model, device, args):
        self.device = device
        self.args = args
        self.model = model.to(device)

    def model_forward(self, x, t):
        if self.args.model != "ot":
            raise ValueError("FlowADMM in the public repo supports only `model=ot`.")
        return self.model(x, t)

    def denoiser(self, x, t):
        v = self.model_forward(x, t)
        return x + (1 - t.view(-1, 1, 1, 1)) * v

    def denoiser_batched(self, x, t):
        chunk_size = int(
            getattr(self.args, "denoiser_chunk_size", getattr(self.args, "batch_size_ip", x.shape[0]))
        )
        chunk_size = max(1, chunk_size)
        outputs = []
        for start in range(0, x.shape[0], chunk_size):
            end = min(start + chunk_size, x.shape[0])
            outputs.append(self.denoiser(x[start:end], t[start:end]))
        return torch.cat(outputs, dim=0)

    def proximal_step(self, b, degradation, noisy_img, sigma_noise):
        nu = sigma_noise ** 2 / (float(self.args.tau) + 1e-10)
        result = degradation.wiener_solve(b, noisy_img, sigma_noise, nu)
        if result is not None:
            return result
        return cg_proximal(
            b,
            degradation.H,
            degradation.H_adj,
            noisy_img,
            sigma_noise,
            float(self.args.tau),
        )

    def build_K_avg_plan(self):
        steps = int(self.args.steps_admm)
        schedule = getattr(self.args, "K_avg_schedule", "constant")
        K_avg = max(1, int(getattr(self.args, "K_avg", 1)))

        if schedule == "constant":
            return [K_avg] * steps

        if schedule != "three_phase":
            raise ValueError(f"Unsupported K_avg_schedule: {schedule}")

        K_avg_early = max(1, int(getattr(self.args, "K_avg_early", K_avg)))
        K_avg_mid = max(1, int(getattr(self.args, "K_avg_mid", K_avg)))
        K_avg_late = max(1, int(getattr(self.args, "K_avg_late", K_avg)))
        switch_1 = float(getattr(self.args, "K_avg_switch_frac", 0.5))
        switch_2 = float(getattr(self.args, "K_avg_switch_frac_2", 0.8))

        idx_1 = int(np.floor(switch_1 * steps))
        idx_2 = int(np.floor(switch_2 * steps))
        idx_1 = min(max(idx_1, 0), steps)
        idx_2 = min(max(idx_2, idx_1), steps)

        plan = []
        for iteration in range(steps):
            if iteration < idx_1:
                plan.append(K_avg_early)
            elif iteration < idx_2:
                plan.append(K_avg_mid)
            else:
                plan.append(K_avg_late)
        return plan

    def solve_ip(self, test_loader, degradation, sigma_noise, H_funcs=None):
        H = degradation.H
        H_adj = degradation.H_adj

        t_min = float(self.args.t_min)
        t_max = float(self.args.t_max)
        gamma = float(self.args.gamma)
        steps = int(self.args.steps_admm)
        K_avg_plan = self.build_K_avg_plan()

        loader = iter(test_loader)
        for batch in range(self.args.max_batch):
            clean_img, _ = next(loader)
            self.args.batch = batch
            print(clean_img.shape)

            torch.manual_seed(batch)
            noisy_img = H(clean_img.clone().to(self.device))
            noisy_img += torch.randn_like(noisy_img) * sigma_noise
            noisy_img = noisy_img.to(self.device)
            clean_img = clean_img.to("cpu")

            x = H_adj(noisy_img).to(self.device)
            z = x.clone()
            u = torch.zeros_like(x)

            if self.args.compute_time:
                torch.cuda.synchronize()
                time_per_batch = 0

            if self.args.compute_memory:
                torch.cuda.reset_max_memory_allocated(self.device)

            with torch.no_grad():
                for iteration in range(steps):
                    if self.args.compute_time:
                        time_counter_1 = perf_counter()

                    n_frac = ((iteration + 1) / steps) ** gamma
                    t_rf = t_min + n_frac * (t_max - t_min)
                    sigma_n = 1.0 - t_rf
                    K_avg_iter = K_avg_plan[iteration]

                    x = self.proximal_step(z - u, degradation, noisy_img, sigma_noise)

                    center = t_rf * (x + u)
                    batch_size = x.shape[0]
                    t_pnp = torch.ones(batch_size, device=self.device) * t_rf

                    eps_all = torch.randn(K_avg_iter, *x.shape, device=x.device)
                    z_tilde_all = center.unsqueeze(0) + sigma_n * eps_all
                    z_flat = z_tilde_all.reshape(K_avg_iter * batch_size, *x.shape[1:])
                    t_flat = t_pnp.repeat(K_avg_iter)
                    z_hat_flat = self.denoiser_batched(z_flat, t_flat)
                    z = z_hat_flat.reshape(K_avg_iter, batch_size, *x.shape[1:]).mean(dim=0)

                    u = u + x - z

                    if self.args.compute_time:
                        torch.cuda.synchronize()
                        time_counter_2 = perf_counter()
                        time_per_batch += time_counter_2 - time_counter_1

                    if self.args.save_results:
                        if iteration % 50 == 0 or self.should_save_image(iteration, steps):
                            restored_img = z.detach().clone()
                            utils.compute_psnr(
                                clean_img, noisy_img, restored_img, self.args, H_adj, iter=iteration
                            )
                            utils.compute_ssim(
                                clean_img, noisy_img, restored_img, self.args, H_adj, iter=iteration
                            )
                            utils.compute_lpips(
                                clean_img, noisy_img, restored_img, self.args, H_adj, iter=iteration
                            )

            if self.args.compute_memory:
                utils.save_memory_use(
                    {"batch": batch, "max_allocated": torch.cuda.max_memory_allocated(self.device)},
                    self.args,
                )

            if self.args.compute_time:
                utils.save_time_use({"batch": batch, "time_per_batch": time_per_batch}, self.args)

            if self.args.save_results:
                restored_img = z.detach().clone()
                if self.args.eval_split == "test":
                    utils.save_images(clean_img, noisy_img, restored_img, self.args, H_adj, iter="final")
                utils.compute_psnr(clean_img, noisy_img, restored_img, self.args, H_adj, iter=iteration)
                utils.compute_ssim(clean_img, noisy_img, restored_img, self.args, H_adj, iter=iteration)
                utils.compute_lpips(clean_img, noisy_img, restored_img, self.args, H_adj, iter=iteration)

        if self.args.save_results:
            utils.compute_average_psnr(self.args)
            utils.compute_average_ssim(self.args)
            utils.compute_average_lpips(self.args)
        if self.args.compute_memory:
            utils.compute_average_memory(self.args)
        if self.args.compute_time:
            utils.compute_average_time(self.args)

    def should_save_image(self, iteration, steps):
        return iteration % max(1, steps // 10) == 0

    def run_method(self, data_loaders, degradation, sigma_noise, H_funcs=None):
        folder = utils.get_save_path_ip(self.args.dict_cfg_method)
        self.args.save_path_ip = os.path.join(self.args.save_path, folder)
        os.makedirs(self.args.save_path_ip, exist_ok=True)
        self.solve_ip(data_loaders[self.args.eval_split], degradation, sigma_noise, H_funcs)
