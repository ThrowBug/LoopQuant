"""Loop-first low-rank initialization with optional block reconstruction."""

from pathlib import Path
import tempfile

import torch
from safetensors.torch import load_file, save_file

from .adapter import run_block
from .modules import iter_linears


@torch.no_grad()
def initialize_increment(module, original, loop, *, seed, iterations, trainable=True, logger=None, label=""):
    residual = original.to(module.weight.device, torch.float32) - module.weight.float()
    if module.sharing == "prefix" and loop:
        a = torch.cat(tuple(module.A[:loop]), 0).float()
        b = torch.cat(tuple(module.B[:loop]), 1).float()
        residual -= b @ a
    rank = module.rank_step
    devices = [residual.device] if residual.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if min(residual.shape) <= 256:
            u, s, vh = torch.linalg.svd(residual, full_matrices=False)
            u, s, vh = u[:, :rank], s[:rank], vh[:rank]
        else:
            u, s, v = torch.svd_lowrank(residual, q=min(rank + 8, min(residual.shape)), niter=iterations)
            order = s.argsort(descending=True)[:rank]
            u, s, vh = u[:, order], s[order], v[:, order].T
    root = s.clamp_min(0).sqrt()
    # FP32 masters during optimization; forward casts them to the base dtype.
    module.A[loop].data = (root[:, None] * vh).float()
    module.B[loop].data = (u * root[None, :]).float()
    module.A[loop].requires_grad_(trainable)
    module.B[loop].requires_grad_(trainable)
    module._inference_cache = None
    if logger is not None:
        values = s.tolist()
        sampled = ", ".join(f"s{i + 1}={values[i]:.6g}" for i in range(0, (len(values) + 1) // 2, 4))
        median = (values[(len(values) - 1) // 2] + values[len(values) // 2]) / 2
        logger.info("[LowRank] %s init_rank=%d singular_values=[%s] median=%.6g",
                    label, rank, sampled, median)


def reconstruct(model, teacher, config, device, logger):
    options = config.method["optimization"]
    samples = teacher.manifest["num_samples"]
    num_blocks = len(model.model.layers)
    logged_blocks = {0, num_blocks // 2, num_blocks - 1}
    history = []
    with tempfile.TemporaryDirectory(prefix=".student-", dir=teacher.directory.parent) as temporary:
        state_dir = Path(temporary)
        for sample in range(samples):
            save_file({"hidden": teacher.read(0, 0, sample)["input"]}, state_dir / f"{sample}.safetensors")
        for loop in range(config.num_loops):
            for block_id, block in enumerate(model.model.layers):
                model.requires_grad_(False)
                originals = teacher.weights(block_id)
                linears = dict(iter_linears(block))
                parameters = []
                for index, (name, module) in enumerate(linears.items()):
                    initialize_increment(module, originals[name], loop,
                                         seed=config.calibration["seed"] + loop * 1009 + block_id * 101 + index,
                                         iterations=options["svd_iterations"], trainable=options["enabled"],
                                         logger=logger if block_id in logged_blocks else None,
                                         label=f"loop={loop + 1}/{config.num_loops} "
                                         f"block={block_id + 1}/{num_blocks} module={name}")
                    parameters.extend((module.A[loop], module.B[loop]))
                losses = []
                if options["enabled"]:
                    optimizer = torch.optim.Adam(parameters, lr=options["learning_rate"])
                    generator = torch.Generator().manual_seed(config.calibration["seed"] + loop * 1009 + block_id)
                    order = torch.randperm(samples, generator=generator).tolist()
                    batch_size = min(options["batch_size"], samples)
                    for step in range(options["steps"]):
                        indices = [order[(step * batch_size + offset) % samples] for offset in range(batch_size)]
                        hidden = torch.cat([load_file(state_dir / f"{i}.safetensors")["hidden"] for i in indices]).to(device)
                        target = torch.cat([teacher.read(block_id, loop, i)["output"] for i in indices]).to(device)
                        prediction = run_block(model, block, hidden, loop)
                        # Equal-loop training budgets; normalize each sample's energy.
                        error = (prediction.float() - target.float()).square().flatten(1).mean(1)
                        energy = target.float().square().flatten(1).mean(1).clamp_min(1e-8)
                        loss = (error / energy).mean()
                        if not torch.isfinite(loss):
                            raise FloatingPointError(f"non-finite reconstruction loss at loop={loop}, block={block_id}")
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(parameters, options["gradient_clip"])
                        optimizer.step()
                        losses.append(float(loss.detach()))
                        logger.info("[LowRank] loop=%d/%d block=%d/%d step=%d/%d loss=%.6g",
                                    loop + 1, config.num_loops, block_id + 1, len(model.model.layers),
                                    step + 1, options["steps"], losses[-1])
                    del optimizer
                with torch.no_grad():
                    for module in linears.values():
                        for bank in (module.A, module.B):
                            bank[loop].requires_grad_(False)
                            bank[loop].data = bank[loop].to(module.weight.dtype)
                    for sample in range(samples):
                        path = state_dir / f"{sample}.safetensors"
                        hidden = load_file(path)["hidden"].to(device)
                        output = run_block(model, block, hidden, loop)
                        save_file({"hidden": output.cpu().contiguous()}, path)
                record = {"loop": loop, "block": block_id, "optimization_enabled": options["enabled"],
                          "steps": len(losses), "initial_loss": losses[0] if losses else None,
                          "final_loss": losses[-1] if losses else None}
                history.append(record)
                if losses:
                    logger.info("[LowRank] loop=%d/%d block=%d/%d loss=%.6g -> %.6g",
                                loop + 1, config.num_loops, block_id + 1, len(model.model.layers), losses[0], losses[-1])
                else:
                    logger.info("[LowRank] loop=%d/%d block=%d/%d initialized; optimization disabled",
                                loop + 1, config.num_loops, block_id + 1, len(model.model.layers))
            with torch.no_grad():
                for sample in range(samples):
                    path = state_dir / f"{sample}.safetensors"
                    hidden = load_file(path)["hidden"].to(device)
                    save_file({"hidden": model.model.norm(hidden).cpu().contiguous()}, path)
    model.requires_grad_(False)
    return history
