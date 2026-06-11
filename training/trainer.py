"""
v1 lean trainer: step-based wave-to-wave autoencoder + discriminator.

No stem-pair logic, no mixing losses, no alpha sweeps, no HF hub, no legacy
epoch paths. Just: load config, build model + disc, iterate the train loader
step-by-step, periodically validate and checkpoint, log to console + TB.
"""

import json
import os
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm

from data.dataset import build_dataloaders
from models.autoencoder import Autoencoder
from models.discriminator import (
    CombinedDiscriminator,
    MultiPeriodDiscriminator,
    MultiScaleSTFTDiscriminator,
    discriminator_loss,
    feature_matching_loss,
    generator_loss,
)
from training.config import build_model_config, get_device, load_config


def _si_sdr(x_hat: torch.Tensor, x: torch.Tensor, eps: float = 1e-8) -> float:
    """Scale-invariant SDR (dB), per-batch mean."""
    x_hat = x_hat.float().reshape(x_hat.size(0), -1)
    x = x.float().reshape(x.size(0), -1)
    x = x - x.mean(dim=1, keepdim=True)
    x_hat = x_hat - x_hat.mean(dim=1, keepdim=True)
    alpha = (x_hat * x).sum(dim=1, keepdim=True) / (x.pow(2).sum(dim=1, keepdim=True) + eps)
    s_target = alpha * x
    e_noise = x_hat - s_target
    sisdr = 10.0 * torch.log10(
        (s_target.pow(2).sum(dim=1) + eps) / (e_noise.pow(2).sum(dim=1) + eps)
    )
    return float(sisdr.mean().item())


class Logger:
    """Tees messages to console and to a per-run text log."""

    def __init__(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        self._path = os.path.join(save_dir, "train.log")
        self._f = open(self._path, "a", encoding="utf-8")

    def info(self, msg: str):
        print(msg)
        self._f.write(msg + "\n")
        self._f.flush()

    def close(self):
        self._f.close()


class Trainer:
    def __init__(self, config_path: str, base_config_path: Optional[str] = None):
        cfg = load_config(config_path, base_config_path)
        self.cfg = cfg
        self.config_path = config_path

        data_cfg = cfg["data"]
        train_cfg = cfg["train"]
        model_cfg = cfg["model"]

        # Device
        self.device = get_device(train_cfg.get("gpu_index", 0))
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        # Run dir
        name = cfg.get("name", "run")
        self.save_dir = os.path.join(train_cfg.get("save_path", "./checkpoints"), name)
        os.makedirs(self.save_dir, exist_ok=True)
        self.logs = Logger(self.save_dir)
        self.logs.info(f"== {name} on {self.device} ==")
        if self.device.type == "cuda":
            self.logs.info(f"GPU: {torch.cuda.get_device_name(self.device)}")

        # Audio
        self.sample_rate = int(data_cfg["sample_rate"])
        self.chunk_seconds = float(data_cfg["chunk_seconds"])
        self.chunk_samples = int(self.sample_rate * self.chunk_seconds)

        # Model config (via shared builder)
        self.model_config = build_model_config(cfg)
        self.architecture = model_cfg.get("architecture", "wave")
        assert self.architecture == "wave", "v1 only supports wave-to-wave"

        # Data loaders
        self.batch_size = int(train_cfg["batch_size"])
        num_workers = int(train_cfg.get("num_workers", 4))
        (
            self.train_loader,
            self.val_loader,
            self.train_dataset,
            self.val_dataset,
            self.train_sampler,
        ) = build_dataloaders(
            chunks_dir=data_cfg["chunks_dir"],
            batch_size=self.batch_size,
            num_workers=num_workers,
            pin_memory=(self.device.type == "cuda"),
            persistent_workers=(num_workers > 0),
            prefetch_factor=train_cfg.get("prefetch_factor", 2) if num_workers > 0 else None,
            cache_size=int(train_cfg.get("dataset_cache_size", 8)),
        )
        self.logs.info(
            f"train: {len(self.train_dataset):,} chunks across "
            f"{len(self.train_dataset.files):,} files"
        )
        self.logs.info(
            f"val:   {len(self.val_dataset):,} chunks across "
            f"{len(self.val_dataset.files):,} files"
        )

        # Loss weights (read at loop time so live-reloads are possible)
        self.mrstft_weight = float(model_cfg.get("mrstft_weight", 1.0))
        self.mel_weight = float(model_cfg.get("mel_weight", 1.0))
        self.latent_l2_weight = float(model_cfg.get("latent_l2_weight", 0.0))

        # Discriminator
        self.use_disc = bool(model_cfg.get("use_discriminator", True))
        self.use_mpd = bool(model_cfg.get("use_mpd", True))
        self.disc_weight = float(model_cfg.get("disc_weight", 0.5))
        self.feat_match_weight = float(model_cfg.get("feat_match_weight", 2.0))
        self.disc_start_step = int(model_cfg.get("disc_start_step", 0))
        # v2: when true and decode_mix_weight > 0, the gen step also pushes
        # g(z̄) (the mixed-decode output) through the discriminator, adding
        # adversarial + feature-matching loss on the mixing path. Disc training
        # itself is unchanged — disc only sees (x_real, x_hat) updates.
        self.disc_on_mix = bool(model_cfg.get("disc_on_mix", False))

        # Optimizer / schedule (step-based)
        self.lr = float(train_cfg["learning_rate"])
        self.disc_lr = float(train_cfg.get("disc_lr", 2e-4))
        self.weight_decay = float(train_cfg.get("weight_decay", 1e-3))
        self.betas = tuple(train_cfg.get("optimizer_betas", [0.8, 0.99]))
        self.max_steps = int(train_cfg["max_steps"])
        self.warmup_steps = int(train_cfg.get("warmup_steps", 0))
        self.save_every_steps = int(train_cfg.get("save_every_steps", 10000))
        self.log_every_steps = int(train_cfg.get("log_every_steps", 100))
        self.grad_clip = float(train_cfg.get("grad_clip", 1.0))
        self.num_val_batches = train_cfg.get("num_val_batches", None)  # None -> all
        # Gradient accumulation: effective_batch = batch_size * grad_accum_steps.
        # Keeps optimizer stepping at the "logical" batch while halving peak
        # memory per forward/backward — required once disc activates if the
        # combined gen+disc memory exceeds GPU capacity at the logical batch.
        self.grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))
        self._accum_counter = 0  # how many micro-batches we've accumulated in the current window

        # v2: optionally freeze encoder (mechanism-isolation ablation —
        # tests whether L_dec works via decoder smoothness or via encoder
        # gradients flowing through z̄). Set in train config.
        self.freeze_encoder_flag = bool(train_cfg.get("freeze_encoder", False))

        # AMP precision. v1 defaults to fp32 for stability; "bf16" restores
        # the mixed-precision path that caused the single-step regression at
        # step 4700 in the first bf16 run. No GradScaler needed either way —
        # bf16 has a wide enough dynamic range that fp16-style scaling isn't
        # required.
        precision = str(train_cfg.get("precision", "fp32")).lower()
        self.use_amp = (precision == "bf16") and (self.device.type == "cuda")
        self.amp_dtype = torch.bfloat16 if self.use_amp else torch.float32
        self.logs.info(f"precision: {precision} (amp={'on' if self.use_amp else 'off'})")

        # Val sample dump (writes .wav files to <save_dir>/samples/step_<N>/)
        self.log_audio = bool(train_cfg.get("log_val_audio", True))
        self.num_test_samples = int(train_cfg.get("num_test_samples", 4))

        self.model: Optional[Autoencoder] = None
        self.disc: Optional[torch.nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.disc_optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler = None

    # ----------------------------------------------------------------------

    def build_model(self):
        self.model = Autoencoder(**self.model_config).to(self.device)

        # Optional encoder freeze (v2 mechanism-isolation ablation).
        if self.freeze_encoder_flag:
            for p in self.model.encoder.parameters():
                p.requires_grad = False
            n_frozen = sum(p.numel() for p in self.model.encoder.parameters())
            n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.logs.info(
                f"freeze_encoder=True: frozen {n_frozen:,} encoder params; "
                f"{n_trainable:,} params remain trainable"
            )

        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )

        # Cosine decay to 10% of peak over max_steps, with linear warmup.
        if self.warmup_steps > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1e-3, end_factor=1.0,
                total_iters=self.warmup_steps,
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, self.max_steps - self.warmup_steps),
                eta_min=self.lr * 0.1,
            )
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup, cosine],
                milestones=[self.warmup_steps],
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(1, self.max_steps), eta_min=self.lr * 0.1,
            )

        total = sum(p.numel() for p in self.model.parameters())
        self.logs.info(f"model params: {total:,}")

        if self.use_disc:
            msstftd = MultiScaleSTFTDiscriminator().to(self.device)
            if self.use_mpd:
                mpd = MultiPeriodDiscriminator().to(self.device)
                self.disc = CombinedDiscriminator(msstftd, mpd).to(self.device)
            else:
                self.disc = msstftd
            self.disc_optimizer = torch.optim.AdamW(
                self.disc.parameters(),
                lr=self.disc_lr,
                betas=(0.5, 0.9),
                weight_decay=0.0,
            )
            disc_params = self.disc.num_parameters() if hasattr(self.disc, "num_parameters") \
                else sum(p.numel() for p in self.disc.parameters())
            self.logs.info(f"disc params: {disc_params:,} (starts at step {self.disc_start_step})")

    # ----------------------------------------------------------------------

    def fit(self, start_step: int = 0, best_val_loss: float = float("inf")):
        assert self.model is not None, "call build_model() first"

        # Persist config next to checkpoints for reproducibility
        import shutil
        dst = os.path.join(self.save_dir, "config.yaml")
        if not os.path.exists(dst):
            shutil.copy2(self.config_path, dst)

        self.logs.info(
            f"fit: max_steps={self.max_steps}, warmup={self.warmup_steps}, "
            f"save_every={self.save_every_steps}, log_every={self.log_every_steps}, "
            f"batch={self.batch_size}, lr={self.lr:.2e}"
        )

        self.model.train()
        if self.disc is not None:
            self.disc.train()

        global_step = start_step
        epoch_ctr = 0
        running: Dict[str, list] = {}

        pbar = tqdm(total=self.max_steps, initial=start_step, desc="steps")
        t0 = time.time()

        while global_step < self.max_steps:
            self.train_sampler.set_epoch(epoch_ctr)

            for batch in self.train_loader:
                if global_step >= self.max_steps:
                    break

                metrics = self._train_step(batch, global_step)

                # Non-boundary micro-batches return None; skip logging/global_step
                # until we complete the accumulation window.
                if metrics is None:
                    continue

                for k, v in metrics.items():
                    running.setdefault(k, []).append(v)

                global_step += 1
                pbar.update(1)

                # Console / TB summary
                if global_step % self.log_every_steps == 0:
                    self._flush_running(running, global_step, t0)
                    running = {}
                    t0 = time.time()

                # Validate + checkpoint
                if global_step % self.save_every_steps == 0:
                    val_metrics = self._validate(global_step)
                    best_val_loss = self._maybe_save(global_step, val_metrics, best_val_loss)
                    self.model.train()
                    if self.disc is not None:
                        self.disc.train()

            epoch_ctr += 1

        pbar.close()
        # Final validation + checkpoint at end of training
        val_metrics = self._validate(global_step)
        best_val_loss = self._maybe_save(global_step, val_metrics, best_val_loss, force_final=True)

        self.logs.info(f"done. final step {global_step}, best_val {best_val_loss:.4f}")
        self.logs.close()
        return best_val_loss

    # ----------------------------------------------------------------------

    def _train_step(self, batch: Dict, global_step: int) -> Optional[Dict[str, float]]:
        """Run one micro-batch of training. Returns metrics dict on an
        accumulation boundary (so the outer loop should treat that as a
        completed optimizer step), or None while still accumulating."""
        x_wave = batch["x_wave"].to(self.device, non_blocking=True)
        if x_wave.dim() == 2:
            x_wave = x_wave.unsqueeze(1)

        disc_active = self.disc is not None and global_step >= self.disc_start_step
        is_first = (self._accum_counter == 0)
        is_boundary = (self._accum_counter + 1 >= self.grad_accum_steps)
        scale = 1.0 / self.grad_accum_steps
        out: Dict[str, float] = {}

        # --- Discriminator micro-step ---
        if disc_active:
            if is_first:
                self.disc_optimizer.zero_grad(set_to_none=True)
            self.disc.requires_grad_(True)
            tgt = self.model.decoder.target_length
            x_real = x_wave[:, :, :tgt] if x_wave.size(2) > tgt else x_wave

            # Generate fake samples without tracking generator gradients.
            # When disc_on_mix is on AND L_dec is active, also generate the
            # mixing-path samples so the disc trains symmetrically with what
            # the gen step asks of it (otherwise disc has bias: gen step's
            # gen_adv_mix targets a "real" the disc never had as a positive).
            do_mix_d = (self.disc_on_mix
                        and self.model.decode_mix_weight > 0.0
                        and x_wave.size(0) >= 2)
            with torch.no_grad():
                with autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                    z = self.model.encoder(x_wave)
                    x_hat_d, _ = self.model.decoder(z)
                    if do_mix_d:
                        B = x_wave.size(0)
                        perm_d = self.model._random_pair_perm(B, x_wave.device)
                        a_d = torch.rand(B, 1, 1, device=x_wave.device)
                        x_mix_real_d = (a_d * x_wave + (1 - a_d) * x_wave[perm_d])
                        if x_mix_real_d.size(2) > tgt:
                            x_mix_real_d = x_mix_real_d[:, :, :tgt]
                        z_interp_d = a_d * z + (1 - a_d) * z[perm_d]
                        x_interp_d, _ = self.model.decoder(z_interp_d)

            with autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                real_logits, _ = self.disc(x_real)
                fake_logits, _ = self.disc(x_hat_d.detach())
                d_loss = discriminator_loss(real_logits, fake_logits)
                d_total = d_loss
                if do_mix_d:
                    real_mix_logits, _ = self.disc(x_mix_real_d)
                    fake_mix_logits, _ = self.disc(x_interp_d.detach())
                    d_loss_mix = discriminator_loss(real_mix_logits, fake_mix_logits)
                    d_total = d_total + d_loss_mix

            (d_total * scale).backward()
            if is_boundary:
                self.disc_optimizer.step()
            out["disc/loss"] = float(d_loss.item())
            if do_mix_d:
                out["disc/loss_mix"] = float(d_loss_mix.item())
                del z_interp_d, x_interp_d, x_mix_real_d, real_mix_logits, fake_mix_logits, d_loss_mix
            del z, x_hat_d, real_logits, fake_logits, d_loss, d_total
            self.disc.requires_grad_(False)

        # --- Generator micro-step ---
        if is_first:
            self.optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
            total, comps, x_hat, mix_aux = self.model(x_wave)

            if disc_active:
                tgt = self.model.decoder.target_length
                x_real = x_wave[:, :, :tgt] if x_wave.size(2) > tgt else x_wave
                fake_logits, fake_feats = self.disc(x_hat)
                with torch.no_grad():
                    _, real_feats = self.disc(x_real)
                gen_adv = generator_loss(fake_logits)
                fm = feature_matching_loss(real_feats, fake_feats)
                total = total + self.disc_weight * gen_adv + self.feat_match_weight * fm
                out["disc/gen_adv"] = float(gen_adv.item())
                out["disc/feat_match"] = float(fm.item())

                # v2: optional adversarial supervision on the mixed-decode
                # path g(z̄). Same disc weights as the regular recon path; the
                # disc isn't retrained on mix-specific data, it just judges
                # any audio coming out of the generator.
                if self.disc_on_mix and mix_aux is not None:
                    x_interp = mix_aux["x_interp"]
                    x_mix_real = mix_aux["x_mix_wave"]
                    if x_mix_real.size(2) > tgt:
                        x_mix_real = x_mix_real[:, :, :tgt]
                    fake_logits_m, fake_feats_m = self.disc(x_interp)
                    with torch.no_grad():
                        _, real_feats_m = self.disc(x_mix_real)
                    gen_adv_m = generator_loss(fake_logits_m)
                    fm_m = feature_matching_loss(real_feats_m, fake_feats_m)
                    total = (total
                             + self.disc_weight * gen_adv_m
                             + self.feat_match_weight * fm_m)
                    out["disc/gen_adv_mix"] = float(gen_adv_m.item())
                    out["disc/feat_match_mix"] = float(fm_m.item())

        (total * scale).backward()

        if not is_boundary:
            self._accum_counter += 1
            return None

        # Accumulation boundary — clip, step, advance scheduler.
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        self._accum_counter = 0

        out["loss"] = float(total.item())
        out["recon/mrstft"] = float(comps.get("ReconSingle/MRSTFT", 0.0))
        out["recon/mel"] = float(comps.get("ReconSingle/Mel", 0.0))
        # v2: surface mixing-loss components so the train log shows whether
        # L_dec / L_enc are firing without manually decomposing total.
        out["mix/decode"] = float(comps.get("DecodeMix/L1", 0.0))
        out["mix/latent"] = float(comps.get("LatentMix/MSE", 0.0))
        out["latent/std"] = float(comps.get("Latent/std", 0.0))
        out["latent/absmax"] = float(comps.get("Latent/absmax", 0.0))
        out["grad_norm"] = float(grad_norm.item())
        out["lr"] = float(self.optimizer.param_groups[0]["lr"])
        return out

    # ----------------------------------------------------------------------

    @torch.no_grad()
    def _validate(self, global_step: int) -> Dict[str, float]:
        self.model.eval()
        if self.disc is not None:
            self.disc.eval()

        agg: Dict[str, list] = {}
        by_source: Dict[str, list] = {}

        for i, batch in enumerate(tqdm(self.val_loader, desc=f"val@{global_step}", leave=False)):
            if self.num_val_batches is not None and i >= self.num_val_batches:
                break
            x_wave = batch["x_wave"].to(self.device, non_blocking=True)
            if x_wave.dim() == 2:
                x_wave = x_wave.unsqueeze(1)

            with autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                total, comps, x_hat, _ = self.model(x_wave)

            tgt = self.model.decoder.target_length
            x_ref = x_wave[:, :, :tgt] if x_wave.size(2) > tgt else x_wave
            sisdr = _si_sdr(x_hat, x_ref)

            agg.setdefault("loss", []).append(float(total.item()))
            agg.setdefault("recon/mrstft", []).append(float(comps.get("ReconSingle/MRSTFT", 0.0)))
            agg.setdefault("recon/mel", []).append(float(comps.get("ReconSingle/Mel", 0.0)))
            agg.setdefault("si_sdr", []).append(sisdr)

            # Per-source breakdown uses the "source" field in the batch (list of strings).
            sources = batch.get("source", None)
            if sources is not None:
                # Default collate turns list-of-str into a list. We want per-sample.
                for s_i, src in enumerate(sources):
                    by_source.setdefault(f"si_sdr/{src}", []).append(
                        _si_sdr(x_hat[s_i:s_i+1], x_ref[s_i:s_i+1])
                    )

        means = {k: float(np.mean(v)) for k, v in agg.items() if v}
        for k, v in by_source.items():
            means[k] = float(np.mean(v))

        self.logs.info(
            f"VAL  step {global_step}: loss={means.get('loss', 0):.4f}  "
            f"MR-STFT={means.get('recon/mrstft', 0):.4f}  "
            f"Mel={means.get('recon/mel', 0):.4f}  "
            f"SI-SDR={means.get('si_sdr', 0):.2f} dB"
        )
        src_lines = [f"{k.split('/')[-1]}={v:.2f}" for k, v in means.items() if k.startswith("si_sdr/")]
        if src_lines:
            self.logs.info("VAL  per-source SI-SDR: " + ", ".join(src_lines))

        if self.log_audio and self.num_test_samples > 0:
            self._save_val_samples(global_step)

        return means

    # ----------------------------------------------------------------------

    @torch.no_grad()
    def _save_val_samples(self, global_step: int):
        """Write ref.wav and hat.wav pairs to <save_dir>/samples/step_<N>/.

        Selects samples spanning every source in the val set so a given dump
        covers musdb / maestro / fma rather than only the alphabetically-first
        source. Indices are deterministic, so the same tracks are dumped at
        every checkpoint — A/B comparison across training is possible by
        playing the same filename from different step_<N>/ folders.
        """
        try:
            import soundfile as sf
        except ImportError:
            self.logs.info("WARN: soundfile not installed; skipping val audio dump")
            return

        out_dir = os.path.join(self.save_dir, "samples", f"step_{global_step:07d}")
        os.makedirs(out_dir, exist_ok=True)

        # Group val files by source, pick deterministic representatives.
        files_by_src: Dict[str, list] = {}
        for f in self.val_dataset.files:
            files_by_src.setdefault(f["source"], []).append(f)

        sources = sorted(files_by_src.keys())
        per_source = max(1, self.num_test_samples // max(1, len(sources)))
        leftover = max(0, self.num_test_samples - per_source * len(sources))

        # Build the list of (chunk_idx, source) to dump.
        picks: List[tuple] = []
        for s_i, source in enumerate(sources):
            n_for_this = per_source + (1 if s_i < leftover else 0)
            files = files_by_src[source]
            # Spread picks across the source's files (first, mid, last, ...).
            for k in range(min(n_for_this, len(files))):
                f = files[k * (len(files) // max(1, n_for_this))]
                picks.append((f["start"], source))  # first chunk of that file

        for idx, src in picks:
            item = self.val_dataset[idx]
            x = item["x_wave"].unsqueeze(0).to(self.device)
            with autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                _, _, x_hat, _ = self.model(x)
            tgt = self.model.decoder.target_length
            ref = x[:, :, :tgt].float().cpu().squeeze().numpy()
            hat = x_hat.float().cpu().squeeze().numpy()
            sf.write(os.path.join(out_dir, f"{src}_{idx:08d}_ref.wav"), ref, self.sample_rate)
            sf.write(os.path.join(out_dir, f"{src}_{idx:08d}_hat.wav"), hat, self.sample_rate)

    # ----------------------------------------------------------------------

    def _flush_running(self, running: Dict[str, list], step: int, t0: float):
        if not running:
            return
        dt = max(1e-6, time.time() - t0)
        steps_per_s = self.log_every_steps / dt

        means = {k: float(np.mean(v)) for k, v in running.items() if v}
        lr = means.get("lr", self.optimizer.param_groups[0]["lr"])

        line_parts = [
            f"step {step:>7d}",
            f"loss={means.get('loss', 0):.4f}",
            f"MR={means.get('recon/mrstft', 0):.4f}",
            f"Mel={means.get('recon/mel', 0):.4f}",
            f"z|std|={means.get('latent/std', 0):.3f}",
            f"gnorm={means.get('grad_norm', 0):.2f}",
            f"lr={lr:.2e}",
            f"sps={steps_per_s:.2f}",
        ]
        # Surface mixing-loss components (only when actually firing)
        if means.get("mix/decode", 0.0) > 0.0:
            line_parts.append(f"L_dec={means['mix/decode']:.3f}")
        if means.get("mix/latent", 0.0) > 0.0:
            line_parts.append(f"L_enc={means['mix/latent']:.4f}")
        if "disc/loss" in means:
            line_parts += [
                f"D={means['disc/loss']:.3f}",
                f"Gadv={means.get('disc/gen_adv', 0):.3f}",
                f"FM={means.get('disc/feat_match', 0):.3f}",
            ]
        if "disc/gen_adv_mix" in means:
            line_parts += [
                f"Gadv_m={means['disc/gen_adv_mix']:.3f}",
                f"FM_m={means.get('disc/feat_match_mix', 0):.3f}",
            ]
        self.logs.info("  ".join(line_parts))

        # Append training-step metrics to JSONL (one line per flush) so runs
        # are plottable without TB.
        with open(os.path.join(self.save_dir, "train_log.jsonl"), "a") as f:
            f.write(json.dumps({"step": step, "steps_per_sec": steps_per_s, **means}) + "\n")

    # ----------------------------------------------------------------------

    def _maybe_save(
        self, step: int, val_metrics: Dict[str, float],
        best_val_loss: float, force_final: bool = False,
    ) -> float:
        val_loss = val_metrics.get("loss", float("inf"))

        ckpt = {
            "global_step": step,
            "best_val_loss": best_val_loss,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "model_config": self.model_config,
            "val_metrics": val_metrics,
            "config": self.cfg,
        }
        if self.disc is not None:
            ckpt["disc"] = self.disc.state_dict()
            ckpt["disc_optimizer"] = self.disc_optimizer.state_dict()

        def _write(path: str):
            tmp = path + ".tmp"
            torch.save(ckpt, tmp)
            if os.path.exists(path):
                os.remove(path)
            os.rename(tmp, path)

        # Update best_val_loss FIRST so latest.pth records the fresh value.
        # (Earlier versions wrote latest.pth before this check, causing
        # best_val_loss in latest.pth to lag by one val cycle and read as inf
        # on the first save. Resumes from latest then kept overwriting best.pth.)
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        ckpt["best_val_loss"] = best_val_loss

        _write(os.path.join(self.save_dir, "latest.pth"))
        if is_best:
            _write(os.path.join(self.save_dir, "best.pth"))
            self.logs.info(f"     new best (val_loss={val_loss:.4f}) -> saved best.pth")

        if force_final:
            _write(os.path.join(self.save_dir, f"step_{step}.pth"))

        # Dump val metrics as JSON for quick inspection / plotting
        with open(os.path.join(self.save_dir, "val_log.jsonl"), "a") as f:
            f.write(json.dumps({"step": step, "best": is_best, **val_metrics}) + "\n")

        return best_val_loss

    # ----------------------------------------------------------------------

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        if self.disc is not None and "disc" in ckpt:
            self.disc.load_state_dict(ckpt["disc"])
            self.disc_optimizer.load_state_dict(ckpt["disc_optimizer"])
        step = int(ckpt.get("global_step", 0))
        best = float(ckpt.get("best_val_loss", float("inf")))
        self.logs.info(f"resumed from {path}: step={step}, best_val={best:.4f}")
        return step, best
