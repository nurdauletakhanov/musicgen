"""Entry point for v1+ training.

Usage:
  # Fresh run from scratch
  python -m training.train --config configs/experiments/v1/v1.1.yaml

  # Resume an interrupted run (loads model + optimizer + scheduler + step)
  python -m training.train --config ... --resume checkpoints/v1.1/latest.pth

  # Warm-start a NEW run from a previously-trained model (model weights only;
  # fresh optimizer state, scheduler, and step counter — used for v2 fine-tunes)
  python -m training.train --config configs/experiments/v2/v2.1_decmix.yaml \
      --warm-start checkpoints/v1.1/best.pth
"""

import argparse

import torch

from training.trainer import Trainer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--base-config", type=str, default=None)
    p.add_argument("--resume", type=str, default=None,
                   help="Resume training: load model + optim + sched + step")
    p.add_argument("--warm-start", type=str, default=None,
                   help="Warm-start a new run: load model (and disc) weights "
                        "only; fresh optim/sched/step. Mutually exclusive with "
                        "--resume.")
    args = p.parse_args()

    if args.resume and args.warm_start:
        raise SystemExit("--resume and --warm-start are mutually exclusive")

    trainer = Trainer(args.config, base_config_path=args.base_config)
    trainer.build_model()

    if args.resume:
        start_step, best = trainer.load_checkpoint(args.resume)
        trainer.fit(start_step=start_step, best_val_loss=best)
    elif args.warm_start:
        ck = torch.load(args.warm_start, map_location=trainer.device, weights_only=False)
        trainer.model.load_state_dict(ck["model"])
        if trainer.disc is not None and "disc" in ck:
            trainer.disc.load_state_dict(ck["disc"])
            trainer.logs.info("warm-start: loaded model + disc weights")
        else:
            trainer.logs.info("warm-start: loaded model weights (no disc in checkpoint)")
        trainer.logs.info(
            f"warm-start: starting at step 0 with fresh optimizer+scheduler "
            f"(weights from {args.warm_start})"
        )
        trainer.fit()  # fresh step=0, fresh optim/sched
    else:
        trainer.fit()


if __name__ == "__main__":
    main()
