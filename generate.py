import argparse
from pathlib import Path

import zebranoise


def main(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    zebranoise.zebra_noise(
        filename=args.output_dir / "zebra_noise",
        xsize=1920,
        ysize=1080,
        tdur=args.duration,
        fps=args.fps,
        seed=args.seed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--duration", type=int, default=5 * 60, help="duration in seconds."
    )
    parser.add_argument("--fps", type=int, default=30, help="frame rate of the video.")
    parser.add_argument("--seed", type=int, default=1234)
    main(parser.parse_args())
