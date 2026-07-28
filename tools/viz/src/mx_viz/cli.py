from __future__ import annotations

import argparse

from mx_viz import io as viz_io
from mx_viz import sweeps

# Only the sweep-comparison plot is exposed as a CLI verb: it's the one case that
# already has a natural persisted artifact (a results JSON file). Field/loss-curve
# plots need a live trained model or loss history -- this repo doesn't checkpoint
# models to disk, so those stay library-only functions called from a research
# script/session rather than CLI verbs (see tools/README.md).


def cmd_sweep(args: argparse.Namespace) -> int:
    payload = viz_io.load_results(args.results)
    title = payload.get("metadata", {}).get("title")
    fig = sweeps.plot_sweep_comparison(payload["results"], kind=args.kind, title=title)
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mx-viz", description="Repo-wide experiment visualization CLI"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sweep_p = sub.add_parser(
        "sweep", help="Plot a sweep-comparison chart from a results JSON file"
    )
    sweep_p.add_argument("results", help="Path to a results JSON file (see mx_viz.io.save_results)")
    sweep_p.add_argument("--out", required=True, help="Output image path (e.g. sweep.png)")
    sweep_p.add_argument("--kind", choices=["box", "bar"], default="box")
    sweep_p.set_defaults(func=cmd_sweep)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
