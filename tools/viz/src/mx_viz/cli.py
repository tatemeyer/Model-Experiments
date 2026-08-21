from __future__ import annotations

import argparse

from mx_viz import feed as viz_feed
from mx_viz import io as viz_io
from mx_viz import sweeps

# The sweep-comparison and field verbs are exposed as CLI commands because both now have a
# natural persisted artifact (a results JSON file, or a field .npz written by
# mx_viz.io.save_field_artifact / em_piml.train's grid-persistence wiring). Loss-curve plots
# still have no persisted artifact to load (no loss-history checkpointing exists), so that one
# stays a library-only function called from a research script/session (see tools/README.md).


def cmd_sweep(args: argparse.Namespace) -> int:
    payload = viz_io.load_results(args.results)
    title = payload.get("metadata", {}).get("title")
    fig = sweeps.plot_sweep_comparison(payload["results"], kind=args.kind, title=title)
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")
    return 0


def cmd_field(args: argparse.Namespace) -> int:
    data = viz_io.load_field_artifact(args.artifact)
    viz_io.validate_field_artifact(data)
    print(
        f"Loaded {args.artifact}: grid shape {data['grid_x'].shape}, "
        f"schema_version={int(data['schema_version'])}"
    )
    return 0


def cmd_feed(args: argparse.Namespace) -> int:
    count = viz_feed.write_feed(args.results, args.out)
    print(f"Wrote {args.out} ({count} records)")
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

    field_p = sub.add_parser(
        "field", help="Validate and summarize a field artifact (see mx_viz.io.save_field_artifact)"
    )
    field_p.add_argument("artifact", help="Path to a field .npz artifact")
    field_p.set_defaults(func=cmd_field)

    feed_p = sub.add_parser(
        "feed",
        help="Publish a tidy results.csv as the JSONL metrics feed parallax.yaml declares",
    )
    feed_p.add_argument("results", help="Path to a long-format results.csv")
    feed_p.add_argument("--out", required=True, help="Output .jsonl path (beside the CSV)")
    feed_p.set_defaults(func=cmd_feed)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
