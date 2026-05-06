import argparse
import json
import logging
from pathlib import Path

import numpy as np

from app.dataset_defaults import apply_dataset_cli_defaults
from app.logging_setup import setup_logging
from config import load_core_config, load_eval_config, load_retrieval_config
from core.contracts import DatasetPaths, RetrievalRequest
from core.ema import EMAConfig
from core.registry import build_core
from data.synthetic import generate_synthetic_dataset
from eval import EvaluationRunner, retrieval_request_from_eval_dict

logger = logging.getLogger(__name__)


def _dataset_from_args(args: argparse.Namespace) -> DatasetPaths:
    return DatasetPaths(
        users_path=args.users_path,
        items_path=args.items_path,
        interactions_path=args.interactions_path,
        item_embeddings_npz=getattr(args, "item_embeddings_npz", None),
        interaction_paths=getattr(args, "_interaction_paths", None),
    )


def _build_core_with_config(args: argparse.Namespace):
    cfg = load_core_config(args.config)
    core_name = args.core or cfg["core_name"]
    params = dict(cfg.get("params", {}))
    if core_name == "ema":
        params = {"config": EMAConfig(**params)}
    return core_name, build_core(core_name, **params), cfg


def cmd_generate(args: argparse.Namespace) -> None:
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    generated = generate_synthetic_dataset(
        output_dir=args.output_dir,
        n_users=args.n_users,
        n_items=args.n_items,
        embedding_dim=args.embedding_dim,
        interactions_per_user=args.interactions_per_user,
        seed=args.seed,
    )
    print(json.dumps(generated, indent=2))


def cmd_train(args: argparse.Namespace) -> None:
    core_name, core, cfg = _build_core_with_config(args)
    summary = core.fit_offline(_dataset_from_args(args), config=cfg)
    core.save(args.model_dir)
    print(json.dumps({"core": core_name, "n_users": summary.n_users, "n_items": summary.n_items}, indent=2))


def cmd_update(args: argparse.Namespace) -> None:
    core_name, core, cfg = _build_core_with_config(args)
    if Path(args.model_dir).exists():
        try:
            core.load(args.model_dir)
        except FileNotFoundError:
            logger.warning("No existing model found in %s; update will run as fresh fit", args.model_dir)
    summary = core.update_incremental(_dataset_from_args(args), config=cfg)
    core.save(args.model_dir)
    print(json.dumps({"core": core_name, "n_users": summary.n_users, "n_items": summary.n_items}, indent=2))


def cmd_recommend(args: argparse.Namespace) -> None:
    core_name, core, _ = _build_core_with_config(args)
    core.load(args.model_dir)
    user_ids = [int(x) for x in args.user_ids.split(",")]
    retrieval_cfg = load_retrieval_config(args.retrieval_config)
    request = RetrievalRequest(
        mode=retrieval_cfg["mode"],
        ann_enabled=retrieval_cfg.get("ann_enabled", False),
        subset_size=retrieval_cfg.get("subset_size"),
        random_seed=retrieval_cfg.get("random_seed", 42),
        extra=retrieval_cfg.get("extra", {}),
    )
    recs = core.recommend(user_ids, k=args.k, retrieval_request=request)
    payload = []
    for result in recs.results:
        payload.append(
            {
                "user_id": result.user_id,
                "item_ids": result.item_ids.astype(np.uint64).tolist(),
                "scores": result.scores.astype(np.float32).tolist(),
            }
        )
    print(json.dumps({"core": core_name, "results": payload}, indent=2))


def cmd_evaluate(args: argparse.Namespace) -> None:
    core_name, core, core_cfg = _build_core_with_config(args)
    if core_name in {"ema", "gcn", "online_kmeans", "two_tower"}:
        if Path(args.model_dir).exists():
            try:
                core.load(args.model_dir)
            except FileNotFoundError:
                core.fit_offline(_dataset_from_args(args), config=core_cfg)
        else:
            core.fit_offline(_dataset_from_args(args), config=core_cfg)
    else:
        core.fit_offline(_dataset_from_args(args), config=core_cfg)

    eval_cfg = load_eval_config(args.eval_config)
    runner = EvaluationRunner(
        k=eval_cfg["k"],
        positive_policy=eval_cfg["positive_policy"],
        metrics_output_dir=eval_cfg["metrics_output_dir"],
        retrieval_request=retrieval_request_from_eval_dict(eval_cfg.get("retrieval")),
        recommend_chunk_size=eval_cfg["recommend_chunk_size"],
        max_eval_users=eval_cfg["max_eval_users"],
    )
    result = runner.evaluate(core, _dataset_from_args(args), core_name=core_name)
    metrics_path = runner.save_result(
        core_name=core_name,
        result=result,
        config={"core": core_cfg, "eval": eval_cfg},
        data_slice=Path(_dataset_from_args(args).interactions_path).name,
    )
    print(
        json.dumps(
            {
                "core": core_name,
                "precision_at_k": result.precision_at_k,
                "recall_at_k": result.recall_at_k,
                "map_at_k": result.map_at_k,
                "ndcg_at_k": result.ndcg_at_k,
                "coverage_at_k": result.coverage_at_k,
                "n_eval_users": result.n_eval_users,
                "metrics_path": metrics_path,
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recsys",
        epilog="Use JSON configs: --config, --retrieval-config, --eval-config",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG-level logs on stderr")
    parser.add_argument("-q", "--quiet", action="store_true", help="fewer logs and no tqdm bars")
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="override log level (default: INFO, or DEBUG with -v, WARNING with -q)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generate-synth")
    p_gen.add_argument("--output-dir", default="./datasets/synthetic")
    p_gen.add_argument("--n-users", type=int, default=2000)
    p_gen.add_argument("--n-items", type=int, default=1000)
    p_gen.add_argument("--embedding-dim", type=int, default=32)
    p_gen.add_argument("--interactions-per-user", type=int, default=20)
    p_gen.add_argument("--seed", type=int, default=42)
    p_gen.set_defaults(func=cmd_generate)

    for name, handler in [("train", cmd_train), ("update", cmd_update), ("evaluate", cmd_evaluate)]:
        p = sub.add_parser(name)
        p.add_argument(
            "--data-root",
            default=None,
            help="Root folder with metadata/ and subsamples/up0.001_ip0.001/ (default: <repo>/datasets/VK-LSVD)",
        )
        p.add_argument("--users-path", default=None, help="Default: <data-root>/metadata/users_metadata.parquet")
        p.add_argument("--items-path", default=None, help="Default: <data-root>/metadata/items_metadata.parquet")
        p.add_argument(
            "--item-embeddings-npz",
            default=None,
            help="Default: <data-root>/metadata/item_embeddings.npz",
        )
        p.add_argument(
            "--interactions-path",
            default=None,
            help="Train/update: single parquet replaces all train weeks. Evaluate: holdout parquet.",
        )
        p.add_argument(
            "--small-sample",
            action="store_true",
            help="Use only train/week_00..05 as a compact shared dataset split for faster iteration.",
        )
        p.add_argument(
            "--core",
            choices=["ema", "popular", "random", "gcn", "online_kmeans", "two_tower"],
            default=None,
        )
        p.add_argument("--config", required=True, help="Path to core JSON config")
        p.add_argument("--model-dir", default="./artifacts")
        if name == "evaluate":
            p.add_argument("--eval-config", required=True, help="Path to eval JSON config")
        p.set_defaults(func=handler)

    p_rec = sub.add_parser("recommend")
    p_rec.add_argument(
        "--core",
        choices=["ema", "popular", "random", "gcn", "online_kmeans", "two_tower"],
        required=True,
    )
    p_rec.add_argument("--config", required=True, help="Path to core JSON config")
    p_rec.add_argument("--retrieval-config", required=True, help="Path to retrieval JSON config")
    p_rec.add_argument("--model-dir", default="./artifacts")
    p_rec.add_argument("--user-ids", required=True)
    p_rec.add_argument("--k", type=int, default=20)
    p_rec.set_defaults(func=cmd_recommend)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(verbose=args.verbose, quiet=args.quiet, log_level=args.log_level)
    if args.command in ("train", "update", "evaluate"):
        apply_dataset_cli_defaults(args)
    args.func(args)


if __name__ == "__main__":
    main()

