from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="lapo-value")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in (
        "preflight",
        "manifest",
        "extract",
        "evaluate",
        "export",
        "prepare-openpi",
        "evaluate-openpi",
        "compare-openpi",
        "select-final",
        "prepare-openpi-finetune",
        "generate-openpi-latents",
        "create-cascaded-init",
        "train-openpi-fold",
        "evaluate-openpi-oof",
        "report-openpi-oof",
        "compare-openpi-oof",
        "plot-openpi-oof-curves",
        "plot-openpi-oof-three-group-values",
        "evaluate-openpi-procvlm-metrics",
        "benchmark-openpi-latency",
        "report-openpi-rollout-success-curves",
    ):
        sub = subparsers.add_parser(name)
        sub.add_argument("--config", required=True)
        if name == "preflight":
            sub.add_argument("--allow-cpu", action="store_true")
        if name == "extract":
            sub.add_argument("--workers", type=int, default=8)
            sub.add_argument("--overwrite", action="store_true")
        if name in ("evaluate", "export"):
            sub.add_argument("--run-name", default="proposed")
            sub.add_argument("--checkpoint", default="best.pt")
        if name == "export":
            sub.add_argument("--output-dir")
        if name == "prepare-openpi":
            sub.add_argument("--workers", type=int, default=4)
        if name == "prepare-openpi-finetune":
            sub.add_argument("--workers", type=int, default=4)
        if name == "create-cascaded-init":
            sub.add_argument("--source-checkpoint", required=True)
            sub.add_argument("--output-checkpoint", required=True)
            sub.add_argument("--seed", type=int, required=True)
        if name == "evaluate-openpi":
            sub.add_argument("--run-name", required=True)
            sub.add_argument("--checkpoint", default="best.pt")
        if name == "train-openpi-fold":
            sub.add_argument("--fold", type=int, required=True)
            sub.add_argument("--starting-checkpoint")
            sub.add_argument("--output-name", default="openpi_finetune_oof")
            sub.add_argument("--lambda-latent", type=float)
            sub.add_argument(
                "--value-path", choices=("direct", "latent_hidden"), default="direct"
            )
        if name == "evaluate-openpi-oof":
            sub.add_argument("--bootstrap-samples", type=int)
            sub.add_argument("--output-name", default="openpi_finetune_oof")
        if name == "report-openpi-oof":
            sub.add_argument("--output-name", default="openpi_finetune_oof")
        if name == "compare-openpi-oof":
            sub.add_argument("--baseline-output-name", default="openpi_finetune_oof")
            sub.add_argument(
                "--proposed-output-name", default="openpi_finetune_oof_proposed"
            )
            sub.add_argument(
                "--comparison-output-name", default="openpi_finetune_oof_comparison"
            )
        if name == "plot-openpi-oof-curves":
            sub.add_argument("--baseline-output-name", default="openpi_finetune_oof")
            sub.add_argument(
                "--proposed-output-name", default="openpi_finetune_oof_proposed"
            )
            sub.add_argument("--output-name", default="openpi_finetune_oof_curves")
        if name == "plot-openpi-oof-three-group-values":
            sub.add_argument("--baseline-output-name", default="openpi_finetune_oof")
            sub.add_argument(
                "--proposed-output-name", default="openpi_finetune_oof_proposed"
            )
            sub.add_argument(
                "--output-name", default="openpi_oof_three_group_value_curves"
            )
            sub.add_argument("--step", type=int, default=2000)
        if name in ("evaluate-openpi-procvlm-metrics", "benchmark-openpi-latency"):
            sub.add_argument("--baseline-output-name", default="openpi_finetune_oof")
            sub.add_argument(
                "--proposed-output-name", default="openpi_finetune_oof_proposed"
            )
            sub.add_argument("--output-name", default="openpi_procvlm_metrics")
        if name == "benchmark-openpi-latency":
            sub.add_argument("--fold", type=int, default=0)
            sub.add_argument("--step", type=int, default=2000)
        if name == "report-openpi-rollout-success-curves":
            sub.add_argument("--baseline-output-name", default="openpi_finetune_oof")
            sub.add_argument(
                "--proposed-output-name", default="openpi_finetune_oof_proposed"
            )
            sub.add_argument("--output-name", default="openpi_rollout_success_group_curves")
            sub.add_argument("--step", type=int, default=2000)
            sub.add_argument("--dataset-name", dest="dataset_names", action="append")
    args = parser.parse_args()
    if args.command == "preflight":
        from .preflight import run_preflight

        run_preflight(args.config, require_gpu=not args.allow_cpu)
    elif args.command == "manifest":
        from .manifest import build_manifest

        build_manifest(args.config)
    elif args.command == "extract":
        from .extract import extract_shards

        extract_shards(args.config, workers=args.workers, overwrite=args.overwrite)
    elif args.command == "evaluate":
        from .evaluate import evaluate

        evaluate(args.config, run_name=args.run_name, checkpoint_name=args.checkpoint)
    elif args.command == "export":
        from .export import export_model

        export_model(
            args.config,
            run_name=args.run_name,
            checkpoint_name=args.checkpoint,
            output_dir=args.output_dir,
        )
    elif args.command == "prepare-openpi":
        from .openpi_eval import prepare_openpi

        prepare_openpi(args.config, workers=args.workers)
    elif args.command == "evaluate-openpi":
        from .openpi_eval import evaluate_openpi

        evaluate_openpi(args.config, run_name=args.run_name, checkpoint_name=args.checkpoint)
    elif args.command == "compare-openpi":
        from .openpi_eval import compare_openpi

        compare_openpi(args.config)
    elif args.command == "select-final":
        from .select_final import select_and_export

        select_and_export(args.config)
    elif args.command == "prepare-openpi-finetune":
        from .openpi_finetune import prepare_openpi_finetune

        prepare_openpi_finetune(args.config, workers=args.workers)
    elif args.command == "generate-openpi-latents":
        from .openpi_finetune import generate_openpi_latents

        generate_openpi_latents(args.config)
    elif args.command == "create-cascaded-init":
        from .checkpoint_init import create_cascaded_init

        create_cascaded_init(
            args.config,
            source_checkpoint=args.source_checkpoint,
            output_checkpoint=args.output_checkpoint,
            seed=args.seed,
        )
    elif args.command == "train-openpi-fold":
        from .openpi_finetune import train_openpi_fold

        train_openpi_fold(
            args.config,
            fold=args.fold,
            starting_checkpoint=args.starting_checkpoint,
            output_name=args.output_name,
            lambda_latent_override=args.lambda_latent,
            value_path=args.value_path,
        )
    elif args.command == "evaluate-openpi-oof":
        from .recap_metrics import evaluate_openpi_oof

        evaluate_openpi_oof(
            args.config,
            bootstrap_samples=args.bootstrap_samples,
            output_name=args.output_name,
        )
    elif args.command == "report-openpi-oof":
        from .recap_metrics import report_openpi_oof

        report_openpi_oof(args.config, output_name=args.output_name)
    elif args.command == "compare-openpi-oof":
        from .recap_metrics import compare_openpi_oof

        compare_openpi_oof(
            args.config,
            baseline_output_name=args.baseline_output_name,
            proposed_output_name=args.proposed_output_name,
            comparison_output_name=args.comparison_output_name,
        )
    elif args.command == "plot-openpi-oof-curves":
        from .curve_report import generate_openpi_curve_report

        generate_openpi_curve_report(
            args.config,
            baseline_output_name=args.baseline_output_name,
            proposed_output_name=args.proposed_output_name,
            output_name=args.output_name,
        )
    elif args.command == "plot-openpi-oof-three-group-values":
        from .curve_report import generate_openpi_three_group_value_report

        generate_openpi_three_group_value_report(
            args.config,
            baseline_output_name=args.baseline_output_name,
            proposed_output_name=args.proposed_output_name,
            output_name=args.output_name,
            step=args.step,
        )
    elif args.command == "evaluate-openpi-procvlm-metrics":
        from .procvlm_metrics import evaluate_openpi_procvlm_metrics

        evaluate_openpi_procvlm_metrics(
            args.config,
            baseline_output_name=args.baseline_output_name,
            proposed_output_name=args.proposed_output_name,
            output_name=args.output_name,
        )
    elif args.command == "benchmark-openpi-latency":
        from .openpi_latency import benchmark_openpi_latency

        benchmark_openpi_latency(
            args.config,
            baseline_output_name=args.baseline_output_name,
            proposed_output_name=args.proposed_output_name,
            output_name=args.output_name,
            fold=args.fold,
            step=args.step,
        )
    elif args.command == "report-openpi-rollout-success-curves":
        from .rollout_success_curves import report_openpi_rollout_success_curves

        report_openpi_rollout_success_curves(
            args.config,
            baseline_output_name=args.baseline_output_name,
            proposed_output_name=args.proposed_output_name,
            output_name=args.output_name,
            step=args.step,
            dataset_names=args.dataset_names,
        )


if __name__ == "__main__":
    main()
