import argparse
import sys
import logging
from bayesprompt.experiments.run_source_training import main as train_source
from bayesprompt.experiments.run_target_adaptation import main as adapt_target

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("BayesPrompt")

def main():
    parser = argparse.ArgumentParser(
        description="BayesPrompt: Uncertainty-Aware Bayesian Prompt Adaptation for Robust Cross-Modality Medical Segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available sub-commands")
    
    # Standard Config-based commands
    for cmd in ["train", "adapt", "ablate", "crossmod", "reproduce", "visualize", "eval", "clean", "sensitivity", "efficiency"]:
        sp = subparsers.add_parser(cmd, help=f"Run {cmd}")
        sp.add_argument("--config", type=str, required=True, help="Path to config file")
        
    # validate-splits
    sp_vs = subparsers.add_parser("validate-splits", help="Validate dataset splits")
    sp_vs.add_argument("--split", type=str, required=True, help="Path to split.json")
    
    # build-tables
    sp_bt = subparsers.add_parser("build-tables", help="Build LaTeX tables")
    sp_bt.add_argument("--metrics-dir", type=str, required=True, help="Path to metrics directory")
    sp_bt.add_argument("--out-dir", type=str, required=True, help="Output directory for tables")
    sp_bt.add_argument("--template", action="store_true", help="Generate empty template tables")
    
    # check-baselines
    sp_cb = subparsers.add_parser("check-baselines", help="Check baseline coverage")
    sp_cb.add_argument("--metrics-dir", type=str, required=True, help="Path to metrics directory")
    sp_cb.add_argument("--out", type=str, required=True, help="Output markdown file path")

    args, unknown_args = parser.parse_known_args()
    
    logger.info(f"🚀 Starting BayesPrompt: {args.command.upper()} mode")
    if hasattr(args, "config"):
        logger.info(f"Dotlist Overrides: {unknown_args}")
    
    try:
        if args.command == "train":
            train_source(args.config, unknown_args)
        elif args.command == "adapt":
            adapt_target(args.config, unknown_args)
        elif args.command == "ablate":
            from bayesprompt.experiments.run_ablation import run_ablation
            run_ablation(args.config, unknown_args)
        elif args.command == "crossmod":
            from bayesprompt.experiments.run_crossmodality import run_crossmodality
            run_crossmodality(args.config, unknown_args)
        elif args.command == "reproduce":
            from bayesprompt.experiments.run_reproduce import run_reproduce
            run_reproduce(args.config, unknown_args)
        elif args.command == "visualize":
            from bayesprompt.visualization.qualitative_grid import run_visualization
            run_visualization(args.config, unknown_args)
        elif args.command == "eval":
            from bayesprompt.experiments.run_evaluation import run_evaluation
            run_evaluation(args.config, unknown_args)
        elif args.command == "clean":
            from bayesprompt.experiments.run_cleanup import run_cleanup
            run_cleanup(args.config, unknown_args)
        elif args.command == "sensitivity":
            from bayesprompt.experiments.run_iteration_sensitivity import run_sensitivity
            run_sensitivity(args.config, unknown_args)
        elif args.command == "efficiency":
            from bayesprompt.evaluation.efficiency_report import run_efficiency
            run_efficiency(args.config, unknown_args)
        elif args.command == "validate-splits":
            from bayesprompt.datasets.validate_splits import run_validation
            run_validation(args.split)
        elif args.command == "build-tables":
            from bayesprompt.evaluation.build_tables import run_build_tables
            run_build_tables(args.metrics_dir, args.out_dir, getattr(args, "template", False))
        elif args.command == "check-baselines":
            from bayesprompt.evaluation.check_baselines import run_check_baselines
            run_check_baselines(args.metrics_dir, args.out)
    except ModuleNotFoundError as e:
        logger.error(f"Module missing: {e}. Please ensure the requested module is implemented.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
