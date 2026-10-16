import argparse
import asyncio
import json
import sys

from app.training.models import TrainingConfig
from app.training.service import training_service


def main():
    parser = argparse.ArgumentParser(description="own-ai Phase 5: Training & Model Deployment CLI")
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to execute")

    # status
    subparsers.add_parser("status", help="View training readiness and active model status")

    # consent
    consent_parser = subparsers.add_parser("consent", help="Manage user training consent")
    consent_group = consent_parser.add_mutually_exclusive_group(required=True)
    consent_group.add_argument("--opt-in", action="store_true", help="Opt-in to behavioral training")
    consent_group.add_argument("--opt-out", action="store_true", help="Opt-out from behavioral training")

    # dataset
    dataset_parser = subparsers.add_parser("dataset", help="Prepare a training dataset")
    dataset_parser.add_argument("--user_id", default="local_user", help="User ID to extract interactions for")

    # run / bundle
    bundle_parser = subparsers.add_parser("bundle", help="Create a training run and export Cloud GPU bundle")
    bundle_parser.add_argument("--dataset", required=True, help="Dataset version identifier (e.g. v_20261016_...)")
    bundle_parser.add_argument("--tag", default="llama3.2:1b-behavior-v1", help="Target candidate Ollama model tag")

    # evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a candidate run")
    eval_parser.add_argument("--run", required=True, help="Training run ID")
    eval_parser.add_argument("--mock", action="store_true", default=True, help="Use mock evaluation for fast verification")

    # activate
    activate_parser = subparsers.add_parser("activate", help="Activate an approved candidate model")
    activate_parser.add_argument("--run", required=True, help="Training run ID to activate")

    # rollback
    subparsers.add_parser("rollback", help="Revert active model to baseline default (llama3.2:1b)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "status":
        status = training_service.get_status()
        print(json.dumps(status, indent=2))

    elif args.command == "consent":
        res = training_service.set_consent(opted_in=args.opt_in)
        print(f"Updated consent: opted_in={res.opted_in}, scope={res.scope}")

    elif args.command == "dataset":
        res = training_service.prepare_dataset(user_id=args.user_id)
        print(json.dumps(res, indent=2))

    elif args.command == "bundle":
        config = TrainingConfig(candidate_tag=args.tag)
        res = training_service.initiate_training_run(dataset_version=args.dataset, config=config)
        print(json.dumps(res, indent=2))

    elif args.command == "evaluate":
        res = asyncio.run(training_service.evaluate_candidate(run_id=args.run, mock_eval=args.mock))
        print(json.dumps(res, indent=2))

    elif args.command == "activate":
        res = asyncio.run(training_service.activate_model(run_id=args.run))
        print(json.dumps(res, indent=2))

    elif args.command == "rollback":
        res = training_service.rollback_model()
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
