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

    # export-colab
    colab_exp = subparsers.add_parser("export-colab", help="Gather all data into one unit and sync directly to Google Colab / Drive")
    colab_exp.add_argument("--user_id", default="local_user", help="User ID to extract interactions for")
    colab_exp.add_argument("--epochs", type=int, default=3, help="Training epochs")
    colab_exp.add_argument("--no-gdrive-sync", action="store_true", help="Skip Google Drive sync")

    # download-model
    dl_parser = subparsers.add_parser("download-model", help="Download latest or versioned model from Google Drive, package, and activate")
    dl_parser.add_argument("--version", default=None, help="Specific model version ID to download (defaults to latest)")
    dl_parser.add_argument("--no-activate", action="store_true", help="Do not activate model automatically")
    dl_parser.add_argument("--live-eval", action="store_true", help="Run live inference evaluation instead of mock")

    # list-models
    subparsers.add_parser("list-models", help="List versioned models available in Google Drive and local registry")

    # sync-cloud
    sc_parser = subparsers.add_parser("sync-cloud", help="Run 15-day delta sync of sanitized profile, interests, and chat turns to PostgreSQL")
    sc_parser.add_argument("--user_id", default="local_user", help="User ID to sync")
    sc_parser.add_argument("--force", action="store_true", help="Force sync even if 15 days have not elapsed")

    # cloud-status
    cs_parser = subparsers.add_parser("cloud-status", help="Check 15-day sync progress and 60-day training readiness")
    cs_parser.add_argument("--user_id", default="local_user", help="User ID to check")

    # cloud-train-prepare
    cp_parser = subparsers.add_parser("cloud-train-prepare", help="Compile accumulated 60-day cloud data into a unified GPU training unit")
    cp_parser.add_argument("--user_id", default="local_user", help="User ID")
    cp_parser.add_argument("--force", action="store_true", help="Force compilation even if 60 days / 4 cycles not fully reached")

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

    elif args.command == "export-colab":
        config = TrainingConfig(num_epochs=args.epochs)
        res = training_service.export_colab_unit(
            config=config,
            user_id=args.user_id,
            sync_to_gdrive=not args.no_gdrive_sync,
        )
        print(json.dumps(res, indent=2))
        print(f"\n🚀 Ready for Google Colab! Open: {res.get('colab_notebook_path')}")

    elif args.command == "download-model":
        res = asyncio.run(training_service.import_colab_model(
            version_id=args.version,
            auto_activate=not args.no_activate,
            mock_eval=not args.live_eval,
        ))
        print(json.dumps(res, indent=2))
        if res.get("success"):
            print(f"\n🎉 Successfully downloaded and deployed model version '{res.get('version_id')}' as '{res.get('candidate_tag')}'!")

    elif args.command == "list-models":
        res = training_service.list_colab_models()
        print(json.dumps(res, indent=2))

    elif args.command == "sync-cloud":
        res = asyncio.run(training_service.sync_cloud_delta(user_id=args.user_id, force=args.force))
        print(json.dumps(res, indent=2))
        if res.get("synced"):
            print(f"\n✅ 15-day sync batch '{res.get('batch_id')}' completed successfully!")
            print(f"📊 Chat turns synced: {res.get('stats', {}).get('chat_turns')}")
            print(f"🎯 Readiness for 60-day training: {res.get('training_pipeline', {}).get('readiness_percentage', 0)}%")
        else:
            print(f"\nℹ️ {res.get('message')}")

    elif args.command == "cloud-status":
        res = training_service.get_cloud_sync_status(user_id=args.user_id)
        print(json.dumps(res, indent=2))

    elif args.command == "cloud-train-prepare":
        res = training_service.prepare_60_day_training(user_id=args.user_id, force=args.force)
        print(json.dumps(res, indent=2))
        if res.get("success"):
            print(f"\n🚀 60-day cumulative training unit created: {res.get('version_id')}")
            print(f"📦 Colab Unit Archive: {res.get('unit_zip_path')}")



if __name__ == "__main__":
    main()
