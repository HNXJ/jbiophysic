import argparse
from jbiophysic.common.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    logger.info("Initializing jbiophysic CLI")
    parser = argparse.ArgumentParser(description="jbiophysic - Biophysical Simulation Platform")
    parser.add_argument("--run", action="store_true", help="Run the default scientific experiment pipeline.")

    args = parser.parse_args()

    if args.run:
        logger.info("Starting Full Scientific Experiment Pipeline...")
        try:
            from jbiophysic.models.pipelines.run_full_experiment import run_pipeline
        except ModuleNotFoundError as exc:
            if exc.name == "jaxley":
                raise ImportError(
                    "The full experiment pipeline requires optional dependency 'jaxley'. "
                    "Install the full dependency set before running --run."
                ) from exc
            raise
        run_pipeline()
    else:
        logger.info("No command specified. Displaying help.")
        parser.print_help()


if __name__ == "__main__":
    main()
