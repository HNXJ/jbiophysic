# src/jbiophysic/cli/main.py
import argparse
from jbiophysic.common.utils.logging import get_logger
from jbiophysic.models.pipelines.run_full_experiment import run_experiment

logger = get_logger(__name__)

def main():
    logger.info("Initializing jbiophysic CLI")
    parser = argparse.ArgumentParser(description="jbiophysic - Biophysical Simulation Platform")
    parser.add_argument("--run", action="store_true", help="Run the default scientific experiment pipeline.")
    
    args = parser.parse_args()
    
    if args.run:
        logger.info("Starting Full Scientific Experiment Pipeline...")
        run_experiment()
    else:
        logger.info("No command specified. Displaying help.")
        parser.print_help()

if __name__ == "__main__":
    main()