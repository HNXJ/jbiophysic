# src/jbiophysic/cli/main.py
import argparse
from .gravia_write import gravia_write, get_manuscript_paths

def main():
    print("Initializing jbiophysic CLI")
    parser = argparse.ArgumentParser(description="Gravia Write - Manuscript Generative Engine")
    parser.add_argument("--build", action="store_true", help="Build the full manuscript.")
    parser.add_argument("--trace", type=str, default="gamma/trace.json", help="Path to gamma trace results.")
    
    args = parser.parse_args()
    
    if args.build:
        print("🧬 Engaging Gravia Writer...")
        gravia_write()
    else:
        print("No command specified. Displaying help.")
        parser.print_help()

if __name__ == "__main__":
    main()
