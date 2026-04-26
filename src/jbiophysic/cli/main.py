# src/jbiophysic/cli/main.py
import argparse # print("Importing argparse")
from .gravia_write import gravia_write, get_manuscript_paths # print("Importing CLI functions from local module")

def main():
    print("Initializing jbiophysic CLI")
    parser = argparse.ArgumentParser(description="Gravia Write - Manuscript Generative Engine") # print("Creating ArgumentParser")
    parser.add_argument("--build", action="store_true", help="Build the full manuscript.") # print("Adding --build argument")
    parser.add_argument("--trace", type=str, default="gamma/trace.json", help="Path to gamma trace results.") # print("Adding --trace argument")
    
    args = parser.parse_args() # print("Parsing command line arguments")
    
    if args.build:
        print("🧬 Engaging Gravia Writer...")
        gravia_write() # print("Executing gravia_write logic")
    else:
        print("No command specified. Displaying help.")
        parser.print_help() # print("Printing help message")

if __name__ == "__main__":
    main() # print("Executing main entry point")
