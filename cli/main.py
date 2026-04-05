# jbiophysics/cli/main.py
import argparse
from cli.gravia_write import write_results, get_manuscript_paths

def main():
    parser = argparse.ArgumentParser(description="Gravia Write - Manuscript Generative Engine")
    parser.add_argument("--build", action="store_true", help="Build the full manuscript.")
    parser.add_argument("--trace", type=str, default="jbiophysics/gamma/trace.json", help="Path to gamma trace results.")
    args = parser.parse_args()
    
    if args.build:
        print("🧬 Engaging Gravia Writer...")
        # Placeholder for actual compile pipeline
        paths = get_manuscript_paths()
        write_results(paths["results_trace"], paths["results_md"])
        print("✅ Manuscript components synchronized.")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
