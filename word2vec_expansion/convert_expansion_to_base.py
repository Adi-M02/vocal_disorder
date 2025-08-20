import argparse
import sys
sys.path.append('../vocal_disorder')
from utils.load_json import expansion_to_base_json

def main():
    parser = argparse.ArgumentParser(description="Convert expansion JSON to base format.")
    parser.add_argument("input_json", help="Path to the input JSON file")
    parser.add_argument("output_directory", help="Directory to save the output")
    args = parser.parse_args()

    expansion_to_base_json(args.input_json, args.output_directory)

if __name__ == "__main__":
    main()
