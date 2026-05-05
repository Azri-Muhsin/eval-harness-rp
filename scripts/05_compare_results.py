import argparse
from pathlib import Path
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_file", required=True)
    p.add_argument("--out_file", required=True)
    args = p.parse_args()

    df = pd.read_csv(args.results_file)
    df = df.sort_values(["macro_f1", "accuracy"], ascending=False)

    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_file, index=False)

    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
