from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf
from pydantic import BaseModel

from sfimwe2sc import OnecpvClustering, fix_seed, write_jsonl


class Args(BaseModel):
    input_dev_file: Path
    input_test_file: Path
    output_dir: Path
    clustering_name: str


# TODO: 修正
def main():
    fix_seed(0)
    args = Args(**OmegaConf.from_cli())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df_dev = pd.read_json(args.input_dev_file)
    df_test = pd.read_json(args.input_test_file)

    if args.clustering_name == "1cpv":
        clustering = OnecpvClustering()

    df_clu_dev = clustering.step(df_dev)
    df_clu_test = clustering.step(df_test)

    write_jsonl(
        df_clu_dev.to_dict("records"),
        args.output_dir / "exemplars_dev.jsonl",
    )
    write_jsonl(
        df_clu_test.to_dict("records"),
        args.output_dir / "exemplars_test.jsonl",
    )

    with open(args.output_dir / "clustering_summary.txt", "w") as f:
        print(args, file=f)


if __name__ == "__main__":
    main()
