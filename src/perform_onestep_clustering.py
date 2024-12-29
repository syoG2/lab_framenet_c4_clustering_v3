import json
from pathlib import Path

from omegaconf import OmegaConf
from pydantic import BaseModel

from sfimwe2sc import OnestepClustering, fix_seed, read_embedding, write_jsonl


class Args(BaseModel):
    input_dir: Path = Path("./embedding/bert-base-uncased")
    output_dir: Path = Path("./clustering/onestep_clustering/bert-base-uncased")
    input_params_file: Path = Path("")
    vec_type: str = "word"  # word, mask, wm
    alpha: float | None = None
    layer: str | None = None
    vec_type2layer: dict[str, str] | None = None
    clustering_method: str = "average"  # average or ward

    def model_post_init(self, __context):
        if self.input_params_file == Path(""):
            self.input_params_file = Path(
                f"./params/bert-base-uncased/{self.vec_type}/onestep_{self.clustering_method}/best_params.json"
            )

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--input_dir", type=Path, required=True)
    # parser.add_argument("--output_dir", type=Path, required=True)

    # parser.add_argument("--input_params_file", type=Path, required=False)

    # parser.add_argument("--alpha", type=float, required=False)
    # parser.add_argument("--layer", type=str, required=False)

    # parser.add_argument(
    #     "--clustering_method",
    #     type=str,
    #     choices=["average", "ward"],
    #     required=False,
    # )
    # args = parser.parse_args()
    # print(args)


def main():
    fix_seed(0)
    args = Args(**OmegaConf.from_cli())
    print(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.input_params_file is not None:
        with open(args.input_params_file, "r") as f:
            best_params = json.load(f)
        for key, value in best_params.items():
            setattr(args, key, value)
    else:
        if args.alpha == 0:
            args.vec_type2layer = {"word": args.layer}
        elif args.alpha == 1:
            args.vec_type2layer = {"mask": args.layer}
        else:
            layer_word, layer_mask = args.layer.split("-")
            args.vec_type2layer = {"word": layer_word, "mask": layer_mask}

    clustering = OnestepClustering(args.clustering_method)

    df_vec, vec_array = read_embedding(args.input_dir, "dev", args.vec_type2layer, args.alpha)
    params = clustering.make_params(df_vec, vec_array)

    df_clu_dev = clustering.step(df_vec, vec_array, params)

    df_vec, vec_array = read_embedding(args.input_dir, "test", args.vec_type2layer, args.alpha)
    df_clu_test = clustering.step(df_vec, vec_array, params)
    write_jsonl(
        df_clu_dev.to_dict("records"),
        (args.output_dir / "exemplars_dev.jsonl"),
    )
    write_jsonl(
        df_clu_test.to_dict("records"),
        (args.output_dir / "exemplars_test.jsonl"),
    )

    with open(args.output_dir / "clustering_summary.txt", "w") as f:
        print(args, file=f)
    # write_json(vars(args), args.output_dir / "params.json")


if __name__ == "__main__":
    main()
