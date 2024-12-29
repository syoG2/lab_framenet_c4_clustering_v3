import json
from pathlib import Path

from omegaconf import OmegaConf
from pydantic import BaseModel
from tqdm import tqdm

from sfimwe2sc import OnestepClustering, calculate_bcubed, fix_seed, read_embedding


class Args(BaseModel):
    input_dir: Path = Path("./embedding/bert-base-uncased")
    output_dir: Path = Path("")
    vec_type: str = "word"
    layers: list[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    clustering_method: str = "average"

    def model_post_init(self, __context):
        if self.output_dir == Path(""):
            self.output_dir = Path(f"./params/bert-base-uncased/{self.vec_type}/onestep_{self.clustering_method}")


def main():
    fix_seed(0)
    args = Args(**OmegaConf.from_cli())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.vec_type == "word":
        vec_types = ["word"]
        alphas = [0]
    elif args.vec_type == "mask":
        vec_types = ["mask"]
        alphas = [1]
    elif args.vec_type == "wm":
        vec_types = ["word", "mask"]
        alphas = [i / 10 for i in range(11)]

    clustering = OnestepClustering(args.clustering_method)

    best_vec_type2layer = {}
    for vec_type in tqdm(vec_types):
        best_bcf = 0
        for layer in tqdm(args.layers):
            vec_type2layer = {vec_type: str(layer)}
            alpha = 0 if vec_type == "word" else 1
            df_vec, vec_array = read_embedding(args.input_dir, "dev", vec_type2layer, alpha)
            params = clustering.make_params(df_vec, vec_array)
            df_output = clustering.step(df_vec, vec_array, params)

            true = df_output.groupby("frame_name")["uuid"].agg(list).tolist()
            pred = df_output.groupby("frame_cluster")["uuid"].agg(list).tolist()
            bcf = calculate_bcubed(true, pred)[2]
            if best_bcf < bcf:
                best_bcf = bcf
                best_vec_type2layer[vec_type] = layer

    if args.vec_type == "wm":
        best_bcf = 0
        for alpha in tqdm(alphas):
            df_vec, vec_array = read_embedding(args.input_dir, "dev", best_vec_type2layer, alpha)
            params = clustering.make_params(df_vec, vec_array)
            df_output = clustering.step(df_vec, vec_array, params)

            true = df_output.groupby("frame_name")["uuid"].agg(list).tolist()
            pred = df_output.groupby("frame_cluster")["uuid"].agg(list).tolist()
            bcf = calculate_bcubed(true, pred)[2]
            if best_bcf < bcf:
                best_bcf = bcf
                best_alpha = alpha
    else:
        best_alpha = alphas[0]

    best_params = {
        "alpha": best_alpha,
        "vec_type2layer": best_vec_type2layer,
    }
    best_params["clustering_method"] = args.clustering_method

    with open(args.output_dir / "best_params.json", "w") as f:
        json.dump(best_params, f)
    # write_json(best_params, args.output_dir / "best_params.json")


if __name__ == "__main__":
    main()
