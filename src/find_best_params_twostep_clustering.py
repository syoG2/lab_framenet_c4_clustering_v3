from pathlib import Path

from omegaconf import OmegaConf
from pydantic import BaseModel
from tqdm import tqdm

from sfimwe2sc import TwostepClustering, calculate_bcubed, fix_seed, read_embedding


class Args(BaseModel):
    input_dir: Path
    output_dir: Path
    vec_type: str
    layers: list
    clustering_method1: str
    clustering_method2: str


def main(args):
    fix_seed(0)
    args = Args(**OmegaConf.from_cli())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    clustering = TwostepClustering(args.clustering_method1, args.clustering_method2)

    if args.vec_type == "word":
        vec_types = ["word"]
    elif args.vec_type == "mask":
        vec_types = ["mask"]
    else:
        vec_types = ["word", "mask"]

    best_vec_type2layer = {}
    for vec_type in tqdm(vec_types):
        best_bcf = 0
        for layer in tqdm(args.layers):
            vec_type2layer = {vec_type: layer}
            alpha = 0 if vec_type == "word" else 1
            df_vec, vec_array = read_embedding(args.input_dir, "dev", vec_type2layer, alpha)
            params = clustering.make_params(df_vec, vec_array)
            df_output = clustering.step(df_vec, vec_array, vec_array, params)

            true = df_output.groupby("frame")["ex_idx"].agg(list).tolist()
            pred = df_output.groupby("frame_cluster")["ex_idx"].agg(list).tolist()
            bcf = calculate_bcubed(true, pred)[2]
            if best_bcf < bcf:
                best_bcf = bcf
                best_vec_type2layer[vec_type] = layer

    if "wm" not in args.vec_type.split("-"):
        if args.vec_type in ["word", "mask"]:
            args.vec_type = f"{args.vec_type}-{args.vec_type}"
        for i, vec_type in enumerate(args.vec_type.split("-")):
            alpha = 0 if vec_type == "word" else 1
            if i == 0:
                best_alpha1 = alpha
            else:
                best_alpha2 = alpha
    else:
        if args.vec_type in ["wm"]:
            best_bcf, best_alpha1, best_alpha2 = 0, -1, -1
            for alpha in tqdm([i / 10 for i in range(11)]):
                df_vec, vec_array = read_embedding(args.input_dir, "dev", best_vec_type2layer, alpha)
                params = clustering.make_params(df_vec, vec_array)
                df_output = clustering.step(df_vec, vec_array, vec_array, params)

                true = df_output.groupby("frame")["ex_idx"].agg(list).tolist()
                pred = df_output.groupby("frame_cluster")["ex_idx"].agg(list).tolist()
                bcf = calculate_bcubed(true, pred)[2]
                if best_bcf < bcf:
                    best_bcf = bcf
                    best_alpha1, best_alpha2 = alpha, alpha
        else:
            for i, vec_type in enumerate(args.vec_type.split("-")):
                if vec_type == "word":
                    alphas = [0]
                elif vec_type == "mask":
                    alphas = [1]
                elif vec_type == "wm":
                    alphas = [i / 10 for i in range(11)]

                if i == 0:
                    alphas1 = alphas
                else:
                    alphas2 = alphas

            best_bcf, best_alpha1, best_alpha2 = 0, -1, -1
            for alpha1 in tqdm(alphas1):
                df_vec, vec_array1 = read_embedding(args.input_dir, "dev", best_vec_type2layer, alpha1)
                params = clustering.make_params(df_vec, vec_array1)
                for alpha2 in tqdm(alphas2):
                    _, vec_array2 = read_embedding(args.input_dir, "dev", best_vec_type2layer, alpha2)
                    df_output = clustering.step(df_vec, vec_array1, vec_array2, params)

                    true = df_output.groupby("frame")["ex_idx"].agg(list).tolist()
                    pred = df_output.groupby("frame_cluster")["ex_idx"].agg(list).tolist()
                    bcf = calculate_bcubed(true, pred)[2]
                    if best_bcf < bcf:
                        best_bcf = bcf
                        best_alpha1 = alpha1
                        best_alpha2 = alpha2

    best_params = {
        "alpha1": best_alpha1,
        "alpha2": best_alpha2,
        "vec_type2layer": best_vec_type2layer,
    }
    best_params["clustering_method1"] = args.clustering_method1
    best_params["clustering_method2"] = args.clustering_method2

    with open(args.output_dir / "best_params.json", "w") as f:
        print(best_params, file=f)
