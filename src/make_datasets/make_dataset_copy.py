import random
from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd

from sfimwe2sc.utils.data_utils import read_jsonl, write_jsonl


def main(args: Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(read_jsonl(args.input_file))
    df = df.sample(frac=1, random_state=args.random_state)
    df = df.reset_index(drop=True).reset_index()
    df = df[df["lu_name"].apply(lambda x: x.split(".")[-1]) == "v"]
    df = df.reset_index(drop=True)

    df["target_widx"] = df["target_widx"].apply(lambda x: x[0])
    df["frame"] = df["frame_name"]
    df["verb"] = df["lu_name"].apply(lambda x: x.split(".v")[0])
    df["verb_frame"] = df["verb"].str.cat(df["frame"], sep=":")
    df.loc[:, "source"] = "framenet"

    df = df[
        [
            "ex_id",
            "ex_idx",
            "verb",
            "frame",
            "verb_frame",
            "lu_id",
            "lu_name",
            "text_widx",
            "target_widx",
        ]
    ]

    df_agg = pd.DataFrame(df.agg("lu_id").value_counts())
    ex_idx_list = []
    # for lu_id in df_agg[df_agg["lu_id"] >= args.min_n_examples].index.to_list():
    for lu_id in df_agg[df_agg["count"] >= args.min_n_examples].index.to_list():
        df_vf = df[df["lu_id"] == lu_id]
        ex_idx_list += df_vf["ex_idx"].to_list()[: args.max_n_examples]
    df_ex = df[df["ex_idx"].isin(ex_idx_list)]

    all_list, v2_list = [], []
    for lu_name, _ in df_ex.groupby(["lu_name", "lu_id"]).count().index:
        if lu_name not in all_list:
            all_list.append(lu_name)
        else:
            if lu_name not in v2_list:
                v2_list.append(lu_name)
    v1_list = sorted(set(all_list) - set(v2_list))

    random.seed(args.random_state)
    random.shuffle(v1_list)
    random.shuffle(v2_list)
    dev_list = sorted(
        v1_list[: -int(len(v1_list) * 0.8)] + v2_list[: -int(len(v2_list) * 0.8)]
    )
    test_list = sorted(
        v1_list[-int(len(v1_list) * 0.8) :] + v2_list[-int(len(v2_list) * 0.8) :]
    )

    for split, verb_list in zip(["dev", "test"], [dev_list, test_list]):
        df_verb = df_ex[df_ex["lu_name"].isin(verb_list)].copy()
        df_verb = df_verb.sort_values("ex_idx")
        write_jsonl(
            df_verb.to_dict("records"),
            args.output_dir / f"exemplars_{split}.jsonl",
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)

    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--min_n_examples", type=int, default=20)
    parser.add_argument("--max_n_examples", type=int, default=100)

    args = parser.parse_args()
    print(args)
    main(args)
