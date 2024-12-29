import random
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf
from pydantic import BaseModel
from utils import read_c4_datasets


class Args(BaseModel):
    input_framenet_file: Path = Path("./datasets/framenet/exemplars.jsonl")
    c4_file_id_list: list[int] = [0]
    c4_part_id_list: list[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    c4_split_name_list: list[str] = ["train"]
    output_dir: Path = Path("./datasets/merged")  # TODO:設定値によって出力先ファイルが変わるように変更する
    seed: int = 42


def main():
    args = Args(**OmegaConf.from_cli())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df_framenet = pd.read_json(args.input_framenet_file, lines=True)

    lu_frame_sample_num = df_framenet.groupby(["lu_name", "frame_name"]).size()
    lu_frame = lu_frame_sample_num[lu_frame_sample_num > 20].index.to_list()
    df_framenet = df_framenet[df_framenet.set_index(["lu_name", "frame_name"]).index.isin(lu_frame)]

    lu_frame_unique = df_framenet[["lu_name", "frame_name"]].drop_duplicates()

    lu_name_counts = lu_frame_unique["lu_name"].value_counts()
    lu_name_single = lu_name_counts[lu_name_counts == 1].index.to_list()
    lu_name_multi = lu_name_counts[lu_name_counts > 1].index.to_list()

    random.seed(args.seed)
    random.shuffle(lu_name_single)
    random.shuffle(lu_name_multi)

    dev_lu_list = lu_name_single[: int(len(lu_name_single) * 0.2)] + lu_name_multi[: int(len(lu_name_multi) * 0.2)]
    test_lu_list = lu_name_single[int(len(lu_name_single) * 0.2) :] + lu_name_multi[int(len(lu_name_multi) * 0.2) :]

    df_framenet_dev = df_framenet[df_framenet["lu_name"].isin(dev_lu_list)]
    df_framenet_test = df_framenet[df_framenet["lu_name"].isin(test_lu_list)]

    df_c4 = read_c4_datasets(args.c4_file_id_list, args.c4_part_id_list, args.c4_split_name_list)
    df_c4 = df_c4[df_c4["lu_name"].isin(test_lu_list)]
    df_c4.reset_index(drop=True, inplace=True)

    for lu_name, count in df_framenet_test["lu_name"].value_counts().items():
        df_framenet_test = pd.concat([df_framenet_test, df_c4[df_c4["lu_name"] == lu_name].sample(n=count)])

    # print(df_c4)
    # print(df_framenet_test["lu_name"].value_counts())

    df_framenet_dev.to_json(args.output_dir / "exemplars_dev.jsonl", orient="records", lines=True)
    df_framenet_test.to_json(args.output_dir / "exemplars_test.jsonl", orient="records", lines=True)

    # TODO: devとtestを分けたデータセットを作成する
    # TODO: FrameNetとC4のデータを混ぜる
    # TODO: C4のデータをどのように混ぜるか
    # TODO: C4のデータの複数ファイルを読み取る関数を作る


if __name__ == "__main__":
    main()
