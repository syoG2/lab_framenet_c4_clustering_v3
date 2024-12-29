from pathlib import Path

import pandas as pd


def read_c4_datasets(file_id_list: list[int], part_id_list: list[int], split_name_list: list[str]) -> pd.DataFrame:
    ret_df = pd.DataFrame()
    for file_id in file_id_list:
        for part_id in part_id_list:
            for split_name in split_name_list:
                input_file: Path = Path(f"./datasets/c4/{split_name}_{file_id:05}/exemplars_{part_id}_token0.jsonl")
                if input_file.exists():
                    df = pd.read_json(input_file, lines=True)
                    ret_df = pd.concat([ret_df, df], ignore_index=True)
                else:
                    print(f"{input_file} is not found.")

    return ret_df
