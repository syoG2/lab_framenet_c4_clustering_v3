from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from pydantic import BaseModel

from sfimwe2sc import BaseEmbedding, BaseNet, write_jsonl


class Args(BaseModel):
    input_file: Path = Path("./datasets/framenet/exemplars.jsonl")
    output_dir: Path = Path("")
    pretrained_model_name: str = "bert-base-uncased"
    vec_type: str = "word"
    layer: int = 0
    normalization: bool = False
    batch_size: int = 32
    device: str = "cuda:0"
    split: str = "dev"

    def model_post_init(self, __context):
        if self.output_dir == Path(""):
            self.output_dir = Path(f"./embeddng/{self.pretrained_model_name}/{self.vec_type}/{self.layer}")


def main():
    # OmegaConfを用いて実験設定を読み込む
    args = Args(**OmegaConf.from_cli())
    print(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_json(args.input_file, lines=True)
    model = BaseNet(args.pretrained_model_name, args.normalization, args.device, args.layer)
    model.to(args.device).eval()

    embedding = BaseEmbedding(model, args.pretrained_model_name, args.vec_type, args.batch_size)
    df_vec, vec_array = embedding.get_embedding(df)
    write_jsonl(
        df_vec.to_dict("records"),
        args.output_dir / f"exemplars_{args.split}.jsonl",
    )
    vec_dict = {"vec": vec_array}
    np.savez_compressed(args.output_dir / f"vec_{args.split}", **vec_dict)


if __name__ == "__main__":
    main()
