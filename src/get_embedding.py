from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf
from pydantic import BaseModel


class Args(BaseModel):
    input_file: Path
    output_exemplar_file: Path
    output_vec_file: Path
    pretrained_model_name: str
    vec_type: str = "word"
    layer: int = 0
    normalization: bool = False
    batch_size: int = 32
    device: str = "cuda:0"
    split: str = "dev"


def main():
    # OmegaConfを用いて実験設定を読み込む
    args = Args(**OmegaConf.from_cli())
    print(args)
    # outputディレクトリの作成
    args.output_exemplar_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_vec_file.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(args.input_file)
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
