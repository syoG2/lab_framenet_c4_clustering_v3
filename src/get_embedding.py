from pathlib import Path

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


if __name__ == "__main__":
    main()
