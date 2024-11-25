from glob import glob
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf
from pydantic import BaseModel
from seqeval.metrics import classification_report
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import set_seed
from util import compute_scores, convert_results_to_labels, extract_entities, id2label, label2id, preprocess_data, run_prediction

from datasets import Dataset, DatasetDict


class Args(BaseModel):
    input_file: Path = Path("./datasets/framenet/exemplars.jsonl")
    output_model_dir: Path = Path("")
    pretrained_model: str = "bert-base-uncased"
    device: str = "cuda:0"
    seed: int = 42

    def model_post_init(self, __context):
        if self.output_model_dir == Path(""):
            self.output_model_dir = Path(f"./src/make_datasets/lu_classifier/models/{self.pretrained_model}")


def main():
    # OmegaConfを用いて実験設定を読み込む
    args = Args(**OmegaConf.from_cli())
    args.output_model_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    # データの読み込み
    df = pd.read_json(args.input_file, orient="records", lines=True)
    df = df[["target_word", "preprocessed_text", "preprocessed_target_widx"]]
    # データのシャッフル
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)
    # データの分割
    train_size = int(len(df) * 0.8)
    test_size = int(len(df) * 0.1)
    valid_size = int(len(df) * 0.1)

    # データセットの作成
    train_dataset = Dataset.from_pandas(df[:train_size])
    test_dataset = Dataset.from_pandas(df[train_size : train_size + test_size])
    validation_dataset = Dataset.from_pandas(df[train_size + test_size :])
    dataset = DatasetDict({"train": train_dataset, "test": test_dataset, "validation": validation_dataset})

    # トークナイザーとモデルの準備
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)

    # 訓練セットに対して前処理を行う
    train_dataset = dataset["train"].map(
        preprocess_data,
        fn_kwargs={
            "tokenizer": tokenizer,
            "label2id": label2id,
        },
        remove_columns=dataset["train"].column_names,
    )
    # 検証セットに対して前処理を行う
    validation_dataset = dataset["validation"].map(
        preprocess_data,
        fn_kwargs={
            "tokenizer": tokenizer,
            "label2id": label2id,
        },
        remove_columns=dataset["validation"].column_names,
    )

    model = AutoModelForTokenClassification.from_pretrained(args.pretrained_model, label2id=label2id, id2label=id2label)

    for param in model.parameters():
        param.data = param.data.contiguous()

    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Trainerに渡す引数を初期化する
    training_args = TrainingArguments(
        output_dir=args.output_model_dir,  # 結果の保存フォルダ
        per_device_train_batch_size=32,  # 訓練時のバッチサイズ
        per_device_eval_batch_size=32,  # 評価時のバッチサイズ
        learning_rate=1e-4,  # 学習率
        lr_scheduler_type="linear",  # 学習率スケジューラ
        warmup_ratio=0.1,  # 学習率のウォームアップ
        num_train_epochs=5,  # 訓練エポック数
        evaluation_strategy="epoch",  # 評価タイミング
        save_strategy="epoch",  # チェックポイントの保存タイミング
        logging_strategy="epoch",  # ロギングのタイミング
        fp16=True,  # 自動混合精度演算の有効化
        report_to="none",  # 外部ツールへのログを無効化
    )

    # Trainerを初期化する
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        args=training_args,
    )

    # 訓練する
    trainer.train()

    # ミニバッチの作成にDataLoaderを用いる
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=data_collator,
    )

    best_score = 0
    # 各チェックポイントで処理する
    for checkpoint in sorted(glob(str(args.output_model_dir / "checkpoint-*"))):
        # モデルを読み込む
        model = AutoModelForTokenClassification.from_pretrained(checkpoint)
        model.to(args.device)  # モデルをGPUに移動
        # 固有表現ラベルを予測する
        predictions = run_prediction(validation_dataloader, model)
        # 固有表現を抽出する
        results = extract_entities(predictions, dataset["validation"], tokenizer, id2label)
        # 正解データと予測データのラベルのlistを作成する
        true_labels, pred_labels = convert_results_to_labels(results)
        # 評価スコアを算出する
        scores = compute_scores(true_labels, pred_labels, "micro")
        if best_score < scores["f1-score"]:
            best_score = scores["f1-score"]
            best_model = model

    best_model.save_pretrained(str(args.output_model_dir / "best_model"))

    best_model = AutoModelForTokenClassification.from_pretrained(args.output_model_dir / "best_model")

    test_dataset = dataset["test"].map(
        preprocess_data,
        fn_kwargs={
            "tokenizer": tokenizer,
            "label2id": label2id,
        },
        remove_columns=dataset["test"].column_names,
    )
    # ミニバッチの作成にDataLoaderを用いる
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=data_collator,
    )
    # 固有表現ラベルを予測する
    predictions = run_prediction(test_dataloader, best_model)
    # 固有表現を抽出する
    results = extract_entities(predictions, dataset["test"], tokenizer, id2label)
    # 正解データと予測データのラベルのlistを作成する
    true_labels, pred_labels = convert_results_to_labels(results)
    # 評価結果を出力する
    print(classification_report(true_labels, pred_labels))


if __name__ == "__main__":
    main()
