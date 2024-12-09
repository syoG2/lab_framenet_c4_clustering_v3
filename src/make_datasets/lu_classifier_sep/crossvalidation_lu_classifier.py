from glob import glob
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf
from pydantic import BaseModel
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
    part: int  # 分割交差検証用のパラメータ(0~n_splits-1)
    n_splits: int = 5  # 分割数
    input_file: Path = Path("./datasets/framenet/exemplars.jsonl")
    output_model_dir: Path = Path("")
    pretrained_model: str = "bert-base-uncased"
    device: str = "cuda:0"
    seed: int = 42

    def model_post_init(self, __context):
        wd = Path(__file__)
        if self.output_model_dir == Path(""):
            self.output_model_dir = wd.parent / f"models/{self.pretrained_model}/{self.n_splits}_{self.part}"


def main():
    # OmegaConfを用いて実験設定を読み込む
    args = Args(**OmegaConf.from_cli())
    print(args)
    args.output_model_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    # データの読み込み
    df = pd.read_json(args.input_file, orient="records", lines=True)
    df = df[["target_word", "preprocessed_text", "preprocessed_lu_idx", "lu_name"]]

    # # データのシャッフル
    # df = df.sample(frac=1, random_state=0).reset_index(drop=True)
    # # データの分割
    # train_size = int(len(df) * 0.8)
    # test_size = int(len(df) * 0.1)
    # # valid_size = int(len(df) * 0.1)

    # 5分割交差検証用にデータを分割する
    # luの単語数が1のものと2以上のもので比率を合わせたい。
    # 学習データとテストデータに同じlu_nameが含まれないようにする
    df_lu_single = df[df["preprocessed_lu_idx"].apply(len) == 1]
    df_lu_multi = df[df["preprocessed_lu_idx"].apply(len) > 1]

    # "lu_name"でグループ化し、数が少ない順にソート
    lu_name_single_counts = df_lu_single["lu_name"].value_counts()
    df_lu_single_list: pd.DataFrame = [pd.DataFrame() for _ in range(args.n_splits)]
    # ソートされた順にfor文を回す
    for lu_name in lu_name_single_counts.index:
        df_lu_single_list.sort(key=len)
        df_lu_single_list[0] = pd.concat(
            [df_lu_single_list[0], df_lu_single[df_lu_single["lu_name"] == lu_name]], ignore_index=True
        )
    df_lu_single_list.sort(key=len)

    # "lu_name"でグループ化し、数が少ない順にソート
    lu_name_multi_counts = df_lu_multi["lu_name"].value_counts()
    df_lu_multi_list: pd.DataFrame = [pd.DataFrame() for _ in range(args.n_splits)]
    # ソートされた順にfor文を回す
    for lu_name in lu_name_multi_counts.index:
        df_lu_multi_list.sort(key=len)
        df_lu_multi_list[0] = pd.concat([df_lu_multi_list[0], df_lu_multi[df_lu_multi["lu_name"] == lu_name]], ignore_index=True)
    df_lu_multi_list.sort(key=len, reverse=True)

    df_list = [pd.concat([df_lu_single_list[i], df_lu_multi_list[i]], ignore_index=True) for i in range(args.n_splits)]

    # データセットの作成
    train_dataset = Dataset.from_pandas(
        pd.concat([df_list[(args.part + i) % args.n_splits] for i in range(args.n_splits - 2)], ignore_index=True)
    )
    test_dataset = Dataset.from_pandas(df_list[(args.part + args.n_splits - 2) % args.n_splits])
    validation_dataset = Dataset.from_pandas(df_list[(args.part + args.n_splits - 1) % args.n_splits])
    dataset = DatasetDict({"train": train_dataset, "test": test_dataset, "validation": validation_dataset})

    # トークナイザーとモデルの準備
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)

    model = AutoModelForTokenClassification.from_pretrained(args.pretrained_model, label2id=label2id, id2label=id2label)

    for param in model.parameters():
        param.data = param.data.contiguous()

    data_collator = DataCollatorForTokenClassification(tokenizer)

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

    # トークナイザーとモデルを保存する
    tokenizer.save_pretrained(str(args.output_model_dir / "tokenizer"))
    best_model.save_pretrained(str(args.output_model_dir / "best_model"))

    # 保存したトークナイザーとモデルを読み込む
    tokenizer = AutoTokenizer.from_pretrained(args.output_model_dir / "tokenizer")
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

    # LUの単語数ごとの正解率を計算する
    lu_len_scores = {}
    for result in results:
        true_length = len(result["preprocessed_lu_idx"])
        if true_length not in lu_len_scores:
            lu_len_scores[true_length] = {"size": 0, "correct": 0}

        lu_len_scores[true_length]["size"] += 1
        if result["preprocessed_lu_idx"] == result["pred_lu_idx"]:
            lu_len_scores[true_length]["correct"] += 1

    all_score = {"size": 0, "correct": 0}
    more_than_2_score = {"size": 0, "correct": 0}
    for key in lu_len_scores:
        correct = lu_len_scores[key]["correct"]
        size = lu_len_scores[key]["size"]
        lu_len_scores[key]["acc"] = correct / size

        all_score["size"] += size
        all_score["correct"] += correct
        if key > 2:
            more_than_2_score["size"] += size
            more_than_2_score["correct"] += correct

    all_score["acc"] = all_score["correct"] / all_score["size"]
    more_than_2_score["acc"] = more_than_2_score["correct"] / more_than_2_score["size"]

    with open(args.output_model_dir / "score.txt", "w") as f:
        for key, value in sorted(lu_len_scores.items(), key=lambda x: x[0]):
            print(
                f"{str(key).rjust(3)}:\tAccuracy=\t{str(round(value['acc'],6)).ljust(9)}=\t{str(value['correct']).rjust(6)}/{str(value['size']).rjust(6)}",
                file=f,
            )

        print(
            f"{">2".rjust(3)}:\tAccuracy=\t{str(round(more_than_2_score['acc'],6)).ljust(9)}=\t{str(more_than_2_score['correct']).rjust(6)}/{str(more_than_2_score['size']).rjust(6)}",
            file=f,
        )
        print(
            f"{"all".rjust(3)}:\tAccuracy=\t{str(round(all_score['acc'],6)).ljust(9)}=\t{str(all_score['correct']).rjust(6)}/{str(all_score['size']).rjust(6)}",
            file=f,
        )


if __name__ == "__main__":
    main()
