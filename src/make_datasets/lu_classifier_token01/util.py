from typing import Any

import torch
from seqeval.metrics import f1_score, precision_score, recall_score
from spacy_alignments import get_alignments
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.tokenization_utils_base import BatchEncoding

# 分類ラベルとIDの対応付け
# labelはLUか否かの2種。"B-"をつけているのは、seqevalの仕様に合わせるため
label2id = {"O": 0, "B-lu": 1}
id2label = {v: k for k, v in label2id.items()}


# FrameNetをbert-base-uncasedでtokenizeした場合、token数が512を超える事例はなかった
def preprocess_data(
    data: dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    label2id: dict[int, str],
    prediction: bool = False,  # 予測時にはラベルを作成しない
) -> BatchEncoding:
    # tokenizer.convert_ids_to_tokens(1)は[unused0]のトークン
    # tokenizer.convert_ids_to_tokens(2)は[unused1]のトークン
    text = (
        data["preprocessed_text"][: data["target_word_idx"][0]]
        + tokenizer.convert_ids_to_tokens(1)
        + " "
        + data["preprocessed_text"][data["target_word_idx"][0] : data["target_word_idx"][1]]
        + " "
        + tokenizer.convert_ids_to_tokens(2)
        + data["preprocessed_text"][data["target_word_idx"][1] :]
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        return_special_tokens_mask=True,
        return_offsets_mapping=True,  # FastTokenizerを使用する必要がある
    )  # [CLS] {文 [unused0] 注目動詞 [unused1]} [SEP]
    inputs = {k: v.squeeze(0) for k, v in inputs.items()}

    char_to_token = [[] for _ in range(len(data["preprocessed_text"]))]
    token_to_char = [[] for _ in range(len(inputs["input_ids"]))]

    shift = 0
    for idx, ((start, end), id) in enumerate(zip(inputs["offset_mapping"], inputs["input_ids"], strict=True)):
        if id == 1 or id == 2:  # [unused0] or [unused1]
            if len(inputs["offset_mapping"]) > idx + 1:
                shift += inputs["offset_mapping"][idx + 1][0] - start
            continue

        for i in range(start - shift, end - shift):
            char_to_token[i].append(idx)
            token_to_char[idx].append(i)

    del inputs["offset_mapping"]

    if prediction:  # 推論をする場合はラベルを作成しない
        return inputs

    # 学習時に必要な正解ラベルを作成(トークンレベル)
    # 全て0のラベルを作成
    labels = torch.zeros_like(inputs["input_ids"])
    for entity in data["preprocessed_lu_idx"]:
        # entityには単語レベルでの開始位置と終了位置が格納されている
        start_token = char_to_token[entity[0]]
        end_token = char_to_token[entity[-1] - 1]

        start, end = start_token[0], end_token[-1]
        entity_type = "B-lu"
        # start,endのインデックスのどちらも含む範囲にラベルを設定
        labels[start : end + 1] = label2id[entity_type]

    # 特殊トークンの位置のIDは-100とする
    labels[torch.where(inputs["special_tokens_mask"])] = -100
    inputs["labels"] = labels
    return inputs


def convert_list_dict_to_dict_list(list_dict: dict[str, list]) -> list[dict[str, list]]:
    """ミニバッチのデータを事例単位のlistに変換"""
    dict_list = []
    # dictのキーのlistを作成する
    keys = list(list_dict.keys())
    for idx in range(len(list_dict[keys[0]])):  # 各事例で処理する
        # dictの各キーからデータを取り出してlistに追加する
        dict_list.append({key: list_dict[key][idx] for key in keys})
    return dict_list


def run_prediction(dataloader: DataLoader, model: PreTrainedModel) -> list[dict[str, Any]]:
    """予測スコアに基づき固有表現ラベルを予測"""
    predictions = []
    for batch in tqdm(dataloader):  # 各ミニバッチを処理する
        inputs = {k: v.to(model.device) for k, v in batch.items() if k != "special_tokens_mask"}
        # 予測スコアを取得する
        logits = model(**inputs).logits
        # 最もスコアの高いIDを取得する
        batch["pred_label_ids"] = logits.argmax(-1)
        batch = {k: v.cpu().tolist() for k, v in batch.items()}
        # ミニバッチのデータを事例単位のlistに変換する
        predictions += convert_list_dict_to_dict_list(batch)
    return predictions


def extract_entities(
    predictions: list[dict[str, Any]],
    dataset: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizer,
    id2label: dict[int, str],
) -> list[dict[str, Any]]:
    """固有表現を抽出"""
    results = []
    for prediction, data in zip(predictions, dataset, strict=True):
        # 文字のlistを取得する
        chars = list(data["preprocessed_text"])

        # 特殊トークンを除いたトークンのlistと予測ラベルのlistを取得する
        tokens, pred_labels = [], []
        all_tokens = tokenizer.convert_ids_to_tokens(prediction["input_ids"])
        for token, label_id in zip(all_tokens, prediction["pred_label_ids"], strict=True):
            # 特殊トークン以外をlistに追加する
            if token not in tokenizer.all_special_tokens:
                tokens.append(token)
                pred_labels.append(id2label[label_id])

        # 文字のlistとトークンのlistのアライメントをとる
        _, token_to_word_indices = get_alignments(chars, tokens)

        # 予測ラベルのlistから固有表現タイプと、
        # トークン単位の開始位置と終了位置を取得して、
        # それらを正解データと同じ形式に変換する
        pred_entities = []
        for i, pred_label in enumerate(pred_labels):
            if pred_label == "B-lu":
                idxs = token_to_word_indices[i]
                if pred_entities == [] or pred_entities[-1][-1] != idxs[0]:
                    pred_entities.append([idxs[0], idxs[-1] + 1])
                else:
                    pred_entities[-1][-1] = idxs[-1] + 1

        data["pred_lu_idx"] = pred_entities
        results.append(data)

    return results


def create_word_labels(text: str, entities: list[list[int]]) -> list[str]:
    """単語ベースでラベルのlistを作成"""
    # "O"のラベルで初期化したラベルのlistを作成する
    labels = ["O"] * len(text)
    for entity in entities:
        for i in range(entity[0], entity[1] + 1):
            labels[i] = "B-lu"
    return labels


def convert_results_to_labels(results: list[dict[str, Any]]) -> tuple[list[list[str]], list[list[str]]]:
    """正解データと予測データのラベルのlistを作成"""
    true_labels, pred_labels = [], []
    for result in results:  # 各事例を処理する
        # 文字ベースでラベルのリストを作成してlistに加える
        true_labels.append(create_word_labels(result["preprocessed_text"], result["preprocessed_lu_idx"]))
        pred_labels.append(create_word_labels(result["preprocessed_text"], result["pred_lu_idx"]))
    return true_labels, pred_labels


def compute_scores(true_labels: list[list[str]], pred_labels: list[list[str]], average: str) -> dict[str, float]:
    """適合率、再現率、F値を算出"""
    scores = {
        "precision": precision_score(true_labels, pred_labels, average=average),
        "recall": recall_score(true_labels, pred_labels, average=average),
        "f1-score": f1_score(true_labels, pred_labels, average=average),
    }
    return scores
