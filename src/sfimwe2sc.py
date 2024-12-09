import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer


def write_jsonl(items, file):
    output_items = []
    for item in items:
        output_item = {}
        for k, v in item.items():
            if isinstance(v, Path):
                output_item[k] = str(v)
            else:
                output_item[k] = v
        output_items.append(output_item)
    with open(file, "w") as fo:
        for item in output_items:
            print(json.dumps(item, ensure_ascii=False), file=fo)


def base_collate_fn(batch):
    output_dict = {"verb": [], "frame": [], "ex_idx": [], "batch_size": len(batch)}
    for i in ["input_ids", "token_type_ids", "attention_mask"]:
        if i in batch[0]:
            output_dict[i] = nn.utils.rnn.pad_sequence([torch.LongTensor(b[i]) for b in batch], batch_first=True)
    output_dict["target_tidx"] = torch.LongTensor([b["target_tidx"] for b in batch])

    for b in batch:
        output_dict["verb"].append(b["verb"])
        output_dict["frame"].append(b["frame"])
        output_dict["ex_idx"].append(b["ex_idx"])
    return output_dict


def tokenize_text_and_target(tokenizer, preprocessed_text, preprocessed_lu_idx, vec_type):
    inputs = tokenizer(
        preprocessed_text,
        return_tensors="pt",
        return_special_tokens_mask=True,
        return_offsets_mapping=True,  # FastTokenizerを使用する必要がある
    )
    inputs = {k: v.squeeze(0) for k, v in inputs.items()}

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"])

    char_to_token = [[] for _ in range(len(preprocessed_text))]
    token_to_char = [[] for _ in range(len(inputs["input_ids"]))]

    for idx, (start, end) in enumerate(inputs["offset_mapping"]):
        for i in range(start, end):
            char_to_token[i].append(idx)
            token_to_char[idx].append(i)

    target_tidx = []
    for lu_idx in preprocessed_lu_idx:
        for tidx in range(char_to_token[lu_idx[0]][0], char_to_token[lu_idx[-1] - 1][-1] + 1):
            target_tidx.append(tidx)

    # alignments, previous_char_idx_list = [], [1]

    # for char_idx_list in get_alignments(preprocessed_text.split(), tokens)[0]:
    #     # get_alignments()は単語列とトークン列の対応関係(現われる位置)を示す配列を返す
    #     # 文章："Dougal started with the body . "
    #     # トークン列：['doug', '##al', 'started', 'with', 'the', 'body', '.']
    #     # 出力：([[0, 1], [2], [3], [4], [5], [6]], [[0], [0], [1], [2], [3], [4], [5]])
    #     if len(char_idx_list) == 0:
    #         # 単語に対応するトークンがない場合、一つ前の単語に統合する
    #         alignments.append(previous_char_idx_list)
    #     else:
    #         # トークン列には先頭に[CLS]が追加されているため、indexを1増加させる
    #         char_idx_list = [c + 1 for c in char_idx_list]
    #         alignments.append(char_idx_list)
    #         previous_char_idx_list = char_idx_list

    # target_tidx = alignments[target_word_idx][0]
    # if vec_type == "mask":
    #     # 動詞をmaskに置き換える際、動詞が複数のトークンに分割されている可能性がある。
    #     # そのため動詞のトークンの先頭をmaskに置き換え、以降の動詞のトークンを削除する。
    #     inputs["input_ids"][target_tidx] = tokenizer.mask_token_id
    #     if len(alignments[target_word_idx]) >= 2:
    #         for _ in alignments[target_word_idx][1:]:
    #             for k in inputs.keys():
    #                 del inputs[k][target_tidx + 1]

    inputs["target_tidx"] = target_tidx

    # TODO:確認
    print(tokenizer)
    print(preprocessed_text)
    print(preprocessed_lu_idx)
    print(vec_type)
    print()
    print(inputs)
    return inputs


class BaseDataset(Dataset):
    def __init__(self, df, pretrained_model_name, vec_type):
        self.df = df
        self.pretrained_model_name = pretrained_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name)
        self.vec_type = vec_type
        self._preprocess()

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        return self.out_inputs[idx]

    def _preprocess(self):
        self.out_inputs = []
        for df_dict in self.df.to_dict("records"):
            inputs = tokenize_text_and_target(
                self.tokenizer,
                df_dict["preprocessed_text"],
                df_dict["preprocessed_lu_idx"],
                self.vec_type,
            )
            inputs.update(
                {
                    # "frame": df_dict["frame"],
                    "verb": df_dict["pred_lu_name"].replace(r"\.v", ""),
                    "ex_idx": df_dict["id_data"],
                }
            )
            self.out_inputs.append(inputs)
        self.data_num = len(self.out_inputs)


class BaseEmbedding:
    def __init__(self, model, pretrained_model_name, vec_type, batch_size):
        self.model = model
        self.pretrained_model_name = pretrained_model_name
        self.vec_type = vec_type
        self.batch_size = batch_size

    def get_embedding(self, df):
        ds = BaseDataset(df, self.pretrained_model_name, self.vec_type)
        dl = DataLoader(
            ds,
            batch_size=self.batch_size,
            collate_fn=base_collate_fn,
            shuffle=False,
        )

        vec_list = []
        for batch in tqdm(dl):
            with torch.no_grad():
                vec_list += list(self.model(batch).cpu().detach().numpy())

        df_vec = df.reset_index(drop=True).reset_index().rename(columns={"index": "vec_id"})
        vec_array = np.array(vec_list)
        return df_vec, vec_array


class BaseNet(nn.Module):
    def __init__(self, pretrained_model_name, normalization, device, layer=-1):
        super(BaseNet, self).__init__()
        config = AutoConfig.from_pretrained(pretrained_model_name, output_hidden_states=True)
        self.pretrained_model = AutoModel.from_pretrained(pretrained_model_name, config=config).to(device)

        self.normalization = normalization
        self.layer = layer
        self.device = device

    def forward(self, inputs):
        if "token_type_ids" in inputs:
            hidden_states = self.pretrained_model(
                input_ids=inputs["input_ids"].to(self.device),
                attention_mask=inputs["attention_mask"].to(self.device),
                token_type_ids=inputs["token_type_ids"].to(self.device),
            )["hidden_states"][self.layer]
        else:
            hidden_states = self.pretrained_model(
                input_ids=inputs["input_ids"].to(self.device),
                attention_mask=inputs["attention_mask"].to(self.device),
            )["hidden_states"][self.layer]
        embeddings = hidden_states[
            torch.LongTensor(range(len(inputs["target_tidx"]))),
            inputs["target_tidx"],
        ]
        if self.normalization == "true":
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings
