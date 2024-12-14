import itertools
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastcluster import linkage
from pyclustering.cluster.xmeans import kmeans_plusplus_initializer, xmeans
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import pdist
from scipy.special import comb
from sklearn.metrics import confusion_matrix
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
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
    # データセットをミニバッチにまとめる関数

    output_dict = {"verb": [], "frame": [], "ex_idx": [], "batch_size": len(batch)}
    for i in ["input_ids", "token_type_ids", "attention_mask"]:
        if i in batch[0]:
            output_dict[i] = nn.utils.rnn.pad_sequence([torch.LongTensor(b[i]) for b in batch], batch_first=True)

    output_dict["target_tidx"] = pad_sequence(
        [torch.tensor(b["target_tidx"]) for b in batch], batch_first=True, padding_value=-1
    ).long()  # サイズに差があるため、padding_value=-1で埋める

    for b in batch:
        output_dict["verb"].append(b["verb"])
        # output_dict["frame"].append(b["frame"])  # TODO: C4のデータにはframeがない
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

    # tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"])

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

    inputs["target_tidx"] = target_tidx

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
                    "verb": df_dict["lu_name"].replace(r"\.v", ""),
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

        inputs["target_tidx"] = torch.transpose(inputs["target_tidx"], 0, 1).to(self.device)  # 転置
        embeddings = hidden_states[
            torch.LongTensor(range(inputs["target_tidx"].size()[1])),
            inputs["target_tidx"],
        ]
        embeddings[torch.where(inputs["target_tidx"] == -1)] = 0
        valid_counts = torch.transpose(torch.where(inputs["target_tidx"] != -1, 1, 0).sum(dim=0, keepdim=True), 0, 1).to(
            self.device
        )

        embeddings = embeddings.sum(dim=0) / valid_counts
        if self.normalization == "true":
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings


def read_embedding(vec_dir, split, vec_type2run_number, alpha):
    if alpha == 0:
        dir1 = vec_dir / "word" / vec_type2run_number["word"]
        vec_array = np.load(dir1 / f"vec_{split}.npz", allow_pickle=True)["vec"]
        ex_file = dir1 / f"exemplars_{split}.jsonl"
    elif alpha == 1:
        dir1 = vec_dir / "mask" / vec_type2run_number["mask"]
        vec_array = np.load(dir1 / f"vec_{split}.npz", allow_pickle=True)["vec"]
        ex_file = dir1 / f"exemplars_{split}.jsonl"
    else:
        dir1 = vec_dir / "word" / vec_type2run_number["word"]
        dir2 = vec_dir / "mask" / vec_type2run_number["mask"]
        word_array = np.load(dir1 / f"vec_{split}.npz", allow_pickle=True)["vec"]
        mask_array = np.load(dir2 / f"vec_{split}.npz", allow_pickle=True)["vec"]
        vec_array = word_array * (1 - alpha) + mask_array * alpha
        ex_file = dir1 / f"exemplars_{split}.jsonl"
    df_vec = pd.read_json(ex_file, lines=True)
    return df_vec, vec_array


class OnestepClustering:
    def __init__(self, clustering):
        self.clustering = clustering

    def make_params(self, df, vec_array):
        params = {}
        z = linkage(pdist(vec_array), method=self.clustering, preserve_input=False)
        # params["th"] = z[-len(set(df["frame"])) + 1][2] + 1e-6 # TODO: frame_nameで機能するか確認
        params["th"] = z[-len(set(df["frame_name"])) + 1][2] + 1e-6
        return params

    def _clustering(self, vec_array, params):
        z = linkage(pdist(vec_array), method=self.clustering, preserve_input=False)
        cluster_array = fcluster(z, t=params["th"], criterion="distance")
        return cluster_array

    def step(self, df, vec_array, params):
        df["frame_cluster"] = self._clustering(vec_array, params)
        return df


def calculate_bcubed(true, pred):
    true_map = defaultdict(set)
    true_map.update({e: set(c) for c in true for e in c})
    pred_map = defaultdict(set)
    pred_map.update({e: set(c) for c in pred for e in c})

    instances = set(itertools.chain(*true, *pred))
    sum_precision, sum_recall = 0, 0
    for instance in instances:
        n_commons = len(true_map[instance] & pred_map[instance])
        n_preds = len(pred_map[instance])
        n_trues = len(true_map[instance])
        if n_preds != 0:
            sum_precision += n_commons / n_preds
        if n_trues != 0:
            sum_recall += n_commons / n_trues

    avg_precision = sum_precision / len(instances)
    avg_recall = sum_recall / len(instances)
    f_score = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)

    return avg_precision, avg_recall, f_score


def fix_seed(random_state: int) -> None:
    random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    torch.backends.cudnn.deterministic = True


class TwostepClustering:
    def __init__(self, clustering_method1, clustering_method2):
        self.clustering_method1 = clustering_method1
        self.clustering_method2 = clustering_method2

    def _make_vec_1st(self, df, vec_array, verb):
        df_verb = df[df["verb"] == verb].copy()
        verb_vec_array = vec_array[df_verb["vec_id"]]
        return df_verb, verb_vec_array

    def _clustering_1st(self, vec_array, params):
        if self.clustering_method1 == "average":
            if len(vec_array) >= 2:
                z = linkage(
                    pdist(vec_array),
                    method="average",
                    metric="euclidean",
                    preserve_input=False,
                )
                cluster_array = fcluster(z, t=params["lth"], criterion="distance")
            else:
                cluster_array = np.array([1])
        elif self.clustering_method1 == "xmeans":
            init_center = kmeans_plusplus_initializer(vec_array, 1).initialize()
            xm = xmeans(
                vec_array,
                init_center,
                ccore=False,
                kmax=params["kmax"],
                random_state=0,
            )
            xm.process()
            cluster_array = np.array([-1] * len(vec_array))
            for idx, clusters in enumerate(xm.get_clusters()):
                for sent_idx in clusters:
                    cluster_array[sent_idx] = idx + 1
        elif self.clustering_method1 == "1cpv":
            cluster_array = np.array([1] * len(vec_array))
        return cluster_array

    def _make_vec_2nd(self, df, vec_array, count):
        vec_list, df_list = [], []
        for plu in sorted(set(df["plu_local"])):
            df_cluster = df[df["plu_local"] == plu].copy()
            vec_list.append(
                np.average(
                    [vec_array[vec_id] for vec_id in df_cluster["vec_id"]],
                    axis=0,
                )
            )
            df_cluster.loc[:, "plu_global"] = count + 1
            df_list.append(df_cluster)
            count += 1
        return vec_list, pd.concat(df_list, axis=0)

    def _clustering_2nd(self, vec_array, params):
        z = linkage(
            pdist(vec_array),
            method=self.clustering_method2,
            preserve_input=False,
        )
        cluster_array = fcluster(z, t=self._decide_t(z, params["gth"]), criterion="distance")
        return cluster_array

    def _decide_t(self, z, gth):
        n_sides, n_points = 0, len(z) + 1
        for i in range(len(z)):
            if n_points <= z[i, 0]:
                pre_n_points0 = int(z[int(z[i, 0] - n_points), 3])
                pre_n_sides0 = comb(pre_n_points0, 2, exact=True)
                n_sides -= pre_n_sides0
            if n_points <= z[i, 1]:
                pre_n_points1 = int(z[int(z[i, 1] - n_points), 3])
                pre_n_sides1 = comb(pre_n_points1, 2, exact=True)
                n_sides -= pre_n_sides1
            n_sides += comb(int(z[i, 3]), 2, exact=True)
            probs = n_sides / comb(n_points, 2, exact=True)
            if probs >= gth:
                t = z[i, 2]
                break
        return t

    def make_confusion_matrix(self, df):
        true = df.groupby("verb").nunique()["frame"].values
        pred = df.groupby("verb").max()["plu_local"].values
        labels = [str(i) for i in range(1, max([max(true), max(pred)]))]
        cm = confusion_matrix(true.astype(str), pred.astype(str), labels=labels)
        return pd.DataFrame(cm, index=labels, columns=labels)

    def make_params(self, df, vec_array):
        params = {}
        if self.clustering_method1 == "average":
            lth_list = []
            for verb in sorted(set(df["verb"])):
                df_verb = df[df["verb"] == verb]
                verb_vec_array = vec_array[df_verb["vec_id"]]
                lth_dict = {
                    "verb": verb,
                    "n_frames": len(set(df_verb["frame"])),
                    "n_texts": len(verb_vec_array),
                }
                if len(verb_vec_array) >= 2:
                    z = linkage(
                        pdist(verb_vec_array),
                        method=self.clustering_method1,
                        metric="euclidean",
                        preserve_input=False,
                    )
                    for _, _, lth, _ in z:
                        lth_dict = lth_dict.copy()
                        lth_dict["lth"] = lth
                        lth_list.append(lth_dict)
                else:
                    lth_dict["lth"] = 0
                    lth_list.append(lth_dict)

            df_lth = pd.DataFrame(lth_list).sort_values("lth", ascending=False)
            params["lth"] = df_lth["lth"][: len(set(df["verb_frame"])) - len(set(df["verb"])) + 1].values[-1] + 1e-6
        elif self.clustering_method1 == "xmeans":
            params["kmax"] = max(df.groupby("verb").agg(set)["frame"].apply(lambda x: len(x)))

        vf2f = {vf: f for vf, f in zip(df["verb_frame"], df["frame"], strict=False)}
        params["gth"] = sum([comb(i, 2, exact=True) for i in Counter(vf2f.values()).values()]) / comb(
            len(vf2f.values()), 2, exact=True
        )
        return params

    def step(self, df, vec_array1, vec_array2, params):
        vec_array_2nd, df_2nd_list = [], []
        for verb in sorted(set(df["verb"])):
            df_1st, vec_array_1st = self._make_vec_1st(df, vec_array1, verb)
            df_1st.loc[:, "plu_local"] = self._clustering_1st(vec_array_1st, params)
            _vec_array_2nd, _df_2nd = self._make_vec_2nd(df_1st, vec_array2, len(vec_array_2nd))
            vec_array_2nd += _vec_array_2nd
            df_2nd_list.append(_df_2nd)
        vec_array_2nd = np.array(vec_array_2nd)
        df_2nd = pd.concat(df_2nd_list, axis=0)

        map_1to2 = {c1 + 1: c2 for c1, c2 in enumerate(self._clustering_2nd(vec_array_2nd, params))}
        df_2nd["frame_cluster"] = df_2nd["plu_global"].map(map_1to2)
        return df_2nd
