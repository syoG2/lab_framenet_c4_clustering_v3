from pathlib import Path
from unicodedata import normalize

import pandas as pd
import stanza
from base import BaseData, WordInfo, WordList
from collect_c4 import C4Id
from lu_classifier_token.util import extract_entities, id2label, label2id, preprocess_data, run_prediction
from omegaconf import OmegaConf
from pydantic import BaseModel
from spacy_alignments import get_alignments
from timeout_decorator import timeout
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification

from datasets import Dataset

# tqdmをpandasのapplyメソッドで使用できるように設定
tqdm.pandas()


class Args(BaseModel):
    input_file: Path = Path("")
    file_id: int = 0  # input_fileが指定されている場合は無視される
    part_id: int  # 前処理に時間がかかるため、part_id*1000~(part_id+1)*1000行目のデータを処理する
    split_name: str = "train"
    output_exemplar_file: Path = Path("")
    output_wordlist_file: Path = Path("")
    device: str = "cuda:0"
    model_name: str = "bert-base-uncased"
    model_path: Path = Path("")
    tokenizer_path: Path = Path("")

    def model_post_init(self, __context):
        output_dir: Path = Path(f"./datasets/c4/{self.split_name}_{self.file_id:05}")
        if self.input_file == Path(""):
            self.input_file = Path(f"./data/c4/{self.split_name}_{self.file_id:05}.jsonl")
        if self.output_exemplar_file == Path(""):
            self.output_exemplar_file = output_dir / Path(f"exemplars_{self.part_id}_token.jsonl")
        if self.output_wordlist_file == Path(""):
            self.output_wordlist_file = output_dir / Path(f"word_list_{self.part_id}_token.jsonl")
        if self.model_path == Path(""):
            self.model_path = Path(f"./src/make_datasets/lu_classifier_token/models/{self.model_name}/5_3/best_model")
        if self.tokenizer_path == Path(""):
            self.tokenizer_path = Path(f"./src/make_datasets/lu_classifier_token/models/{self.model_name}/5_3/tokenizer")


def get_pred_lu_name(preprocessed_text, doc_sentence, pred_lu_idx):
    # 前処理後のtextのsplit()とdoc_sentence.wordsの対応を取る必要がある
    doc_words = [word.text for word in doc_sentence.words]
    char_to_word, _ = get_alignments(list(preprocessed_text), doc_words)
    return (
        " ".join(
            [
                doc_sentence.words[i].lemma
                for idx in pred_lu_idx
                for i in range(char_to_word[idx[0]][0], char_to_word[idx[-1] - 1][-1] + 1)
            ]
        )
        + ".v"
    )


def get_target_word_idxs(preprocessed_text, doc_sentence):
    doc_words = [word.text for word in doc_sentence.words]
    _, doc_to_char = get_alignments(list(preprocessed_text), doc_words)
    return [[doc_to_char[word.id - 1][0], doc_to_char[word.id - 1][-1] + 1] for word in doc_sentence.words if word.upos == "VERB"]


class C4Data(BaseData):
    id_data: C4Id  # 元データの参照に必要な情報を格納
    part_id: int


class C4WordList(WordList):
    id_data: C4Id
    # words: list[WordInfo]


def make_word_list(id_data: C4Id, doc_sentence: list[list], sequence_number: int) -> WordList:
    # 構文解析の結果を整理して返す
    ret: C4WordList = C4WordList(id_data=id_data, words=[])

    for word in doc_sentence.words:
        try:
            word_info: WordInfo = WordInfo(
                id=len(ret.words),  # 複数sentence全体の連番に変更
                text=word.text,
                lemma=word.lemma,
                upos=word.upos,
                xpos=word.xpos,
                feats=word.feats,
                head=len(ret.words) + (word.head - word.id) if word.head != 0 else -1,  # idの変更に合わせる
                deprel=word.deprel,
                start_char=word.start_char,
                end_char=word.end_char,
                children=[],
                word_idx=word.id - 1,
                sent_id=sequence_number,
            )
            ret.words.append(word_info)
        except KeyError as e:
            print(f"key:'{e}'が存在しません。")
    for word_info in ret.words:
        ret.words[word_info.head].children.append(word_info.id)  # childrenを作成
    return ret


@timeout(100)
def nlp_with_timeout(nlp, text):  # nlp(text)のタイムアウトを設定
    return nlp(text)


def main():
    # OmegaConfを用いて実験設定を読み込む
    args = Args(**OmegaConf.from_cli())
    print(args)  # 引数を表示
    # outputディレクトリの作成
    args.output_exemplar_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_wordlist_file.parent.mkdir(parents=True, exist_ok=True)

    nlp = stanza.Pipeline(
        "en",
        processors="tokenize,mwt,pos,lemma,depparse",
        use_gpu=True,
        device=args.device,
        pos_batch_size=3000,
    )

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_path))
    model = AutoModelForTokenClassification.from_pretrained(str(args.model_path))
    data_collator = DataCollatorForTokenClassification(tokenizer)

    df = pd.read_json(args.input_file, lines=True)
    df = df[args.part_id * 1000 : min((args.part_id + 1) * 1000, len(df))]

    tqdm.pandas(desc="normalize_text")
    df["normalize_text"] = df["text"].progress_apply(lambda x: normalize("NFKC", x))  # Unicode正規化

    tqdm.pandas(desc="doc_sentence")
    df["doc_sentence"] = df["normalize_text"].progress_apply(
        lambda x: [(seq_id, sentence) for seq_id, sentence in enumerate(nlp(x).sentences)]
    )

    df = df.explode("doc_sentence", True)
    # 連番をつける
    df["sequence_number"] = df["doc_sentence"].apply(lambda x: x[0])
    df["doc_sentence"] = df["doc_sentence"].apply(lambda x: x[1])

    tqdm.pandas(desc="preprocessed_text")
    df["preprocessed_text"] = df["doc_sentence"].progress_apply(
        lambda x: " ".join([word.text for word in x.words])
        + " "  # 末尾に空白を追加しないと[,,,"a"," "]と[,,,," ","a"]のアラインメントをとる時におかしくなる)
    )

    tqdm.pandas(desc="token_length")
    df = df[df["preprocessed_text"].progress_apply(lambda x: len(tokenizer(x)["input_ids"]) <= tokenizer.model_max_length)]

    tqdm.pandas(desc="target_word_idx")
    df["target_word_idx"] = df.progress_apply(
        lambda row: get_target_word_idxs(row["preprocessed_text"], row["doc_sentence"]), axis=1
    )
    df = df.explode("target_word_idx", True)
    df = df.dropna(subset=["target_word_idx"])

    tqdm.pandas(desc="target_word")
    df["target_word"] = df.progress_apply(
        lambda row: row["preprocessed_text"][row["target_word_idx"][0] : row["target_word_idx"][-1]], axis=1
    )
    # df = df.dropna(subset=["target_word"])
    # df["preprocessed_target_widx"] = [[0, 0] for _ in range(len(df))]

    dataset = Dataset.from_pandas(df[["target_word", "preprocessed_text", "target_word_idx"]])
    preprocessed_dataset = dataset.map(
        preprocess_data,
        fn_kwargs={"tokenizer": tokenizer, "label2id": label2id, "prediction": True},
        remove_columns=dataset.column_names,
    )
    dataloader = DataLoader(
        preprocessed_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=data_collator,
    )

    predictions = run_prediction(dataloader, model)

    results = extract_entities(predictions, dataset, tokenizer, id2label)
    df["pred_lu_idx"] = [result["pred_lu_idx"] for result in results]

    tqdm.pandas(desc="pred_lu_name")
    df["pred_lu_name"] = df.progress_apply(
        lambda row: get_pred_lu_name(row["preprocessed_text"], row["doc_sentence"], row["pred_lu_idx"]),
        axis=1,
    )

    preprocessed_exemplars: list[C4Data] = [
        C4Data(
            source=row["source"],
            id_data=C4Id(**row["id_data"]),
            target_word=row["target_word"],
            target_word_idx=row["target_word_idx"],
            text=row["text"],
            preprocessed_text=row["preprocessed_text"],
            preprocessed_lu_idx=row["pred_lu_idx"],
            part_id=args.part_id,
            lu_name=row["pred_lu_name"],
        )
        for _, row in df.iterrows()
    ]

    with open(args.output_exemplar_file, "w") as f:
        with tqdm(preprocessed_exemplars) as pbar:
            pbar.set_description("[write preprocessed_exemplars]")
            for exemplar in pbar:
                print(exemplar.model_dump_json(), file=f)

    df = df.drop_duplicates(subset=["preprocessed_text"])  # 重複を削除
    preprocessed_word_lists: list[C4WordList] = [
        make_word_list(C4Id(**row["id_data"]), row["doc_sentence"], row["sequence_number"]) for _, row in df.iterrows()
    ]

    with open(args.output_wordlist_file, "w") as f:
        with tqdm(preprocessed_word_lists) as pbar:
            pbar.set_description("[write word_list]")
            for word_list in pbar:
                print(word_list.model_dump_json(), file=f)


if __name__ == "__main__":
    main()
