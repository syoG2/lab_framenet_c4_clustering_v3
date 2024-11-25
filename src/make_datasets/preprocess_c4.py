from pathlib import Path
from unicodedata import normalize

import pandas as pd
import stanza
from base import BaseData, WordInfo, WordList
from collect_c4 import C4Id
from lu_classifier.util import extract_entities, id2label, label2id, preprocess_data, run_prediction
from omegaconf import OmegaConf
from pydantic import BaseModel
from timeout_decorator import timeout
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification

from datasets import Dataset


# TODO:引数の初期値の見直し
# TODO:入力された引数をterminalに表示するようにする
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

    def model_post_init(self, __context):
        output_dir: Path = Path(f"./datasets/c4/{self.split_name}_{self.file_id:05}")
        if self.input_file == Path(""):
            self.input_file = Path(f"./data/c4/{self.split_name}_{self.file_id:05}.jsonl")
        if self.output_exemplar_file == Path(""):
            self.output_exemplar_file = output_dir / Path(f"exemplars_{self.part_id}.jsonl")
        if self.output_wordlist_file == Path(""):
            self.output_wordlist_file = output_dir / Path(f"word_list_{self.part_id}.jsonl")
        if self.model_path == Path(""):
            self.model_path = Path(f"./src/make_datasets/lu_classifier/models/{self.model_name}/best_model")


class C4Data(BaseData):
    # source: str  # データの取得元(e.g. framenet)
    id_data: C4Id  # 元データの参照に必要な情報を格納
    # target_word: str  # 注目する語(基本形)
    # text: str  # 前処理前のtext
    # preprocessed_text: str  # 前処理後のtext
    # preprocessed_target_widx: list[int]  # 前処理後のLUの位置(単語レベル)[開始位置,終了位置,主となる語の位置(構文木)]
    # uuid: UUID = Field(default_factory=lambda: uuid4())  # 分割後のデータに対して一意に与える
    part_id: int
    pred_lu_name: str


class C4WordList(WordList):
    id_data: C4Id
    # words: list[WordInfo]


def make_word_list(id_data: C4Id, doc: list[list]) -> WordList:
    # 構文解析の結果を整理して返す
    ret: C4WordList = C4WordList(id_data=id_data, words=[])
    for sent_id, sentence in enumerate(doc.sentences):
        for word in sentence.words:
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
                    sent_id=sent_id,
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

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(str(args.model_path))
    data_collator = DataCollatorForTokenClassification(tokenizer)

    df = pd.read_json(args.input_file, lines=True)
    df = df[args.part_id * 1000 : min((args.part_id + 1) * 1000, len(df))]

    df["normalize_text"] = df["text"].apply(lambda x: normalize("NFKC", x))  # Unicode正規化

    df["doc_sentence"] = df["normalize_text"].apply(lambda x: [sentence for sentence in nlp(x).sentences])
    df = df.explode("doc_sentence")
    df["preprocessed_text"] = df["doc_sentence"].apply(lambda x: x.text)
    df = df[df["preprocessed_text"].apply(lambda x: len(tokenizer(x)["input_ids"]) <= tokenizer.model_max_length)]
    df["target_word"] = df["doc_sentence"].apply(lambda x: [word.lemma for word in x.words if word.upos == "VERB"])
    df = df.explode("target_word")
    df = df.dropna(subset=["target_word"])
    # df["preprocessed_target_widx"] = [[0, 0] for _ in range(len(df))]

    dataset = Dataset.from_pandas(df[["target_word", "preprocessed_text"]])
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
    df["preprocessed_target_widx"] = pd.Series([result["pred_target_widx"] for result in results])
    df["pred_lu_name"] = df.apply(
        lambda x: " ".join([x["doc_sentence"].words[idx[0]].lemma for idx in x["preprocessed_target_widx"]]) + ".v", axis=1
    )

    # exemplars: list[RawC4Data] = []
    # with open(args.input_file, "r") as f:
    #     exemplars = [RawC4Data.model_validate_json(line) for line in f]
    #     exemplars = exemplars[args.part_id * 1000 : min((args.part_id + 1) * 1000, len(exemplars))]

    # preprocessed_exemplars: list[C4Data] = []
    # preprocessed_word_lists: list[C4WordList] = []  # 一文ごとのword_list

    # with tqdm(exemplars) as pbar:
    #     pbar.set_description("[preprocessed]")
    #     for exemplar in pbar:
    #         cleaned_text: str = clean_text(exemplar.text)
    #         # doc: list = nlp(cleaned_text)
    #         try:
    #             doc = nlp_with_timeout(nlp, cleaned_text)
    #         except TimeoutError:
    #             print(f"timeout: {exemplar.id_data}")
    #             continue

    #         word_list: C4WordList = make_word_list(exemplar.id_data, doc)

    #         # c4のデータは複数文が含まれるため、文ごとに分割して処理する
    #         separate_idxes: list[int] = [0]  # 文ごとの境目のindex
    #         for i in range(1, len(word_list.words)):
    #             if word_list.words[i - 1].sent_id != word_list.words[i].sent_id:
    #                 separate_idxes.append(i)
    #         separate_idxes.append(-1)

    #         for i in range(1, len(separate_idxes)):
    #             # c4のデータは複数文が含まれるため、文ごとに分割して処理する
    #             words: list[WordInfo] = word_list.words[separate_idxes[i - 1] : separate_idxes[i]]
    #             for word_info in words:
    #                 if word_info.upos == "VERB":
    #                     data: C4Data = C4Data(
    #                         source=exemplar.source,
    #                         id_data=exemplar.id_data,
    #                         target_word=word_info.lemma,
    #                         text=exemplar.text,  # 前処理前の文として分割前の文を格納
    #                         preprocessed_text=" ".join([word.text for word in words]),
    #                         preprocessed_target_widx=[
    #                             word_info.word_idx,
    #                             word_info.word_idx,
    #                             word_info.word_idx,
    #                         ],  # TODO: どこまでを動詞句とするか
    #                         part_id=args.part_id,
    #                     )
    #                     preprocessed_exemplars.append(data)

    #             one_sentence_word_list: C4WordList = C4WordList(id_data=exemplar.id_data, words=words)
    #             preprocessed_word_lists.append(one_sentence_word_list)

    preprocessed_exemplars: list[C4Data] = [
        C4Data(
            source=row["source"],
            id_data=C4Id(**row["id_data"]),
            target_word=row["target_word"],
            text=row["text"],
            preprocessed_text=row["preprocessed_text"],
            preprocessed_target_widx=row["preprocessed_target_widx"],
            part_id=args.part_id,
            pred_lu_name=row["pred_lu_name"],
        )
        for _, row in df.iterrows()
    ]

    with open(args.output_exemplar_file, "w") as f:
        with tqdm(preprocessed_exemplars) as pbar:
            pbar.set_description("[write preprocessed_exemplars]")
            for exemplar in pbar:
                print(exemplar.model_dump_json(), file=f)

    # with open(args.output_wordlist_file, "w") as f:
    #     with tqdm(preprocessed_word_lists) as pbar:
    #         pbar.set_description("[write word_list]")
    #         for word_list in pbar:
    #             print(word_list.model_dump_json(), file=f)


if __name__ == "__main__":
    main()
