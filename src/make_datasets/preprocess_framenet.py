from pathlib import Path
from unicodedata import normalize

import pandas as pd
from base import BaseData, WordInfo, WordList
from collect_framenet import FramenetId
from omegaconf import OmegaConf
from pydantic import BaseModel
from spacy_alignments import get_alignments
from stanza.pipeline.core import Pipeline
from tqdm import tqdm


class Args(BaseModel):
    input_file: Path = Path("./data/framenet/exemplars.jsonl")
    output_exemplar_file: Path = Path("./datasets/framenet/exemplars.jsonl")
    output_wordlist_file: Path = Path("./datasets/framenet/word_list.jsonl")
    device: str = "cuda:0"


class FramenetData(BaseData):
    # 前処理後のframenet
    # source: str  # データの取得元(e.g. framenet)
    id_data: FramenetId  # 元データの参照に必要な情報を格納
    # text: str  # 前処理前のtext
    # target_word: str  # 注目する語(基本形)
    # preprocessed_text: str  # 前処理後のtext
    # preprocessed_target_widx: list[int]  # 前処理後のLUの位置(単語レベル)[開始位置,終了位置,主となる語の位置(構文木)]
    frame_name: str
    frame_id: int
    lu_name: str
    lu_id: int
    fe_widx: list[
        list[list[int | str]] | dict[str, str]
    ]  # 前処理後のfeの位置(単語レベル)[[開始位置,終了位置,主となる語の位置(構文木)],{省略されているfe名,省略の種類}]


class FramenetWordList(WordList):
    id_data: FramenetId
    # words: list[WordInfo]


def get_target_word_idx(text: str, preprocessed_text: str, target: list[list[int]]) -> list[list[int]]:
    char_to_word, _ = get_alignments(list(text), preprocessed_text.split())
    return [[char_to_word[t[0]][0], char_to_word[t[1] - 1][0]] for t in target]


def get_fe_word_idx(text: str, preprocessed_text: str, fe: list[list[list[int | str]] | dict]) -> list[list[int]]:
    char_to_word, _ = get_alignments(list(text), preprocessed_text.split())
    ret: list[list[list[int | str]] | dict] = [[], {}]
    ret[0] = [[char_to_word[f[0]][0], char_to_word[f[1] - 1][0], f[2]] for f in fe[0]]
    ret[1] = fe[1]
    return ret


def make_word_list(id_data: FramenetId, doc: list[list]) -> FramenetWordList:
    # 構文解析の結果を整理して返す
    ret: FramenetWordList = FramenetWordList(id_data=id_data, words=[])
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


def main():
    # OmegaConfを用いて実験設定を読み込む
    args = Args(**OmegaConf.from_cli())
    # outputディレクトリの作成
    args.output_exemplar_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_wordlist_file.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_json(args.input_file, lines=True)
    df = df[df["lu_name"].str.contains(r"\.v")]  # 動詞を抽出
    df = df[
        df["lu_name"].apply(lambda lu_name: len(lu_name.split())) == df["target"].apply(lambda target: len(target))
    ]  # LUの単語数とtargetの単語数が一致するものを抽出

    df["preprocessed_text"] = df["text"].apply(lambda x: normalize("NFKC", x))  # Unicode正規化
    df["preprocessed_target_widx"] = df.apply(
        lambda row: get_target_word_idx(row["text"], row["preprocessed_text"], row["target"]), axis=1
    )  # targetの位置を単語単位に変換
    df["fe_widx"] = df.apply(
        lambda row: get_fe_word_idx(row["text"], row["preprocessed_text"], row["fe"]), axis=1
    )  # feの位置を単語単位に変換

    preprocessed_exemplars: list[FramenetData] = [
        FramenetData(
            source=row["source"],
            id_data=FramenetId(id=row["id_data"]["id"]),
            text=row["text"],
            target_word=row["lu_name"],
            preprocessed_text=row["preprocessed_text"],
            preprocessed_target_widx=row["preprocessed_target_widx"],
            frame_name=row["frame_name"],
            frame_id=row["frame_id"],
            lu_name=row["lu_name"],
            lu_id=row["lu_id"],
            fe_widx=row["fe_widx"],
        )
        for _, row in df.iterrows()
    ]

    nlp = Pipeline(
        "en",
        processors="tokenize,mwt,pos,lemma,depparse",
        use_gpu=True,
        device=args.device,
        pos_batch_size=9000,
    )

    docs = [nlp(text) for text in tqdm(df["preprocessed_text"], desc="Processing texts with NLP")]

    word_lists: list[FramenetWordList] = [
        make_word_list(FramenetId(id=row["id_data"]["id"]), doc) for doc, (_, row) in zip(docs, df.iterrows(), strict=True)
    ]

    with open(args.output_exemplar_file, "w") as f:
        with tqdm(preprocessed_exemplars) as pbar:
            pbar.set_description("[write preprocessed_exemplars]")
            for exemplar in pbar:
                print(exemplar.model_dump_json(), file=f)

    with open(args.output_wordlist_file, "w") as f:
        with tqdm(word_lists) as pbar:
            pbar.set_description("[write word_list]")
            for word_list in pbar:
                print(word_list.model_dump_json(), file=f)


if __name__ == "__main__":
    main()
