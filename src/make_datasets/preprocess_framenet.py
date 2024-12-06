import re
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

# tqdmをpandasのapplyメソッドで使用できるように設定
tqdm.pandas()


class Args(BaseModel):
    input_file: Path = Path("./data/framenet/exemplars.jsonl")
    output_exemplar_file: Path = Path("./datasets/framenet/exemplars.jsonl")
    output_wordlist_file: Path = Path("./datasets/framenet/word_list.jsonl")
    device: str = "cuda:0"


class FramenetData(BaseData):
    # 前処理後のframenet
    # source: str  # データの取得元(e.g. framenet)
    id_data: FramenetId  # 元データの参照に必要な情報を格納
    # target_word: str  # 注目する語(text内に現れる形)
    # target_word_idx: list[int]  # 注目する語の位置(文字レベル)[開始位置,終了位置]
    # preprocessed_text: str  # 前処理後のtext
    # preprocessed_lu_idx: list[list[int]]  # 前処理後のLUの位置(文字レベル)[開始位置,終了位置]
    # text: str  # 前処理前のtext
    # uuid: UUID = Field(default_factory=lambda: uuid4())  # 分割後のデータに対して一意に与える
    frame_name: str
    frame_id: int
    lu_name: str
    lu_id: int
    fe_idx: list[
        list[list[int | str]] | dict[str, str]
    ]  # 前処理後のfeの位置(文字レベル)[[開始位置,終了位置],{省略されているfe名,省略の種類}]


class FramenetWordList(WordList):
    id_data: FramenetId
    # words: list[WordInfo]


def get_lu_idx(text: str, preprocessed_text: str, target: list[list[int]]) -> list[list[int]]:
    # 前処理前後でtargetの位置を揃える(文字レベル)
    text_to_preprocessed_text, _ = get_alignments(list(text), list(preprocessed_text))
    return [[text_to_preprocessed_text[t[0]][0], text_to_preprocessed_text[t[1] - 1][-1] + 1] for t in target]


def get_fe_idx(text: str, preprocessed_text: str, fe: list[list[list[int | str]] | dict]) -> list[list[int]]:
    # 前処理前後でfeの位置を揃える(文字レベル)
    text_to_preprocessed_text, preprocessed_text_to_text = get_alignments(list(text), list(preprocessed_text))
    ret: list[list[list[int | str]] | dict] = [[], {}]
    ret[0] = [[text_to_preprocessed_text[f[0]][0], text_to_preprocessed_text[f[1] - 1][-1] + 1, f[2]] for f in fe[0]]
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


def get_verb_idx(doc: list[list]) -> int:
    # lu_nameをstanzaにかけたものからrootを取得
    for sentence in doc.sentences:
        for word in sentence.words:
            if word.deprel == "root":
                return word.id - 1


def main():
    # OmegaConfを用いて実験設定を読み込む
    args = Args(**OmegaConf.from_cli())
    print(args)  # 引数を表示
    # outputディレクトリの作成
    args.output_exemplar_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_wordlist_file.parent.mkdir(parents=True, exist_ok=True)

    nlp = Pipeline(
        "en",
        processors="tokenize,mwt,pos,lemma,depparse",
        use_gpu=True,
        device=args.device,
        pos_batch_size=9000,
    )

    df = pd.read_json(args.input_file, lines=True)

    df = df[df["lu_name"].str.contains(r"\.v")]  # 動詞を抽出
    df = df[
        df["lu_name"].apply(lambda lu_name: len(re.sub(r"(\.v)|(\[.*?\])|(\(.*?\))", "", lu_name).split()))
        == df["target"].apply(lambda target: len(target))
    ]  # LUの単語数とtargetの単語数が一致するものを抽出(アノテーションミスと見られるものを省く)

    tqdm.pandas(desc="doc")
    df["doc"] = df["text"].progress_apply(lambda x: nlp(normalize("NFKC", x)))  # Unicode正規化

    tqdm.pandas(desc="preprocessed_text")
    df["preprocessed_text"] = df["doc"].progress_apply(
        lambda x: " ".join([word.text for sent in x.sentences for word in sent.words])
        + " "  # 末尾に空白を追加しないと[,,,"a"," "]と[,,,," ","a"]のアラインメントをとる時におかしくなる
    )

    tqdm.pandas(desc="preprocessed_lu_idx")
    df["preprocessed_lu_idx"] = df.progress_apply(
        lambda row: get_lu_idx(row["text"], row["preprocessed_text"], row["target"]), axis=1
    )

    tqdm.pandas(desc="fe_idx")
    df["fe_idx"] = df.progress_apply(
        lambda row: get_fe_idx(row["text"], row["preprocessed_text"], row["fe"]), axis=1
    )  # feの位置を単語単位に変換

    tqdm.pandas(desc="target_word_idx")
    df["target_word_idx"] = df.progress_apply(
        lambda row: row["preprocessed_lu_idx"][0]
        if " " not in row["lu_name"]
        else row["preprocessed_lu_idx"][get_verb_idx(nlp(re.sub(r"(\.v)|(\[.*?\])|(\(.*?\))", "", row["lu_name"]).strip()))],
        axis=1,
    )  # 注目する単語(動詞)の位置を取得

    tqdm.pandas(desc="target_word")
    df["target_word"] = df.progress_apply(
        lambda x: x["preprocessed_text"][x["target_word_idx"][0] : x["target_word_idx"][1]], axis=1
    )  # 注目する単語(動詞)を取得

    df = df.drop_duplicates(subset=["preprocessed_text", "target_word"])  # 重複を削除

    preprocessed_exemplars: list[FramenetData] = [
        FramenetData(
            source=row["source"],
            id_data=FramenetId(id=row["id_data"]["id"]),
            text=row["text"],
            target_word=row["target_word"],
            target_word_idx=row["target_word_idx"],
            preprocessed_text=row["preprocessed_text"],
            preprocessed_lu_idx=row["preprocessed_lu_idx"],
            frame_name=row["frame_name"],
            frame_id=row["frame_id"],
            lu_name=row["lu_name"],
            lu_id=row["lu_id"],
            fe_idx=row["fe_idx"],
        )
        for _, row in df.iterrows()
    ]

    word_lists: list[FramenetWordList] = [
        make_word_list(FramenetId(id=dictionary["id_data"]["id"]), dictionary["doc"])
        for dictionary in df[["doc", "id_data"]].to_dict(orient="records")
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
