from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class BaseData(BaseModel):
    source: str  # データの取得元(e.g. framenet)
    id_data: object  # 元データの参照に必要な情報を格納
    target_word: str  # 注目する語(基本形)
    target_word_idx: int  # 注目する語の位置(単語レベル)
    preprocessed_text: str  # 前処理後のtext
    preprocessed_target_widx: list[list[int]]  # 前処理後のLUの位置(単語レベル)[開始位置,終了位置,主となる語の位置(構文木)]
    text: str  # 前処理前のtext
    uuid: UUID = Field(default_factory=lambda: uuid4())  # 分割後のデータに対して一意に与える


class WordInfo(BaseModel):
    id: int  # 与えた文章全体で連番(0index)
    text: str
    lemma: str
    upos: str
    xpos: str
    feats: str | None
    head: int
    deprel: str
    start_char: int | None
    end_char: int | None
    children: list[int]
    word_idx: int  # 1つの文章内での連番(0index)
    sent_id: int  # 何番目の文章か(0index)


class WordList(BaseModel):
    id_data: object  # 元データの参照に必要な情報を格納
    words: list[WordInfo]
