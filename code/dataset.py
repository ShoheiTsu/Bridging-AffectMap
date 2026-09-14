"""
OASIS 画像＋ラベルのデータセット（論文1用）
同一テーマ（Acorns 1, Acorns 2 等）は必ず train または val のどちらか一方にまとめる。
"""
from pathlib import Path
import pandas as pd
import numpy as np


def theme_base(theme):
    """'Acorns 1' -> 'Acorns'。同一タイトル群を一つの theme_base にまとめる。"""
    s = str(theme).strip()
    if " " in s:
        parts = s.rsplit(" ", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0]
    return s


def load_oasis_meta(csv_path):
    """oasis_scores.csv を読み込み、image_path が存在する行だけ返す。"""
    df = pd.read_csv(csv_path)
    df = df[df["image_path"].apply(lambda p: Path(p).exists())].reset_index(drop=True)
    return df


def add_theme_base(df):
    """df に theme 列があるとき theme_base 列を追加する。"""
    if "theme" not in df.columns:
        return df
    df = df.copy()
    df["theme_base"] = df["theme"].map(theme_base)
    return df


def train_val_split(df, train_ratio=0.8, random_state=42, return_indices=False):
    """ランダムに train/val に分割（画像単位）。return_indices=True で train_idx, val_idx も返す。"""
    n = len(df)
    idx = np.arange(n)
    np.random.seed(random_state)
    np.random.shuffle(idx)
    n_train = int(n * train_ratio)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]
    if return_indices:
        return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True), train_idx, val_idx
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


def train_val_split_by_theme(df, train_ratio=0.8, random_state=42, return_indices=False):
    """
    テーマ単位で train/val に分割する。
    同一 theme_base（Acorns, Alcohol 等）に属する画像はすべて train かすべて val のどちらかに入る。
    評価時に「Acorns を評価するときは Acorns 1, Acorns 2 を訓練から除く」を満たす。
    """
    df = add_theme_base(df)
    bases = df["theme_base"].unique()
    n_bases = len(bases)
    order = np.arange(n_bases)
    np.random.seed(random_state)
    np.random.shuffle(order)
    n_train_bases = max(1, int(n_bases * train_ratio))
    train_bases = set(bases[order[:n_train_bases]])
    val_bases = set(bases[order[n_train_bases:]])

    train_mask = df["theme_base"].isin(train_bases).values
    val_mask = ~train_mask
    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)
    if return_indices:
        return df_train, df_val, train_idx, val_idx
    return df_train, df_val
