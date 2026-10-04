#!/usr/bin/env python3
"""
Step 1 보조 통계: 신뢰구간과 조건 간 차이.

기존 평가 출력(results/step1/<model>/<original|augmented>/eval/<eval_name>/predictions.csv)을
읽기만 하고, 기존 출력에 없는 통계를 results/stats/ 아래에 추가로 만든다.
지표 정의(confusion_matrix labels=[0,1], zero_division=0, 분모 0이면 0.0)와
changed 정규화 규칙은 evaluate.py / analyze_obfuscated_eval.py와 같다.

계산:
  1. 조건별(model x training) clean / obfuscated(changed=true) Recall, FPR, F1, Precision
     - clean 계열(원문 1문장 = 1행): Recall, FPR은 Wilson 95% 구간
     - obfuscated 계열(원문 1개 = 변형 여러 행): Recall, FPR도 seed_id 클러스터 부트스트랩이
       기본 구간이고 Wilson은 참고 열(*_wilson_*)로만 저장
     - 성공이 0건이거나 전체이면 부트스트랩 구간이 0폭이 되므로 seed_id 수를 n으로 한 Wilson 구간으로
       대체한다(ci_method = wilson_cluster_fallback, markdown 표에 †).
     - F1, Precision: seed_id 클러스터 부트스트랩 95% 구간
     - 재표본 결과가 전부 같아 구간 폭이 0이면(F1·Precision·paired 차이) '산출 불가'(not_computable)
  2. Original vs Augmented (같은 모델, 같은 eval_name) paired 차이
     (Augmented - Original) - 같은 id로 결합, seed_id 클러스터 부트스트랩
  3. 원문 vs 변형 paired: 원문(clean, id)에서 맞힌 공격 중 변형(obfuscated, seed_id) 후에도
     맞힌 비율 (원문 1 : 변형 다)
  4. 기법 분해: 학습 기법(같은 기법·같은 강도) / 같은 기법·다른 강도 / 나머지 기법
  5. source별 Recall, FPR
  6. 정상 문장(label 0)의 난독화 오탐률 (changed=true)

행 키는 입력 파일에서 복사된 id이다. 평가 입력 파일도 함께 받아 predictions.csv의 id와
대조하고, 빠진 행 / 중복 id / 여분 행이 있으면 목록을 보여주고 멈춘다.

사용법:
    python scripts/step1_stats.py \\
        --clean-input data/step1/test.jsonl \\
        --obfuscated-input data/step1/obfuscated_test.jsonl
"""

import argparse
import hashlib
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = {
    "koelectra": "KoELECTRA",
    "mdeberta": "mDeBERTa",
}

TRAININGS = {
    "original": "Original",
    "augmented": "Augmented",
}

# (suite 이름, clean 계열 eval_name, obfuscated 계열 eval_name)
SUITES = (
    ("main", "clean", "obfuscated"),
    ("kg", "kg_clean", "kg_obfuscated"),
)

WILSON_Z = 1.96

# 구간 산출 방식 (csv의 *ci_method 열)
METHOD_WILSON = "wilson"
METHOD_BOOTSTRAP = "cluster_bootstrap"
METHOD_FALLBACK = "wilson_cluster_fallback"
METHOD_NA = "not_computable"

PRINT_LIMIT = 20


class StatsError(Exception):
    pass


# --------------------------------------------------
# changed 정규화 (analyze_obfuscated_eval.py의 normalize_changed와 같은 규칙)
# --------------------------------------------------

def normalize_changed(series):
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "1.0": True,
        "0.0": False,
    }

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )

    if normalized.isna().any():
        bad = sorted(
            series.loc[normalized.isna()]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            "Invalid changed values: "
            f"{bad}"
        )

    return normalized.astype(bool)


# --------------------------------------------------
# 지표 (evaluate.py의 compute_binary_metrics와 같은 정의)
# --------------------------------------------------

def safe_ratio(numerator, denominator):
    """분모가 0이면 0.0. 스칼라와 배열 모두 지원."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)

    out = np.zeros(
        np.broadcast(numerator, denominator).shape,
        dtype=float,
    )

    np.divide(
        numerator,
        denominator,
        out=out,
        where=denominator > 0,
    )

    return out


def metrics_from_counts(tp, fp, fn, tn):
    """tp/fp/fn/tn(스칼라 또는 배열)에서 precision, recall, f1, fpr, fnr."""
    tp = np.asarray(tp, dtype=float)
    fp = np.asarray(fp, dtype=float)
    fn = np.asarray(fn, dtype=float)
    tn = np.asarray(tn, dtype=float)

    return {
        "precision": safe_ratio(tp, tp + fp),
        "recall": safe_ratio(tp, tp + fn),
        "f1": safe_ratio(2 * tp, 2 * tp + fp + fn),
        "fpr": safe_ratio(fp, fp + tn),
        "fnr": safe_ratio(fn, fn + tp),
    }


def confusion_counts(y_true, y_pred):
    """confusion_matrix(labels=[0,1]).ravel()와 같은 순서가 아니라 dict로 반환."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return {
        "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
        "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
        "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
        "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
    }


def point_metrics(df):
    counts = confusion_counts(
        df["label"].to_numpy(),
        df["prediction"].to_numpy(),
    )

    metrics = metrics_from_counts(
        counts["tp"],
        counts["fp"],
        counts["fn"],
        counts["tn"],
    )

    out = {"n": int(len(df))}
    out.update(counts)
    out.update({k: float(v) for k, v in metrics.items()})

    return out


# --------------------------------------------------
# 구간 추정
# --------------------------------------------------

def wilson_interval(successes, n, z=WILSON_Z):
    """Wilson score 구간. n=0이면 (nan, nan)."""
    if n <= 0:
        return (float("nan"), float("nan"))

    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (
        z
        * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
        / denom
    )

    return (
        float(max(0.0, center - half)),
        float(min(1.0, center + half)),
    )


def make_rng(seed, key):
    """분석 단위마다 독립적이고 실행 순서와 무관한 난수 생성기."""
    return np.random.default_rng(
        [seed, zlib.crc32(key.encode("utf-8"))]
    )


def cluster_bootstrap(count_arrays, stat_fn, n_boot, rng, chunk=250):
    """
    클러스터(seed_id) 부트스트랩.

    count_arrays: 같은 클러스터 순서의 (K, m) 정수 배열 목록.
    stat_fn: 재표본 합계 (B, m) 배열들을 받아 (B, q) 통계를 반환.
    반환: (n_boot, q)
    """
    n_clusters = count_arrays[0].shape[0]

    if n_clusters == 0:
        width = stat_fn(*[np.zeros((1, arr.shape[1]), dtype=np.int64) for arr in count_arrays]).shape[1]

        return np.full((n_boot, width), np.nan)

    chunks = []
    done = 0

    while done < n_boot:
        size = min(chunk, n_boot - done)

        idx = rng.integers(
            0,
            n_clusters,
            size=(size, n_clusters),
        )

        sums = [arr[idx].sum(axis=1) for arr in count_arrays]

        chunks.append(stat_fn(*sums))
        done += size

    return np.concatenate(chunks, axis=0)


def percentile_interval(samples):
    """부트스트랩 표본 (n_boot, q)에서 열별 95% 백분위 구간."""
    lo = np.percentile(samples, 2.5, axis=0)
    hi = np.percentile(samples, 97.5, axis=0)

    return lo, hi


def bootstrap_ci(samples):
    """
    부트스트랩 표본 (n_boot, q)에서 열별 95% 백분위 구간과 산출 방식.
    표본이 전부 같은 값(폭 0)이거나 nan이면 구간을 만들지 않고 not_computable로 표시한다.
    반환: lo (q,), hi (q,), methods (길이 q 목록)
    """
    lo = np.percentile(samples, 2.5, axis=0).astype(float)
    hi = np.percentile(samples, 97.5, axis=0).astype(float)

    methods = []

    for j in range(samples.shape[1]):
        col = samples[:, j]

        if np.isnan(col).any() or np.ptp(col) == 0:
            lo[j], hi[j] = float("nan"), float("nan")
            methods.append(METHOD_NA)
        else:
            methods.append(METHOD_BOOTSTRAP)

    return lo, hi, methods


def counts_by_cluster(df):
    """seed_id별 [tp, fp, fn, tn] 배열 (seed_id 정렬)."""
    work = pd.DataFrame({
        "seed_id": df["seed_id"].to_numpy(),
        "tp": ((df["label"] == 1) & (df["prediction"] == 1)).astype(int).to_numpy(),
        "fp": ((df["label"] == 0) & (df["prediction"] == 1)).astype(int).to_numpy(),
        "fn": ((df["label"] == 1) & (df["prediction"] == 0)).astype(int).to_numpy(),
        "tn": ((df["label"] == 0) & (df["prediction"] == 0)).astype(int).to_numpy(),
    })

    grouped = work.groupby("seed_id", sort=True)[["tp", "fp", "fn", "tn"]].sum()

    return grouped


def ci_method_for(kind):
    """clean 계열(원문 1문장 = 1행)은 Wilson, obfuscated 계열(원문 1개 = 변형 여러 행)은 클러스터 부트스트랩."""
    return METHOD_WILSON if kind == "clean" else METHOD_BOOTSTRAP


def rate_counts_by_cluster(df, label):
    """label 행에서 seed_id별 [예측 1 건수, 행 수] 배열 (Recall: label=1, FPR: label=0)."""
    part = df.loc[df["label"] == label]

    hit = (part["prediction"] == 1).astype(int)

    grouped = (
        pd.DataFrame({"seed_id": part["seed_id"].to_numpy(), "hit": hit.to_numpy()})
        .groupby("seed_id", sort=True)["hit"]
        .agg(["sum", "count"])
    )

    return grouped.to_numpy(dtype=np.int64)


def stats_rate(sums):
    """합계 (B, 2: hit, n) -> (B, 1) 비율."""
    return safe_ratio(sums[:, 0], sums[:, 1])[:, None]


def proportion_interval(hits, n, n_clusters, samples_fn):
    """
    비율의 95% 구간. 성공이 0건이거나 전체(hits == n)이면 클러스터 부트스트랩 표본이 전부 같아져
    구간 폭이 0이 되므로, 클러스터(seed_id) 수를 n으로 한 Wilson 구간을 대신 쓴다.
    반환: (lo, hi, method)
    """
    if n_clusters == 0:
        return (float("nan"), float("nan"), METHOD_NA)

    if hits == 0 or hits == n:
        lo, hi = wilson_interval(n_clusters if hits == n else 0, n_clusters)

        return (lo, hi, METHOD_FALLBACK)

    lo, hi, methods = bootstrap_ci(samples_fn())

    return (float(lo[0]), float(hi[0]), methods[0])


def rate_interval(df, label, method, args, key):
    """
    label 행 중 예측 1의 비율(Recall 또는 FPR)의 95% 구간 (lo, hi, method).
    method='wilson'은 행을 독립으로 보고, 'cluster_bootstrap'은 seed_id를 클러스터로 재표본한다.
    클러스터 부트스트랩에서 성공이 0건이거나 전체이면 Wilson(n = 클러스터 수)으로 대체한다.
    """
    part = df.loc[df["label"] == label]

    if method == METHOD_WILSON:
        lo, hi = wilson_interval(int((part["prediction"] == 1).sum()), len(part))

        return (lo, hi, METHOD_WILSON)

    arr = rate_counts_by_cluster(df, label)

    hits = int(arr[:, 0].sum()) if len(arr) else 0
    n = int(arr[:, 1].sum()) if len(arr) else 0

    def samples_fn():
        return cluster_bootstrap(
            [arr],
            stats_rate,
            args.n_boot,
            make_rng(args.seed, f"rate|{label}|{key}"),
        )

    return proportion_interval(hits, n, len(arr), samples_fn)


def rate_block(df, method, args, key):
    """Recall / FPR의 기본 구간(method)과 참고용 Wilson 구간."""
    attack = df.loc[df["label"] == 1]
    benign = df.loc[df["label"] == 0]

    n_attack, n_benign = len(attack), len(benign)

    tp = int((attack["prediction"] == 1).sum())
    fp = int((benign["prediction"] == 1).sum())

    rec_ci = rate_interval(df, 1, method, args, key)
    fpr_ci = rate_interval(df, 0, method, args, key)

    rec_w = wilson_interval(tp, n_attack)
    fpr_w = wilson_interval(fp, n_benign)

    return {
        "recall_ci_method": rec_ci[2],
        "recall_ci_lo": rec_ci[0],
        "recall_ci_hi": rec_ci[1],
        "recall_wilson_lo": rec_w[0],
        "recall_wilson_hi": rec_w[1],
        "fpr_ci_method": fpr_ci[2],
        "fpr_ci_lo": fpr_ci[0],
        "fpr_ci_hi": fpr_ci[1],
        "fpr_wilson_lo": fpr_w[0],
        "fpr_wilson_hi": fpr_w[1],
    }


def stats_single(sums):
    """합계 (B, 4: tp, fp, fn, tn) -> (B, 4: recall, fpr, f1, precision)."""
    m = metrics_from_counts(
        sums[:, 0],
        sums[:, 1],
        sums[:, 2],
        sums[:, 3],
    )

    return np.stack(
        [m["recall"], m["fpr"], m["f1"], m["precision"]],
        axis=1,
    )


def stats_delta(base, other):
    """(base, other) 합계 -> (B, 3: dRecall, dFPR, dF1) = other - base."""
    mb = metrics_from_counts(base[:, 0], base[:, 1], base[:, 2], base[:, 3])
    mo = metrics_from_counts(other[:, 0], other[:, 1], other[:, 2], other[:, 3])

    return np.stack(
        [
            mo["recall"] - mb["recall"],
            mo["fpr"] - mb["fpr"],
            mo["f1"] - mb["f1"],
        ],
        axis=1,
    )


# --------------------------------------------------
# 입력 읽기 / 검증
# --------------------------------------------------

def load_table(path):
    path = Path(path)

    if not path.exists():
        raise StatsError(f"파일이 없음: {path}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(
            path,
            dtype={"id": str, "seed_id": str},
        )
    elif suffix == ".jsonl":
        rows = []

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))

        df = pd.DataFrame(rows)
    else:
        raise StatsError(
            f"지원하는 형식은 .csv, .jsonl 입니다: {path}"
        )

    for col in ("id", "seed_id"):
        if col in df.columns:
            df[col] = df[col].where(df[col].isna(), df[col].astype(str))

    return df


def prepare_input(df, name):
    """evaluate.py의 prepare_dataframe과 같은 방식으로 행을 정리(text/label 결측 행 제거)."""
    required = {"id", "text", "label"}

    if not required.issubset(df.columns):
        raise StatsError(
            f"[{name}] 필수 열이 없음. 필요: {sorted(required)}, "
            f"있음: {sorted(df.columns.tolist())}"
        )

    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    df = df.copy()
    df["label"] = df["label"].astype(int)

    if df["id"].isna().any():
        raise StatsError(f"[{name}] id가 비어 있는 행이 있음")

    df["id"] = df["id"].astype(str)

    return df


def _show(items, limit=PRINT_LIMIT):
    items = list(items)
    head = ", ".join(str(x) for x in items[:limit])

    if len(items) > limit:
        head += f", ... (총 {len(items)}건)"

    return head


def canonical(col, series):
    """입력 파일과 predictions.csv의 같은 열을 비교하기 위한 표준형."""
    if col == "changed":
        return normalize_changed(series).astype(str)

    if col == "intensity":
        return pd.to_numeric(series, errors="coerce").round(9).astype(str)

    if col == "label":
        return series.astype(int).astype(str)

    return series.where(series.isna(), series.astype(str)).fillna("")


def check_alignment(name, pred, source, compare_cols):
    """
    predictions.csv와 평가 입력 파일의 id를 대조한다.
    빠진 행 / 중복 id / 여분 행이 있으면 목록을 보여주고 멈춘다.
    같은 id의 label과 compare_cols 값이 다르면 그것도 멈춘다.
    """
    problems = []

    dup_input = sorted(set(source.loc[source["id"].duplicated(), "id"]))
    dup_pred = sorted(set(pred.loc[pred["id"].duplicated(), "id"]))

    source_ids = set(source["id"])
    pred_ids = set(pred["id"])

    missing = sorted(source_ids - pred_ids)
    extra = sorted(pred_ids - source_ids)

    if dup_input:
        problems.append(f"입력 파일에 중복 id {len(dup_input)}건: {_show(dup_input)}")

    if dup_pred:
        problems.append(f"predictions.csv에 중복 id {len(dup_pred)}건: {_show(dup_pred)}")

    if missing:
        problems.append(f"predictions.csv에 빠진 행 {len(missing)}건: {_show(missing)}")

    if extra:
        problems.append(f"predictions.csv에 여분 행 {len(extra)}건: {_show(extra)}")

    if not problems:
        merged = source.merge(
            pred,
            on="id",
            suffixes=("_in", "_pred"),
        )

        for col in ["label"] + list(compare_cols):
            col_in = f"{col}_in"
            col_pred = f"{col}_pred"

            if col_in not in merged.columns or col_pred not in merged.columns:
                continue

            a = canonical(col, merged[col_in])
            b = canonical(col, merged[col_pred])
            diff = merged.loc[a != b, "id"].tolist()

            if diff:
                problems.append(
                    f"'{col}' 값이 입력 파일과 다른 id {len(diff)}건: {_show(diff)}"
                )

    if problems:
        raise StatsError(
            f"[{name}] predictions.csv와 평가 입력 파일이 맞지 않음\n  - "
            + "\n  - ".join(problems)
        )


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)

    return h.hexdigest()


# --------------------------------------------------
# 조건 로딩
# --------------------------------------------------

def build_frame(pred, kind, source_lookup=None):
    """
    분석용 프레임을 만든다.
    clean: 모든 행, seed_id = id.
    obfuscated: changed=true 행만(원문 행처럼 seed_id가 비어 있으면 seed_id = id).
    """
    df = pred.copy()

    df["label"] = df["label"].astype(int)
    df["prediction"] = df["prediction"].astype(int)

    bad_pred = sorted(set(df["prediction"].unique().tolist()) - {0, 1})

    if bad_pred:
        raise StatsError(f"prediction은 0 또는 1이어야 함. 발견: {bad_pred}")

    if "seed_id" not in df.columns:
        df["seed_id"] = np.nan

    empty = df["seed_id"].isna() | (df["seed_id"].astype(str).str.strip() == "")

    df["seed_id"] = df["seed_id"].astype(object)
    df.loc[empty, "seed_id"] = df.loc[empty, "id"]
    df["seed_id"] = df["seed_id"].astype(str)

    if kind == "obfuscated":
        df["_changed"] = normalize_changed(df["changed"])
        df = df.loc[df["_changed"]].copy()
        df["intensity"] = pd.to_numeric(df["intensity"], errors="coerce")
    else:
        df["_changed"] = True

    if "source" not in df.columns:
        df["source"] = np.nan

    if source_lookup is not None:
        missing = df["source"].isna()
        df.loc[missing, "source"] = df.loc[missing, "seed_id"].map(source_lookup)

    df["source"] = df["source"].fillna("(unknown)").astype(str)

    return df.reset_index(drop=True)


def crosscheck_with_existing(eval_dir, kind, pred_df, frame):
    """채원님 metrics.json / analysis csv의 confusion count와 점추정이 같은지 확인."""
    problems = []

    metrics_path = eval_dir / "metrics.json"

    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as f:
            stored = json.load(f)

        mine = confusion_counts(
            pred_df["label"].astype(int).to_numpy(),
            pred_df["prediction"].astype(int).to_numpy(),
        )

        for key in ("tp", "fp", "fn", "tn"):
            if int(stored[key]) != mine[key]:
                problems.append(f"metrics.json {key}={stored[key]} / 재계산 {mine[key]}")

        if int(stored["n"]) != len(pred_df):
            problems.append(f"metrics.json n={stored['n']} / 재계산 {len(pred_df)}")

    if kind == "obfuscated":
        overall_path = eval_dir / "analysis" / "obfuscated_overall.csv"

        if overall_path.exists():
            overall = pd.read_csv(overall_path)
            row = overall.loc[overall["scope"] == "changed_only"]

            if len(row) == 1:
                row = row.iloc[0]
                mine = confusion_counts(
                    frame["label"].to_numpy(),
                    frame["prediction"].to_numpy(),
                )

                for key in ("tp", "fp", "fn", "tn"):
                    if int(row[key]) != mine[key]:
                        problems.append(
                            f"obfuscated_overall.csv changed_only {key}={row[key]} / 재계산 {mine[key]}"
                        )

    if problems:
        raise StatsError(
            f"[{eval_dir}] 기존 결과와 점추정이 다름\n  - " + "\n  - ".join(problems)
        )


def load_conditions(args, inputs, log):
    """results/step1/<model>/<training>/eval/<eval_name>/predictions.csv를 모두 읽어 검증."""
    root = Path(args.results_root)

    conditions = {}
    run_files = []

    for suite, clean_name, obf_name in active_suites(args):
        for model in args.models:
            for training in args.trainings:
                for eval_name, kind in ((clean_name, "clean"), (obf_name, "obfuscated")):
                    eval_dir = root / model / training / "eval" / eval_name
                    pred_path = eval_dir / "predictions.csv"

                    if not pred_path.exists():
                        raise StatsError(f"predictions.csv가 없음: {pred_path}")

                    pred = load_table(pred_path)

                    if "prediction" not in pred.columns:
                        raise StatsError(
                            f"[{pred_path}] 'prediction' 열이 없음. 열: {pred.columns.tolist()}"
                        )

                    if "id" not in pred.columns:
                        raise StatsError(f"[{pred_path}] 'id' 열이 없음")

                    pred = pred.copy()
                    pred["id"] = pred["id"].astype(str)

                    source_df = inputs[eval_name]

                    compare = (
                        ["seed_id", "technique", "intensity", "changed"]
                        if kind == "obfuscated"
                        else []
                    )

                    if kind == "obfuscated":
                        for col in ("technique", "intensity", "changed"):
                            if col not in pred.columns:
                                raise StatsError(f"[{pred_path}] '{col}' 열이 없음")

                    check_alignment(
                        f"{model}/{training}/{eval_name}",
                        pred,
                        source_df,
                        compare,
                    )

                    conditions[(suite, model, training, kind)] = {
                        "pred": pred,
                        "eval_dir": eval_dir,
                        "eval_name": eval_name,
                    }

                    run_files.append({
                        "path": str(pred_path),
                        "rows": int(len(pred)),
                        "sha256": sha256_file(pred_path),
                    })

    # 프레임 구성 (obfuscated는 같은 조건의 clean에서 source를 보충할 수 있음)
    for (suite, model, training, kind), cond in conditions.items():
        if kind == "clean":
            cond["frame"] = build_frame(cond["pred"], "clean")

    for (suite, model, training, kind), cond in conditions.items():
        if kind == "obfuscated":
            clean = conditions[(suite, model, training, "clean")]["frame"]
            lookup = dict(zip(clean["id"], clean["source"]))
            cond["frame"] = build_frame(cond["pred"], "obfuscated", lookup)

    if not args.skip_crosscheck:
        for (suite, model, training, kind), cond in conditions.items():
            crosscheck_with_existing(
                cond["eval_dir"],
                kind,
                cond["pred"],
                cond["frame"],
            )

        log.append("기존 metrics.json / obfuscated_overall.csv와 confusion count 일치 확인")

    return conditions, run_files


def ordered_conditions(conditions, args):
    """표와 csv의 행 순서: suite, --models 순서, --trainings 순서, clean 다음 obfuscated."""
    for suite, _, _ in active_suites(args):
        for model in args.models:
            for training in args.trainings:
                for kind in ("clean", "obfuscated"):
                    key = (suite, model, training, kind)

                    yield key, conditions[key]


def active_suites(args):
    suites = [SUITES[0]]

    if args.kg_clean_input and args.kg_obfuscated_input:
        suites.append(SUITES[1])

    return suites


# --------------------------------------------------
# 분석 1: 조건별 지표
# --------------------------------------------------

def analyze_conditions(conditions, args):
    rows = []

    for (suite, model, training, kind), cond in ordered_conditions(conditions, args):
        df = cond["frame"]
        point = point_metrics(df)

        method = ci_method_for(kind)
        key = f"cond|{suite}|{model}|{training}|{kind}"

        block = rate_block(df, method, args, key)

        counts = counts_by_cluster(df)

        # F1, Precision은 두 클래스를 함께 쓰므로 seed_id 클러스터를 재표본한다.
        samples = cluster_bootstrap(
            [counts.to_numpy()],
            stats_single,
            args.n_boot,
            make_rng(args.seed, key),
        )

        lo, hi, methods = bootstrap_ci(samples)

        row = {
            "suite": suite,
            "model": model,
            "training": training,
            "eval_name": cond["eval_name"],
            "scope": "clean_all_rows" if kind == "clean" else "changed_only",
            "n": point["n"],
            "n_attack": point["tp"] + point["fn"],
            "n_benign": point["fp"] + point["tn"],
            "n_clusters": int(len(counts)),
            "tp": point["tp"],
            "fp": point["fp"],
            "fn": point["fn"],
            "tn": point["tn"],
            "recall": point["recall"],
            "fpr": point["fpr"],
            "f1": point["f1"],
            "f1_ci_method": methods[2],
            "f1_boot_lo": lo[2],
            "f1_boot_hi": hi[2],
            "precision": point["precision"],
            "precision_ci_method": methods[3],
            "precision_boot_lo": lo[3],
            "precision_boot_hi": hi[3],
        }

        row.update(block)

        rows.append(row)

    return pd.DataFrame(rows)


# --------------------------------------------------
# 분석 2: Original vs Augmented paired 차이
# --------------------------------------------------

def analyze_original_vs_augmented(conditions, args):
    rows = []

    if not {"original", "augmented"}.issubset(set(args.trainings)):
        return pd.DataFrame(rows)

    for suite, clean_name, obf_name in active_suites(args):
        for model in args.models:
            for kind, eval_name in (("clean", clean_name), ("obfuscated", obf_name)):
                base = conditions[(suite, model, "original", kind)]["frame"]
                other = conditions[(suite, model, "augmented", kind)]["frame"]

                merged = base.merge(
                    other[["id", "prediction"]],
                    on="id",
                    suffixes=("_orig", "_aug"),
                )

                if len(merged) != len(base) or len(merged) != len(other):
                    raise StatsError(
                        f"[{model}/{eval_name}] Original/Augmented의 평가 행 집합이 다름"
                    )

                paired_base = merged.rename(columns={"prediction_orig": "prediction"})
                paired_other = merged.rename(columns={"prediction_aug": "prediction"})

                counts_base = counts_by_cluster(paired_base)
                counts_other = counts_by_cluster(paired_other)

                pm_base = point_metrics(paired_base)
                pm_other = point_metrics(paired_other)

                samples = cluster_bootstrap(
                    [counts_base.to_numpy(), counts_other.to_numpy()],
                    stats_delta,
                    args.n_boot,
                    make_rng(args.seed, f"delta|{suite}|{model}|{kind}"),
                )

                # 두 조건의 결과가 모두 한쪽으로 같아 차이가 재표본마다 같은 값이면 구간을 만들지 않는다
                lo, hi, methods = bootstrap_ci(samples)

                for i, metric in enumerate(("recall", "fpr", "f1")):
                    rows.append({
                        "suite": suite,
                        "model": model,
                        "eval_name": eval_name,
                        "scope": "clean_all_rows" if kind == "clean" else "changed_only",
                        "metric": metric,
                        "n": int(len(merged)),
                        "n_clusters": int(len(counts_base)),
                        "original": pm_base[metric],
                        "augmented": pm_other[metric],
                        "delta_aug_minus_orig": pm_other[metric] - pm_base[metric],
                        "ci_method": methods[i],
                        "delta_boot_lo": lo[i],
                        "delta_boot_hi": hi[i],
                    })

    return pd.DataFrame(rows)


# --------------------------------------------------
# 분석 3: 원문 vs 변형 paired
# --------------------------------------------------

def analyze_original_vs_variant(conditions, args):
    rows = []

    for suite, clean_name, obf_name in active_suites(args):
        for model in args.models:
            for training in args.trainings:
                clean = conditions[(suite, model, training, "clean")]["frame"]
                obf = conditions[(suite, model, training, "obfuscated")]["frame"]

                original = clean.set_index("id")

                unknown = sorted(set(obf["seed_id"]) - set(original.index))

                if unknown:
                    raise StatsError(
                        f"[{model}/{training}/{obf_name}] 원문(clean)에 없는 seed_id "
                        f"{len(unknown)}건: {_show(unknown)}"
                    )

                seed_label = original.loc[obf["seed_id"], "label"].to_numpy()
                label_diff = obf.loc[seed_label != obf["label"].to_numpy(), "id"].tolist()

                if label_diff:
                    raise StatsError(
                        f"[{model}/{training}/{obf_name}] 변형의 label이 원문과 다른 id "
                        f"{len(label_diff)}건: {_show(label_diff)}"
                    )

                attack = obf.loc[obf["label"] == 1].copy()

                attack["orig_correct"] = (
                    original.loc[attack["seed_id"], "prediction"].to_numpy() == 1
                )
                attack["kept"] = attack["orig_correct"] & (attack["prediction"] == 1)

                per_seed = attack.groupby("seed_id", sort=True).agg(
                    den=("orig_correct", "sum"),
                    kept=("kept", "sum"),
                )

                n_den = int(per_seed["den"].sum())
                n_kept = int(per_seed["kept"].sum())

                rate = n_kept / n_den if n_den > 0 else float("nan")

                def stat_ratio(sums):
                    return safe_ratio(sums[:, 1], sums[:, 0])[:, None]

                def samples_fn():
                    return cluster_bootstrap(
                        [per_seed[["den", "kept"]].to_numpy(dtype=np.int64)],
                        stat_ratio,
                        args.n_boot,
                        make_rng(args.seed, f"retain|{suite}|{model}|{training}"),
                    )

                # 유지가 0건이거나 전체이면 Recall·FPR과 같은 규칙(seed 수 기준 Wilson)을 쓴다
                ci = proportion_interval(
                    n_kept,
                    n_den,
                    int((per_seed["den"] > 0).sum()),
                    samples_fn,
                )

                rows.append({
                    "suite": suite,
                    "model": model,
                    "training": training,
                    "n_attack_variants": int(len(attack)),
                    "n_seeds": int(len(per_seed)),
                    "n_variants_orig_correct": n_den,
                    "n_kept": n_kept,
                    "retention_rate": rate,
                    "ci_method": ci[2],
                    "retention_boot_lo": ci[0],
                    "retention_boot_hi": ci[1],
                    "n_flipped_to_wrong": n_den - n_kept,
                })

    return pd.DataFrame(rows)


# --------------------------------------------------
# 분석 4: 기법 분해
# --------------------------------------------------

def parse_train_techniques(text):
    pairs = []

    for item in text.split(","):
        item = item.strip()

        if not item:
            continue

        if ":" not in item:
            raise StatsError(f"--train-techniques 형식은 기법:강도 (예: yamin_swap:0.7): {item}")

        name, intensity = item.split(":", 1)
        pairs.append((name.strip(), float(intensity)))

    if not pairs:
        raise StatsError("--train-techniques가 비어 있음")

    return pairs


def technique_group(technique, intensity, train_pairs):
    for name, value in train_pairs:
        if technique == name and abs(float(intensity) - value) < 1e-9:
            return "trained_technique_same_intensity"

    if technique in {name for name, _ in train_pairs}:
        return "trained_technique_other_intensity"

    return "other_techniques"


def analyze_technique_groups(conditions, args, train_pairs):
    rows = []

    for suite, clean_name, obf_name in active_suites(args):
        for model in args.models:
            for training in args.trainings:
                obf = conditions[(suite, model, training, "obfuscated")]["frame"].copy()

                obf["group"] = [
                    technique_group(t, i, train_pairs)
                    for t, i in zip(obf["technique"], obf["intensity"])
                ]

                for group, part in obf.groupby("group", sort=True):
                    point = point_metrics(part)

                    block = rate_block(
                        part,
                        METHOD_BOOTSTRAP,
                        args,
                        f"grp|{suite}|{model}|{training}|{group}",
                    )

                    row = {
                        "suite": suite,
                        "model": model,
                        "training": training,
                        "group": group,
                        "n_attack": point["tp"] + point["fn"],
                        "recall": point["recall"],
                        "n_benign": point["fp"] + point["tn"],
                        "fpr": point["fpr"],
                    }

                    row.update(block)

                    rows.append(row)

    return pd.DataFrame(rows)


# --------------------------------------------------
# 분석 5: source별
# --------------------------------------------------

def analyze_sources(conditions, args):
    rows = []

    for (suite, model, training, kind), cond in ordered_conditions(conditions, args):
        df = cond["frame"]

        for source, part in df.groupby("source", sort=True):
            point = point_metrics(part)

            block = rate_block(
                part,
                ci_method_for(kind),
                args,
                f"src|{suite}|{model}|{training}|{kind}|{source}",
            )

            row = {
                "suite": suite,
                "model": model,
                "training": training,
                "eval_name": cond["eval_name"],
                "scope": "clean_all_rows" if kind == "clean" else "changed_only",
                "source": source,
                "n_attack": point["tp"] + point["fn"],
                "recall": point["recall"],
                "n_benign": point["fp"] + point["tn"],
                "fpr": point["fpr"],
            }

            row.update(block)

            rows.append(row)

    return pd.DataFrame(rows)


# --------------------------------------------------
# 분석 6: 정상 문장의 난독화 오탐률
# --------------------------------------------------

def analyze_benign_obfuscation(conditions, args):
    rows = []

    for suite, clean_name, obf_name in active_suites(args):
        for model in args.models:
            for training in args.trainings:
                clean = conditions[(suite, model, training, "clean")]["frame"]
                obf = conditions[(suite, model, training, "obfuscated")]["frame"]

                clean_benign = clean.loc[clean["label"] == 0]
                clean_fp = int((clean_benign["prediction"] == 1).sum())
                clean_ci = wilson_interval(clean_fp, len(clean_benign))

                benign = obf.loc[obf["label"] == 0]

                scopes = [("overall", benign)]

                for technique, part in benign.groupby("technique", sort=True):
                    scopes.append((f"technique:{technique}", part))

                for scope, part in scopes:
                    n = int(len(part))
                    fp = int((part["prediction"] == 1).sum())

                    ci = rate_interval(
                        part,
                        0,
                        METHOD_BOOTSTRAP,
                        args,
                        f"benign|{suite}|{model}|{training}|{scope}",
                    )

                    wilson = wilson_interval(fp, n)

                    rows.append({
                        "suite": suite,
                        "model": model,
                        "training": training,
                        "scope": scope,
                        "n_benign_changed": n,
                        "n_seeds": int(part["seed_id"].nunique()),
                        "false_positives": fp,
                        "fpr": fp / n if n > 0 else float("nan"),
                        "ci_method": ci[2],
                        "fpr_ci_lo": ci[0],
                        "fpr_ci_hi": ci[1],
                        "fpr_wilson_lo": wilson[0],
                        "fpr_wilson_hi": wilson[1],
                        "clean_n_benign": int(len(clean_benign)),
                        "clean_fpr": clean_fp / len(clean_benign) if len(clean_benign) else float("nan"),
                        "clean_fpr_ci_lo": clean_ci[0],
                        "clean_fpr_ci_hi": clean_ci[1],
                    })

    return pd.DataFrame(rows)


# --------------------------------------------------
# 출력
# --------------------------------------------------

def pct(value):
    if value is None or pd.isna(value):
        return "–"

    return f"{float(value) * 100:.1f}"


def pct_signed(value):
    if value is None or pd.isna(value):
        return "–"

    return f"{float(value) * 100:+.1f}"


def with_ci(value, lo, hi, signed=False, method=None):
    """값 (구간). fallback 구간에는 †, 구간을 산출할 수 없으면 '산출 불가'."""
    fmt = pct_signed if signed else pct

    if method == METHOD_NA:
        return f"{fmt(value)} (산출 불가)"

    if pd.isna(lo) or pd.isna(hi):
        return f"{fmt(value)} (–)"

    # 차이(signed)는 음수가 있으므로 구간을 쉼표로 구분
    sep = ", " if signed else "–"

    text = f"{fmt(value)} ({pct_signed(lo) if signed else pct(lo)}{sep}{pct_signed(hi) if signed else pct(hi)})"

    if method == METHOD_FALLBACK:
        text += "†"

    return text


def rate_ci(value, lo, hi, n, method=None):
    """비율 표시. 분모(n)가 0이면 값이 정의되지 않으므로 '–'."""
    if n == 0:
        return "–"

    return with_ci(value, lo, hi, method=method)


def md_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")

    return "\n".join(lines)


def label_of(model, training):
    return MODELS.get(model, model), TRAININGS.get(training, training)


def write_markdown(path, cond_df, delta_df, retain_df, group_df, source_df, benign_df, args, train_pairs):
    parts = []

    parts.append("# Step 1 보조 통계")
    parts.append(
        "값은 %, 소수 첫째 자리. 괄호는 95% 구간. "
        "구간: clean 계열의 Recall·FPR은 Wilson(원문 1문장 = 1행), "
        "obfuscated 계열의 Recall·FPR과 모든 F1·Precision은 seed_id 클러스터 부트스트랩"
        f"(반복 {args.n_boot}, 시드 {args.seed}). 난독화 평가는 changed=true 행 기준. "
        "† 0건 또는 전체 성공으로 부트스트랩 구간을 산출할 수 없어 원문 수 기준 Wilson 구간 사용. "
        "'산출 불가'는 재표본 결과가 모두 같은 값이라 구간 폭이 0이 되는 경우."
    )

    for suite, clean_name, obf_name in active_suites(args):
        sc = cond_df.loc[cond_df["suite"] == suite]

        parts.append(f"## [{suite}] 조건별 성능")

        for kind, title in (("clean_all_rows", f"{clean_name} (전체 행)"), ("changed_only", f"{obf_name} (changed=true)")):
            part = sc.loc[sc["scope"] == kind]

            rows = []

            for _, r in part.iterrows():
                model, training = label_of(r["model"], r["training"])

                rows.append([
                    model,
                    training,
                    int(r["n"]),
                    rate_ci(r["recall"], r["recall_ci_lo"], r["recall_ci_hi"], r["n_attack"], method=r["recall_ci_method"]),
                    rate_ci(r["fpr"], r["fpr_ci_lo"], r["fpr_ci_hi"], r["n_benign"], method=r["fpr_ci_method"]),
                    with_ci(r["f1"], r["f1_boot_lo"], r["f1_boot_hi"], method=r["f1_ci_method"]),
                    with_ci(r["precision"], r["precision_boot_lo"], r["precision_boot_hi"], method=r["precision_ci_method"]),
                ])

            parts.append(f"### {title}")
            parts.append(md_table(
                ["Model", "Training", "n", "Recall", "FPR", "F1", "Precision"],
                rows,
            ))

        if not delta_df.empty:
            sd = delta_df.loc[delta_df["suite"] == suite]

            rows = []

            for (model, eval_name), part in sd.groupby(["model", "eval_name"], sort=False):
                cells = {r["metric"]: r for _, r in part.iterrows()}

                rows.append([
                    MODELS.get(model, model),
                    eval_name,
                    int(part["n"].iloc[0]),
                    with_ci(cells["recall"]["delta_aug_minus_orig"], cells["recall"]["delta_boot_lo"], cells["recall"]["delta_boot_hi"], signed=True, method=cells["recall"]["ci_method"]),
                    with_ci(cells["fpr"]["delta_aug_minus_orig"], cells["fpr"]["delta_boot_lo"], cells["fpr"]["delta_boot_hi"], signed=True, method=cells["fpr"]["ci_method"]),
                    with_ci(cells["f1"]["delta_aug_minus_orig"], cells["f1"]["delta_boot_lo"], cells["f1"]["delta_boot_hi"], signed=True, method=cells["f1"]["ci_method"]),
                ])

            parts.append("### Augmented − Original (%p, paired 부트스트랩)")
            parts.append(md_table(
                ["Model", "Eval", "n", "ΔRecall", "ΔFPR", "ΔF1"],
                rows,
            ))

        sr = retain_df.loc[retain_df["suite"] == suite]

        rows = []

        for _, r in sr.iterrows():
            model, training = label_of(r["model"], r["training"])

            rows.append([
                model,
                training,
                int(r["n_seeds"]),
                int(r["n_variants_orig_correct"]),
                int(r["n_kept"]),
                with_ci(r["retention_rate"], r["retention_boot_lo"], r["retention_boot_hi"], method=r["ci_method"]),
            ])

        parts.append("### 원문에서 맞힌 공격 중 변형 후에도 맞힌 비율")
        parts.append(md_table(
            ["Model", "Training", "seed 수", "변형 수(원문 정답)", "유지", "유지율"],
            rows,
        ))

        sg = group_df.loc[group_df["suite"] == suite]

        rows = []

        for _, r in sg.iterrows():
            model, training = label_of(r["model"], r["training"])

            rows.append([
                model,
                training,
                r["group"],
                int(r["n_attack"]),
                rate_ci(r["recall"], r["recall_ci_lo"], r["recall_ci_hi"], r["n_attack"], method=r["recall_ci_method"]),
                int(r["n_benign"]),
                rate_ci(r["fpr"], r["fpr_ci_lo"], r["fpr_ci_hi"], r["n_benign"], method=r["fpr_ci_method"]),
            ])

        trained = ", ".join(f"{n}:{i}" for n, i in train_pairs)

        parts.append(f"### 기법 분해 (학습 기법: {trained})")
        parts.append(md_table(
            ["Model", "Training", "그룹", "n(attack)", "Recall", "n(benign)", "FPR"],
            rows,
        ))

        ss = source_df.loc[source_df["suite"] == suite]

        rows = []

        for _, r in ss.iterrows():
            model, training = label_of(r["model"], r["training"])

            rows.append([
                model,
                training,
                r["eval_name"],
                r["source"],
                int(r["n_attack"]),
                rate_ci(r["recall"], r["recall_ci_lo"], r["recall_ci_hi"], r["n_attack"], method=r["recall_ci_method"]),
                int(r["n_benign"]),
                rate_ci(r["fpr"], r["fpr_ci_lo"], r["fpr_ci_hi"], r["n_benign"], method=r["fpr_ci_method"]),
            ])

        parts.append("### source별 Recall / FPR")
        parts.append(md_table(
            ["Model", "Training", "Eval", "source", "n(attack)", "Recall", "n(benign)", "FPR"],
            rows,
        ))

        sb = benign_df.loc[(benign_df["suite"] == suite) & (benign_df["scope"] == "overall")]

        rows = []

        for _, r in sb.iterrows():
            model, training = label_of(r["model"], r["training"])

            rows.append([
                model,
                training,
                int(r["clean_n_benign"]),
                rate_ci(r["clean_fpr"], r["clean_fpr_ci_lo"], r["clean_fpr_ci_hi"], r["clean_n_benign"]),
                int(r["n_benign_changed"]),
                int(r["false_positives"]),
                rate_ci(r["fpr"], r["fpr_ci_lo"], r["fpr_ci_hi"], r["n_benign_changed"], method=r["ci_method"]),
            ])

        parts.append("### 정상 문장의 난독화 오탐률 (label 0 & changed=true)")
        parts.append(md_table(
            ["Model", "Training", "n(clean benign)", "clean FPR", "n(변형 정상)", "오탐 수", "난독화 FPR"],
            rows,
        ))

    Path(path).write_text("\n\n".join(parts) + "\n", encoding="utf-8")


def save_csv(df, path):
    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Step 1 supplementary statistics: confidence intervals and "
            "paired differences between conditions."
        )
    )

    parser.add_argument(
        "--results-root",
        default="results/step1",
        help="results/step1/<model>/<original|augmented>/eval/<eval_name>/ 구조의 루트",
    )

    parser.add_argument(
        "--clean-input",
        required=True,
        help="clean 평가에 쓴 입력 파일 (.csv/.jsonl)",
    )

    parser.add_argument(
        "--obfuscated-input",
        required=True,
        help="obfuscated 평가에 쓴 입력 파일 (.csv/.jsonl)",
    )

    parser.add_argument("--kg-clean-input", default=None)
    parser.add_argument("--kg-obfuscated-input", default=None)

    parser.add_argument(
        "--models",
        default="koelectra,mdeberta",
        help="쉼표로 구분한 model_key",
    )

    parser.add_argument(
        "--trainings",
        default="original,augmented",
        help="쉼표로 구분한 training_type",
    )

    parser.add_argument(
        "--train-techniques",
        default="yamin_swap:0.7,symbol_insert:0.3",
        help="증강 학습에 쓴 기법:강도 (쉼표로 구분)",
    )

    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261002)

    parser.add_argument(
        "--output-dir",
        default="results/stats",
    )

    parser.add_argument(
        "--skip-crosscheck",
        action="store_true",
        help="기존 metrics.json / obfuscated_overall.csv와의 대조를 건너뜀",
    )

    args = parser.parse_args()

    args.models = [m.strip() for m in args.models.split(",") if m.strip()]
    args.trainings = [t.strip() for t in args.trainings.split(",") if t.strip()]

    if bool(args.kg_clean_input) != bool(args.kg_obfuscated_input):
        parser.error("--kg-clean-input과 --kg-obfuscated-input은 함께 지정해야 함")

    try:
        run(args)
    except StatsError as e:
        print("\n[중단] " + str(e), file=sys.stderr)
        sys.exit(1)


def run(args):
    train_pairs = parse_train_techniques(args.train_techniques)

    log = []

    input_paths = {
        "clean": args.clean_input,
        "obfuscated": args.obfuscated_input,
        "kg_clean": args.kg_clean_input,
        "kg_obfuscated": args.kg_obfuscated_input,
    }

    inputs = {}
    input_info = {}

    for name, path in input_paths.items():
        if not path:
            continue

        df = prepare_input(load_table(path), name)
        inputs[name] = df

        input_info[name] = {
            "path": str(path),
            "rows_after_dropna": int(len(df)),
            "sha256": sha256_file(path),
        }

    conditions, run_files = load_conditions(args, inputs, log)

    cond_df = analyze_conditions(conditions, args)
    delta_df = analyze_original_vs_augmented(conditions, args)
    retain_df = analyze_original_vs_variant(conditions, args)
    group_df = analyze_technique_groups(conditions, args, train_pairs)
    source_df = analyze_sources(conditions, args)
    benign_df = analyze_benign_obfuscation(conditions, args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_csv(cond_df, output_dir / "condition_metrics.csv")
    save_csv(delta_df, output_dir / "paired_original_vs_augmented.csv")
    save_csv(retain_df, output_dir / "paired_original_vs_variant.csv")
    save_csv(group_df, output_dir / "technique_groups.csv")
    save_csv(source_df, output_dir / "source_metrics.csv")
    save_csv(benign_df, output_dir / "benign_obfuscation_fpr.csv")

    write_markdown(
        output_dir / "step1_stats_tables.md",
        cond_df,
        delta_df,
        retain_df,
        group_df,
        source_df,
        benign_df,
        args,
        train_pairs,
    )

    info = {
        "n_boot": args.n_boot,
        "seed": args.seed,
        "train_techniques": [f"{n}:{i}" for n, i in train_pairs],
        "inputs": input_info,
        "prediction_files": run_files,
        "notes": log,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }

    (output_dir / "run_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("===== STEP 1 STATS =====")

    for line in log:
        print(line)

    print()

    show = cond_df[[
        "suite", "model", "training", "scope", "n",
        "recall", "fpr", "f1", "precision",
    ]]

    print(show.round(4).to_string(index=False))

    print()
    print("Saved:")

    for name in (
        "condition_metrics.csv",
        "paired_original_vs_augmented.csv",
        "paired_original_vs_variant.csv",
        "technique_groups.csv",
        "source_metrics.csv",
        "benign_obfuscation_fpr.csv",
        "step1_stats_tables.md",
        "run_info.json",
    ):
        print(output_dir / name)


if __name__ == "__main__":
    main()
