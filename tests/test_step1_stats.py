"""
scripts/step1_stats.py 테스트.

저장소의 기존 테스트(src/classifier/test_*.py)처럼 스크립트로 바로 실행할 수 있고,
pytest로도 돌아간다 (pytest는 필수가 아님).

    python tests/test_step1_stats.py

가짜 데이터만 쓴다. 채원님 evaluate.py / analyze_obfuscated_eval.py로 predictions.csv를
직접 만들어서 형식 차이가 없게 하고, 점추정이 기존 결과(metrics.json,
obfuscated_overall.csv, obfuscated_by_technique.csv)와 sklearn 계산과 같은지 확인한다.
evaluate.py를 돌리려면 torch / transformers가 필요하며, 없으면 그 테스트는 건너뛴다.
"""

import atexit
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


REPO = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


stats = load_module("step1_stats", REPO / "scripts" / "step1_stats.py")
analyze = load_module("analyze_obfuscated_eval", REPO / "scripts" / "analyze_obfuscated_eval.py")


# --------------------------------------------------
# 순수 함수 테스트
# --------------------------------------------------

def test_normalize_changed_same_as_repo():
    cases = [
        pd.Series([True, False, True]),
        pd.Series(["True", "false", "TRUE", " 1 ", "0.0", "1.0", "0"]),
        pd.Series([1, 0, 1]),
        pd.Series([1.0, 0.0]),
        pd.Series([True, False], dtype=object),
        pd.Series(["True", "yes"]),
        pd.Series(["True", np.nan]),
        pd.Series([1.0, np.nan]),
    ]

    for series in cases:
        try:
            expected = analyze.normalize_changed(series)
            expected_error = None
        except ValueError as e:
            expected, expected_error = None, str(e)

        try:
            actual = stats.normalize_changed(series)
            actual_error = None
        except ValueError as e:
            actual, actual_error = None, str(e)

        assert expected_error == actual_error, (series.tolist(), expected_error, actual_error)

        if expected is not None:
            assert expected.tolist() == actual.tolist(), series.tolist()
            assert expected.dtype == actual.dtype


def test_wilson_known_values():
    lo, hi = stats.wilson_interval(5, 10)
    assert abs(lo - 0.2366) < 1e-3 and abs(hi - 0.7634) < 1e-3, (lo, hi)

    lo, hi = stats.wilson_interval(0, 10)
    assert lo == 0.0 and abs(hi - 3.8416 / 13.8416) < 1e-4, (lo, hi)

    lo, hi = stats.wilson_interval(10, 10)
    assert hi == 1.0 and abs(lo - 10 / 13.8416) < 1e-4, (lo, hi)

    lo, hi = stats.wilson_interval(0, 0)
    assert np.isnan(lo) and np.isnan(hi)

    try:
        from scipy.stats import binomtest
    except ImportError:
        return

    for k, n in [(3, 20), (57, 100), (0, 44), (44, 44), (81, 263)]:
        ref = binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
        lo, hi = stats.wilson_interval(k, n)
        assert abs(lo - ref.low) < 1e-4 and abs(hi - ref.high) < 1e-4, (k, n, lo, hi, ref)


def random_frame(rng, n, p_attack=0.5, accuracy=0.8):
    label = (rng.random(n) < p_attack).astype(int)
    flip = rng.random(n) > accuracy
    prediction = np.where(flip, 1 - label, label)

    return pd.DataFrame({"label": label, "prediction": prediction})


def test_point_metrics_match_repo_and_sklearn():
    rng = np.random.default_rng(0)

    frames = [random_frame(rng, n, p, a) for n, p, a in [
        (50, 0.5, 0.8), (200, 0.3, 0.6), (17, 0.5, 0.9), (400, 0.9, 0.7),
    ]]

    # 극단 경우: 예측이 전부 0 / 전부 1 / 공격 없음 / 정상 없음
    frames.append(pd.DataFrame({"label": [1, 1, 0, 0], "prediction": [0, 0, 0, 0]}))
    frames.append(pd.DataFrame({"label": [1, 1, 0, 0], "prediction": [1, 1, 1, 1]}))
    frames.append(pd.DataFrame({"label": [0, 0, 0], "prediction": [0, 1, 0]}))
    frames.append(pd.DataFrame({"label": [1, 1, 1], "prediction": [1, 0, 1]}))

    for df in frames:
        mine = stats.point_metrics(df)
        repo = analyze.compute_metrics(df)

        y, p = df["label"], df["prediction"]
        tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()

        for key in ("n", "tp", "fp", "fn", "tn"):
            assert mine[key] == repo[key], (key, mine[key], repo[key])

        assert (mine["tp"], mine["fp"], mine["fn"], mine["tn"]) == (tp, fp, fn, tn)

        for key in ("precision", "recall", "f1", "fpr", "fnr"):
            assert abs(mine[key] - repo[key]) < 1e-12, (key, mine[key], repo[key])

        assert abs(mine["f1"] - f1_score(y, p, pos_label=1, zero_division=0)) < 1e-12
        assert abs(mine["precision"] - precision_score(y, p, pos_label=1, zero_division=0)) < 1e-12
        assert abs(mine["recall"] - recall_score(y, p, pos_label=1, zero_division=0)) < 1e-12


def test_bootstrap_counts_match_sklearn_on_resample():
    rng = np.random.default_rng(1)

    df = random_frame(rng, 60, 0.5, 0.7)
    df["seed_id"] = [f"s{i // 3}" for i in range(len(df))]  # 클러스터당 3행

    counts = stats.counts_by_cluster(df)
    arr = counts.to_numpy()

    idx = np.array([[0, 0, 5, 7, 19, 3, 3, 12]])
    sums = arr[idx].sum(axis=1)

    vec = stats.stats_single(sums)[0]  # recall, fpr, f1, precision

    names = counts.index.tolist()
    rows = pd.concat([df.loc[df["seed_id"] == names[i]] for i in idx[0]])

    y, p = rows["label"], rows["prediction"]
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()

    expected = [
        recall_score(y, p, pos_label=1, zero_division=0),
        fp / (fp + tn) if fp + tn else 0.0,
        f1_score(y, p, pos_label=1, zero_division=0),
        precision_score(y, p, pos_label=1, zero_division=0),
    ]

    assert np.allclose(vec, expected, atol=1e-12), (vec, expected)


def test_bootstrap_reproducible_and_seed_dependent():
    rng = np.random.default_rng(2)

    df = random_frame(rng, 120, 0.5, 0.75)
    df["seed_id"] = [f"s{i}" for i in range(len(df))]
    arr = stats.counts_by_cluster(df).to_numpy()

    a = stats.cluster_bootstrap([arr], stats.stats_single, 300, stats.make_rng(7, "k"))
    b = stats.cluster_bootstrap([arr], stats.stats_single, 300, stats.make_rng(7, "k"))
    c = stats.cluster_bootstrap([arr], stats.stats_single, 300, stats.make_rng(8, "k"))
    d = stats.cluster_bootstrap([arr], stats.stats_single, 300, stats.make_rng(7, "other"))

    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert not np.array_equal(a, d)


def test_bootstrap_interval_close_to_wilson_for_singletons():
    # 클러스터가 전부 1행이면 recall의 부트스트랩 구간은 Wilson 구간과 비슷해야 한다.
    n = 400
    label = np.ones(n, dtype=int)
    prediction = np.array([1] * 320 + [0] * 80)

    df = pd.DataFrame({"label": label, "prediction": prediction, "seed_id": [f"s{i}" for i in range(n)]})
    arr = stats.counts_by_cluster(df).to_numpy()

    samples = stats.cluster_bootstrap([arr], stats.stats_single, 2000, stats.make_rng(1, "x"))
    lo, hi = stats.percentile_interval(samples)

    w_lo, w_hi = stats.wilson_interval(320, 400)

    assert abs(lo[0] - w_lo) < 0.02 and abs(hi[0] - w_hi) < 0.02, (lo[0], hi[0], w_lo, w_hi)
    assert lo[0] < 0.8 < hi[0]


def test_rate_counts_match_resample():
    # Recall/FPR 클러스터 부트스트랩의 재표본 합계가 행 단위 계산과 같은지 확인
    rng = np.random.default_rng(3)

    df = random_frame(rng, 90, 0.5, 0.7)
    df["seed_id"] = [f"s{i // 3}" for i in range(len(df))]

    for label in (1, 0):
        arr = stats.rate_counts_by_cluster(df, label)

        names = sorted(df.loc[df["label"] == label, "seed_id"].unique())

        assert arr.shape == (len(names), 2)

        idx = np.array([[0, 0, 2, len(names) - 1, 1]])
        value = stats.stats_rate(arr[idx].sum(axis=1))[0, 0]

        rows = pd.concat([df.loc[(df["seed_id"] == names[i]) & (df["label"] == label)] for i in idx[0]])

        assert abs(value - (rows["prediction"] == 1).mean()) < 1e-12


def test_rate_interval_clustered_is_wider_when_variants_are_correlated():
    # 한 원문의 변형 10개가 모두 같은 예측이면 행을 독립으로 본 Wilson 구간은 너무 좁다.
    rows = []

    for i in range(40):
        for j in range(10):
            rows.append({"seed_id": f"s{i}", "label": 1, "prediction": int(i < 24)})

    df = pd.DataFrame(rows)

    class A:
        n_boot = 2000
        seed = 1

    w_lo, w_hi, _ = stats.rate_interval(df, 1, "wilson", A, "k")
    b_lo, b_hi, b_method = stats.rate_interval(df, 1, "cluster_bootstrap", A, "k")

    assert b_method == "cluster_bootstrap"

    assert (b_hi - b_lo) > 2 * (w_hi - w_lo), (b_lo, b_hi, w_lo, w_hi)
    assert b_lo < 0.6 < b_hi

    # 원문 1문장 = 1행이면 두 방법이 비슷하다
    single = pd.DataFrame({
        "seed_id": [f"s{i}" for i in range(400)],
        "label": 1,
        "prediction": [1] * 320 + [0] * 80,
    })

    s_lo, s_hi, _ = stats.rate_interval(single, 1, "wilson", A, "k")
    c_lo, c_hi, _ = stats.rate_interval(single, 1, "cluster_bootstrap", A, "k")

    assert abs(s_lo - c_lo) < 0.02 and abs(s_hi - c_hi) < 0.02

    # 해당 label 행이 없으면 nan
    none_lo, none_hi, none_method = stats.rate_interval(single, 0, "cluster_bootstrap", A, "k")

    assert np.isnan(none_lo) and np.isnan(none_hi) and none_method == "not_computable"


def test_rate_interval_falls_back_to_wilson_on_seed_count_when_all_or_none():
    class A:
        n_boot = 500
        seed = 1

    z2 = stats.WILSON_Z ** 2

    # 공격 30개 seed x 변형 5개가 전부 맞음 -> 행 150개가 아니라 seed 30개 기준 Wilson
    rows = [{"seed_id": f"a{i}", "label": 1, "prediction": 1} for i in range(30) for _ in range(5)]
    df = pd.DataFrame(rows)

    lo, hi, method = stats.rate_interval(df, 1, "cluster_bootstrap", A, "k")

    assert method == "wilson_cluster_fallback"

    w_lo, w_hi = stats.wilson_interval(30, 30)

    assert (lo, hi) == (w_lo, w_hi)
    assert abs(lo - 30 / (30 + z2)) < 1e-9 and hi == 1.0

    row_lo, _ = stats.wilson_interval(150, 150)

    assert lo < row_lo  # 행 수 기준 Wilson보다 넓다

    # 정상 25개 seed x 변형 4개가 전부 정상 판정 -> 오탐 0건, seed 25개 기준
    rows = [{"seed_id": f"b{i}", "label": 0, "prediction": 0} for i in range(25) for _ in range(4)]
    df = pd.DataFrame(rows)

    lo, hi, method = stats.rate_interval(df, 0, "cluster_bootstrap", A, "k")

    assert method == "wilson_cluster_fallback"
    assert (lo, hi) == stats.wilson_interval(0, 25)
    assert lo == 0.0 and abs(hi - z2 / (25 + z2)) < 1e-9

    # clean 계열(wilson)은 그대로 Wilson, 일부만 맞으면 부트스트랩
    lo, hi, method = stats.rate_interval(pd.DataFrame(rows), 0, "wilson", A, "k")

    assert method == "wilson" and (lo, hi) == stats.wilson_interval(0, 100)

    mixed = pd.DataFrame([{"seed_id": f"c{i}", "label": 1, "prediction": int(i % 3 != 0)} for i in range(30) for _ in range(2)])

    assert stats.rate_interval(mixed, 1, "cluster_bootstrap", A, "k")[2] == "cluster_bootstrap"


def test_bootstrap_ci_marks_zero_width_as_not_computable():
    samples = np.column_stack([
        np.full(200, 1.0),                                   # 항상 1.0 (예: 전부 맞힌 F1)
        np.linspace(0.2, 0.8, 200),                          # 정상
        np.zeros(200),                                       # 항상 0 (예: 분모 0)
        np.where(np.arange(200) % 2 == 0, np.nan, 0.5),      # nan 포함
    ])

    lo, hi, methods = stats.bootstrap_ci(samples)

    assert methods == ["not_computable", "cluster_bootstrap", "not_computable", "not_computable"]
    assert np.isnan(lo[0]) and np.isnan(hi[0]) and np.isnan(lo[2]) and np.isnan(lo[3])
    assert 0.2 < lo[1] < hi[1] < 0.8

    # F1/Precision: 정상만 있고 전부 정상 판정이면(분모 0) 재표본마다 0 -> 산출 불가
    df = pd.DataFrame({"seed_id": [f"s{i}" for i in range(20)], "label": 0, "prediction": 0})

    sums = stats.cluster_bootstrap([stats.counts_by_cluster(df).to_numpy()], stats.stats_single, 100, stats.make_rng(1, "x"))

    assert stats.bootstrap_ci(sums)[2] == ["not_computable", "not_computable", "not_computable", "not_computable"]


def test_with_ci_marks_fallback_and_not_computable():
    assert stats.with_ci(1.0, 0.887, 1.0, method="wilson_cluster_fallback") == "100.0 (88.7–100.0)†"
    assert stats.with_ci(1.0, 0.887, 1.0, method="cluster_bootstrap") == "100.0 (88.7–100.0)"
    assert stats.with_ci(0.0, float("nan"), float("nan"), signed=True, method="not_computable") == "+0.0 (산출 불가)"
    assert stats.with_ci(0.5, float("nan"), float("nan")) == "50.0 (–)"


def write_degenerate_results(root):
    """
    torch 없이 만드는 퇴화 사례.
    koelectra: Original / Augmented 모두 전부 맞힘(Recall 100%, FPR 0%) -> 구간 fallback, 차이는 항상 0.
    mdeberta: 무작위 예측(퇴화 없음) -> 부트스트랩, 차이 구간 산출.
    """
    rng = np.random.default_rng(5)

    seeds = [
        {"id": f"d_{i:03d}", "text": f"문장 {i}", "label": int(i % 2 == 0), "source": "src_a" if i < 30 else "src_b"}
        for i in range(60)
    ]

    obf = []

    for sd in seeds:
        for j in range(3):
            obf.append({
                "id": f"{sd['id']}__v{j}", "text": f"{sd['text']} 변형{j}", "label": sd["label"], "source": sd["source"],
                "seed_id": sd["id"], "technique": ["yamin_swap", "symbol_insert", "chosung"][j],
                "intensity": 0.7 if j == 0 else 0.3, "changed": True, "n_changed": 1,
            })

    data = root / "data"
    data.mkdir(parents=True)

    write_jsonl(data / "test.jsonl", seeds)
    write_jsonl(data / "obfuscated_test.jsonl", obf)

    for model in MODEL_KEYS:
        for training in TRAININGS:
            for name, rows in (("clean", seeds), ("obfuscated", obf)):
                df = pd.DataFrame(rows)

                if model == "koelectra":
                    df["prediction"] = df["label"]
                else:
                    df["prediction"] = rng.integers(0, 2, size=len(df))

                df["attack_score"] = df["prediction"].astype(float)

                out = root / "results" / "step1" / model / training / "eval" / name
                out.mkdir(parents=True)

                df.to_csv(out / "predictions.csv", index=False)

    return data


def test_degenerate_cases_end_to_end():
    root = Path(tempfile.mkdtemp(prefix="step1_stats_degenerate_"))
    atexit.register(shutil.rmtree, root, ignore_errors=True)

    data = write_degenerate_results(root)
    out_dir = root / "stats"

    proc = subprocess.run(
        [
            sys.executable, str(REPO / "scripts" / "step1_stats.py"),
            "--results-root", str(root / "results" / "step1"),
            "--clean-input", str(data / "test.jsonl"),
            "--obfuscated-input", str(data / "obfuscated_test.jsonl"),
            "--n-boot", "300",
            "--output-dir", str(out_dir),
        ],
        capture_output=True, text=True, cwd=str(REPO),
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout

    z2 = stats.WILSON_Z ** 2
    cond = pd.read_csv(out_dir / "condition_metrics.csv")

    ko = cond.loc[(cond["model"] == "koelectra") & (cond["scope"] == "changed_only")]

    assert len(ko) == 2

    for _, row in ko.iterrows():
        # 전부 맞힘: Recall 100%, FPR 0% -> 공격·정상 seed 30개씩 기준 Wilson
        assert row["recall_ci_method"] == row["fpr_ci_method"] == "wilson_cluster_fallback"
        assert np.allclose([row["recall_ci_lo"], row["recall_ci_hi"]], stats.wilson_interval(30, 30), atol=1e-6)
        assert np.allclose([row["fpr_ci_lo"], row["fpr_ci_hi"]], stats.wilson_interval(0, 30), atol=1e-6)
        assert abs(row["recall_ci_lo"] - 30 / (30 + z2)) < 1e-6
        assert abs(row["fpr_ci_hi"] - z2 / (30 + z2)) < 1e-6

        # F1/Precision도 재표본마다 1.0이라 구간 폭 0 -> 산출 불가
        assert row["f1_ci_method"] == row["precision_ci_method"] == "not_computable"
        assert np.isnan(row["f1_boot_lo"]) and np.isnan(row["precision_boot_hi"])

    # clean 계열은 Wilson 그대로 (행 30/30)
    ko_clean = cond.loc[(cond["model"] == "koelectra") & (cond["scope"] == "clean_all_rows")]

    assert (ko_clean["recall_ci_method"] == "wilson").all()
    assert (ko_clean["recall_ci_lo"] == ko_clean["recall_wilson_lo"]).all()

    # mDeBERTa(무작위)는 부트스트랩 구간이 정상 산출
    md_obf = cond.loc[(cond["model"] == "mdeberta") & (cond["scope"] == "changed_only")]

    assert (md_obf["recall_ci_method"] == "cluster_bootstrap").all()
    assert (md_obf["f1_ci_method"] == "cluster_bootstrap").all()
    assert (md_obf["recall_ci_hi"] > md_obf["recall_ci_lo"]).all()

    # paired 차이: koelectra는 두 조건 모두 전부 맞힘이라 항상 0 -> 산출 불가, mdeberta는 구간 산출
    delta = pd.read_csv(out_dir / "paired_original_vs_augmented.csv")

    ko_delta = delta.loc[delta["model"] == "koelectra"]

    assert (ko_delta["ci_method"] == "not_computable").all()
    assert ko_delta["delta_boot_lo"].isna().all() and ko_delta["delta_boot_hi"].isna().all()
    assert (ko_delta["delta_aug_minus_orig"] == 0).all()

    md_delta = delta.loc[delta["model"] == "mdeberta"]

    assert (md_delta["ci_method"] == "cluster_bootstrap").all()
    assert (md_delta["delta_boot_hi"] >= md_delta["delta_boot_lo"]).all()

    # 원문 -> 변형 유지율: 전부 유지 -> seed 30개 기준 Wilson
    retain = pd.read_csv(out_dir / "paired_original_vs_variant.csv")
    ko_retain = retain.loc[retain["model"] == "koelectra"]

    assert (ko_retain["ci_method"] == "wilson_cluster_fallback").all()
    assert ((ko_retain["retention_boot_lo"] - 30 / (30 + z2)).abs() < 1e-6).all()

    # 정상 문장 난독화 오탐률: koelectra는 오탐 0건 -> fallback
    benign = pd.read_csv(out_dir / "benign_obfuscation_fpr.csv")
    ko_overall = benign.loc[(benign["model"] == "koelectra") & (benign["scope"] == "overall")]

    assert (ko_overall["false_positives"] == 0).all()
    assert (ko_overall["ci_method"] == "wilson_cluster_fallback").all()
    assert (ko_overall["fpr_ci_lo"] == 0.0).all()

    # markdown: fallback에는 †, 산출 불가 표기, 머리말 설명
    md = (out_dir / "step1_stats_tables.md").read_text(encoding="utf-8")

    assert "† 0건 또는 전체 성공으로 부트스트랩 구간을 산출할 수 없어 원문 수 기준 Wilson 구간 사용" in md
    assert "(88.6–100.0)†" in md  # Wilson(30, 30) 하한
    assert "(0.0–11.4)†" in md  # Wilson(0, 30) 상한
    assert "(산출 불가)" in md
    assert "0.0–0.0" not in md and "100.0–100.0" not in md

    # mDeBERTa 줄에는 † 가 없다
    for line in md.splitlines():
        if line.startswith("| mDeBERTa") and "obfuscated" not in line:
            assert "†" not in line, line


def test_rate_ci_shows_dash_when_denominator_is_zero():
    nan = float("nan")

    assert stats.rate_ci(0.0, nan, nan, 0) == "–"
    assert stats.rate_ci(0.0, nan, nan, 0, method="not_computable") == "–"
    assert stats.rate_ci(0.0, 0.0, 0.0, 0, method="wilson_cluster_fallback") == "–"
    assert stats.rate_ci(0.5, 0.4, 0.6, 10) == "50.0 (40.0–60.0)"
    assert stats.rate_ci(1.0, 0.887, 1.0, 30, method="wilson_cluster_fallback") == "100.0 (88.7–100.0)†"
    assert stats.rate_ci(0.2, nan, nan, 5, method="not_computable") == "20.0 (산출 불가)"


def write_source_only_results(root):
    """source별로 공격만 / 정상만 / 둘 다 있는 평가 파일. torch 없이 만든 무작위 예측."""
    rng = np.random.default_rng(11)

    seeds = []

    for i in range(90):
        if i < 30:
            label, source = 1, "src_attack_only"
        elif i < 60:
            label, source = 0, "src_benign_only"
        else:
            label, source = i % 2, "src_both"

        seeds.append({"id": f"s_{i:03d}", "text": f"문장 {i}", "label": label, "source": source})

    obf = []

    for sd in seeds:
        for j in range(3):
            obf.append({
                "id": f"{sd['id']}__v{j}", "text": f"{sd['text']} 변형{j}", "label": sd["label"], "source": sd["source"],
                "seed_id": sd["id"], "technique": "yamin_swap", "intensity": 0.7, "changed": True, "n_changed": 1,
            })

    data = root / "data"
    data.mkdir(parents=True)

    write_jsonl(data / "test.jsonl", seeds)
    write_jsonl(data / "obfuscated_test.jsonl", obf)

    for training in TRAININGS:
        for name, rows in (("clean", seeds), ("obfuscated", obf)):
            df = pd.DataFrame(rows)
            df["prediction"] = rng.integers(0, 2, size=len(df))
            df["attack_score"] = df["prediction"].astype(float)

            out = root / "results" / "step1" / "koelectra" / training / "eval" / name
            out.mkdir(parents=True)

            df.to_csv(out / "predictions.csv", index=False)

    return data


def test_markdown_shows_dash_for_zero_denominator_cells():
    root = Path(tempfile.mkdtemp(prefix="step1_stats_dash_"))
    atexit.register(shutil.rmtree, root, ignore_errors=True)

    data = write_source_only_results(root)
    out_dir = root / "stats"

    proc = subprocess.run(
        [
            sys.executable, str(REPO / "scripts" / "step1_stats.py"),
            "--results-root", str(root / "results" / "step1"),
            "--clean-input", str(data / "test.jsonl"),
            "--obfuscated-input", str(data / "obfuscated_test.jsonl"),
            "--models", "koelectra",
            "--n-boot", "200",
            "--output-dir", str(out_dir),
        ],
        capture_output=True, text=True, cwd=str(REPO),
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout

    src = pd.read_csv(out_dir / "source_metrics.csv")

    assert (src.loc[src["source"] == "src_attack_only", "n_benign"] == 0).all()
    assert (src.loc[src["source"] == "src_benign_only", "n_attack"] == 0).all()

    md = (out_dir / "step1_stats_tables.md").read_text(encoding="utf-8")

    assert "0.0 (–)" not in md and "0.0 (산출 불가)" not in md

    def cells(prefix):
        line = next(l for l in md.splitlines() if l.startswith(prefix))

        return [c.strip() for c in line.strip().strip("|").split("|")]

    # | Model | Training | Eval | source | n(attack) | Recall | n(benign) | FPR |
    for kind in ("clean", "obfuscated"):
        a = cells(f"| KoELECTRA | Original | {kind} | src_attack_only |")
        b = cells(f"| KoELECTRA | Original | {kind} | src_benign_only |")
        c = cells(f"| KoELECTRA | Original | {kind} | src_both |")

        assert a[6] == "0" and a[7] == "–" and a[5] != "–"
        assert b[4] == "0" and b[5] == "–" and b[7] != "–"
        assert c[5] != "–" and c[7] != "–"


def test_technique_group_assignment():
    pairs = stats.parse_train_techniques("yamin_swap:0.7,symbol_insert:0.3")

    assert pairs == [("yamin_swap", 0.7), ("symbol_insert", 0.3)]

    assert stats.technique_group("yamin_swap", 0.7, pairs) == "trained_technique_same_intensity"
    assert stats.technique_group("symbol_insert", 0.3, pairs) == "trained_technique_same_intensity"
    assert stats.technique_group("yamin_swap", 0.3, pairs) == "trained_technique_other_intensity"
    assert stats.technique_group("symbol_insert", 0.7, pairs) == "trained_technique_other_intensity"
    assert stats.technique_group("chosung", 0.7, pairs) == "other_techniques"


# --------------------------------------------------
# 가짜 데이터 + 채원님 스크립트로 만든 predictions.csv
# --------------------------------------------------

POOL = list("가나다라마바사아자차카타파하거너더러머버서어저처커터퍼허")
SYMBOLS = ["#", "@"]
TECHNIQUES = ["yamin_swap", "symbol_insert", "chosung", "qwerty", "tensify"]
SOURCES = ["src_a", "src_b", "src_c"]

MODEL_KEYS = ["koelectra", "mdeberta"]
TRAININGS = ["original", "augmented"]


def make_text(rng, low=6, high=14):
    n = int(rng.integers(low, high))
    return " ".join(rng.choice(POOL, size=n))


def make_variant_text(rng, text):
    tokens = text.split(" ")

    for i in range(len(tokens)):
        if rng.random() < 0.3:
            tokens[i] = str(rng.choice(POOL + SYMBOLS))

    return " ".join(tokens)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_dataset(rng, prefix, n_seeds):
    """서형님 형식: 원문 (id, text, label, source) + 변형 (seed_id, technique, intensity, changed, n_changed)."""
    clean, obf = [], []

    for i in range(n_seeds):
        seed = f"{prefix}_{i:04d}"
        label = int(i % 2 == 0)
        text = make_text(rng)

        clean.append({
            "id": seed,
            "text": text,
            "label": label,
            "source": SOURCES[i % len(SOURCES)],
        })

        n_var = int(rng.integers(3, 7))

        for j in range(n_var):
            technique = TECHNIQUES[int(rng.integers(len(TECHNIQUES)))]
            intensity = [0.3, 0.7][int(rng.integers(2))]
            changed = bool(rng.random() > 0.15)
            variant_text = make_variant_text(rng, text) if changed else text

            obf.append({
                "id": f"{seed}__v{j}",
                "text": variant_text,
                "label": label,
                "source": SOURCES[i % len(SOURCES)],
                "seed_id": seed,
                "technique": technique,
                "intensity": intensity,
                "changed": changed,
                "n_changed": int(changed) * int(rng.integers(1, 5)),
            })

    return clean, obf


def build_tiny_model(directory, vocab, seed, texts):
    """학습하지 않은 작은 BERT 분류기. 두 클래스가 모두 예측되는 seed를 찾는다."""
    import torch
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    directory.mkdir(parents=True, exist_ok=True)

    vocab_path = directory / "vocab.txt"
    vocab_path.write_text("\n".join(vocab) + "\n", encoding="utf-8")

    tokenizer = BertTokenizerFast(vocab_file=str(vocab_path), do_lower_case=False)
    tokenizer.save_pretrained(directory)

    enc = tokenizer(texts, padding=True, truncation=True, max_length=32, return_tensors="pt")

    for candidate in range(seed, seed + 200):
        torch.manual_seed(candidate)

        config = BertConfig(
            vocab_size=len(vocab),
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=64,
            num_labels=2,
            initializer_range=1.0,
        )

        model = BertForSequenceClassification(config)
        model.eval()

        with torch.no_grad():
            pred = model(**enc).logits.argmax(-1).numpy()

        if 0.25 < pred.mean() < 0.75:
            model.save_pretrained(directory)
            return candidate

    raise RuntimeError("두 클래스를 모두 예측하는 tiny 모델을 찾지 못함")


_FIXTURE = {}


def get_fixture():
    """가짜 데이터로 evaluate.py / analyze_obfuscated_eval.py를 실제로 돌려 결과 폴더를 만든다 (한 번만)."""
    if _FIXTURE:
        return _FIXTURE

    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        raise unittest.SkipTest("torch/transformers가 없어 evaluate.py를 돌릴 수 없음")

    root = Path(tempfile.mkdtemp(prefix="step1_stats_test_"))
    atexit.register(shutil.rmtree, root, ignore_errors=True)

    rng = np.random.default_rng(20261002)

    clean, obf = make_dataset(rng, "main", 120)
    kg_clean, kg_obf = make_dataset(rng, "kg", 40)

    data_dir = root / "data"
    data_dir.mkdir()

    inputs = {
        "clean": data_dir / "test.jsonl",
        "obfuscated": data_dir / "obfuscated_test.jsonl",
        "kg_clean": data_dir / "kg_test.jsonl",
        "kg_obfuscated": data_dir / "kg_obfuscated_test.jsonl",
    }

    write_jsonl(inputs["clean"], clean)
    write_jsonl(inputs["obfuscated"], obf)
    write_jsonl(inputs["kg_clean"], kg_clean)
    write_jsonl(inputs["kg_obfuscated"], kg_obf)

    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + POOL + SYMBOLS

    texts = [r["text"] for r in clean + obf + kg_clean + kg_obf]

    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", TOKENIZERS_PARALLELISM="false")

    results = root / "results" / "step1"

    for m_i, model_key in enumerate(MODEL_KEYS):
        for t_i, training in enumerate(TRAININGS):
            model_dir = root / "models" / model_key / training
            build_tiny_model(model_dir, vocab, 100 * (2 * m_i + t_i) + 1, texts)

            for eval_name, input_path in inputs.items():
                out_dir = results / model_key / training / "eval" / eval_name

                subprocess.run(
                    [
                        sys.executable, str(REPO / "src" / "classifier" / "evaluate.py"),
                        "--model", str(model_dir),
                        "--input", str(input_path),
                        "--output-dir", str(out_dir),
                        "--batch-size", "64",
                        "--max-length", "32",
                    ],
                    check=True,
                    capture_output=True,
                    env=env,
                    cwd=str(REPO),
                )

                if eval_name in ("obfuscated", "kg_obfuscated"):
                    subprocess.run(
                        [
                            sys.executable, str(REPO / "scripts" / "analyze_obfuscated_eval.py"),
                            "--predictions", str(out_dir / "predictions.csv"),
                            "--output-dir", str(out_dir / "analysis"),
                        ],
                        check=True,
                        capture_output=True,
                        env=env,
                        cwd=str(REPO),
                    )

    _FIXTURE.update({"root": root, "results": results, "inputs": inputs})

    return _FIXTURE


def run_stats(fixture, results=None, out_name="stats", extra=None, kg=True):
    out_dir = fixture["root"] / out_name

    cmd = [
        sys.executable, str(REPO / "scripts" / "step1_stats.py"),
        "--results-root", str(results or fixture["results"]),
        "--clean-input", str(fixture["inputs"]["clean"]),
        "--obfuscated-input", str(fixture["inputs"]["obfuscated"]),
        "--n-boot", "300",
        "--output-dir", str(out_dir),
    ]

    if kg:
        cmd += [
            "--kg-clean-input", str(fixture["inputs"]["kg_clean"]),
            "--kg-obfuscated-input", str(fixture["inputs"]["kg_obfuscated"]),
        ]

    cmd += extra or []

    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))

    return proc, out_dir


def read_pred(fixture, model, training, eval_name):
    path = fixture["results"] / model / training / "eval" / eval_name / "predictions.csv"

    return pd.read_csv(path, dtype={"id": str, "seed_id": str})


def test_predictions_have_expected_columns():
    fx = get_fixture()

    df = read_pred(fx, "koelectra", "original", "obfuscated")

    for col in ("id", "text", "label", "source", "seed_id", "technique", "intensity",
                "changed", "n_changed", "prediction", "attack_score"):
        assert col in df.columns, col

    clean = read_pred(fx, "koelectra", "original", "clean")

    assert "seed_id" not in clean.columns


def test_end_to_end_point_estimates_match_existing_results_and_sklearn():
    fx = get_fixture()

    proc, out_dir = run_stats(fx)

    assert proc.returncode == 0, proc.stderr + proc.stdout

    cond = pd.read_csv(out_dir / "condition_metrics.csv")

    assert len(cond) == 2 * 2 * 2 * 2  # suite(main, kg) x model x training x (clean, obfuscated)

    for _, row in cond.iterrows():
        suite = row["suite"]
        eval_name = row["eval_name"]
        eval_dir = fx["results"] / row["model"] / row["training"] / "eval" / eval_name

        pred = read_pred(fx, row["model"], row["training"], eval_name)

        if row["scope"] == "changed_only":
            changed = analyze.normalize_changed(pred["changed"])
            pred = pred.loc[changed]

        y, p = pred["label"].astype(int), pred["prediction"].astype(int)

        tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()

        assert (row["tp"], row["fp"], row["fn"], row["tn"]) == (tp, fp, fn, tn)
        assert row["n"] == len(pred)

        assert abs(row["recall"] - recall_score(y, p, pos_label=1, zero_division=0)) < 1e-6
        assert abs(row["f1"] - f1_score(y, p, pos_label=1, zero_division=0)) < 1e-6
        assert abs(row["precision"] - precision_score(y, p, pos_label=1, zero_division=0)) < 1e-6
        assert abs(row["fpr"] - (fp / (fp + tn))) < 1e-6

        if row["scope"] == "clean_all_rows":
            with open(eval_dir / "metrics.json", encoding="utf-8") as f:
                stored = json.load(f)

            for key in ("recall", "f1", "precision", "fpr", "tp", "fp", "fn", "tn"):
                assert abs(float(stored[key]) - float(row[key])) < 1e-6, (key, stored[key], row[key])
        else:
            overall = pd.read_csv(eval_dir / "analysis" / "obfuscated_overall.csv")
            stored = overall.loc[overall["scope"] == "changed_only"].iloc[0]

            for key in ("recall", "f1", "precision", "fpr", "tp", "fp", "fn", "tn"):
                assert abs(float(stored[key]) - float(row[key])) < 1e-5, (key, stored[key], row[key])

            by_tech = pd.read_csv(eval_dir / "analysis" / "obfuscated_by_technique.csv")

            frame = pred.copy()

            for _, t_row in by_tech.iterrows():
                part = frame.loc[frame["technique"] == t_row["technique"]]
                mine = stats.point_metrics(part)

                for key in ("recall", "f1", "precision", "fpr"):
                    assert abs(float(t_row[key]) - mine[key]) < 1e-5, (t_row["technique"], key)

        # Wilson 구간은 직접 계산한 값과 같아야 함
        lo, hi = stats.wilson_interval(tp, tp + fn)

        assert abs(row["recall_wilson_lo"] - lo) < 1e-6 and abs(row["recall_wilson_hi"] - hi) < 1e-6

        lo, hi = stats.wilson_interval(fp, fp + tn)

        assert abs(row["fpr_wilson_lo"] - lo) < 1e-6 and abs(row["fpr_wilson_hi"] - hi) < 1e-6

        # 기본 구간: clean은 Wilson, obfuscated는 클러스터 부트스트랩 (Wilson은 참고 열)
        if row["scope"] == "clean_all_rows":
            assert row["recall_ci_method"] == row["fpr_ci_method"] == "wilson"
            assert row["recall_ci_lo"] == row["recall_wilson_lo"] and row["recall_ci_hi"] == row["recall_wilson_hi"]
            assert row["fpr_ci_lo"] == row["fpr_wilson_lo"] and row["fpr_ci_hi"] == row["fpr_wilson_hi"]
        else:
            for metric in ("recall", "fpr"):
                assert row[f"{metric}_ci_method"] in ("cluster_bootstrap", "wilson_cluster_fallback")

                lo, hi = row[f"{metric}_ci_lo"], row[f"{metric}_ci_hi"]

                assert lo <= hi
                assert lo - 0.05 <= row[metric] <= hi + 0.05, (metric, lo, row[metric], hi)

        # 부트스트랩 구간이 점추정을 감싸는지(대략)
        for metric in ("f1", "precision"):
            if row[f"{metric}_ci_method"] == "not_computable":
                continue

            assert row[f"{metric}_boot_lo"] <= row[f"{metric}_boot_hi"]
            assert row[f"{metric}_boot_lo"] - 0.05 <= row[metric] <= row[f"{metric}_boot_hi"] + 0.05

    md = (out_dir / "step1_stats_tables.md").read_text(encoding="utf-8")

    assert (out_dir / "run_info.json").exists()

    # 논문용 표에는 기본 구간만 표시한다 (참고용 Wilson 값은 표에 나오지 않음)
    for _, row in cond.iterrows():
        text = stats.with_ci(row["recall"], row["recall_ci_lo"], row["recall_ci_hi"], method=row["recall_ci_method"])

        assert text in md, text

        if row["scope"] == "changed_only":
            wilson_text = stats.with_ci(row["recall"], row["recall_wilson_lo"], row["recall_wilson_hi"])

            if wilson_text != text:
                assert wilson_text not in md.split("### " + row["eval_name"])[1].split("###")[0], wilson_text


def test_end_to_end_paired_and_group_tables_match_independent_computation():
    fx = get_fixture()

    proc, out_dir = run_stats(fx)

    assert proc.returncode == 0, proc.stderr + proc.stdout

    # --- 원문 vs 변형 paired ---
    retain = pd.read_csv(out_dir / "paired_original_vs_variant.csv")

    names = {"main": ("clean", "obfuscated"), "kg": ("kg_clean", "kg_obfuscated")}

    for _, row in retain.iterrows():
        clean_name, obf_name = names[row["suite"]]

        clean = read_pred(fx, row["model"], row["training"], clean_name).set_index("id")
        obf = read_pred(fx, row["model"], row["training"], obf_name)

        obf = obf.loc[analyze.normalize_changed(obf["changed"])]
        obf = obf.loc[obf["label"] == 1]

        orig_pred = clean.loc[obf["seed_id"], "prediction"].to_numpy()

        den = int((orig_pred == 1).sum())
        kept = int(((orig_pred == 1) & (obf["prediction"].to_numpy() == 1)).sum())

        assert row["n_variants_orig_correct"] == den
        assert row["n_kept"] == kept
        assert abs(row["retention_rate"] - kept / den) < 1e-6
        assert row["ci_method"] in ("cluster_bootstrap", "wilson_cluster_fallback", "not_computable")

        if row["ci_method"] != "not_computable":
            assert row["retention_boot_lo"] <= row["retention_rate"] + 0.05
            assert row["retention_boot_hi"] >= row["retention_rate"] - 0.05

    # --- Original vs Augmented ---
    delta = pd.read_csv(out_dir / "paired_original_vs_augmented.csv")

    assert len(delta) == 2 * 2 * 2 * 3  # suite x model x (clean, obf) x (recall, fpr, f1)

    for _, row in delta.iterrows():
        kind = "obfuscated" if row["scope"] == "changed_only" else "clean"
        name = row["eval_name"]

        parts = {}

        for training in ("original", "augmented"):
            pred = read_pred(fx, row["model"], training, name)

            if kind == "obfuscated":
                pred = pred.loc[analyze.normalize_changed(pred["changed"])]

            parts[training] = stats.point_metrics(pred)[row["metric"]]

        assert abs(row["original"] - parts["original"]) < 1e-6
        assert abs(row["augmented"] - parts["augmented"]) < 1e-6
        assert abs(row["delta_aug_minus_orig"] - (parts["augmented"] - parts["original"])) < 1e-6
        if row["ci_method"] == "not_computable":
            assert np.isnan(row["delta_boot_lo"]) and np.isnan(row["delta_boot_hi"])
        else:
            assert row["delta_boot_lo"] <= row["delta_boot_hi"]
            assert row["delta_boot_lo"] - 0.1 <= row["delta_aug_minus_orig"] <= row["delta_boot_hi"] + 0.1

    # --- 기법 분해 ---
    groups = pd.read_csv(out_dir / "technique_groups.csv")
    pairs = stats.parse_train_techniques("yamin_swap:0.7,symbol_insert:0.3")

    for _, row in groups.loc[groups["suite"] == "main"].iterrows():
        obf = read_pred(fx, row["model"], row["training"], "obfuscated")
        obf = obf.loc[analyze.normalize_changed(obf["changed"])]

        mask = [stats.technique_group(t, i, pairs) == row["group"] for t, i in zip(obf["technique"], obf["intensity"])]
        part = obf.loc[mask]

        mine = stats.point_metrics(part)

        assert row["n_attack"] == mine["tp"] + mine["fn"]
        assert row["n_benign"] == mine["fp"] + mine["tn"]
        assert abs(row["recall"] - mine["recall"]) < 1e-6
        assert abs(row["fpr"] - mine["fpr"]) < 1e-6

    for metric in ("recall", "fpr"):
        assert groups[f"{metric}_ci_method"].isin(["cluster_bootstrap", "wilson_cluster_fallback", "not_computable"]).all()

    # 세 그룹 합이 전체 changed=true 행과 같아야 함
    one = groups.loc[(groups["suite"] == "main") & (groups["model"] == "koelectra") & (groups["training"] == "original")]
    obf = read_pred(fx, "koelectra", "original", "obfuscated")
    total = int(analyze.normalize_changed(obf["changed"]).sum())

    assert int((one["n_attack"] + one["n_benign"]).sum()) == total

    # --- source별 ---
    source = pd.read_csv(out_dir / "source_metrics.csv")

    for _, row in source.loc[(source["suite"] == "main") & (source["scope"] == "changed_only")].iterrows():
        obf = read_pred(fx, row["model"], row["training"], "obfuscated")
        obf = obf.loc[analyze.normalize_changed(obf["changed"])]
        part = obf.loc[obf["source"] == row["source"]]
        mine = stats.point_metrics(part)

        assert row["n_attack"] == mine["tp"] + mine["fn"]
        assert abs(row["recall"] - mine["recall"]) < 1e-6
        assert abs(row["fpr"] - mine["fpr"]) < 1e-6

    clean_rows = source["scope"] == "clean_all_rows"

    for metric in ("recall", "fpr"):
        assert (source.loc[clean_rows, f"{metric}_ci_method"] == "wilson").all()
        assert source.loc[~clean_rows, f"{metric}_ci_method"].isin(
            ["cluster_bootstrap", "wilson_cluster_fallback", "not_computable"]
        ).all()

    # --- 정상 문장의 난독화 오탐률 ---
    benign = pd.read_csv(out_dir / "benign_obfuscation_fpr.csv")

    assert benign["ci_method"].isin(["cluster_bootstrap", "wilson_cluster_fallback", "not_computable"]).all()

    for _, row in benign.loc[(benign["suite"] == "main") & (benign["scope"] == "overall")].iterrows():
        obf = read_pred(fx, row["model"], row["training"], "obfuscated")
        obf = obf.loc[analyze.normalize_changed(obf["changed"]) & (obf["label"] == 0)]

        assert row["n_benign_changed"] == len(obf)
        assert row["false_positives"] == int((obf["prediction"] == 1).sum())


def test_end_to_end_reproducible_with_fixed_seed():
    fx = get_fixture()

    proc_a, out_a = run_stats(fx, out_name="stats_a")
    proc_b, out_b = run_stats(fx, out_name="stats_b")
    proc_c, out_c = run_stats(fx, out_name="stats_c", extra=["--seed", "1"])

    assert proc_a.returncode == proc_b.returncode == proc_c.returncode == 0

    a = (out_a / "condition_metrics.csv").read_bytes()
    b = (out_b / "condition_metrics.csv").read_bytes()

    assert a == b

    ca = pd.read_csv(out_a / "condition_metrics.csv")
    cc = pd.read_csv(out_c / "condition_metrics.csv")

    assert (ca["recall"] == cc["recall"]).all()  # 점추정은 시드와 무관
    assert not (ca["f1_boot_lo"] == cc["f1_boot_lo"]).all()  # 구간은 시드에 따라 달라짐


def copy_results(fixture, name):
    dst = fixture["root"] / name

    if dst.exists():
        shutil.rmtree(dst)

    shutil.copytree(fixture["results"], dst)

    return dst


def stop_message(proc):
    return proc.stderr + proc.stdout


def test_missing_row_stops():
    fx = get_fixture()

    results = copy_results(fx, "results_missing")
    path = results / "koelectra" / "original" / "eval" / "obfuscated" / "predictions.csv"

    df = pd.read_csv(path, dtype={"id": str, "seed_id": str})
    dropped = df["id"].iloc[3]

    df.drop(index=3).to_csv(path, index=False)

    proc, out_dir = run_stats(fx, results=results, out_name="stats_missing")

    assert proc.returncode != 0
    assert "빠진 행" in stop_message(proc) and dropped in stop_message(proc), stop_message(proc)
    assert not (out_dir / "condition_metrics.csv").exists()


def test_clean_missing_row_stops():
    fx = get_fixture()

    results = copy_results(fx, "results_clean_missing")
    path = results / "mdeberta" / "augmented" / "eval" / "clean" / "predictions.csv"

    df = pd.read_csv(path, dtype={"id": str})
    dropped = df["id"].iloc[0]

    df.iloc[1:].to_csv(path, index=False)

    proc, _ = run_stats(fx, results=results, out_name="stats_clean_missing")

    assert proc.returncode != 0
    assert "빠진 행" in stop_message(proc) and dropped in stop_message(proc), stop_message(proc)


def test_duplicate_id_stops():
    fx = get_fixture()

    results = copy_results(fx, "results_duplicate")
    path = results / "mdeberta" / "original" / "eval" / "obfuscated" / "predictions.csv"

    df = pd.read_csv(path, dtype={"id": str, "seed_id": str})
    dup = df["id"].iloc[5]

    pd.concat([df, df.iloc[[5]]]).to_csv(path, index=False)

    proc, _ = run_stats(fx, results=results, out_name="stats_duplicate")

    assert proc.returncode != 0
    assert "중복 id" in stop_message(proc) and dup in stop_message(proc), stop_message(proc)


def test_extra_row_stops():
    fx = get_fixture()

    results = copy_results(fx, "results_extra")
    path = results / "koelectra" / "augmented" / "eval" / "obfuscated" / "predictions.csv"

    df = pd.read_csv(path, dtype={"id": str, "seed_id": str})

    extra = df.iloc[[0]].copy()
    extra["id"] = "not_in_input_0001"

    pd.concat([df, extra]).to_csv(path, index=False)

    proc, _ = run_stats(fx, results=results, out_name="stats_extra")

    assert proc.returncode != 0
    assert "여분 행" in stop_message(proc) and "not_in_input_0001" in stop_message(proc), stop_message(proc)


def test_changed_mismatch_with_input_stops():
    fx = get_fixture()

    results = copy_results(fx, "results_changed")
    path = results / "koelectra" / "original" / "eval" / "obfuscated" / "predictions.csv"

    df = pd.read_csv(path, dtype={"id": str, "seed_id": str})

    flipped = df["id"].iloc[2]
    df.loc[2, "changed"] = not bool(analyze.normalize_changed(df["changed"]).iloc[2])
    df.to_csv(path, index=False)

    proc, _ = run_stats(fx, results=results, out_name="stats_changed")

    assert proc.returncode != 0
    assert "changed" in stop_message(proc) and flipped in stop_message(proc), stop_message(proc)


def test_crosscheck_detects_changed_prediction():
    fx = get_fixture()

    results = copy_results(fx, "results_crosscheck")
    path = results / "koelectra" / "original" / "eval" / "clean" / "predictions.csv"

    df = pd.read_csv(path, dtype={"id": str})
    df.loc[0, "prediction"] = 1 - int(df.loc[0, "prediction"])
    df.to_csv(path, index=False)

    proc, _ = run_stats(fx, results=results, out_name="stats_crosscheck")

    assert proc.returncode != 0
    assert "기존 결과와 점추정이 다름" in stop_message(proc), stop_message(proc)


# --------------------------------------------------
# Runner
# --------------------------------------------------

def main():
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]

    passed = failed = skipped = 0

    for name, fn in tests:
        try:
            fn()
        except unittest.SkipTest as e:
            skipped += 1
            print(f"SKIP  {name}: {e}")
        except Exception:
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
        else:
            passed += 1
            print(f"PASS  {name}")

    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
