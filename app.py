"""
粗利最大化ポートフォリオ最適化 兼 媒体ポテンシャル診断Webアプリ
"""

import re

import streamlit as st
import pandas as pd
import numpy as np
from pulp import (
    LpProblem, LpMaximize, LpVariable,
    lpSum, LpStatus, value,
)
import plotly.graph_objects as go

st.set_page_config(
    page_title="広告ポートフォリオ最適化 & 媒体ポテンシャル診断",
    page_icon="📊",
    layout="wide",
)

MENYO_CPA_PENALTY = 9_999_999  # 面予率=0 / CTR=0 / CVR=0 時のペナルティ値

# ── シナリオ定義メタデータ ─────────────────────────────────────
SCENARIO_META = {
    "profit_uncap": {
        "label":       "🚀 粗利MAX（上限突破型）",
        "tab_label":   "🚀 粗利MAX",
        "objective":   "全媒体の粗利合計（売上 − 広告費）を最大化",
        "constraints": "獲得件数 ±50件（目標件数の範囲内）　面予CPAの制約なし",
        "note":        "面予CPAを無視して指定件数レンジの中で最大いくらの粗利が出せるかのポテンシャル上限を示す。",
        "badge":       "#636EFA",
    },
    "menyo_min": {
        "label":       "⚡ 面予CPA最小化（効率特化型）",
        "tab_label":   "⚡ 面予CPA最小",
        "objective":   "全体の加重平均面予CPAを最小化",
        "constraints": "獲得件数 ±50件（目標件数の範囲内）　面予CPA上限の制約なし",
        "note":        "指定件数レンジをクリアしつつ最もコストを抑えられた場合の究極の効率を示す。コスト効率のベンチマーク値として活用。",
        "badge":       "#00CC96",
    },
    "target_strict": {
        "label":       "🎯 目標忠実型（実務メイン）",
        "tab_label":   "🎯 目標忠実",
        "objective":   "全媒体の粗利合計（売上 − 広告費）を最大化",
        "constraints": "面予CPA ≤ 目標面予CPA（絶対遵守・バッファなし） ＋ 獲得件数 ±50件",
        "note":        "目標CPA・目標件数を厳守しながら最大の利益を出す、最も実務で重視するシナリオ。日々の入稿調整の基準として使用。",
        "badge":       "#FFA15A",
    },
}


def fmt_man(val: float) -> str:
    """金額を万円表記で返す（1万円以上の場合）"""
    if abs(val) >= 10_000:
        return f"¥{val / 10_000:,.1f}万"
    return f"¥{val:,.0f}"


# 除去対象の記号パターン：¥ ￥ , ， 円 % ％ 全角/半角スペース
_SYMBOL_RE = re.compile(r"[¥￥,，円%％\s\u3000]")


def _clean_col(series: pd.Series, default: float = 0.0) -> pd.Series:
    """
    スプレッドシートからコピペした「¥15,000」「5.5%」などの
    文字列を数値に変換する。変換不能・None・NaN は default にフォールバック。
    """
    def _parse(val) -> float:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return default
        try:
            return float(_SYMBOL_RE.sub("", str(val)))
        except (ValueError, TypeError):
            return default
    return series.map(_parse)


def _safe_coeff(val: float, na_replacement: float = MENYO_CPA_PENALTY) -> float:
    """NaN / inf を LP に渡す前に安全な値へ変換する"""
    if val is None:
        return 0.0
    if isinstance(val, float) and (np.isinf(val) or np.isnan(val)):
        return na_replacement
    return float(val)


def calc_contribution_scores(cpas: list, menyo_rates: list) -> list:
    """
    媒体貢献度スコア（100点満点）を Min-Max 正規化で計算する。
      CPAスコア（50点満点）    : CPA が低いほど高評価 → 反転正規化
      面予率スコア（50点満点）  : 面予率が高いほど高評価 → 正規化
    全媒体が同値でゼロ割りになる場合はその項目を 25点 固定とする。
    戻り値: list[dict]  各要素 {"cpa_score": int, "mr_score": int, "total": int}
    """
    n = len(cpas)
    if n == 0:
        return []

    max_cpa, min_cpa = max(cpas), min(cpas)
    max_mr,  min_mr  = max(menyo_rates), min(menyo_rates)

    result = []
    for cpa, mr in zip(cpas, menyo_rates):
        cpa_score = round(
            50.0 * (max_cpa - cpa) / (max_cpa - min_cpa)
            if max_cpa - min_cpa > 0 else 25.0
        )
        mr_score = round(
            50.0 * (mr - min_mr) / (max_mr - min_mr)
            if max_mr - min_mr > 0 else 25.0
        )
        result.append({"cpa_score": cpa_score, "mr_score": mr_score, "total": cpa_score + mr_score})
    return result


# ──────────────────────────────────────────────
# LP ソルバー（月次計画）
# ──────────────────────────────────────────────
def solve_portfolio(media, target_cpa, target_acq, scenario):
    """
    【全シナリオ共通の絶対制約】
      target_acq - 50 <= 総獲得件数 <= target_acq + 50

    【区分線形近似（収穫逓減）】
      x_base[i] : 安全圏内 (0 ~ safe_cap[i])      → 通常CPA・通常面予率
      x_extra[i]: 超過分   (0 ~ cap[i]-safe_cap[i]) → 悪化後CPA・低下後面予率

    3シナリオ:
      'profit_uncap'  🚀 粗利MAX  — 目的: maximize Σ(rev-cost), 追加制約なし
      'menyo_min'     ⚡ 面予CPA最小化 — 目的: minimize 加重平均面予CPA, 追加制約なし
      'target_strict' 🎯 目標忠実型 — 目的: maximize Σ(rev-cost),
                                        追加制約: 全体面予CPA ≤ target_cpa（絶対）
    """
    n    = len(media)
    prob = LpProblem(f"Portfolio_{scenario}", LpMaximize)

    # ── 決定変数: 安全圏内 / 超過分 ─────────────────────────
    x_base  = [
        LpVariable(f"xb_{i}", lowBound=0,
                   upBound=media[i]["safe_cap"], cat="Continuous")
        for i in range(n)
    ]
    x_extra = [
        LpVariable(f"xe_{i}", lowBound=0,
                   upBound=max(0.0, media[i]["cap"] - media[i]["safe_cap"]),
                   cat="Continuous")
        for i in range(n)
    ]

    # ── LP係数 NaN/inf 安全処理 ──────────────────────────────
    cpas_b   = [_safe_coeff(media[i]["cpa"],            0.0)               for i in range(n)]
    cpas_e   = [_safe_coeff(media[i]["cpa_extra"],      0.0)               for i in range(n)]
    rewards  = [_safe_coeff(media[i]["reward"],         0.0)               for i in range(n)]
    mcps_b   = [_safe_coeff(media[i]["menyo_cpa"],      MENYO_CPA_PENALTY) for i in range(n)]
    mcps_e   = [_safe_coeff(media[i]["menyo_cpa_extra"],MENYO_CPA_PENALTY) for i in range(n)]

    total_cost    = lpSum([cpas_b[i] * x_base[i] + cpas_e[i] * x_extra[i]    for i in range(n)])
    total_revenue = lpSum([rewards[i] * (x_base[i] + x_extra[i])              for i in range(n)])
    total_acq_lp  = lpSum([x_base[i] + x_extra[i]                             for i in range(n)])

    # ── 目的関数（シナリオ別）────────────────────────────────
    if scenario == "profit_uncap":
        prob += total_revenue - total_cost

    elif scenario == "menyo_min":
        prob += lpSum([
            -(mcps_b[i] * x_base[i] + mcps_e[i] * x_extra[i])
            for i in range(n)
        ])

    else:  # target_strict
        prob += total_revenue - total_cost
        # 追加制約: 全体面予CPA ≤ 目標面予CPA（バッファなし絶対制約）
        prob += (
            lpSum([
                (mcps_b[i] - target_cpa) * x_base[i] +
                (mcps_e[i] - target_cpa) * x_extra[i]
                for i in range(n)
            ]) <= 0,
            "menyo_cpa_strict",
        )

    # ── 共通制約: 全シナリオで目標件数 ±50件 ──────────────────
    prob += total_acq_lp >= max(0, target_acq - 50), "acq_lower"
    prob += total_acq_lp <= target_acq + 50,         "acq_upper"

    # ── 媒体ごとの最低獲得件数制約 ────────────────────────────
    for i in range(n):
        if media[i]["min_acq"] > 0:
            prob += x_base[i] + x_extra[i] >= media[i]["min_acq"], f"min_acq_{i}"

    prob.solve()

    if LpStatus[prob.status] != "Optimal":
        return None

    result = []
    for i in range(n):
        acq_b = max(value(x_base[i])  or 0, 0)
        acq_e = max(value(x_extra[i]) or 0, 0)
        acq   = acq_b + acq_e
        cost  = acq_b * cpas_b[i] + acq_e * cpas_e[i]
        rev   = acq * rewards[i]

        mr_b            = _safe_coeff(media[i]["menyo_rate"],       0.0)
        mr_e            = _safe_coeff(media[i]["menyo_rate_extra"], 0.0)
        eff_menyo_rate  = (mr_b * acq_b + mr_e * acq_e) / acq if acq > 0 else mr_b
        eff_menyo_cpa   = rewards[i] / eff_menyo_rate if eff_menyo_rate > 0 else MENYO_CPA_PENALTY
        eff_cpa         = cost / acq if acq > 0 else 0.0

        result.append(dict(
            name=media[i]["name"],
            acquisitions=round(acq, 1),
            acquisitions_base=round(acq_b, 1),
            acquisitions_extra=round(acq_e, 1),
            cost=round(cost),
            revenue=round(rev),
            gross_profit=round(rev - cost),
            cpa=cpas_b[i],           # 元の（安全圏）CPA（参考値）
            eff_cpa=round(eff_cpa),  # 実効CPA
            menyo_cpa=mcps_b[i],             # 元の面予CPA（参考値）
            eff_menyo_cpa=round(eff_menyo_cpa),  # 実効面予CPA
            menyo_rate=mr_b,                 # 元の面予率（参考値）
            eff_menyo_rate=eff_menyo_rate,   # 実効面予率
            roas=rev / cost if cost > 0 else 0.0,
        ))
    return result


def summarize(result):
    if result is None:
        return None
    df = pd.DataFrame(result)
    total_acq  = df["acquisitions"].sum()
    total_cost = df["cost"].sum()
    total_rev  = df["revenue"].sum()
    total_gp   = df["gross_profit"].sum()
    avg_menyo_cpa = (
        (df["acquisitions"] * df["eff_menyo_cpa"]).sum() / total_acq
        if total_acq > 0 else 0
    )
    return dict(
        df=df,
        total_acq=total_acq,
        total_cost=total_cost,
        total_rev=total_rev,
        total_gp=total_gp,
        avg_cpa=total_cost / total_acq  if total_acq  > 0 else 0,
        avg_roas=total_rev  / total_cost if total_cost > 0 else 0,
        avg_menyo_cpa=avg_menyo_cpa,
    )


# ──────────────────────────────────────────────
# LP ソルバー（月中着地調整）
# ──────────────────────────────────────────────
def solve_mid_portfolio(mid_media, total_actual_acq, total_actual_cost,
                        target_cpa, target_acq, scenario):
    """
    月中着地調整モードの最適化
    変数: 各媒体の残り獲得件数 x_i（0 〜 残り獲得上限数）
    着地予測 = 実績 + 残り最適化結果

    CPA制約: 全体着地予測CPA <= target_cpa（線形変換済み）
    獲得件数制約: target_acq - 50 <= 着地予測件数 <= target_acq + 50
    """
    n    = len(mid_media)
    prob = LpProblem(f"Mid_{scenario}", LpMaximize)

    x = [
        LpVariable(f"xm_{i}", lowBound=0,
                   upBound=mid_media[i]["remaining_cap"], cat="Continuous")
        for i in range(n)
    ]

    # ── LP係数 NaN/inf 安全処理 ──────────────────────────────
    rem_cpas = [_safe_coeff(mid_media[i]["remaining_cpa"], MENYO_CPA_PENALTY) for i in range(n)]
    rewards  = [_safe_coeff(mid_media[i]["reward"],        0.0)               for i in range(n)]

    cpa_upper = target_cpa

    # 目的関数
    if scenario == "profit":
        prob += lpSum([(rewards[i] - rem_cpas[i]) * x[i] for i in range(n)])

    elif scenario == "menyo_cpa":
        prob += lpSum([-rem_cpas[i] * x[i] for i in range(n)])

    else:  # balanced: (reward - cpa) * (reward/cpa) * x  ← menyo_rate_proxy = reward/rem_cpa
        coeffs = []
        for i in range(n):
            if rem_cpas[i] > 0 and rem_cpas[i] < MENYO_CPA_PENALTY:
                menyo_rt = _safe_coeff(rewards[i] / rem_cpas[i], 0.0)
            else:
                menyo_rt = 0.0
            coeffs.append((rewards[i] - rem_cpas[i]) * menyo_rt)
        prob += lpSum([coeffs[i] * x[i] for i in range(n)])

    # 制約1: 全体着地CPA <= cpa_upper
    #   (total_actual_cost + Σ(rem_cpa_i * x_i)) / (total_actual_acq + Σ(x_i)) <= cpa_upper
    #   → Σ((rem_cpa_i - cpa_upper) * x_i) <= cpa_upper * total_actual_acq - total_actual_cost
    rhs_cpa = cpa_upper * total_actual_acq - total_actual_cost
    prob += (
        lpSum([(rem_cpas[i] - cpa_upper) * x[i] for i in range(n)]) <= rhs_cpa,
        "landing_cpa_upper",
    )

    # 制約2-3: 着地予測件数 = 実績 + Σ(x_i) → 残り件数に換算
    remaining_lower = max(0.0, target_acq - 50 - total_actual_acq)
    remaining_upper = max(0.0, target_acq + 50 - total_actual_acq)
    total_remaining = lpSum(x)
    if remaining_lower > 0:
        prob += total_remaining >= remaining_lower, "remaining_lower"
    prob += total_remaining <= remaining_upper, "remaining_upper"

    # 制約4: 媒体ごとの残り最低獲得件数 ──────────────────────────
    for i in range(n):
        if mid_media[i]["min_remaining_acq"] > 0:
            prob += x[i] >= mid_media[i]["min_remaining_acq"], f"min_rem_acq_{i}"

    prob.solve()

    if LpStatus[prob.status] != "Optimal":
        return None

    result = []
    for i in range(n):
        rem_acq   = max(value(x[i]) or 0, 0)
        rem_cost  = rem_acq * rem_cpas[i]
        rem_rev   = rem_acq * rewards[i]
        act_acq   = mid_media[i]["actual_acq"]
        act_cost  = mid_media[i]["actual_cost"]
        land_acq  = act_acq + rem_acq
        land_cost = act_cost + rem_cost
        land_rev  = act_acq * rewards[i] + rem_rev
        land_cpa  = land_cost / land_acq if land_acq > 0 else 0.0
        result.append(dict(
            name=mid_media[i]["name"],
            rem_acquisitions=round(rem_acq, 1),
            rem_cost=round(rem_cost),
            actual_acq=act_acq,
            actual_cost=round(act_cost),
            land_acq=round(land_acq, 1),
            land_cost=round(land_cost),
            land_cpa=round(land_cpa),
            land_rev=round(land_rev),
            land_gp=round(land_rev - land_cost),
            remaining_cpa=rem_cpas[i],
            reward=rewards[i],
        ))
    return result


def summarize_mid(result):
    if result is None:
        return None
    df = pd.DataFrame(result)
    total_land_acq  = df["land_acq"].sum()
    total_land_cost = df["land_cost"].sum()
    total_land_rev  = df["land_rev"].sum()
    total_land_gp   = df["land_gp"].sum()
    total_rem_acq   = df["rem_acquisitions"].sum()
    return dict(
        df=df,
        total_land_acq=total_land_acq,
        total_land_cost=total_land_cost,
        total_land_rev=total_land_rev,
        total_land_gp=total_land_gp,
        total_rem_acq=total_rem_acq,
        avg_land_cpa=total_land_cost / total_land_acq if total_land_acq > 0 else 0,
        avg_roas=total_land_rev / total_land_cost     if total_land_cost > 0 else 0,
    )


INFEASIBLE_MSGS = {
    "profit_uncap": (
        "🚀 **粗利MAX** — 解が見つかりません。  \n"
        "- 有効な媒体（CPA > 0, 上限件数 > 0）が1件以上必要です。"
    ),
    "menyo_min": (
        "⚡ **面予CPA最小化** — 解が見つかりません。  \n"
        "- **全体の目標獲得件数** を下げるか  \n"
        "- 各媒体の **上限件数** を増やしてください。"
    ),
    "target_strict": (
        "🎯 **目標忠実型** — 現在の条件では解が見つかりません。  \n"
        "- サイドバーの **目標 面予CPA** を引き上げる  \n"
        "- サイドバーの **全体の目標獲得件数** を下げる  \n"
        "- 各媒体の **上限件数** を増やす  \n"
        "- 面予率の低い媒体のCPAを下げる"
    ),
}

INFEASIBLE_MID_MSG = (
    "現在の条件では解が見つかりません。以下を調整してください：  \n"
    "- サイドバーの **目標 面予CPA** を引き上げる  \n"
    "- サイドバーの **全体の目標獲得件数** を下げる  \n"
    "- 各媒体の **残り獲得上限数** を増やす"
)

# ──────────────────────────────────────────────
# サイドバー
# ──────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ 目標パラメータ")
    st.divider()

    target_cpa = st.number_input(
        "🎯 目標 面予CPA（円）", min_value=1, value=100_000, step=1_000,
        help="シナリオ③でのみ使用。全体の加重平均面予CPA ≤ この値（バッファなし絶対制約）",
    )
    st.caption(f"シナリオ③制約：面予CPA ≤ ¥{target_cpa:,}（絶対遵守）")

    st.divider()

    target_acq = st.number_input(
        "🎯 全体の目標獲得件数（件）", min_value=1, value=200, step=10,
        help="シナリオ②：下限のみ適用。シナリオ③：±50件の範囲で解を探します",
    )
    st.caption(
        f"全シナリオ共通: {max(0, target_acq - 50):,} 〜 {target_acq + 50:,} 件"
    )

    st.divider()
    st.markdown("**シナリオ制約ガイド**")
    st.markdown(
        "| シナリオ | 面予CPA | 件数 |\n"
        "|:--|:--:|:--:|\n"
        "| 🚀 粗利MAX | ─ | ±50件 |\n"
        "| ⚡ 面予CPA最小 | ─ | ±50件 |\n"
        "| 🎯 目標忠実 | ≤ 目標値 | ±50件 |"
    )

# ──────────────────────────────────────────────
# ヘッダー
# ──────────────────────────────────────────────
st.title("📊 広告ポートフォリオ最適化 & 媒体ポテンシャル診断")
st.caption("🚀 粗利MAX（上限突破型）　⚡ 面予CPA最小化（効率特化型）　🎯 目標忠実型（実務メイン）　の3シナリオで最適配分を算出します。")
st.divider()

# ──────────────────────────────────────────────
# メインタブ
# ──────────────────────────────────────────────
tab_plan, tab_mid = st.tabs(["① 月次計画作成", "② 月中着地調整"])

# ══════════════════════════════════════════════════════════
# TAB 1: 月次計画作成（既存機能）
# ══════════════════════════════════════════════════════════
with tab_plan:

    # ──────────────────────────────────────────────
    # ① 媒体データ入力
    # ──────────────────────────────────────────────
    st.subheader("① 媒体データ入力")
    st.caption("上限件数・CPA・報酬単価・面予率のみ入力してください。広告費・売上・粗利・ROAS・面予CPAは自動計算します。")

    DEFAULT_DATA = pd.DataFrame({
        "媒体名":              ["Google", "Yahoo", "Meta", "LINE"],
        "上限件数":            [80,       50,      120,    40],
        "最低獲得件数":        [0,        0,       0,      0],
        "CPA（円）":           [60_000,   55_000,  45_000, 40_000],
        "報酬単価（円）":       [15_000,   15_000,  15_000, 15_000],
        "面予率（%）":         [60.0,     60.0,    55.0,   50.0],
        "安全拡大件数":        [80,       50,      120,    40],
        "超過時CPA悪化率(%)":  [120.0,    120.0,   120.0,  120.0],
        "超過時面予低下率(%)": [80.0,     80.0,    80.0,   80.0],
    })

    edited_df = st.data_editor(
        DEFAULT_DATA,
        num_rows="dynamic",
        use_container_width=True,
        key="media_editor",
        column_config={
            "媒体名":              st.column_config.TextColumn("媒体名", width="small"),
            "上限件数":            st.column_config.NumberColumn("上限件数", min_value=0, step=1, format="%d 件"),
            "最低獲得件数":        st.column_config.NumberColumn(
                "最低獲得件数", min_value=0, step=1, format="%d 件",
                help="この件数以上は必ず獲得する（0=制約なし）。上限件数を超えた値は自動で丸められます。"),
            "CPA（円）":           st.column_config.NumberColumn("CPA（円）", min_value=0, step=1_000, format="¥%d"),
            "報酬単価（円）":       st.column_config.NumberColumn("報酬単価（円）", min_value=0, step=1_000, format="¥%d"),
            "面予率（%）":         st.column_config.NumberColumn("面予率（%）", min_value=0.0, max_value=100.0, step=0.5, format="%.1f %%"),
            "安全拡大件数":        st.column_config.NumberColumn(
                "安全拡大件数", min_value=0, step=1, format="%d 件",
                help="この件数までは設定CPA・面予率を維持。超えると収穫逓減ペナルティが発動。"),
            "超過時CPA悪化率(%)":  st.column_config.NumberColumn(
                "超過時CPA悪化率(%)", min_value=100.0, max_value=500.0, step=5.0, format="%.0f %%",
                help="安全拡大件数を超えた分のCPA割増率。120% → CPA 20%増し。"),
            "超過時面予低下率(%)": st.column_config.NumberColumn(
                "超過時面予低下率(%)", min_value=0.0, max_value=100.0, step=5.0, format="%.0f %%",
                help="安全拡大件数を超えた分の面予率低下率。80% → 面予率 20%減。"),
        },
    )

    if edited_df.empty or len(edited_df) == 0:
        st.warning("媒体データを1行以上入力してください。")
        st.stop()

    # ── データクレンジング ──
    for _col in ["上限件数", "最低獲得件数", "CPA（円）", "面予率（%）", "安全拡大件数"]:
        edited_df[_col] = _clean_col(edited_df[_col], default=0.0)
    edited_df["報酬単価（円）"]      = _clean_col(edited_df["報酬単価（円）"],      default=15_000.0)
    edited_df["超過時CPA悪化率(%)"]  = _clean_col(edited_df["超過時CPA悪化率(%)"],  default=120.0)
    edited_df["超過時面予低下率(%)"] = _clean_col(edited_df["超過時面予低下率(%)"], default=80.0)

    # ── 内部データ構築 ──
    media = []
    for _, row in edited_df.iterrows():
        name = str(row["媒体名"]).strip()
        if not name or name in ("nan", "None", ""):
            continue
        cap            = int(row["上限件数"])
        cpa            = float(row["CPA（円）"])
        reward         = float(row["報酬単価（円）"])
        menyo_rate_pct = float(row["面予率（%）"])
        menyo_rate     = menyo_rate_pct / 100.0

        # ゼロ割り防止：面予率=0 の場合はペナルティ値を使用
        menyo_cpa    = reward / menyo_rate if menyo_rate > 0 else MENYO_CPA_PENALTY
        cost         = cpa * cap
        revenue      = reward * cap
        gross_profit = revenue - cost
        roas         = revenue / cost if cost > 0 else 0.0

        # ── 最低獲得件数（安全処理：上限件数を超えられない）────
        min_acq_val = int(float(row["最低獲得件数"]))
        min_acq_val = min(min_acq_val, cap)

        # ── 収穫逓減パラメータ ─────────────────────────────────
        safe_cap_raw = float(row["安全拡大件数"])
        safe_cap_val = int(safe_cap_raw) if safe_cap_raw > 0 else cap
        safe_cap_val = min(safe_cap_val, cap)          # 上限件数を超えられない
        cpa_pen_pct  = float(row["超過時CPA悪化率(%)"])
        cpa_pen_pct  = max(cpa_pen_pct, 100.0)         # 100%未満（改善）は無効
        mr_fac_pct   = float(row["超過時面予低下率(%)"])
        cpa_extra_val        = cpa * (cpa_pen_pct / 100.0)
        menyo_rate_extra_val = menyo_rate * (mr_fac_pct / 100.0)
        menyo_cpa_extra_val  = (
            reward / menyo_rate_extra_val
            if menyo_rate_extra_val > 0 else MENYO_CPA_PENALTY
        )

        media.append(dict(
            name=name, cap=cap, cpa=cpa, reward=reward,
            menyo_rate=menyo_rate, menyo_cpa=menyo_cpa,
            cost=cost, revenue=revenue,
            gross_profit=gross_profit, roas=roas,
            min_acq=min_acq_val,
            # 収穫逓減
            safe_cap=safe_cap_val,
            cpa_extra=cpa_extra_val,
            menyo_rate_extra=menyo_rate_extra_val,
            menyo_cpa_extra=menyo_cpa_extra_val,
        ))

    if not media:
        st.warning("有効な媒体データがありません。")
        st.stop()

    # ── 貢献度スコア付与（入力値の CPA と 面予率 で計算）──
    _sc = calc_contribution_scores(
        [m["cpa"]        for m in media],
        [m["menyo_rate"] for m in media],
    )
    for i, m in enumerate(media):
        m["score"]     = _sc[i]["total"]
        m["cpa_score"] = _sc[i]["cpa_score"]
        m["mr_score"]  = _sc[i]["mr_score"]

    # ──────────────────────────────────────────────
    # 現状サマリー
    # ──────────────────────────────────────────────
    total_cap_now  = sum(m["cap"]          for m in media)
    total_cost_now = sum(m["cost"]         for m in media)
    total_rev_now  = sum(m["revenue"]      for m in media)
    total_gp_now   = sum(m["gross_profit"] for m in media)
    avg_cpa_now    = total_cost_now / total_cap_now  if total_cap_now  > 0 else 0
    avg_roas_now   = total_rev_now  / total_cost_now if total_cost_now > 0 else 0

    st.subheader("現状サマリー（全媒体・上限件数ベース）")

    with st.expander("📐 各指標の計算式"):
        st.markdown(
            "| 指標 | 計算式 |\n"
            "|------|--------|\n"
            "| **面予CPA** | 報酬単価 ÷ 面予率 |\n"
            "| **広告費** | CPA × 獲得件数 |\n"
            "| **売上** | 報酬単価 × 獲得件数 |\n"
            "| **粗利** | 売上 − 広告費 |\n"
            "| **ROAS** | 売上 ÷ 広告費 × 100（%） |\n"
            "| **全体面予CPA** | Σ(面予CPA × 獲得件数) ÷ 総獲得件数（加重平均） |"
        )

    kr1, kr2, kr3 = st.columns(3)
    kr1.metric("総上限件数", f"{total_cap_now:,} 件")
    kr2.metric("総広告費",   fmt_man(total_cost_now))
    kr3.metric("総売上",     fmt_man(total_rev_now))

    kr4, kr5, kr6 = st.columns(3)
    kr4.metric("総粗利",   fmt_man(total_gp_now))
    kr5.metric("平均CPA",  f"¥{avg_cpa_now:,.0f}")
    kr6.metric("平均ROAS", f"{avg_roas_now * 100:.1f}%")

    # ── 現状の媒体貢献度スコア（積み上げ棒グラフ）──────────────
    st.markdown("##### 現状の媒体貢献度スコア（CPA × 面予率 / 100点満点）")
    _score_df = pd.DataFrame({
        "媒体名":     [m["name"]      for m in media],
        "CPAスコア":  [m["cpa_score"] for m in media],
        "面予率スコア": [m["mr_score"]  for m in media],
        "合計":       [m["score"]     for m in media],
    }).sort_values("合計", ascending=True)

    fig_score = go.Figure()
    fig_score.add_trace(go.Bar(
        name="CPAスコア（50点満点）",
        x=_score_df["CPAスコア"],
        y=_score_df["媒体名"],
        orientation="h",
        marker_color="#636EFA",
        text=[f"CPA:{s}" for s in _score_df["CPAスコア"]],
        textposition="inside",
        insidetextanchor="middle",
    ))
    fig_score.add_trace(go.Bar(
        name="面予率スコア（50点満点）",
        x=_score_df["面予率スコア"],
        y=_score_df["媒体名"],
        orientation="h",
        marker_color="#00CC96",
        text=[f"面予:{s}" for s in _score_df["面予率スコア"]],
        textposition="inside",
        insidetextanchor="middle",
    ))
    for _, row in _score_df.iterrows():
        fig_score.add_annotation(
            x=row["合計"] + 1, y=row["媒体名"],
            text=f"<b>{row['合計']}点</b>",
            showarrow=False, xanchor="left", font=dict(size=12),
        )
    fig_score.update_layout(
        barmode="stack",
        xaxis=dict(range=[0, 118], title="スコア（100点満点）", showgrid=True),
        height=max(180, len(media) * 58),
        margin=dict(t=10, b=10, l=10, r=60),
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig_score, use_container_width=True)
    st.caption("🔵 CPAスコア（50点満点：CPA低いほど高評価）　🟢 面予率スコア（50点満点：面予率高いほど高評価）")

    with st.expander("媒体別 詳細指標を見る"):
        detail_rows = [{
            "媒体名":        m["name"],
            "上限件数":      m["cap"],
            "CPA（円）":     m["cpa"],
            "報酬単価（円）": m["reward"],
            "面予率":        m["menyo_rate"],
            "面予CPA（円）": m["menyo_cpa"] if m["menyo_cpa"] < MENYO_CPA_PENALTY else float("inf"),
            "広告費（円）":  m["cost"],
            "売上（円）":    m["revenue"],
            "粗利（円）":    m["gross_profit"],
            "ROAS（%）":     m["roas"] * 100,
        } for m in media]
        st.dataframe(
            pd.DataFrame(detail_rows).style.format({
                "上限件数":      "{:,}",
                "CPA（円）":     "¥{:,.0f}",
                "報酬単価（円）": "¥{:,.0f}",
                "面予率":        "{:.1%}",
                "面予CPA（円）": "¥{:,.0f}",
                "広告費（円）":  "¥{:,.0f}",
                "売上（円）":    "¥{:,.0f}",
                "粗利（円）":    "¥{:,.0f}",
                "ROAS（%）":     "{:.1f}%",
            }),
            use_container_width=True, hide_index=True,
        )

    st.divider()

    # ── 3シナリオ計算 ──
    res_profit  = solve_portfolio(media, target_cpa, target_acq, "profit_uncap")
    res_menyo   = solve_portfolio(media, target_cpa, target_acq, "menyo_min")
    res_target  = solve_portfolio(media, target_cpa, target_acq, "target_strict")

    info_profit  = summarize(res_profit)
    info_menyo   = summarize(res_menyo)
    info_target  = summarize(res_target)

    SCENARIOS = [
        ("profit_uncap",  info_profit, res_profit),
        ("menyo_min",     info_menyo,  res_menyo),
        ("target_strict", info_target, res_target),
    ]

    # ──────────────────────────────────────────────
    # ② 最適化シミュレーション
    # ──────────────────────────────────────────────
    st.subheader("② 最適化シミュレーション（3シナリオ比較）")

    st.markdown("##### シナリオ横断サマリー")
    col_p, col_c, col_b = st.columns(3)
    for col, (sc_key, info, _) in zip([col_p, col_c, col_b], SCENARIOS):
        meta = SCENARIO_META[sc_key]
        with col:
            st.markdown(f"**{meta['label']}**")
            st.caption(meta['constraints'])
            if info is None:
                st.error("解なし")
            else:
                ref_gp        = info_profit["total_gp"]     if info_profit else None
                ref_menyo_cpa = info_menyo["avg_menyo_cpa"] if info_menyo  else None
                gp_delta        = f"+¥{info['total_gp'] - ref_gp:,.0f}" if ref_gp and info is not info_profit else None
                menyo_cpa_delta = f"{info['avg_menyo_cpa'] - ref_menyo_cpa:+,.0f}円" if ref_menyo_cpa and info is not info_menyo else None
                st.metric("総粗利",    fmt_man(info['total_gp']),        delta=gp_delta)
                st.metric("平均CPA",   f"¥{info['avg_cpa']:,.0f}")
                st.metric("面予CPA",   f"¥{info['avg_menyo_cpa']:,.0f}", delta=menyo_cpa_delta, delta_color="inverse")
                st.metric("総獲得件数", f"{info['total_acq']:,.1f} 件")
                st.metric("ROAS",      f"{info['avg_roas'] * 100:.1f}%")

    st.markdown("---")

    def render_detail(info, sc_key):
        meta = SCENARIO_META[sc_key]
        # シナリオ情報カード
        st.markdown(
            f"> 📌 **目的** : {meta['objective']}  \n"
            f"> 🔒 **制約** : {meta['constraints']}"
        )
        st.caption(f"💡 {meta['note']}")
        st.markdown("---")

        if info is None:
            st.error(INFEASIBLE_MSGS[sc_key])
            return

        r1c1, r1c2, r1c3 = st.columns(3)
        r1c1.metric("総獲得件数", f"{info['total_acq']:,.1f} 件")
        r1c2.metric("総広告費",   fmt_man(info['total_cost']))
        r1c3.metric("総売上",     fmt_man(info['total_rev']))

        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        r2c1.metric("総粗利",     fmt_man(info['total_gp']))
        r2c2.metric("平均CPA",    f"¥{info['avg_cpa']:,.0f}")
        r2c3.metric("全体面予CPA", f"¥{info['avg_menyo_cpa']:,.0f}")
        r2c4.metric("ROAS",       f"{info['avg_roas'] * 100:.1f}%")

        raw = info["df"]

        # ── 貢献度スコア列を追加（実効値ベースで計算）──
        _sc = calc_contribution_scores(
            list(raw["eff_cpa"]),
            list(raw["eff_menyo_rate"]),
        )

        disp = raw.copy()
        disp["roas"]           = disp["roas"]           * 100
        disp["eff_menyo_rate"] = disp["eff_menyo_rate"] * 100
        disp["貢献度スコア"] = [
            f"{s['total']}点 (CPA:{s['cpa_score']} / 面予:{s['mr_score']})"
            for s in _sc
        ]
        disp = disp.rename(columns={
            "name":               "媒体名",
            "acquisitions":       "目標獲得件数",
            "acquisitions_extra": "超過件数",
            "cost":               "広告費（円）",
            "revenue":            "売上（円）",
            "gross_profit":       "粗利（円）",
            "eff_cpa":            "実効CPA（円）",
            "eff_menyo_cpa":      "実効面予CPA（円）",
            "eff_menyo_rate":     "実効面予率（%）",
            "roas":               "ROAS（%）",
        })

        # ── 合計行の追加 ──────────────────────────────
        tot_acq        = disp["目標獲得件数"].sum()
        tot_extra      = disp["超過件数"].sum()
        tot_cost       = disp["広告費（円）"].sum()
        tot_rev        = disp["売上（円）"].sum()
        tot_gp         = disp["粗利（円）"].sum()
        tot_eff_cpa    = tot_cost / tot_acq if tot_acq > 0 else 0.0
        tot_menyo_cpa  = (raw["acquisitions"] * raw["eff_menyo_cpa"]).sum()  / tot_acq if tot_acq > 0 else 0.0
        tot_menyo_rate = (raw["acquisitions"] * raw["eff_menyo_rate"]).sum() / tot_acq * 100 if tot_acq > 0 else 0.0
        tot_roas       = tot_rev / tot_cost * 100 if tot_cost > 0 else 0.0
        total_row = pd.DataFrame([{
            "媒体名":           "合計",
            "目標獲得件数":      tot_acq,
            "超過件数":          tot_extra,
            "広告費（円）":      tot_cost,
            "売上（円）":        tot_rev,
            "粗利（円）":        tot_gp,
            "実効CPA（円）":     tot_eff_cpa,
            "実効面予CPA（円）": tot_menyo_cpa,
            "実効面予率（%）":   tot_menyo_rate,
            "ROAS（%）":         tot_roas,
            "貢献度スコア":      "―",
        }])
        disp = pd.concat([disp, total_row], ignore_index=True)

        col_order = ["媒体名", "貢献度スコア", "目標獲得件数", "超過件数", "広告費（円）",
                     "売上（円）", "粗利（円）", "実効CPA（円）", "実効面予CPA（円）",
                     "実効面予率（%）", "ROAS（%）"]
        st.dataframe(
            disp[col_order].style.format({
                "目標獲得件数":      "{:,.1f}",
                "超過件数":          "{:,.1f}",
                "広告費（円）":      "¥{:,.0f}",
                "売上（円）":        "¥{:,.0f}",
                "粗利（円）":        "¥{:,.0f}",
                "実効CPA（円）":     "¥{:,.0f}",
                "実効面予CPA（円）": "¥{:,.0f}",
                "実効面予率（%）":   "{:.1f}%",
                "ROAS（%）":         "{:.1f}%",
            }),
            use_container_width=True, hide_index=True,
        )
        if tot_extra > 0:
            st.caption(
                "⚠️ **超過件数** は安全拡大件数を超えて配分された件数です。"
                "実効CPA・実効面予率はペナルティ後の加重平均値を表示しています。"
            )

        # ── 円グラフ（合計行を除いた媒体のみ）──────────────────
        _pie = disp[disp["媒体名"] != "合計"]
        if _pie["目標獲得件数"].sum() > 0:
            pie_c1, pie_c2 = st.columns(2)
            with pie_c1:
                st.markdown("**獲得件数 媒体別内訳（%）**")
                fig_pie_acq = go.Figure(go.Pie(
                    labels=_pie["媒体名"],
                    values=_pie["目標獲得件数"],
                    hole=0.38,
                    textinfo="label+percent",
                    textposition="outside",
                    marker=dict(colors=["#636EFA","#EF553B","#00CC96","#AB63FA",
                                        "#FFA15A","#19D3F3","#FF6692","#B6E880"]),
                ))
                fig_pie_acq.update_layout(
                    height=300, showlegend=False,
                    margin=dict(t=10, b=10, l=10, r=10),
                )
                st.plotly_chart(fig_pie_acq, use_container_width=True, key=f"pie_acq_{sc_key}")
            with pie_c2:
                st.markdown("**広告費 媒体別内訳（%）**")
                fig_pie_cost = go.Figure(go.Pie(
                    labels=_pie["媒体名"],
                    values=_pie["広告費（円）"],
                    hole=0.38,
                    textinfo="label+percent",
                    textposition="outside",
                    marker=dict(colors=["#636EFA","#EF553B","#00CC96","#AB63FA",
                                        "#FFA15A","#19D3F3","#FF6692","#B6E880"]),
                ))
                fig_pie_cost.update_layout(
                    height=300, showlegend=False,
                    margin=dict(t=10, b=10, l=10, r=10),
                )
                st.plotly_chart(fig_pie_cost, use_container_width=True, key=f"pie_cost_{sc_key}")

    sc_tab_p, sc_tab_m, sc_tab_b = st.tabs([
        SCENARIO_META["profit_uncap"]["tab_label"],
        SCENARIO_META["menyo_min"]["tab_label"],
        SCENARIO_META["target_strict"]["tab_label"],
    ])
    with sc_tab_p:
        render_detail(info_profit, "profit_uncap")
    with sc_tab_m:
        render_detail(info_menyo,  "menyo_min")
    with sc_tab_b:
        render_detail(info_target, "target_strict")

    st.divider()

    # ──────────────────────────────────────────────
    # ③ 媒体ポテンシャル診断
    # ──────────────────────────────────────────────
    st.subheader("③ 媒体ポテンシャル診断")

    for m in media:
        disp_menyo_cpa = m["menyo_cpa"] if m["menyo_cpa"] < MENYO_CPA_PENALTY else float("inf")
        with st.expander(f"**{m['name']}**　CPA ¥{m['cpa']:,.0f} / 面予CPA ¥{disp_menyo_cpa:,.0f}", expanded=True):
            col_txt, col_gauge = st.columns([3, 2])

            with col_txt:
                if m["menyo_cpa"] >= MENYO_CPA_PENALTY:
                    st.warning("面予率が0のため面予CPAを計算できません。面予率を入力してください。")
                else:
                    gap = m["menyo_cpa"] - m["cpa"]
                    if m["cpa"] <= m["menyo_cpa"]:
                        st.success(
                            f"CPA余力 **¥{gap:,.0f}**（現状 ¥{m['cpa']:,.0f} ≤ 面予CPA ¥{m['menyo_cpa']:,.0f}）"
                        )
                        st.markdown(f"- **月間粗利**: ¥{m['gross_profit']:,.0f} ／ 年間換算: ¥{m['gross_profit']*12:,.0f}")
                        st.markdown(f"- **ROAS**: {m['roas'] * 100:.1f}%")
                    else:
                        over = -gap
                        st.error(
                            f"CPA超過 **¥{over:,.0f}**（現状 ¥{m['cpa']:,.0f} > 面予CPA ¥{m['menyo_cpa']:,.0f}）"
                        )
                        st.markdown(
                            f"  - CPA を **{over / m['cpa'] * 100:.1f}%** 削減して "
                            f"¥{m['menyo_cpa']:,.0f} 以下を目指してください。"
                        )

            with col_gauge:
                if m["menyo_cpa"] < MENYO_CPA_PENALTY:
                    gauge_max = max(m["menyo_cpa"] * 1.5, m["cpa"] * 1.2)
                    fig_g = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=m["cpa"],
                        number={"prefix": "¥", "valueformat": ",.0f"},
                        gauge={
                            "axis": {"range": [0, gauge_max], "tickformat": ",.0f"},
                            "bar":  {"color": "#EF553B" if m["cpa"] > m["menyo_cpa"] else "#00CC96"},
                            "threshold": {
                                "line": {"color": "#636EFA", "width": 3},
                                "thickness": 0.75,
                                "value": m["menyo_cpa"],
                            },
                            "steps": [
                                {"range": [0, m["menyo_cpa"]], "color": "rgba(0,204,150,0.15)"},
                                {"range": [m["menyo_cpa"], gauge_max], "color": "rgba(239,85,59,0.15)"},
                            ],
                        },
                        title={"text": "現状CPA　🔵=面予CPA"},
                    ))
                    fig_g.update_layout(height=200, margin=dict(t=40, b=5, l=20, r=20))
                    st.plotly_chart(fig_g, use_container_width=True)

    st.divider()

    # ──────────────────────────────────────────────
    # ④ グラフで可視化
    # ──────────────────────────────────────────────
    st.subheader("④ グラフで可視化")

    names      = [m["name"]         for m in media]
    cpas       = [m["cpa"]          for m in media]
    menyo_cpas = [min(m["menyo_cpa"], target_cpa * 3) for m in media]  # ペナルティ値を表示用にキャップ
    gross_prs  = [m["gross_profit"] for m in media]
    roas_vals  = [m["roas"]         for m in media]

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown("**現状CPA vs 面予CPA**")
        fig_cpa = go.Figure()
        fig_cpa.add_trace(go.Bar(name="現状CPA",        x=names, y=cpas,       marker_color="#EF553B"))
        fig_cpa.add_trace(go.Bar(name="面予CPA（目安）", x=names, y=menyo_cpas, marker_color="#636EFA", opacity=0.7))
        fig_cpa.add_hline(
            y=target_cpa, line_dash="dash", line_color="#FFA15A",
            annotation_text=f"🎯 目標面予CPA ¥{target_cpa:,}（シナリオ③制約ライン）",
        )
        fig_cpa.update_layout(barmode="group", yaxis_title="CPA（円）", height=360, margin=dict(t=20, b=10))
        st.plotly_chart(fig_cpa, use_container_width=True)

    with col_g2:
        st.markdown("**媒体別 月間粗利 ＋ ROAS**")
        fig_gp = go.Figure()
        fig_gp.add_trace(go.Bar(
            name="月間粗利", x=names, y=gross_prs,
            marker_color="#00CC96",
            text=[f"¥{gp:,.0f}" for gp in gross_prs], textposition="outside",
            yaxis="y",
        ))
        roas_pct = [r * 100 for r in roas_vals]
        fig_gp.add_trace(go.Scatter(
            name="ROAS", x=names, y=roas_pct,
            mode="lines+markers+text",
            text=[f"{r:.1f}%" for r in roas_pct], textposition="top center",
            marker=dict(size=10, color="#AB63FA"),
            line=dict(color="#AB63FA", width=2),
            yaxis="y2",
        ))
        fig_gp.update_layout(
            yaxis=dict(title="粗利（円）"),
            yaxis2=dict(title="ROAS（%）", overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h", y=1.08),
            height=360, margin=dict(t=20, b=10),
        )
        st.plotly_chart(fig_gp, use_container_width=True)

    valid = [(sc_key, res) for sc_key, info, res in SCENARIOS if res is not None]
    if valid:
        st.markdown("**シナリオ別 獲得件数ポートフォリオ**")
        fig_port = go.Figure()
        for sc_key_v, res in valid:
            meta_v = SCENARIO_META[sc_key_v]
            fig_port.add_trace(go.Bar(
                name=meta_v["label"],
                x=[r["name"] for r in res],
                y=[r["acquisitions"] for r in res],
                marker_color=meta_v["badge"],
                text=[f"{r['acquisitions']:,.1f}" for r in res],
                textposition="outside",
            ))
        fig_port.update_layout(
            barmode="group", yaxis_title="獲得件数",
            height=360, margin=dict(t=20, b=10),
        )
        st.plotly_chart(fig_port, use_container_width=True)

    st.divider()

    # ──────────────────────────────────────────────
    # ⑤ ネクストアクション
    # ──────────────────────────────────────────────
    st.subheader("⑤ ネクストアクション")

    actions = []
    for m in media:
        if m["menyo_cpa"] >= MENYO_CPA_PENALTY:
            actions.append(dict(
                name=m["name"], priority=0, label="⚠️ 設定確認",
                action="面予率が0のため面予CPAを計算できません。面予率を入力してください。",
                sort_key=0,
            ))
        elif m["cpa"] > m["menyo_cpa"]:
            over = m["cpa"] - m["menyo_cpa"]
            actions.append(dict(
                name=m["name"], priority=1, label="🔴 要改善",
                action=(
                    f"CPA削減が必要: 現状 ¥{m['cpa']:,.0f} → 目標 ¥{m['menyo_cpa']:,.0f} 以下 "
                    f"（{over / m['cpa'] * 100:.1f}% 改善必要）"
                ),
                sort_key=-over * m["cap"],
            ))
        else:
            headroom = m["menyo_cpa"] - m["cpa"]
            actions.append(dict(
                name=m["name"], priority=2, label="🟢 拡大推奨",
                action=(
                    f"上限まで積極配分: CPA余力 ¥{headroom:,.0f}、"
                    f"月間粗利 ¥{m['gross_profit']:,.0f}（年間 ¥{m['gross_profit']*12:,.0f}）"
                ),
                sort_key=m["gross_profit"],
            ))

    actions.sort(key=lambda a: (a["priority"], -a["sort_key"]))

    for i, act in enumerate(actions, 1):
        st.markdown(f"**{i}. {act['label']}　{act['name']}**")
        st.markdown(f"&emsp;{act['action']}")

    st.divider()
    st.caption(
        "本ツールは入力された指標に基づくシミュレーションです。"
        "実際の広告運用では、市場環境・季節要因・競合状況等により結果が異なる場合があります。"
    )


# ══════════════════════════════════════════════════════════
# TAB 2: 月中着地調整
# ══════════════════════════════════════════════════════════
with tab_mid:
    st.subheader("② 月中着地調整モード")
    st.caption(
        "これまでの実績と残り期間の予測を入力し、「月次目標を達成するための残り配分」を最適化します。  \n"
        "着地予測 ＝ 実績 ＋ 残り期間の最適化結果として計算します。"
    )

    with st.expander("📐 残り期間CPA の計算式"):
        st.markdown(
            "| 指標 | 計算式 |\n"
            "|------|--------|\n"
            "| **残り期間 見込みCPA** | 予測CPM × 10 ÷ （予測CTR × 予測CVR） |\n"
            "| **着地予測件数** | 実績獲得件数 ＋ 残り目標獲得件数 |\n"
            "| **着地予測広告費** | 実績消化広告費 ＋ 残り期間CPA × 残り目標獲得件数 |\n"
            "| **着地予測CPA** | 着地予測広告費 ÷ 着地予測件数 |\n"
            "| **着地予測粗利** | 着地予測件数 × 報酬単価 ＋ 着地予測広告費 |"
        )

    # ──────────────────────────────────────────────
    # 月中データ入力
    # ──────────────────────────────────────────────
    st.markdown("#### 媒体別 実績・予測データ入力")

    DEFAULT_MID_DATA = pd.DataFrame({
        "媒体名":               ["Google",    "Yahoo",     "Meta",      "LINE"],
        "実績獲得件数":          [50,          30,          70,          20],
        "実績消化広告費（円）":   [3_000_000,  1_650_000,  3_150_000,   800_000],
        "予測CPM（円）":         [6_000,       5_500,       4_500,       4_000],
        "予測CTR（%）":          [0.5,         0.5,         0.5,         0.5],
        "予測CVR（%）":          [2.0,         2.0,         2.0,         2.0],
        "残り獲得上限数（件）":   [40,          25,          60,          25],
        "最低残り獲得件数（件）": [0,           0,           0,           0],
        "報酬単価（円）":         [15_000,      15_000,      15_000,      15_000],
    })

    mid_edited_df = st.data_editor(
        DEFAULT_MID_DATA,
        num_rows="dynamic",
        use_container_width=True,
        key="mid_editor",
        column_config={
            "媒体名":               st.column_config.TextColumn("媒体名", width="small"),
            "実績獲得件数":          st.column_config.NumberColumn("実績獲得件数",        min_value=0, step=1,     format="%d 件"),
            "実績消化広告費（円）":   st.column_config.NumberColumn("実績消化広告費（円）", min_value=0, step=10_000, format="¥%d"),
            "予測CPM（円）":         st.column_config.NumberColumn("予測CPM（円）",        min_value=0, step=100,   format="¥%d"),
            "予測CTR（%）":          st.column_config.NumberColumn("予測CTR（%）",         min_value=0.0, step=0.1, format="%.2f %%"),
            "予測CVR（%）":          st.column_config.NumberColumn("予測CVR（%）",         min_value=0.0, step=0.1, format="%.2f %%"),
            "残り獲得上限数（件）":   st.column_config.NumberColumn("残り獲得上限数（件）", min_value=0, step=1, format="%d 件"),
            "最低残り獲得件数（件）": st.column_config.NumberColumn(
                "最低残り獲得件数（件）", min_value=0, step=1, format="%d 件",
                help="残り期間でこの件数以上は必ず獲得する（0=制約なし）。残り上限数を超えた値は自動で丸められます。"),
            "報酬単価（円）":         st.column_config.NumberColumn("報酬単価（円）",       min_value=0, step=1_000, format="¥%d"),
        },
    )

    if mid_edited_df.empty or len(mid_edited_df) == 0:
        st.warning("媒体データを1行以上入力してください。")
        st.stop()

    # ── データクレンジング ──
    for _col in ["実績獲得件数", "実績消化広告費（円）", "予測CPM（円）",
                 "予測CTR（%）", "予測CVR（%）", "残り獲得上限数（件）", "最低残り獲得件数（件）"]:
        mid_edited_df[_col] = _clean_col(mid_edited_df[_col], default=0.0)
    mid_edited_df["報酬単価（円）"] = _clean_col(mid_edited_df["報酬単価（円）"], default=15_000.0)

    # ── 内部データ構築 ──
    mid_media = []
    for _, row in mid_edited_df.iterrows():
        name = str(row["媒体名"]).strip()
        if not name or name in ("nan", "None", ""):
            continue
        actual_acq   = float(row["実績獲得件数"])
        actual_cost  = float(row["実績消化広告費（円）"])
        cpm          = float(row["予測CPM（円）"])
        ctr_pct      = float(row["予測CTR（%）"])
        cvr_pct      = float(row["予測CVR（%）"])
        remaining_cap    = float(row["残り獲得上限数（件）"])
        min_rem_acq_raw  = int(float(row["最低残り獲得件数（件）"]))
        min_rem_acq_val  = min(min_rem_acq_raw, int(remaining_cap))  # 上限を超えられない
        reward           = float(row["報酬単価（円）"])

        # 残り期間の見込みCPA（ゼロ割り防止）
        if ctr_pct > 0 and cvr_pct > 0:
            remaining_cpa = cpm * 10.0 / (ctr_pct * cvr_pct)
        else:
            remaining_cpa = MENYO_CPA_PENALTY

        mid_media.append(dict(
            name=name,
            actual_acq=actual_acq,
            actual_cost=actual_cost,
            cpm=cpm,
            ctr_pct=ctr_pct,
            cvr_pct=cvr_pct,
            remaining_cap=remaining_cap,
            min_remaining_acq=min_rem_acq_val,
            reward=reward,
            remaining_cpa=remaining_cpa,
        ))

    if not mid_media:
        st.warning("有効な媒体データがありません。")
        st.stop()

    # ── 実績サマリー ──
    total_actual_acq  = sum(m["actual_acq"]  for m in mid_media)
    total_actual_cost = sum(m["actual_cost"] for m in mid_media)
    actual_avg_cpa    = total_actual_cost / total_actual_acq if total_actual_acq > 0 else 0

    st.markdown("#### 実績サマリー")
    ac1, ac2, ac3 = st.columns(3)
    ac1.metric("実績獲得件数（累計）", f"{total_actual_acq:,.0f} 件",
               delta=f"目標まで {max(0, target_acq - total_actual_acq):,.0f} 件")
    ac2.metric("実績消化広告費（累計）", fmt_man(total_actual_cost))
    ac3.metric("実績平均CPA", f"¥{actual_avg_cpa:,.0f}")

    with st.expander("媒体別 残り期間 見込みCPA を確認する"):
        preview_rows = []
        for m in mid_media:
            rem_cpa_disp = m["remaining_cpa"] if m["remaining_cpa"] < MENYO_CPA_PENALTY else float("inf")
            preview_rows.append({
                "媒体名":               m["name"],
                "実績獲得件数":          m["actual_acq"],
                "実績消化広告費（円）":   m["actual_cost"],
                "予測CPM（円）":         m["cpm"],
                "予測CTR（%）":          m["ctr_pct"],
                "予測CVR（%）":          m["cvr_pct"],
                "残り見込みCPA（円）":    rem_cpa_disp,
                "残り獲得上限数（件）":   m["remaining_cap"],
            })
        st.dataframe(
            pd.DataFrame(preview_rows).style.format({
                "実績獲得件数":          "{:,.0f}",
                "実績消化広告費（円）":   "¥{:,.0f}",
                "予測CPM（円）":         "¥{:,.0f}",
                "予測CTR（%）":          "{:.2f}%",
                "予測CVR（%）":          "{:.2f}%",
                "残り見込みCPA（円）":    "¥{:,.0f}",
                "残り獲得上限数（件）":   "{:,.0f}",
            }),
            use_container_width=True, hide_index=True,
        )

    st.divider()

    # ── 月中3シナリオ計算 ──
    mid_res_profit   = solve_mid_portfolio(mid_media, total_actual_acq, total_actual_cost, target_cpa, target_acq, "profit")
    mid_res_menyo    = solve_mid_portfolio(mid_media, total_actual_acq, total_actual_cost, target_cpa, target_acq, "menyo_cpa")
    mid_res_balanced = solve_mid_portfolio(mid_media, total_actual_acq, total_actual_cost, target_cpa, target_acq, "balanced")

    mid_info_profit   = summarize_mid(mid_res_profit)
    mid_info_menyo    = summarize_mid(mid_res_menyo)
    mid_info_balanced = summarize_mid(mid_res_balanced)

    MID_SCENARIOS = [
        ("🏆 粗利最大化",    mid_info_profit,   mid_res_profit),
        ("💴 面予CPA最小化", mid_info_menyo,    mid_res_menyo),
        ("⚖️ バランス型",    mid_info_balanced, mid_res_balanced),
    ]

    # ──────────────────────────────────────────────
    # 月中 最適化結果
    # ──────────────────────────────────────────────
    st.subheader("残り期間の最適配分（3シナリオ比較）")

    st.markdown("##### シナリオ横断サマリー（着地予測ベース）")
    mid_col_p, mid_col_c, mid_col_b = st.columns(3)
    for col, (label, info, _) in zip([mid_col_p, mid_col_c, mid_col_b], MID_SCENARIOS):
        with col:
            st.markdown(f"**{label}**")
            if info is None:
                st.error("解なし")
            else:
                ref_gp = mid_info_profit["total_land_gp"] if mid_info_profit else None
                gp_delta = f"+¥{info['total_land_gp'] - ref_gp:,.0f}" if ref_gp and info is not mid_info_profit else None
                st.metric("着地予測粗利",   fmt_man(info['total_land_gp']),       delta=gp_delta)
                st.metric("着地予測件数",   f"{info['total_land_acq']:,.1f} 件")
                st.metric("着地予測CPA",    f"¥{info['avg_land_cpa']:,.0f}")
                st.metric("残り獲得件数",   f"{info['total_rem_acq']:,.1f} 件")
                st.metric("ROAS（着地）",   f"{info['avg_roas'] * 100:.1f}%")

    st.markdown("---")

    def render_mid_detail(label, info):
        if info is None:
            st.error(INFEASIBLE_MID_MSG)
            return

        r1c1, r1c2, r1c3 = st.columns(3)
        r1c1.metric("着地予測件数",   f"{info['total_land_acq']:,.1f} 件")
        r1c2.metric("着地予測広告費", fmt_man(info['total_land_cost']))
        r1c3.metric("着地予測売上",   fmt_man(info['total_land_rev']))

        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        r2c1.metric("着地予測粗利",  fmt_man(info['total_land_gp']))
        r2c2.metric("着地予測CPA",   f"¥{info['avg_land_cpa']:,.0f}")
        r2c3.metric("残り獲得件数",  f"{info['total_rem_acq']:,.1f} 件")
        r2c4.metric("ROAS（着地）",  f"{info['avg_roas'] * 100:.1f}%")

        disp = info["df"].copy()
        disp = disp.rename(columns={
            "name":             "媒体名",
            "rem_acquisitions": "残り目標獲得件数",
            "rem_cost":         "残り投下広告費（円）",
            "actual_acq":       "実績獲得件数",
            "actual_cost":      "実績消化広告費（円）",
            "land_acq":         "着地予測件数",
            "land_cost":        "着地予測広告費（円）",
            "land_cpa":         "着地予測CPA（円）",
            "land_rev":         "着地予測売上（円）",
            "land_gp":          "着地予測粗利（円）",
        })
        show_cols = [
            "媒体名", "残り目標獲得件数", "残り投下広告費（円）",
            "着地予測件数", "着地予測広告費（円）", "着地予測CPA（円）", "着地予測粗利（円）",
        ]
        st.dataframe(
            disp[show_cols].style.format({
                "残り目標獲得件数":    "{:,.1f}",
                "残り投下広告費（円）": "¥{:,.0f}",
                "着地予測件数":        "{:,.1f}",
                "着地予測広告費（円）": "¥{:,.0f}",
                "着地予測CPA（円）":   "¥{:,.0f}",
                "着地予測粗利（円）":  "¥{:,.0f}",
            }),
            use_container_width=True, hide_index=True,
        )

    mid_tab_p, mid_tab_m, mid_tab_b = st.tabs(["🏆 粗利最大化", "💴 面予CPA最小化", "⚖️ バランス型"])
    with mid_tab_p:
        render_mid_detail("粗利最大化",    mid_info_profit)
    with mid_tab_m:
        render_mid_detail("面予CPA最小化", mid_info_menyo)
    with mid_tab_b:
        render_mid_detail("バランス型",    mid_info_balanced)

    st.divider()
    st.caption(
        "本ツールは入力された指標に基づくシミュレーションです。"
        "実際の広告運用では、市場環境・季節要因・競合状況等により結果が異なる場合があります。"
    )
