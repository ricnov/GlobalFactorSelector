#!/usr/bin/env python
"""
tca_flow_persistence.py v0.2 — the persistence of the platform's own order flow, measured from the inside.

Items 1–3 of the Tóth–Palit–Lillo–Farmer programme ("Why is equity order flow so persistent?", JEDC 2015) that the
BTCA stores support without a public trade tape:

  1. THE SPLITTING MECHANISM, OBSERVED.  Directional episodes per name (runs of trading days with the same sign of the
     platform's net flow) → size and duration distributions, Clauset–Shalizi–Newman power-law tail fit (exponent α of the
     complementary CDF), sign autocorrelation C(τ) ∝ τ^(−γ) of the platform's own flow at several resolutions (account
     orders in arrival order, same-minute bursts, placements, daily net sign, daily signed participation), γ by log–log
     fit with a day-block bootstrap and a DFA cross-check, the Lillo–Mike–Farmer test γ = α − 1, the dependence between
     the signs of successive episodes, and the persistence of the flow by origin (a reason/program code when the store
     has one, otherwise an account-day proxy: same-side account days = cash-flow-like, mixed = rebalance-like).
  2. THE BROKERAGE DISTORTION, MEASURED.  From placements: brokers per episode, the probability that consecutive child
     orders go to the same broker, P_same(τ) along the platform's placement sequence (the quantity Tóth et al. had to
     model), broker shares by name and by algorithm, and the share of the platform's own splitting that a member-code
     decomposition would classify as herding.
  3. THE LIQUIDITY PROVIDERS' RESPONSE TO PERSISTENT FLOW.  Fill mark-outs at 5 / 30 / 60 minutes (fill against the
     later mid on the desk's own quote tape, the realized half-spread; reversion = realized − effective) against the
     platform's cumulative same-direction flow over the preceding k days, the position of the day inside its episode
     and the cumulative size already executed, binned and in a dollar-weighted regression with name-day-clustered
     standard errors; first-day against last-day fills at matched participation; names where the platform is a large
     share of the volume against the rest.
  4. ANTICIPATION AND COUNTERPARTIES (v0.2).  Does the price move in the platform's direction BEFORE the day's orders
     arrive (previous close → arrival, in excess of the cross-section of names that day) as a function of the platform's
     own flow over the preceding days, on the first day of an episode (the decision day, where the model's own signal can
     explain a drift) against the continuation days (where the schedule is mechanical and a drift points at anticipation
     by others)? Does it depend on which broker executed the previous day's flow (the information-leakage channel)? Does
     the executing broker's fill reversion depend on the persistence of the flow (the internalisation channel)? What
     happens to the price after an episode ends (continuation = others positioned along the flow and hold; reversion =
     impact decay or unwinding)? Needs the arrival price and the previous close on the orders; the close for the
     post-episode returns; placements for the broker breakdowns.

    python tca_flow_persistence.py                                   # full history of the store, ADR + Common Stock
    python tca_flow_persistence.py --start 2025-01-01 --end 2026-10-31 --sec-type adr
    python tca_flow_persistence.py --col ACCOUNT=ACCT_ID --col ORIGIN=ORDER_REASON_CD --col BROKER=BRKR_CD
    python tca_flow_persistence.py --print-schema                    # show the columns found in the stores and stop
    python tca_flow_persistence.py --demo                             # synthetic store with known α, γ and reversion: the self-test

Run it from the dashboard folder (it imports data_access; tca_report is used for the security-type aliases when
present). It reads the stores only; nothing in the pipeline or the dashboard is touched.

Conventions: + = cost; side +1 buy / −1 sell; participation in percent of ADV; bp of the reference price; every
aggregate dollar-weighted unless the column says otherwise. Exponents: the CCDF of episode sizes is P(X > x) ∝ x^(−α)
(Clauset et al. fit the density exponent a = α + 1; both are reported). The Lillo–Mike–Farmer prediction is γ = α − 1.

Outputs (in --out): report.md, tables/*.csv, charts/*.png, run.json. Interpretation: see the reading guide
(tca_flow_persistence_reading_guide.md).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")
VERSION = "0.2"
LOG = lambda *a: print("[flow]", *a, flush=True)  # noqa: E731

# ----------------------------------------------------------------------------------------------------------- data access
QUERY = None          # callable(sql, params) -> DataFrame, set in main() (data_access.query or the demo engine)
VIEWS: list[str] = []

CANDIDATES = {   # logical role -> candidate column names in the stores (first match wins; --col ROLE=NAME overrides)
    "ORDER_ID": ["ORDER_ID"], "TICKER": ["TICKER"], "SIDE": ["BUY_SELL_CD", "SIDE", "SIDE_CD"], "TRADE_DT": ["TRADE_DT"],
    "QTY": ["SHARES_FILLED", "FILLED_QTY", "EXEC_QTY", "EXECUTED_QTY"], "ORDER_QTY": ["ORDER_QTY"],
    "PCT_ADV": ["PCT_ADV", "PCT_OF_ADV", "ADV_PCT"], "ADV": ["ADV_IMPLIED", "ADV_20D_SHARES", "ADV_SHARES", "AVG_DAILY_VOLUME", "ADV"],
    "NOTIONAL": ["FILLED_NOTIONAL", "EXEC_NOTIONAL", "EXECUTED_NOTIONAL", "NOTIONAL"],
    "ACCOUNT": ["ACCOUNT_ID", "ACCT_ID", "ACCOUNT_CD", "ACCOUNT_NUM", "ACCT_NUM", "PORTFOLIO_ID", "PORTFOLIO_CD", "ACCOUNT"],
    "ARRIVAL": ["ORDER_ARRIVAL_TMSTMP", "ARRIVAL_TMSTMP", "ORDER_CREATE_TMSTMP", "ORDER_TMSTMP", "ORDER_START_TMSTMP", "START_TMSTMP", "ARRIVAL_TIME", "ORDER_TIME"],
    "ORIGIN": ["ORDER_REASON_CD", "REASON_CD", "PROGRAM_CD", "PROGRAM_ID", "STRATEGY_CD", "MANDATE_CD", "ORDER_SOURCE_CD", "SOURCE_CD", "ORIGIN"],
    "SEC_TYPE": ["SECURITY_TYPE_CD", "SEC_TYPE_CD", "SECURITY_TYPE"], "DAILY_VOL": ["DAILY_VOL_BPS"],
    "QUOTED_HS": ["QUOTED_HALF_SPREAD_BPS"], "ARR_SPREAD": ["ARR_SPREAD_BPS", "ARRIVAL_SPREAD_BPS"],
    "BROKER": ["BROKER_CD", "BROKER", "BROKER_NM", "EXECUTING_BROKER", "EXEC_BROKER", "DESTINATION"],
    "ALGO": ["ALGO_NM", "ALGO_NAME", "STRATEGY_NM", "STRATEGY_NAME", "STRATEGY", "ALGO_CD"],
    "VENUE": ["VENUE", "VENUE_CD", "EXCHANGE_CD", "MARKET_CD", "LAST_MARKET"],
    "PLACEMENT_ID": ["PLACEMENT_ID", "PLCMNT_ID", "ROUTE_ID"], "PLACEMENT_TS": ["PLACEMENT_TMSTMP", "PLCMNT_TMSTMP", "ROUTE_TMSTMP", "PLACEMENT_TIME", "CREATE_TMSTMP", "TMSTMP"],
    "PLACEMENT_QTY": ["PLACEMENT_QTY", "PLCMNT_QTY", "ROUTED_QTY", "ROUTE_QTY", "QTY"],
    "FILL_TS": ["FILL_TMSTMP", "EXEC_TMSTMP", "FILL_TIME"], "FILL_PRC": ["FILL_PRC", "EXEC_PRC", "FILL_PRICE"], "FILL_QTY": ["FILL_VOLUME_QTY", "FILL_QTY", "EXEC_QTY"],
    "FILL_BID": ["ORDER_BID_PRC", "BID_PRC", "BID"], "FILL_ASK": ["ORDER_ASK_PRC", "ASK_PRC", "ASK"],
    "ARRIVAL_PRC": ["ARRIVAL_PRC", "ARRIVAL_PRICE", "ARR_PRC", "ARRIVAL_MID", "ARRIVAL_PX", "BENCHMARK_ARRIVAL_PX", "ARRIVAL_MID_PRC"],
    "PREV_CLOSE_PRC": ["PREV_CLOSE_PRC", "PRIOR_CLOSE_PRC", "PREVIOUS_CLOSE_PRC", "PREV_CLOSE", "PRIOR_CLOSE", "YEST_CLOSE_PRC", "PRIOR_DAY_CLOSE_PRC", "PREV_DAY_CLOSE_PRC"],
    "CLOSE_PRC": ["CLOSE_PRC", "CLOSE_PRICE", "DAY_CLOSE_PRC", "CLOSING_PRC", "CLOSE_PX", "CLOSE"],
}
BUY_CODES = {"B", "CS", "BUY", "BC", "1", "+1"}


def q(sql: str, params=None) -> pd.DataFrame:
    return QUERY(sql, list(params) if params is not None else [])


def columns_of(view: str) -> list[str]:
    try:
        return [c.upper() for c in q(f"SELECT * FROM {view} LIMIT 0").columns]
    except Exception as e:  # pragma: no cover
        LOG(f"cannot read {view}: {e}"); return []


def resolve_schema(overrides: dict, orders_view: str, placements_view: str | None, fills_view: str | None) -> dict:
    """Map logical roles to the columns present in each store; overrides win; roles not found are None."""
    cols = {"orders": columns_of(orders_view), "placements": columns_of(placements_view) if placements_view else [], "fills": columns_of(fills_view) if fills_view else []}
    def pick(role, where):
        if role in overrides: return overrides[role] if overrides[role].upper() in cols[where] else None
        for c in CANDIDATES[role]:
            if c in cols[where]: return c
        return None
    S = {"orders_view": orders_view, "placements_view": placements_view, "fills_view": fills_view, "columns": cols}
    for role in ("ORDER_ID", "TICKER", "SIDE", "TRADE_DT", "QTY", "ORDER_QTY", "PCT_ADV", "ADV", "NOTIONAL", "ACCOUNT", "ARRIVAL", "ORIGIN", "SEC_TYPE", "DAILY_VOL", "QUOTED_HS", "ARR_SPREAD", "ARRIVAL_PRC", "PREV_CLOSE_PRC", "CLOSE_PRC"):
        S["o_" + role] = pick(role, "orders")
    S["o_BROKER"] = pick("BROKER", "orders")
    for role in ("ORDER_ID", "PLACEMENT_ID", "BROKER", "ALGO", "VENUE", "PLACEMENT_TS", "PLACEMENT_QTY"):
        S["p_" + role] = pick(role, "placements") if placements_view else None
    for role in ("ORDER_ID", "TRADE_DT", "FILL_TS", "FILL_PRC", "FILL_QTY", "FILL_BID", "FILL_ASK"):
        S["f_" + role] = pick(role, "fills") if fills_view else None
    return S


def print_schema(S: dict) -> None:
    for store in ("orders", "placements", "fills"):
        LOG(f"{store} view = {S.get(store + '_view')}: {', '.join(S['columns'][store]) if S['columns'][store] else '(not available)'}")
    for k, v in S.items():
        if k[:2] in ("o_", "p_", "f_"): LOG(f"  {k:18s} -> {v}")


# ----------------------------------------------------------------------------------------------------------- helpers
def w_avg(x, w):
    x = np.asarray(x, float); w = np.asarray(w, float); ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return float(np.sum(x[ok] * w[ok]) / np.sum(w[ok])) if ok.any() and np.sum(w[ok]) > 0 else np.nan


def side_of(series: pd.Series, buy_codes: set) -> np.ndarray:
    s = series.astype(str).str.strip().str.upper()
    return np.where(s.isin(buy_codes), 1, -1)


def pct_ci(vals, lo=2.5, hi=97.5):
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    return (float(np.percentile(v, lo)), float(np.percentile(v, hi))) if len(v) >= 10 else (np.nan, np.nan)


def ols_cluster(y, X, w, cluster):
    """Weighted least squares with cluster-robust (sandwich) standard errors; returns beta, se, t, n, n_clusters."""
    y = np.asarray(y, float); X = np.asarray(X, float); w = np.asarray(w, float)
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1) & np.isfinite(w) & (w > 0)
    y, X, w = y[ok], X[ok], w[ok]; cl = pd.factorize(pd.Series(np.asarray(cluster)[ok]).astype(str))[0]
    n, k = X.shape
    if n <= k + 1: return np.full(k, np.nan), np.full(k, np.nan), np.full(k, np.nan), n, 0
    sw = np.sqrt(w / w.mean()); Xw = X * sw[:, None]; yw = y * sw
    XtX_inv = np.linalg.pinv(Xw.T @ Xw); beta = XtX_inv @ (Xw.T @ yw); e = yw - Xw @ beta
    G = cl.max() + 1; meat = np.zeros((k, k)); U = Xw * e[:, None]
    for g in range(G):
        ug = U[cl == g].sum(axis=0); meat += np.outer(ug, ug)
    adj = (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1)); V = adj * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.clip(np.diag(V), 0, None)); t = beta / np.where(se > 0, se, np.nan)
    return beta, se, t, n, G


# ----------------------------------------------------------------------------------------------------------- loading
def load_orders(S: dict, args, where_extra: str, params_extra: list) -> pd.DataFrame:
    o = S["orders_view"]
    sel = [f"{S['o_ORDER_ID']} AS ORDER_ID", f"{S['o_TICKER']} AS TICKER", f"{S['o_SIDE']} AS SIDE_RAW", f"CAST({S['o_TRADE_DT']} AS DATE) AS TRADE_DT",
           f"TRY_CAST({S['o_ORDER_QTY']} AS DOUBLE) AS ORDER_QTY"]
    sel.append(f"TRY_CAST({S['o_QTY']} AS DOUBLE) AS QTY" if S["o_QTY"] else "NULL AS QTY")
    sel.append(f"TRY_CAST({S['o_PCT_ADV']} AS DOUBLE) AS PCT_ADV" if S["o_PCT_ADV"] else "NULL AS PCT_ADV")
    sel.append(f"TRY_CAST({S['o_ADV']} AS DOUBLE) AS ADV_RAW" if S["o_ADV"] else "NULL AS ADV_RAW")
    sel.append(f"TRY_CAST({S['o_NOTIONAL']} AS DOUBLE) AS NOTIONAL" if S["o_NOTIONAL"] else "NULL AS NOTIONAL")
    sel.append(f"CAST({S['o_ACCOUNT']} AS VARCHAR) AS ACCOUNT" if S["o_ACCOUNT"] else "NULL AS ACCOUNT")
    sel.append(f"TRY_CAST({S['o_ARRIVAL']} AS TIMESTAMP) AS ARRIVAL" if S["o_ARRIVAL"] else "NULL AS ARRIVAL")
    sel.append(f"CAST({S['o_ORIGIN']} AS VARCHAR) AS ORIGIN" if S["o_ORIGIN"] else "NULL AS ORIGIN")
    sel.append(f"CAST({S['o_SEC_TYPE']} AS VARCHAR) AS SEC_TYPE" if S["o_SEC_TYPE"] else "'all' AS SEC_TYPE")
    sel.append(f"TRY_CAST({S['o_DAILY_VOL']} AS DOUBLE) AS DAILY_VOL" if S["o_DAILY_VOL"] else "NULL AS DAILY_VOL")
    sel.append(f"TRY_CAST({S['o_QUOTED_HS']} AS DOUBLE) AS QUOTED_HS" if S["o_QUOTED_HS"] else (f"TRY_CAST({S['o_ARR_SPREAD']} AS DOUBLE) / 2 AS QUOTED_HS" if S["o_ARR_SPREAD"] else "NULL AS QUOTED_HS"))
    sel.append(f"CAST({S['o_BROKER']} AS VARCHAR) AS BROKER_O" if S["o_BROKER"] else "NULL AS BROKER_O")
    for role in ("ARRIVAL_PRC", "PREV_CLOSE_PRC", "CLOSE_PRC"):
        sel.append(f"TRY_CAST({S['o_' + role]} AS DOUBLE) AS {role}" if S["o_" + role] else f"NULL AS {role}")
    where = [f"{S['o_TRADE_DT']} BETWEEN ? AND ?"]; params = [args.start, args.end]
    if where_extra: where.append(where_extra); params += params_extra
    d = q(f"SELECT {', '.join(sel)} FROM {o} WHERE {' AND '.join(where)}", params)
    if d.empty: sys.exit("[flow] no orders in the window")
    d["TICKER"] = d["TICKER"].astype(str); d["TRADE_DT"] = pd.to_datetime(d["TRADE_DT"]).dt.date
    d["side"] = side_of(d["SIDE_RAW"], set(args.buy_codes.split(",")))
    d["q"] = pd.to_numeric(d["QTY"], errors="coerce").where(lambda s: s > 0).fillna(pd.to_numeric(d["ORDER_QTY"], errors="coerce"))
    pct = pd.to_numeric(d["PCT_ADV"], errors="coerce") * (100.0 if args.pct_adv_is_fraction else 1.0)
    adv_from_pct = pd.to_numeric(d["ORDER_QTY"], errors="coerce") / (pct / 100.0).replace(0, np.nan)
    d["adv"] = pd.to_numeric(d["ADV_RAW"], errors="coerce").where(lambda s: s > 0).fillna(adv_from_pct)
    d["notional"] = pd.to_numeric(d["NOTIONAL"], errors="coerce")
    d = d[(d["q"] > 0) & (d["adv"] > 0)].copy()
    d["ARRIVAL"] = pd.to_datetime(d["ARRIVAL"], errors="coerce")
    for role in ("ARRIVAL_PRC", "PREV_CLOSE_PRC", "CLOSE_PRC"): d[role] = pd.to_numeric(d[role], errors="coerce").where(lambda x: x > 0)
    return d.drop(columns=["SIDE_RAW", "QTY", "ADV_RAW", "NOTIONAL", "PCT_ADV"])


def load_placements(S: dict, d: pd.DataFrame, args) -> pd.DataFrame:
    if not S["placements_view"] or not S["p_ORDER_ID"] or not S["p_BROKER"]:
        return pd.DataFrame()
    v = S["placements_view"]
    sel = [f"p.{S['p_ORDER_ID']} AS ORDER_ID", f"CAST(p.{S['p_BROKER']} AS VARCHAR) AS BROKER"]
    sel.append(f"CAST(p.{S['p_PLACEMENT_ID']} AS VARCHAR) AS PLACEMENT_ID" if S["p_PLACEMENT_ID"] else "NULL AS PLACEMENT_ID")
    sel.append(f"CAST(p.{S['p_ALGO']} AS VARCHAR) AS ALGO" if S["p_ALGO"] else "NULL AS ALGO")
    sel.append(f"CAST(p.{S['p_VENUE']} AS VARCHAR) AS VENUE" if S["p_VENUE"] else "NULL AS VENUE")
    sel.append(f"TRY_CAST(p.{S['p_PLACEMENT_TS']} AS TIMESTAMP) AS TS" if S["p_PLACEMENT_TS"] else "NULL AS TS")
    sel.append(f"TRY_CAST(p.{S['p_PLACEMENT_QTY']} AS DOUBLE) AS PQTY" if S["p_PLACEMENT_QTY"] else "NULL AS PQTY")
    sql = f"SELECT {', '.join(sel)} FROM {v} p JOIN (SELECT {S['o_ORDER_ID']} AS OID FROM {S['orders_view']} WHERE {S['o_TRADE_DT']} BETWEEN ? AND ?) o ON p.{S['p_ORDER_ID']} = o.OID"
    p = q(sql, [args.start, args.end])
    if p.empty: return p
    p = p.merge(d[["ORDER_ID", "TICKER", "TRADE_DT", "side", "q", "notional", "ARRIVAL"]], on="ORDER_ID", how="inner")
    p["TS"] = pd.to_datetime(p["TS"], errors="coerce").fillna(p["ARRIVAL"])
    p["BROKER"] = p["BROKER"].fillna("?").astype(str)
    return p


MARKOUT_SQL = """
WITH o AS (
  SELECT {OID} AS ORDER_ID, {TICKER} AS TICKER, CASE WHEN UPPER(CAST({SIDE} AS VARCHAR)) IN ({BUYS}) THEN 1 ELSE -1 END AS side
  FROM {ORDERS} WHERE {TRADE_DT} BETWEEN ? AND ? {WHERE_EXTRA}),
f AS (
  SELECT f.{FOID} AS ORDER_ID, o.TICKER, o.side, TRY_CAST(f.{FTS} AS TIMESTAMP) AS ts,
         TRY_CAST(f.{FPX} AS DOUBLE) AS px, TRY_CAST(f.{FQTY} AS DOUBLE) AS qty,
         (TRY_CAST(f.{FBID} AS DOUBLE) + TRY_CAST(f.{FASK} AS DOUBLE)) / 2 AS mid0
  FROM {FILLS} f JOIN o ON f.{FOID} = o.ORDER_ID
  WHERE {FDATE} AND TRY_CAST(f.{FPX} AS DOUBLE) > 0 AND TRY_CAST(f.{FQTY} AS DOUBLE) > 0),
tape AS (
  SELECT t.TICKER, TRY_CAST(f.{FTS} AS TIMESTAMP) AS ts, (TRY_CAST(f.{FBID} AS DOUBLE) + TRY_CAST(f.{FASK} AS DOUBLE)) / 2 AS mid
  FROM {FILLS} f JOIN (SELECT {OID} AS OID, {TICKER} AS TICKER FROM {ORDERS} WHERE {TRADE_DT} BETWEEN ? AND ?) t ON f.{FOID} = t.OID
  WHERE {FDATE} AND TRY_CAST(f.{FBID} AS DOUBLE) > 0 AND TRY_CAST(f.{FASK} AS DOUBLE) >= TRY_CAST(f.{FBID} AS DOUBLE)
    AND t.TICKER IN (SELECT DISTINCT TICKER FROM o)),
probe AS (SELECT f.*, ts + {H} * INTERVAL 1 MINUTE AS tgt, GREATEST(30, {H} * 30) AS tol_s FROM f WHERE ts IS NOT NULL),
m AS (
  SELECT p.ORDER_ID, p.px * p.qty AS dol,
         CASE WHEN t.ts IS NOT NULL AND t.ts > p.ts AND EXTRACT(EPOCH FROM p.tgt - t.ts) <= p.tol_s
              THEN p.side * (p.px - t.mid) / p.px * 1e4 END AS realized,
         CASE WHEN p.mid0 > 0 THEN p.side * (p.px - p.mid0) / p.mid0 * 1e4 END AS eff
  FROM probe p ASOF LEFT JOIN tape t ON p.TICKER = t.TICKER AND p.tgt >= t.ts)
SELECT ORDER_ID,
       SUM(realized * dol) / NULLIF(SUM(CASE WHEN realized IS NOT NULL THEN dol END), 0) AS realized_bps,
       SUM(CASE WHEN realized IS NOT NULL THEN eff * dol END) / NULLIF(SUM(CASE WHEN realized IS NOT NULL AND eff IS NOT NULL THEN dol END), 0) AS eff_bps,
       SUM(eff * dol) / NULLIF(SUM(CASE WHEN eff IS NOT NULL THEN dol END), 0) AS eff_all_bps,
       SUM(CASE WHEN realized IS NOT NULL THEN dol END) AS covered_dol, SUM(dol) AS dol, COUNT(*) AS n_fills
FROM m GROUP BY 1
"""


def load_markouts(S: dict, args, horizons: list[int], where_extra: str, params_extra: list) -> pd.DataFrame:
    """Per order and horizon: dollar-weighted realized half-spread at +H minutes (fill against the later mid on the tape
    reconstructed from the desk's own fill records), the effective half-spread of the same fills, coverage."""
    if not S["fills_view"] or not all(S[k] for k in ("f_ORDER_ID", "f_FILL_TS", "f_FILL_PRC", "f_FILL_QTY", "f_FILL_BID", "f_FILL_ASK")):
        return pd.DataFrame()
    fdate = f"f.{S['f_TRADE_DT']} BETWEEN '{(args.start - dt.timedelta(days=5)).isoformat()}'::DATE AND '{(args.end + dt.timedelta(days=5)).isoformat()}'::DATE" if S["f_TRADE_DT"] else "TRUE"
    buys = ", ".join(f"'{c}'" for c in args.buy_codes.split(","))
    out = []
    for H in horizons:
        sql = MARKOUT_SQL.format(OID=S["o_ORDER_ID"], TICKER=S["o_TICKER"], SIDE=S["o_SIDE"], BUYS=buys, ORDERS=S["orders_view"], TRADE_DT=S["o_TRADE_DT"],
                                 WHERE_EXTRA=(" AND " + where_extra) if where_extra else "", FOID=S["f_ORDER_ID"], FTS=S["f_FILL_TS"], FPX=S["f_FILL_PRC"], FQTY=S["f_FILL_QTY"],
                                 FBID=S["f_FILL_BID"], FASK=S["f_FILL_ASK"], FILLS=S["fills_view"], FDATE=fdate, H=H)
        t0 = dt.datetime.now()
        r = q(sql, [args.start, args.end] + list(params_extra) + [args.start, args.end])
        r["H"] = H; out.append(r)
        LOG(f"mark-outs +{H}m: {len(r):,} orders, {int((r['covered_dol'] > 0).sum()):,} with a covered fill, in {(dt.datetime.now() - t0).seconds}s")
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ----------------------------------------------------------------------------------------------------------- name-days and episodes
def build_name_days(d: pd.DataFrame, calendar: list) -> pd.DataFrame:
    """One row per name-day: net signed shares Q, gross shares G, ADV (median of the day's orders), net and gross
    participation in % ADV, orders, accounts, notional."""
    g = d.assign(sq=d["side"] * d["q"]).groupby(["TICKER", "TRADE_DT"], sort=True)
    nd = g.agg(Q=("sq", "sum"), G=("q", "sum"), adv=("adv", "median"), n_orders=("q", "size"), notional=("notional", "sum")).reset_index()
    nd["n_accounts"] = g["ACCOUNT"].nunique().to_numpy() if d["ACCOUNT"].notna().any() else np.nan
    tidx = {dte: i for i, dte in enumerate(calendar)}
    nd["t"] = nd["TRADE_DT"].map(tidx).astype(int)
    nd["p_net"] = nd["Q"].abs() / nd["adv"] * 100; nd["p_gross"] = nd["G"] / nd["adv"] * 100
    nd["sign"] = np.sign(nd["Q"]).astype(int)
    return nd


def build_episodes(nd: pd.DataFrame, quiet_pct: float, gap_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Directional episodes per name: maximal runs of active days (net participation ≥ quiet_pct) with the same sign of
    the net flow, allowing up to gap_days trading days without active flow inside the run. Returns (episodes, nd with
    episode ids)."""
    nd = nd.sort_values(["TICKER", "t"]).reset_index(drop=True)
    active = (nd["p_net"].to_numpy() >= quiet_pct) & (nd["sign"].to_numpy() != 0)
    tick = nd["TICKER"].to_numpy(); t = nd["t"].to_numpy(); sg = nd["sign"].to_numpy(); pn = nd["p_net"].to_numpy()
    Q = nd["Q"].to_numpy(float); G = nd["G"].to_numpy(float); adv = nd["adv"].to_numpy(float); no = nd["n_orders"].to_numpy(); na = nd["n_accounts"].to_numpy(float); usd = nd["notional"].to_numpy(float)
    ep_id = np.full(len(nd), -1); rows = []; cur = None; k = 0
    def close(c):
        rows.append({"episode": c["id"], "TICKER": c["tk"], "sign": c["sign"], "start_t": c["t0"], "end_t": c["t1"], "n_days": c["n"], "span_days": c["t1"] - c["t0"] + 1,
                     "size_adv_pct": c["size"], "size_shares": c["sh"], "gross_shares": c["gr"], "size_usd": c["usd"], "n_orders": c["no"], "accounts_mean": c["na"] / c["n"] if c["n"] else np.nan,
                     "max_day_pct": c["pmax"], "mean_day_pct": c["size"] / c["n"] if c["n"] else np.nan})
    for i in range(len(nd)):
        if cur is not None and (tick[i] != cur["tk"] or (active[i] and (sg[i] != cur["sign"] or t[i] - cur["t1"] - 1 > gap_days)) or (not active[i] and t[i] - cur["t1"] - 1 > gap_days)):
            close(cur); cur = None
        if not active[i]:
            if cur is not None and t[i] - cur["t1"] - 1 <= gap_days: ep_id[i] = cur["id"]     # quiet day inside a run keeps its id (position bookkeeping)
            continue
        if cur is None:
            cur = {"id": k, "tk": tick[i], "sign": int(sg[i]), "t0": int(t[i]), "t1": int(t[i]), "n": 0, "size": 0.0, "sh": 0.0, "gr": 0.0, "usd": 0.0, "no": 0, "na": 0.0, "pmax": 0.0}; k += 1
        cur["t1"] = int(t[i]); cur["n"] += 1; cur["size"] += float(pn[i]); cur["sh"] += abs(Q[i]); cur["gr"] += G[i]; cur["usd"] += (usd[i] if np.isfinite(usd[i]) else 0.0)
        cur["no"] += int(no[i]); cur["na"] += (na[i] if np.isfinite(na[i]) else 0.0); cur["pmax"] = max(cur["pmax"], float(pn[i])); ep_id[i] = cur["id"]
    if cur is not None: close(cur)
    nd["episode"] = ep_id; nd["active"] = active
    ep = pd.DataFrame(rows)
    if len(ep):
        end_of = nd["episode"].map(ep.set_index("episode")["end_t"]); nd.loc[(nd["episode"] >= 0) & (nd["t"] > end_of), "episode"] = -1
    if len(ep):
        ep["size_days"] = ep["n_days"]
    return ep, nd


# ----------------------------------------------------------------------------------------------------------- power-law tail (Clauset–Shalizi–Newman)
def fit_powerlaw(x, xmin_grid: int = 60, min_tail: int = 50, discrete: bool = False) -> dict:
    """Power law p(x) ∝ x^(−a) above x_min, x_min by the KS-minimising rule (Clauset, Shalizi, Newman 2009). Continuous
    MLE a = 1 + n / Σ ln(x_i / x_min); for integer data the approximate discrete MLE with x_min − 1/2 in place of x_min
    (adequate for x_min ≥ 6, rough below). Returns a (density exponent), alpha = a − 1 (exponent of the CCDF), x_min,
    n_tail, KS, SE(a) = (a − 1)/√n."""
    x = np.asarray(x, float); x = x[np.isfinite(x) & (x > 0)]
    shift = 0.5 if discrete else 0.0
    if len(x) < min_tail: return {"a": np.nan, "alpha": np.nan, "xmin": np.nan, "n_tail": len(x), "ks": np.nan, "se": np.nan}
    xs = np.sort(x); cand = np.unique(np.quantile(xs, np.linspace(0.0, 1 - min_tail / len(xs), xmin_grid))) if len(xs) > 2 * min_tail else np.unique(xs[: max(1, len(xs) - min_tail)])
    best = None
    for xm in cand:
        tail = xs[xs >= xm]; n = len(tail)
        if n < min_tail or xm <= 0: continue
        if discrete and xm - shift <= 0: continue
        a = 1 + n / np.sum(np.log(tail / (xm - shift)))
        if not np.isfinite(a) or a <= 1: continue
        emp = np.arange(1, n + 1) / n; fit = 1 - ((tail + shift) / (xm - shift + 2 * shift)) ** (1 - a) if discrete else 1 - (tail / xm) ** (1 - a)
        ks = float(np.max(np.abs(emp - fit)))
        if best is None or ks < best["ks"]: best = {"a": float(a), "alpha": float(a - 1), "xmin": float(xm), "n_tail": int(n), "ks": ks, "se": float((a - 1) / np.sqrt(n))}
    return best or {"a": np.nan, "alpha": np.nan, "xmin": np.nan, "n_tail": 0, "ks": np.nan, "se": np.nan}


def powerlaw_gof(x, fit: dict, reps: int, rng) -> float:
    """Clauset et al. semi-parametric bootstrap p-value: fraction of synthetic data sets with KS ≥ the observed one."""
    x = np.asarray(x, float); x = x[np.isfinite(x) & (x > 0)]
    if reps <= 0 or not np.isfinite(fit.get("ks", np.nan)): return np.nan
    below = x[x < fit["xmin"]]; n = len(x); ptail = fit["n_tail"] / n; worse = 0
    for _ in range(reps):
        ntail = rng.binomial(n, ptail)
        synth_tail = fit["xmin"] * (1 - rng.random(ntail)) ** (-1 / (fit["a"] - 1))
        synth = np.concatenate([synth_tail, rng.choice(below, n - ntail, replace=True) if len(below) else synth_tail[:0]])
        f2 = fit_powerlaw(synth)
        if np.isfinite(f2["ks"]) and f2["ks"] >= fit["ks"]: worse += 1
    return worse / reps


def lognormal_lr_test(x, fit: dict):
    """Vuong-type likelihood-ratio test of the fitted power law against a lognormal truncated at x_min (Clauset et al.
    §5): R > 0 favours the power law; p is the two-sided p-value of R = 0. Needs scipy; returns (nan, nan) otherwise."""
    try:
        from scipy import optimize, stats
    except Exception:
        return np.nan, np.nan
    x = np.asarray(x, float); tail = x[np.isfinite(x) & (x >= fit["xmin"])]
    if len(tail) < 50 or not np.isfinite(fit["a"]): return np.nan, np.nan
    lx = np.log(tail); xm = fit["xmin"]
    def nll(params):
        mu, ls = params; s = np.exp(ls)
        z = (np.log(xm) - mu) / s; tailp = stats.norm.sf(z)
        if tailp <= 1e-300: return 1e300
        return -np.sum(stats.norm.logpdf((lx - mu) / s) - np.log(s) - lx - np.log(tailp))
    res = optimize.minimize(nll, x0=[lx.mean(), np.log(lx.std() + 1e-6)], method="Nelder-Mead")
    mu, s = res.x[0], np.exp(res.x[1]); z = (np.log(xm) - mu) / s
    ll_ln = stats.norm.logpdf((lx - mu) / s) - np.log(s) - lx - np.log(stats.norm.sf(z))
    ll_pl = np.log((fit["a"] - 1) / xm) - fit["a"] * np.log(tail / xm)
    diff = ll_pl - ll_ln; R = diff.sum(); sd = diff.std(ddof=1) * np.sqrt(len(diff))
    if sd <= 0: return float(R), np.nan
    return float(R), float(2 * stats.norm.sf(abs(R / sd)))


# ----------------------------------------------------------------------------------------------------------- autocorrelation, γ, DFA
def acf_fft(x: np.ndarray, max_lag: int, mu: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Autocovariance sums of x at lags 0..max_lag via FFT, and the number of pairs per lag. The mean removed is `mu`
    (the pooled mean over all sequences) when given: removing each short sequence's own mean biases the autocorrelation
    of a long-memory sequence downwards by roughly Σ_τ C(τ) / n, which is large for daily sequences of a few hundred days."""
    x = np.asarray(x, float); n = len(x); x = x - (x.mean() if mu is None else mu)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(x, nfft); ac = np.fft.irfft(f * np.conj(f), nfft)[: max_lag + 1]
    pairs = n - np.arange(max_lag + 1)
    return ac, pairs


def pooled_mean(seqs: list[np.ndarray]) -> float:
    n = sum(len(s) for s in seqs); return float(sum(np.sum(s) for s in seqs) / n) if n else 0.0


def pooled_acf(seqs: list[np.ndarray], max_lag: int) -> np.ndarray:
    """Pooled autocorrelation C(τ) = ⟨ε_t ε_{t+τ}⟩ − ⟨ε⟩²: autocovariance sums over all sequences divided by the pooled
    number of pairs, normalised by the pooled variance; the mean ⟨ε⟩ is the pooled mean over all sequences."""
    mu = pooled_mean(seqs); num = np.zeros(max_lag + 1); den = np.zeros(max_lag + 1); var = 0.0; nvar = 0
    for s in seqs:
        if len(s) < 3: continue
        L = min(max_lag, len(s) - 2); ac, pairs = acf_fft(s, L, mu)
        num[: L + 1] += ac; den[: L + 1] += pairs; var += ac[0]; nvar += len(s)
    if nvar == 0 or var <= 0: return np.full(max_lag + 1, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        C = (num / den) / (var / nvar)
    C[den < 50] = np.nan
    return C


def fit_gamma(C: np.ndarray, lag_min: int = 1, lag_max: int | None = None, min_points: int = 4) -> dict:
    """γ from log C(τ) = log c − γ log τ, OLS over log-spaced lag bins where C > 0, from lag_min up to the first lag at
    which the binned C drops to zero or below (or lag_max)."""
    L = len(C) - 1; lag_max = min(lag_max or L, L)
    edges = np.unique(np.round(np.logspace(np.log10(lag_min), np.log10(lag_max), 40)).astype(int))
    xs, ys = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        seg = C[a:b]; seg = seg[np.isfinite(seg)]
        if not len(seg): continue
        c = seg.mean()
        if c <= 0: break
        xs.append(np.log(np.sqrt(a * max(b - 1, a)))); ys.append(np.log(c))
    if len(xs) < min_points: return {"gamma": np.nan, "c": np.nan, "lag_hi": np.nan, "points": len(xs), "r2": np.nan}
    X = np.column_stack([np.ones(len(xs)), xs]); b, res, *_ = np.linalg.lstsq(X, np.asarray(ys), rcond=None)
    yhat = X @ b; ss = np.sum((np.asarray(ys) - np.mean(ys)) ** 2); r2 = 1 - np.sum((np.asarray(ys) - yhat) ** 2) / ss if ss > 0 else np.nan
    return {"gamma": float(-b[1]), "c": float(np.exp(b[0])), "lag_hi": float(np.exp(max(xs))), "points": len(xs), "r2": float(r2)}


def dfa_hurst(x: np.ndarray, min_box: int = 8, n_boxes: int = 20) -> tuple[float, np.ndarray, np.ndarray]:
    """DFA-1 Hurst exponent of a sequence: F(s) ∝ s^H over log-spaced box sizes s ∈ [min_box, n/4]."""
    x = np.asarray(x, float); n = len(x)
    if n < 8 * min_box: return np.nan, np.array([]), np.array([])
    y = np.cumsum(x - x.mean()); sizes = np.unique(np.round(np.logspace(np.log10(min_box), np.log10(n // 4), n_boxes)).astype(int))
    F = []
    for s in sizes:
        nb = n // s; segs = y[: nb * s].reshape(nb, s); tt = np.arange(s)
        A = np.column_stack([np.ones(s), tt]); coef = np.linalg.lstsq(A, segs.T, rcond=None)[0]; resid = segs - (A @ coef).T
        F.append(np.sqrt(np.mean(resid ** 2)))
    F = np.asarray(F); ok = F > 0
    if ok.sum() < 4: return np.nan, sizes, F
    H = np.polyfit(np.log(sizes[ok]), np.log(F[ok]), 1)[0]
    return float(H), sizes, F


def day_block_bootstrap_calendar(T: int, block: int, rng) -> np.ndarray:
    """A resampled calendar: blocks of `block` consecutive day indices drawn with replacement until T days are covered."""
    out = []
    while sum(len(b) for b in out) < T:
        s = rng.integers(0, max(T - block, 0) + 1); out.append(np.arange(s, min(s + block, T)))
    return np.concatenate(out)[:T]


class SignSequences:
    """Per-name sign sequences at one resolution, stored per day so that a day-block bootstrap can rebuild them."""
    def __init__(self, name: str, by_day: dict):            # by_day: {TICKER: {t: np.ndarray of ±1}}
        self.name = name; self.by_day = by_day
    def sequences(self, calendar_idx=None) -> list[np.ndarray]:
        out = []
        for tk, days in self.by_day.items():
            if calendar_idx is None: parts = [days[t] for t in sorted(days)]
            else: parts = [days[t] for t in calendar_idx if t in days]
            if parts:
                s = np.concatenate(parts)
                if len(s) >= 20: out.append(s)
        return out
    def n_events(self) -> int:
        return int(sum(len(v) for days in self.by_day.values() for v in days.values()))


def sequences_from_orders(d: pd.DataFrame, tidx: dict, burst_minutes: int) -> dict:
    """Account-order signs in arrival order, same-minute same-side bursts collapsed to one event, per name and day."""
    d = d.sort_values(["TICKER", "TRADE_DT", "ARRIVAL", "ORDER_ID"], na_position="last")
    orders, bursts = {}, {}
    has_ts = d["ARRIVAL"].notna().mean() > 0.5
    for (tk, dte), g in d.groupby(["TICKER", "TRADE_DT"], sort=False):
        t = tidx[dte]; s = g["side"].to_numpy()
        orders.setdefault(tk, {})[t] = s
        if has_ts:
            key = g["ARRIVAL"].dt.floor(f"{burst_minutes}min").astype("int64").to_numpy() if g["ARRIVAL"].notna().all() else np.arange(len(g))
            change = np.r_[True, (key[1:] != key[:-1]) | (s[1:] != s[:-1])]
        else:
            change = np.r_[True, s[1:] != s[:-1]]
        bursts.setdefault(tk, {})[t] = s[change]
    return {"orders": SignSequences("account orders (arrival order)", orders), "bursts": SignSequences(f"bursts (same minute, same side)", bursts)}


def sequences_from_placements(p: pd.DataFrame, tidx: dict) -> SignSequences | None:
    if p.empty: return None
    p = p.sort_values(["TICKER", "TRADE_DT", "TS", "PLACEMENT_ID"], na_position="last"); by = {}
    for (tk, dte), g in p.groupby(["TICKER", "TRADE_DT"], sort=False):
        by.setdefault(tk, {})[tidx[dte]] = g["side"].to_numpy()
    return SignSequences("placements (child orders, routing order)", by)


def sequences_daily(nd: pd.DataFrame) -> tuple[SignSequences, SignSequences]:
    """Daily net sign on active days (event time in days) and signed net participation on every trading day."""
    by_sign, by_part = {}, {}
    for tk, g in nd.groupby("TICKER", sort=False):
        for t, s, a, pnet in zip(g["t"], g["sign"], g["active"], g["p_net"]):
            if a: by_sign.setdefault(tk, {})[int(t)] = np.array([s], float)
            by_part.setdefault(tk, {})[int(t)] = np.array([s * pnet], float)
    return SignSequences("daily net sign (active days)", by_sign), SignSequences("daily signed participation (all days, % ADV)", by_part)


def acf_cache(seqs: list[np.ndarray], max_lag: int) -> list[tuple]:
    """Per-sequence autocovariance sums (about the pooled mean), pair counts, sum of squares and length: the additive
    pieces of the pooled C(τ), so that a bootstrap over names is a sum over cached arrays."""
    mu = pooled_mean(seqs); out = []
    for s in seqs:
        if len(s) < 3: continue
        L = min(max_lag, len(s) - 2); ac, pairs = acf_fft(s, L, mu); out.append((ac, pairs, float(ac[0]), len(s)))
    return out


def pooled_from_cache(cache: list[tuple], idx: np.ndarray, max_lag: int) -> np.ndarray:
    num = np.zeros(max_lag + 1); den = np.zeros(max_lag + 1); var = 0.0; nvar = 0
    for i in idx:
        ac, pairs, v0, n = cache[i]; L = len(ac) - 1
        num[: L + 1] += ac; den[: L + 1] += pairs; var += v0; nvar += n
    if nvar == 0 or var <= 0: return np.full(max_lag + 1, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        C = (num / den) / (var / nvar)
    C[den < 50] = np.nan
    return C


def analyse_persistence(levels: dict, T: int, max_lag_trade: int, max_lag_day: int, boots: int, block: int, rng, max_names: int, boot_mode: str = "name", dfa_min_events: int = 5000) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """C(τ) per level, γ with a bootstrap CI (names resampled with replacement by default; --boot-mode day resamples
    the calendar in blocks, which cuts runs longer than the block and biases γ upwards), DFA H on long sequences."""
    acf_rows, gam_rows, curves = [], [], {}
    for key, S in levels.items():
        if S is None: continue
        seqs = S.sequences()
        if not seqs: continue
        daily = key.startswith("daily"); max_lag = max_lag_day if daily else max_lag_trade
        if len(seqs) > max_names:
            order = np.argsort([-len(s) for s in seqs]); seqs = [seqs[i] for i in order[:max_names]]
        cache = acf_cache(seqs, max_lag); all_idx = np.arange(len(cache))
        C = pooled_from_cache(cache, all_idx, max_lag); fit = fit_gamma(C, lag_min=1, lag_max=max_lag)
        fit2 = fit_gamma(C, lag_min=2, lag_max=max_lag); fit100 = fit_gamma(C, lag_min=1, lag_max=min(100, max_lag)); fit1000 = fit_gamma(C, lag_min=1, lag_max=min(1000, max_lag))
        Hs = [dfa_hurst(s)[0] for s in seqs if len(s) >= dfa_min_events]; Hs = [h for h in Hs if np.isfinite(h)]
        H = float(np.median(Hs)) if Hs else np.nan
        gb = []
        if boot_mode == "day":
            names = list(S.by_day.keys())
            keep = set(sorted(names, key=lambda k: -sum(len(v) for v in S.by_day[k].values()))[:max_names]); sub = SignSequences(S.name, {k: S.by_day[k] for k in keep})
            for _ in range(boots):
                sq = sub.sequences(day_block_bootstrap_calendar(T, block, rng))
                if sq: gb.append(fit_gamma(pooled_acf(sq, max_lag), lag_min=1, lag_max=max_lag)["gamma"])
        else:
            for _ in range(boots):
                gb.append(fit_gamma(pooled_from_cache(cache, rng.integers(0, len(cache), len(cache)), max_lag), lag_min=1, lag_max=max_lag)["gamma"])
        lo, hi = pct_ci(gb)
        curves[key] = C
        for tau in sorted(set([1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 500, 1000, 2000, 5000, 10000]) & set(range(1, len(C)))):
            acf_rows.append({"level": key, "lag": tau, "C": C[tau]})
        gam_rows.append({"level": key, "description": S.name, "names": len(seqs), "events": int(sum(len(s) for s in seqs)), "gamma": fit["gamma"], "gamma_lo": lo, "gamma_hi": hi, "gamma_from_lag2": fit2["gamma"], "gamma_lags_1_100": fit100["gamma"], "gamma_lags_1_1000": fit1000["gamma"],
                         "fit_lag_hi": fit["lag_hi"], "fit_points": fit["points"], "fit_r2": fit["r2"], "C1": C[1] if len(C) > 1 else np.nan, "C10": C[10] if len(C) > 10 else np.nan,
                         "C100": C[100] if len(C) > 100 else np.nan, "dfa_H_median": H, "gamma_dfa": 2 - 2 * H if np.isfinite(H) else np.nan, "names_dfa": len(Hs), "boot_reps": len(gb), "boot_mode": boot_mode})
    return pd.DataFrame(acf_rows), pd.DataFrame(gam_rows), curves


# ----------------------------------------------------------------------------------------------------------- origin proxy
def origin_proxy(d: pd.DataFrame, min_names: int = 3) -> pd.Series:
    """Account-day proxy for the origin of the flow: 'flow-like' when the account trades ≥ min_names names all on the same
    side that day (funding, withdrawal, liquidation, cash management), 'rebalance-like' when its day mixes sides, 'single'
    when it trades one or two names."""
    if d["ACCOUNT"].isna().all(): return pd.Series("unknown", index=d.index)
    g = d.assign(isbuy=(d["side"] > 0).astype(float)).groupby(["ACCOUNT", "TRADE_DT"])
    n_names = g["TICKER"].transform("nunique"); buy_share = g["isbuy"].transform("mean")
    lab = np.where(n_names < min_names, "single", np.where((buy_share == 0) | (buy_share == 1), "flow-like (same-side account day)", "rebalance-like (mixed account day)"))
    return pd.Series(lab, index=d.index)


# ----------------------------------------------------------------------------------------------------------- covariates (items 3 and 4)
def build_covariates(nd: pd.DataFrame, ep: pd.DataFrame, T: int, lookback: int) -> pd.DataFrame:
    """Per name-day: the platform's signed net flow over the preceding 1, 5, 20 and `lookback` trading days (F1, F5, F20,
    F<lookback>, % ADV, sign of the flow itself), the next 5 days (Ffwd5), and the day's position inside its episode."""
    ndx = nd.sort_values(["TICKER", "t"]).copy(); ndx["sp"] = ndx["sign"] * ndx["p_net"]
    full = pd.DataFrame({"TICKER": np.repeat(ndx["TICKER"].unique(), T), "t": np.tile(np.arange(T), ndx["TICKER"].nunique())}).merge(ndx[["TICKER", "t", "sp"]], on=["TICKER", "t"], how="left").fillna({"sp": 0.0})
    for k in sorted(set([1, 5, 20, lookback])):
        full[f"F{k}"] = full.groupby("TICKER")["sp"].transform(lambda s: s.shift(1).rolling(k, min_periods=1).sum())
    full["Ffwd5"] = full.groupby("TICKER")["sp"].transform(lambda s: s[::-1].shift(1).rolling(5, min_periods=1).sum()[::-1])
    ndx = ndx.merge(full.drop(columns="sp"), on=["TICKER", "t"], how="left")
    if len(ep):
        ndx = ndx.merge(ep[["episode", "start_t", "end_t", "n_days", "span_days", "size_adv_pct"]].rename(columns={"size_adv_pct": "ep_size"}), on="episode", how="left")
    else:
        for c in ("start_t", "end_t", "n_days", "span_days", "ep_size"): ndx[c] = np.nan
    ndx["ep_pos"] = ndx["t"] - ndx["start_t"] + 1; ndx["ep_frac"] = (ndx["ep_pos"] - 1) / (ndx["span_days"] - 1).replace(0, np.nan)
    ndx["ep_done"] = ndx.groupby("episode")["p_net"].cumsum() - ndx["p_net"]
    ndx.loc[ndx["episode"] < 0, ["ep_pos", "ep_frac", "ep_done", "ep_size", "n_days"]] = np.nan
    ndx["ep_position"] = np.where(ndx["episode"] < 0, "outside episode", np.where(ndx["n_days"] < 3, "short episode (<3 days)", np.where(ndx["ep_pos"] == 1, "first day", np.where(ndx["t"] == ndx["end_t"], "last day", "middle"))))
    return ndx


def order_covariates(d: pd.DataFrame, ndx: pd.DataFrame, kcol: str) -> pd.DataFrame:
    """Join the name-day covariates onto the orders; the flow columns are re-signed in the order's own direction, so a
    positive F means the platform had been trading the same way as this order."""
    fcols = [c for c in ndx.columns if re.fullmatch(r"F\d+|Ffwd5", c)]
    cov = d[["ORDER_ID", "TICKER", "TRADE_DT", "side", "q", "notional", "SEC_TYPE", "DAILY_VOL", "QUOTED_HS", "origin", "ARRIVAL_PRC", "PREV_CLOSE_PRC", "CLOSE_PRC"]].merge(
        ndx[["TICKER", "TRADE_DT", "t", "p_net", "p_gross", "sign", "episode", "ep_pos", "ep_frac", "ep_done", "ep_size", "ep_position"] + fcols], on=["TICKER", "TRADE_DT"], how="left")
    for c in fcols: cov[c] = cov["side"] * cov[c]
    return cov


# ----------------------------------------------------------------------------------------------------------- charts
def chart_ccdf(ep: pd.DataFrame, fit: dict, path: Path, col: str, label: str, title: str = "Episode size distribution (CCDF)") -> None:
    x = np.sort(ep[col].to_numpy(float)); x = x[x > 0]
    if len(x) < 20: return
    ccdf = 1 - np.arange(len(x)) / len(x)
    fig, ax = plt.subplots(figsize=(6.5, 4.6)); ax.loglog(x, ccdf, ".", ms=3, alpha=.5, label="episodes")
    if np.isfinite(fit.get("alpha", np.nan)):
        xx = np.logspace(np.log10(fit["xmin"]), np.log10(x.max()), 50); c = ccdf[np.searchsorted(x, fit["xmin"])]
        ax.loglog(xx, c * (xx / fit["xmin"]) ** (-fit["alpha"]), "r-", lw=1.5, label=f"power law α={fit['alpha']:.2f} (x_min={fit['xmin']:.2g})")
    ax.set_xlabel(label); ax.set_ylabel("P(X > x)"); ax.legend(fontsize=8); ax.set_title(title, fontsize=10); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def chart_acf(curves: dict, path: Path) -> None:
    """Pooled C(τ) per level, averaged in geometric lag bins for display (every lag is in tables/acf.csv)."""
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for key, C in curves.items():
        edges = np.unique(np.round(np.logspace(0, np.log10(len(C) - 1), 60)).astype(int)); xs, ys = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            seg = C[a:max(b, a + 1)]; seg = seg[np.isfinite(seg)]
            if len(seg) and seg.mean() > 0: xs.append(np.sqrt(a * max(b - 1, a))); ys.append(seg.mean())
        if xs: ax.loglog(xs, ys, "o-", ms=3, lw=.8, alpha=.8, label=key)
    for g_, ls in ((0.5, ":"), (1.0, "--")):
        xx = np.logspace(0, 3, 20); ax.loglog(xx, 0.5 * xx ** (-g_), color="grey", ls=ls, lw=.8, label=f"τ^-{g_:g} (reference)")
    ax.set_xlabel("lag τ (events or days)"); ax.set_ylabel("C(τ)"); ax.legend(fontsize=7); ax.set_title("Sign autocorrelation of the platform's own flow", fontsize=10); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def chart_bins(tbl: pd.DataFrame, path: Path, x: str, y: str, hue: str, title: str, ylabel: str, err: str | None = None) -> None:
    if tbl.empty: return
    fig, ax = plt.subplots(figsize=(7.5, 4.6)); hues = list(tbl[hue].unique())
    for i, h in enumerate(hues):
        g = tbl[tbl[hue] == h]; xs = np.arange(len(g)) + (i - (len(hues) - 1) / 2) * 0.15
        ax.errorbar(xs, g[y], yerr=g[err] if err and err in g else None, fmt="o-", ms=4, lw=1, capsize=2, label=str(h))
        ax.set_xticks(np.arange(len(g))); ax.set_xticklabels(g[x].astype(str), rotation=20, fontsize=8)
    ax.axhline(0, color="grey", lw=.6, ls=":"); ax.set_ylabel(ylabel); ax.legend(fontsize=8, title=hue); ax.set_title(title, fontsize=9); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ----------------------------------------------------------------------------------------------------------- demo store (self-test)
def make_demo_store(seed: int = 7, n_names: int = 30, n_days: int = 400, alpha: float = 1.5, lean: float = 0.6, leak: float = 12.0):
    """A synthetic BTCA-like store with a known structure: metaorders whose duration in days is Pareto with CCDF exponent
    `alpha` from one day (so their sizes have the same tail), executed at a daily rate drawn in 0.3–2 % ADV and split across accounts, a two-sided background
    of tiny cash-flow orders, 3 brokers per name with a 0.7 probability of keeping the previous broker, and fills whose
    reversion after the fill grows with the platform's same-direction flow over the preceding 5 days (`lean` bp per % ADV)
    and with the order's own size. Prices: a common market return per day, the previous close, the arrival price and the
    close per order; when the previous day's flow in a name was executed mainly through BRK_C, the next day's price moves in
    the platform's direction before its orders arrive by `leak` bp per % ADV of that flow (the information-leakage pattern
    item 4 is built to detect); the other brokers leak nothing. Returns a DuckDB connection with tables tca, placements, fills."""
    import duckdb
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2025-01-02", periods=n_days).date; mkt = rng.normal(0, 60, n_days)           # market return per day, bp
    orders, placements, fills = [], [], []
    oid = 0; pid = 0
    for i in range(n_names):
        tk = f"SYN{i:03d}"; adv = float(rng.choice([2e5, 1e6, 5e6, 2e7])); sec = "ADR" if i % 4 == 0 else "Common Stock"
        vol_bps = float(rng.uniform(120, 300)); hs_bp = float(rng.uniform(1.5, 6)); brokers = ["BRK_A", "BRK_B", "BRK_C"]; pref = rng.dirichlet([3, 3, 3])
        close_prev = float(rng.uniform(10, 200)); dom_prev = None; flow_prev = 0.0
        sched = np.zeros(n_days)
        t = int(rng.integers(0, 5))
        while t < n_days:
            nd_ = int(min(np.floor((1 - rng.random()) ** (-1 / alpha)), 120)); rate = float(rng.uniform(0.3, 2.0)); sign = int(rng.choice([-1, 1]))
            per_day = rate                                                                    # duration in days is Pareto from 1 day; size = duration × daily rate
            for k in range(nd_):
                if t + k < n_days: sched[t + k] += sign * per_day
            t += nd_ + int(rng.geometric(0.35))
        flow = np.zeros(n_days); broker_prev = None
        for t in range(n_days):
            day = days[t]; base = dt.datetime.combine(day, dt.time(9, 35)); day_brk = {}
            anticip = (leak * abs(flow_prev) * np.sign(flow_prev)) if dom_prev == "BRK_C" else 0.0          # the leak: yesterday's broker C → today's pre-arrival drift along the flow
            mid0 = close_prev * (1 + (mkt[t] + anticip + rng.normal(0, 15)) / 1e4)                            # pre-trade mid of the day
            close_t = mid0 * (1 + (rng.normal(0, 40) + 0.3 * np.sign(sched[t]) * np.sqrt(abs(sched[t]))) / 1e4)   # close: noise plus a small permanent impact
            n_bg = rng.poisson(2); lots = [(int(s_), float(qq), "CASH") for s_, qq in zip(rng.choice([-1, 1], n_bg), rng.uniform(3e-6, 4e-5, n_bg) * adv)]
            if sched[t] != 0:
                n_acc = 1 + rng.poisson(8); qs = rng.dirichlet(np.ones(n_acc)) * abs(sched[t]) / 100 * adv
                lots += [(int(np.sign(sched[t])), float(qq), "MODEL") for qq in qs]
            F5 = float(flow[max(0, t - 5):t].sum())
            for s_, qq, origin in lots:
                oid += 1; arr = base + dt.timedelta(minutes=int(rng.integers(0, 300))); pct = qq / adv * 100
                orders.append({"ORDER_ID": oid, "TICKER": tk, "BUY_SELL_CD": "B" if s_ > 0 else "S", "TRADE_DT": day, "ORDER_QTY": round(qq), "SHARES_FILLED": round(qq), "PCT_ADV": pct,
                               "ACCOUNT_ID": f"ACC{int(rng.integers(0, 4000)):05d}", "ORDER_ARRIVAL_TMSTMP": arr, "SECURITY_TYPE_CD": sec, "DAILY_VOL_BPS": vol_bps,
                               "FILLED_NOTIONAL": round(qq) * mid0, "QUOTED_HALF_SPREAD_BPS": hs_bp, "ORDER_REASON_CD": origin, "ORDER_EXEC_DURATION_TS": "01:30:00",
                               "ARRIVAL_PRC": mid0 * (1 + rng.normal(0, 3) / 1e4), "PREV_CLOSE_PRC": close_prev, "CLOSE_PRC": close_t})
                temp_bp = 1.0 + 0.3 * np.sqrt(pct) + lean * max(s_ * F5, 0.0) + rng.normal(0, 1.0)     # temporary impact: size and the lean against persistent flow
                n_pl = 1 + int(rng.integers(0, 2))
                for j in range(n_pl):
                    pid += 1
                    brk = broker_prev if (broker_prev is not None and rng.random() < 0.7) else str(rng.choice(brokers, p=pref)); broker_prev = brk; day_brk[brk] = day_brk.get(brk, 0.0) + qq / n_pl
                    pts = arr + dt.timedelta(minutes=5 * j)
                    placements.append({"ORDER_ID": oid, "PLACEMENT_ID": f"P{pid}", "BROKER_CD": brk, "ALGO_NM": str(rng.choice(["VWAP", "POV", "IS"])), "PLACEMENT_TMSTMP": pts, "PLACEMENT_QTY": round(qq / n_pl)})
                    pre = mid0 * (1 + rng.normal(0, 2) / 1e4)                                   # pre-trade mid
                    mid_fill = pre * (1 + s_ * temp_bp / 1e4)                                     # the quotes at the fill already carry the impact
                    fts = pts + dt.timedelta(minutes=int(rng.integers(1, 4))); fpx = mid_fill * (1 + s_ * hs_bp / 1e4)
                    fills.append({"ORDER_ID": oid, "TRADE_DT": day, "FILL_TMSTMP": fts, "FILL_PRC": fpx, "FILL_VOLUME_QTY": round(qq / n_pl), "ORDER_BID_PRC": mid_fill * (1 - hs_bp / 1e4), "ORDER_ASK_PRC": mid_fill * (1 + hs_bp / 1e4)})
                    for h in (5, 30, 60):                                                          # later quote snapshots: 90 % of the temporary impact has reverted
                        mid_h = pre * (1 + s_ * 0.1 * temp_bp / 1e4 + rng.normal(0, 1.0) / 1e4)
                        fills.append({"ORDER_ID": oid, "TRADE_DT": day, "FILL_TMSTMP": fts + dt.timedelta(minutes=h - int(rng.integers(0, 2))), "FILL_PRC": -1.0, "FILL_VOLUME_QTY": 0.0,
                                      "ORDER_BID_PRC": mid_h * (1 - hs_bp / 1e4), "ORDER_ASK_PRC": mid_h * (1 + hs_bp / 1e4)})
            flow[t] = sum(s_ * qq for s_, qq, _ in lots) / adv * 100
            dom_prev = max(day_brk, key=day_brk.get) if day_brk else None; flow_prev = flow[t]; close_prev = close_t
    con = duckdb.connect()
    con.register("tca_df", pd.DataFrame(orders)); con.register("placements_df", pd.DataFrame(placements)); con.register("fills_df", pd.DataFrame(fills))
    con.execute("CREATE TABLE tca AS SELECT * FROM tca_df"); con.execute("CREATE TABLE placements AS SELECT * FROM placements_df"); con.execute("CREATE TABLE fills AS SELECT * FROM fills_df")
    return con


# ----------------------------------------------------------------------------------------------------------- main analysis
def build(args) -> Path:
    global QUERY, VIEWS
    rng = np.random.default_rng(args.seed)
    if args.demo:
        con = make_demo_store(seed=args.seed, n_names=args.demo_names, n_days=args.demo_days, alpha=args.demo_alpha, lean=args.demo_lean, leak=args.demo_leak)
        QUERY = lambda sql, params=None: con.execute(sql, params or []).df(); VIEWS = ["tca", "placements", "fills"]   # noqa: E731
        LOG(f"demo store: {q('SELECT COUNT(*) AS n FROM tca')['n'].iloc[0]:,} orders, {q('SELECT COUNT(*) AS n FROM placements')['n'].iloc[0]:,} placements, {q('SELECT COUNT(*) AS n FROM fills')['n'].iloc[0]:,} fill rows")
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import data_access
        QUERY = data_access.query; VIEWS = list(data_access.available_views())
    orders_view = args.orders_view if args.orders_view in VIEWS or args.demo else ("tca" if "tca" in VIEWS else VIEWS[0])
    placements_view = next((v for v in [args.placements_view, "placements", "placement"] if v and v in VIEWS), None)
    fills_view = next((v for v in [args.fills_view, "fills", "fill"] if v and v in VIEWS), None)
    overrides = dict(kv.split("=", 1) for kv in (args.col or []))
    S = resolve_schema({k.upper(): v for k, v in overrides.items()}, orders_view, placements_view, fills_view)
    if args.print_schema: print_schema(S); sys.exit(0)
    for role in ("o_ORDER_ID", "o_TICKER", "o_SIDE", "o_TRADE_DT", "o_ORDER_QTY"):
        if not S[role]: print_schema(S); sys.exit(f"[flow] required column for {role} not found; pass --col {role[2:]}=<column>")
    if not (S["o_ADV"] or S["o_PCT_ADV"]): print_schema(S); sys.exit("[flow] neither an ADV column nor PCT_ADV found; pass --col ADV=<column> or --col PCT_ADV=<column>")
    # window
    if args.end is None or args.start is None:
        r = q(f"SELECT MIN({S['o_TRADE_DT']}) AS a, MAX({S['o_TRADE_DT']}) AS b FROM {orders_view}")
        args.start = args.start or pd.Timestamp(r["a"].iloc[0]).date(); args.end = args.end or pd.Timestamp(r["b"].iloc[0]).date()
    # security types: tca_report's aliases when available, raw codes otherwise
    where_extra, params_extra, sec_label = "", [], "all types"
    if S["o_SEC_TYPE"] and args.sec_type:
        try:
            from tca_report import resolve_sec_types, sec_type_clause
            codes, other = resolve_sec_types(args.sec_type); where_extra, params_extra = sec_type_clause(codes, other); sec_label = ", ".join(codes)
        except Exception:
            where_extra = f"{S['o_SEC_TYPE']} IN ({', '.join('?' for _ in args.sec_type)})"; params_extra = list(args.sec_type); sec_label = ", ".join(args.sec_type)
    out = Path(args.out) if args.out else Path("reports") / f"flow_persistence_{args.start}_{args.end}"
    (out / "tables").mkdir(parents=True, exist_ok=True); (out / "charts").mkdir(parents=True, exist_ok=True)
    t0 = dt.datetime.now(); LOG(f"window {args.start} → {args.end}, {sec_label}; stores {VIEWS}; out {out}")
    print_schema(S)

    # ---- orders, calendar, name-days, episodes
    d = load_orders(S, args, where_extra, params_extra)
    if args.max_names and d["TICKER"].nunique() > args.max_names:
        keep = d.groupby("TICKER")["q"].size().sort_values(ascending=False).index[: args.max_names]; d = d[d["TICKER"].isin(keep)].copy()
    calendar = sorted(d["TRADE_DT"].unique()); T = len(calendar); tidx = {dte: i for i, dte in enumerate(calendar)}
    LOG(f"{len(d):,} orders, {d['TICKER'].nunique():,} names, {T} trading days, account column {'present' if d['ACCOUNT'].notna().any() else 'absent'}, arrival timestamps {'present' if d['ARRIVAL'].notna().mean() > 0.5 else 'absent'}, origin column {'present' if d['ORIGIN'].notna().any() else 'absent'} in {(dt.datetime.now() - t0).seconds}s")
    nd = build_name_days(d, calendar)
    ep, nd = build_episodes(nd, args.quiet_pct, args.gap_days)
    sec_of = d.groupby("TICKER")["SEC_TYPE"].agg(lambda s: s.mode().iloc[0] if s.notna().any() else "all"); ep["sec_type"] = ep["TICKER"].map(sec_of)
    nd["sec_type"] = nd["TICKER"].map(sec_of)
    LOG(f"{len(nd):,} name-days ({int(nd['active'].sum()):,} active at ≥ {args.quiet_pct:g}% ADV), {len(ep):,} episodes")
    nd.to_csv(out / "tables" / "name_days.csv", index=False); ep.to_csv(out / "tables" / "episodes.csv", index=False)

    # ---- item 1a: size and duration distributions, power-law tails
    tail_rows = []
    for col, label, disc in (("size_adv_pct", "episode size, % ADV (sum of daily net participation)", False), ("n_days", "episode duration, active days (discrete MLE; rough when x_min < 6)", True), ("n_orders", "episode size, account orders (discrete MLE)", True)):
        for grp, g in [("all", ep)] + [(f"type={s}", ep[ep["sec_type"] == s]) for s in ep["sec_type"].dropna().unique()]:
            x = g[col].to_numpy(float); fit = fit_powerlaw(x, discrete=disc, min_tail=100 if disc else 50)
            if not np.isfinite(fit["alpha"]): continue
            bs = [fit_powerlaw(rng.choice(x, len(x), replace=True), discrete=disc, min_tail=100 if disc else 50)["alpha"] for _ in range(args.tail_boot)]; lo, hi = pct_ci(bs)
            pgof = powerlaw_gof(x, fit, args.gof_reps, rng) if grp == "all" else np.nan
            R, pR = lognormal_lr_test(x, fit) if grp == "all" else (np.nan, np.nan)
            tail_rows.append({"variable": col, "label": label, "group": grp, "n": len(x), "median": float(np.median(x)), "p90": float(np.percentile(x, 90)), "max": float(x.max()),
                              "alpha_ccdf": fit["alpha"], "alpha_lo": lo, "alpha_hi": hi, "a_density": fit["a"], "se_a": fit["se"], "xmin": fit["xmin"], "n_tail": fit["n_tail"], "tail_share": fit["n_tail"] / len(x), "ks": fit["ks"],
                              "gof_p": pgof, "LR_vs_lognormal": R, "LR_p": pR, "gamma_LMF_pred": fit["alpha"] - 1})
    tails = pd.DataFrame(tail_rows); tails.to_csv(out / "tables" / "tail_fit.csv", index=False)
    main_fit = next((r for r in tail_rows if r["variable"] == "size_adv_pct" and r["group"] == "all"), None)
    if main_fit: chart_ccdf(ep, {"alpha": main_fit["alpha_ccdf"], "xmin": main_fit["xmin"]}, out / "charts" / "ccdf_episode_size.png", "size_adv_pct", "episode size, % ADV")
    dur_fit = next((r for r in tail_rows if r["variable"] == "n_days" and r["group"] == "all"), None)
    if dur_fit: chart_ccdf(ep, {"alpha": dur_fit["alpha_ccdf"], "xmin": dur_fit["xmin"]}, out / "charts" / "ccdf_episode_days.png", "n_days", "episode duration, active days", "Episode duration distribution (CCDF)")

    # ---- item 1b: sign autocorrelation at several resolutions, γ, DFA
    p = load_placements(S, d, args) if not args.no_placements else pd.DataFrame()
    levels = sequences_from_orders(d, tidx, args.burst_minutes)
    levels["placements"] = sequences_from_placements(p, tidx)
    levels["daily_sign"], levels["daily_part"] = sequences_daily(nd)
    acf, gam, curves = analyse_persistence(levels, T, args.max_lag, args.max_lag_days, args.bootstrap, args.block_days, rng, args.boot_names, args.boot_mode, args.dfa_min_events)
    types_present = [t_ for t_ in sec_of.unique() if (sec_of == t_).sum() >= 10]
    if len(types_present) > 1:                                                    # γ by security type for the burst and daily-sign levels
        for t_ in types_present:
            names_t = set(sec_of.index[sec_of == t_])
            sub = {f"{k}|type={t_}": SignSequences(f"{levels[k].name} — {t_}", {n: v for n, v in levels[k].by_day.items() if n in names_t}) for k in ("bursts", "daily_sign") if levels.get(k) is not None}
            acf_t, gam_t, _ = analyse_persistence(sub, T, args.max_lag, args.max_lag_days, max(args.bootstrap // 2, 20), args.block_days, rng, args.boot_names, args.boot_mode, args.dfa_min_events)
            acf = pd.concat([acf, acf_t], ignore_index=True); gam = pd.concat([gam, gam_t], ignore_index=True)
    acf.to_csv(out / "tables" / "acf.csv", index=False); gam.to_csv(out / "tables" / "gamma.csv", index=False); chart_acf(curves, out / "charts" / "acf_loglog.png")
    LOG("γ by level: " + "; ".join(f"{r['level']} {r['gamma']:.2f} [{r['gamma_lo']:.2f}, {r['gamma_hi']:.2f}] (DFA {r['gamma_dfa']:.2f})" for _, r in gam.iterrows()))

    # ---- item 1c: the LMF test γ = α − 1, episode-sign dependence, sensitivity to the quiet threshold
    lmf_rows = []
    if main_fit is not None and len(gam):
        for _, r in gam.iterrows():
            lmf_rows.append({"level": r["level"], "gamma": r["gamma"], "gamma_lo": r["gamma_lo"], "gamma_hi": r["gamma_hi"], "gamma_lags_1_100": r["gamma_lags_1_100"], "gamma_lags_1_1000": r["gamma_lags_1_1000"], "alpha_size": main_fit["alpha_ccdf"], "alpha_lo": main_fit["alpha_lo"], "alpha_hi": main_fit["alpha_hi"],
                             "alpha_days": dur_fit["alpha_ccdf"] if dur_fit else np.nan, "gamma_pred_from_size": main_fit["alpha_ccdf"] - 1, "gamma_pred_from_days": (dur_fit["alpha_ccdf"] - 1) if dur_fit else np.nan,
                             "gap_size": r["gamma"] - (main_fit["alpha_ccdf"] - 1), "gap_days": r["gamma"] - (dur_fit["alpha_ccdf"] - 1) if dur_fit else np.nan})
    lmf = pd.DataFrame(lmf_rows); lmf.to_csv(out / "tables" / "lmf_test.csv", index=False)
    dep_rows = []
    if len(ep):
        e2 = ep.sort_values(["TICKER", "start_t"]); nxt = e2.groupby("TICKER")[["sign", "start_t"]].shift(-1)
        same = (e2["sign"] == nxt["sign"]); gap = nxt["start_t"] - e2["end_t"] - 1; ok = nxt["sign"].notna() & (gap > args.gap_days)   # within the tolerance two same-sign runs are one episode by construction
        g1 = args.gap_days + 1
        for lbl, m in ((f"all (gap > {args.gap_days} days)", ok), (f"gap {g1}–5 days", ok & (gap <= 5)), ("gap 6–20 days", ok & (gap >= 6) & (gap <= 20)), ("gap > 20 days", ok & (gap > 20))):
            n = int(m.sum()); ps = float(same[m].mean()) if n else np.nan
            dep_rows.append({"subset": lbl, "pairs": n, "P(next episode same sign)": ps, "se": np.sqrt(ps * (1 - ps) / n) if n else np.nan, "lag1_sign_corr": 2 * ps - 1 if n else np.nan})
    dep = pd.DataFrame(dep_rows); dep.to_csv(out / "tables" / "episode_sign_dependence.csv", index=False)
    sens_rows = []
    for thr in sorted(set([args.quiet_pct] + [float(x) for x in args.quiet_sensitivity.split(",") if x])):
        e_, nd_ = build_episodes(nd.drop(columns=["episode", "active"], errors="ignore"), thr, args.gap_days)
        f_ = fit_powerlaw(e_["size_adv_pct"].to_numpy(float)) if len(e_) else {"alpha": np.nan, "xmin": np.nan}
        ds, _ = sequences_daily(nd_); C_ = pooled_acf(ds.sequences(), args.max_lag_days); g_ = fit_gamma(C_, 1, args.max_lag_days)
        sens_rows.append({"quiet_pct": thr, "episodes": len(e_), "active_name_days": int(nd_["active"].sum()), "median_size_pct": float(e_["size_adv_pct"].median()) if len(e_) else np.nan, "alpha_size": f_["alpha"], "gamma_daily_sign": g_["gamma"]})
    sens = pd.DataFrame(sens_rows); sens.to_csv(out / "tables" / "quiet_threshold_sensitivity.csv", index=False)

    # ---- item 1d: persistence by origin (code or proxy)
    d["origin"] = d["ORIGIN"].astype(str).where(d["ORIGIN"].notna(), None)
    origin_kind = "code"
    if d["origin"].isna().all():
        d["origin"] = origin_proxy(d); origin_kind = "proxy"
    org_rows = []
    for org, g in d.groupby("origin"):
        if len(g) < 500: continue
        nd_o = build_name_days(g, calendar); e_o, nd_o = build_episodes(nd_o, args.quiet_pct, args.gap_days)
        f_o = fit_powerlaw(e_o["size_adv_pct"].to_numpy(float)) if len(e_o) else {"alpha": np.nan}
        ds_o, _ = sequences_daily(nd_o); C_o = pooled_acf(ds_o.sequences(), args.max_lag_days); g_o = fit_gamma(C_o, 1, args.max_lag_days)
        org_rows.append({"origin": org, "kind": origin_kind, "orders": len(g), "share_of_shares": float(g["q"].sum() / d["q"].sum()), "episodes": len(e_o), "median_episode_days": float(e_o["n_days"].median()) if len(e_o) else np.nan,
                         "p90_episode_days": float(e_o["n_days"].quantile(.9)) if len(e_o) else np.nan, "alpha_size": f_o["alpha"], "gamma_daily_sign": g_o["gamma"], "C1_daily": C_o[1] if len(C_o) > 1 else np.nan, "C5_daily": C_o[5] if len(C_o) > 5 else np.nan})
    org = pd.DataFrame(org_rows); org.to_csv(out / "tables" / "origin_persistence.csv", index=False)

    # ---- item 2: brokerage distortion from placements
    brk_summary = {}
    if not p.empty:
        p = p.merge(nd[["TICKER", "TRADE_DT", "episode"]], on=["TICKER", "TRADE_DT"], how="left").sort_values(["TICKER", "TS", "PLACEMENT_ID"], na_position="last")
        p["w"] = p["PQTY"].where(p["PQTY"] > 0, p["q"])
        # per episode: brokers, concentration, consecutive-same-broker share
        rows = []
        for e_id, g in p[p["episode"] >= 0].groupby("episode"):
            sh = g.groupby("BROKER")["w"].sum(); sh = sh / sh.sum(); b = g["BROKER"].to_numpy()
            rows.append({"episode": e_id, "n_placements": len(g), "n_brokers": int(sh.size), "hhi": float((sh ** 2).sum()), "top_broker_share": float(sh.max()),
                         "p_same_consecutive": float((b[1:] == b[:-1]).mean()) if len(b) > 1 else np.nan, "p_same_random_pair": float((sh ** 2).sum())})
        ep_b = pd.DataFrame(rows).merge(ep[["episode", "n_days", "size_adv_pct", "sec_type"]], on="episode", how="left"); ep_b.to_csv(out / "tables" / "broker_by_episode.csv", index=False)
        # P_same(τ) along the platform's placement sequence per name (the member-code view of a single investor)
        max_tau = 50; num = np.zeros(max_tau + 1); den = np.zeros(max_tau + 1)
        for tk, g in p.groupby("TICKER"):
            b = pd.factorize(g["BROKER"])[0]
            for tau in range(1, min(max_tau, len(b) - 1) + 1):
                num[tau] += (b[tau:] == b[:-tau]).sum(); den[tau] += len(b) - tau
        psame = pd.DataFrame({"lag": np.arange(1, max_tau + 1), "P_same_broker": num[1:] / np.where(den[1:] > 0, den[1:], np.nan), "pairs": den[1:]}); psame.to_csv(out / "tables" / "broker_psame_by_lag.csv", index=False)
        # the share of the platform's own autocorrelation that a member-code decomposition would call herding
        Cpl = curves.get("placements"); herd_share = np.nan
        if Cpl is not None and len(Cpl) > max_tau:
            c = Cpl[1:max_tau + 1]; ps_ = psame["P_same_broker"].to_numpy(); ok = np.isfinite(c) & np.isfinite(ps_) & (c > 0)
            herd_share = float(np.sum(c[ok] * (1 - ps_[ok])) / np.sum(c[ok])) if ok.any() else np.nan
        # broker shares by name and overall, and by algo
        share_name = p.groupby(["TICKER", "BROKER"])["w"].sum().reset_index(); share_name["share"] = share_name["w"] / share_name.groupby("TICKER")["w"].transform("sum")
        share_name.to_csv(out / "tables" / "broker_share_by_name.csv", index=False)
        overall = p.groupby("BROKER")["w"].sum().sort_values(ascending=False); overall = (overall / overall.sum()).reset_index().rename(columns={"w": "share"}); overall.to_csv(out / "tables" / "broker_share.csv", index=False)
        if p["ALGO"].notna().any():
            by_algo = p.groupby(["ALGO", "BROKER"])["w"].sum().reset_index(); by_algo["share"] = by_algo["w"] / by_algo.groupby("ALGO")["w"].transform("sum"); by_algo.to_csv(out / "tables" / "broker_share_by_algo.csv", index=False)
        nb_day = p.groupby(["TICKER", "TRADE_DT"])["BROKER"].nunique()
        brk_summary = {"placements": int(len(p)), "brokers": int(p["BROKER"].nunique()), "brokers_per_episode_median": float(ep_b["n_brokers"].median()) if len(ep_b) else np.nan,
                       "brokers_per_episode_p90": float(ep_b["n_brokers"].quantile(.9)) if len(ep_b) else np.nan, "p_same_consecutive_wmean": w_avg(ep_b["p_same_consecutive"], ep_b["n_placements"]) if len(ep_b) else np.nan,
                       "p_same_random_pair_wmean": w_avg(ep_b["p_same_random_pair"], ep_b["n_placements"]) if len(ep_b) else np.nan, "p_same_lag1": float(psame["P_same_broker"].iloc[0]), "p_same_lag10": float(psame["P_same_broker"].iloc[9]) if len(psame) > 9 else np.nan,
                       "p_same_lag50": float(psame["P_same_broker"].iloc[-1]), "brokers_per_name_day_mean": float(nb_day.mean()), "top_broker_share_overall": float(overall["share"].iloc[0]), "herding_share_of_own_acf_50_lags": herd_share}
        fig, ax = plt.subplots(figsize=(6.5, 4.2)); ax.plot(psame["lag"], psame["P_same_broker"], "o-", ms=3); ax.axhline(float((overall["share"] ** 2).sum()), color="grey", ls=":", label="Σ share² (random pair, overall)")
        ax.set_xlabel("lag along the platform's placement sequence"); ax.set_ylabel("P(same broker)"); ax.set_title("P_same(τ): the member-code view of one investor's flow", fontsize=10); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out / "charts" / "broker_psame.png", dpi=130); plt.close(fig)
        LOG("brokers: " + ", ".join(f"{k} {v:.3g}" if isinstance(v, float) else f"{k} {v}" for k, v in brk_summary.items()))
    else:
        LOG("placements not available (no view or no broker column): item 2 skipped" + ("; the orders view has a broker column, used for shares only" if d["BROKER_O"].notna().any() else ""))
        if d["BROKER_O"].notna().any():
            overall = d.groupby("BROKER_O")["q"].sum().sort_values(ascending=False); (overall / overall.sum()).reset_index().rename(columns={"q": "share"}).to_csv(out / "tables" / "broker_share.csv", index=False)

    # ---- covariates per name-day and per order (items 3 and 4)
    ndx = build_covariates(nd, ep, T, args.lookback_days); kcol = f"F{args.lookback_days}"
    cov = order_covariates(d, ndx, kcol)
    name_part = ndx.groupby("TICKER")["p_gross"].mean(); hi_share = name_part >= name_part.median()
    cov["platform_share"] = np.where(cov["TICKER"].map(hi_share).fillna(False), "high-share names", "low-share names")
    edges = [-np.inf, 0, 0.1, 0.5, 2, 5, np.inf]; labels = ["≤ 0 (opposite)", "0–0.1", "0.1–0.5", "0.5–2", "2–5", "> 5"]
    cov["F_bin"] = pd.cut(cov[kcol], bins=edges, labels=labels, right=True); cov["F1_bin"] = pd.cut(cov["F1"], bins=edges, labels=labels, right=True)
    if not p.empty:                                                                  # broker attribution: the order's main broker, the name-day's main broker, yesterday's
        ob = p.groupby(["ORDER_ID", "BROKER"])["w"].sum().reset_index().sort_values("w").drop_duplicates("ORDER_ID", keep="last")[["ORDER_ID", "BROKER"]].rename(columns={"BROKER": "broker"})
        nb = p.groupby(["TICKER", "TRADE_DT", "BROKER"])["w"].sum().reset_index().sort_values("w").drop_duplicates(["TICKER", "TRADE_DT"], keep="last")[["TICKER", "TRADE_DT", "BROKER"]].rename(columns={"BROKER": "dom_broker"})
        nb = nb.merge(nd[["TICKER", "TRADE_DT", "t"]], on=["TICKER", "TRADE_DT"], how="left"); nb["t"] = nb["t"] + 1
        cov = cov.merge(ob, on="ORDER_ID", how="left").merge(nb.rename(columns={"dom_broker": "broker_prev"})[["TICKER", "t", "broker_prev"]], on=["TICKER", "t"], how="left")
        cov["broker_prev"] = cov["broker_prev"].fillna("none (no own flow the day before)")
    else:
        cov["broker"] = np.nan; cov["broker_prev"] = np.nan

    def wtab(df, by, val, wcol):
        """Dollar-weighted mean of `val` per group with a day-cluster bootstrap standard error (200 draws over trading days)."""
        rows = []
        for keys, g in df.groupby(by, observed=True):
            keys = keys if isinstance(keys, tuple) else (keys,)
            ok_ = g[val].notna() & (g[wcol] > 0); gb = []
            if ok_.any():
                gd = g[ok_].assign(wx=g.loc[ok_, val] * g.loc[ok_, wcol]).groupby("TRADE_DT")[["wx", wcol]].sum()
                if len(gd) >= 5:
                    idx = rng.integers(0, len(gd), (200, len(gd))); sx = gd["wx"].to_numpy()[idx].sum(axis=1); sw = gd[wcol].to_numpy()[idx].sum(axis=1); gb = list(sx / np.where(sw > 0, sw, np.nan))
            row = dict(zip(by, keys), orders=int(ok_.sum()), usd=float(g.loc[ok_, wcol].sum()), mean=w_avg(g[val], g[wcol]), se=float(np.nanstd(gb, ddof=1)) if len(gb) > 5 else np.nan,
                       participation_pct=w_avg(g["p_gross"], g[wcol]), flow_pct=w_avg(g[kcol], g[wcol]))
            if "realized_bps" in g: row.update(realized_bps=w_avg(g["realized_bps"], g[wcol]), eff_bps=w_avg(g["eff_bps"], g[wcol]), quoted_hs_bps=w_avg(g["QUOTED_HS"], g[wcol]))
            rows.append(row)
        return pd.DataFrame(rows)

    def regress(g, y, w, extra_terms=(), flow_col=kcol, flow_label=f"prev {args.lookback_days} d", reduced=False):
        """WLS of y on the platform's own flow (same-direction and opposite-direction parts), forward own flow, size, position, spread, σ, type; name-day clusters.
        reduced=True drops the episode-position terms and the forward flow, which are collinear with the flow on continuation days (used for the per-broker tests)."""
        terms = [(f"same-direction flow, {flow_label} (% ADV, ≥ 0 part)", g[flow_col].clip(lower=0, upper=20)), (f"opposite-direction flow, {flow_label} (% ADV, magnitude)", (-g[flow_col]).clip(lower=0, upper=20))]
        terms += list(extra_terms)
        if not reduced:
            terms += [("forward own flow, next 5 d (% ADV)", g["Ffwd5"].clip(lower=-20, upper=20)), ("fraction of episode elapsed", g["ep_frac"].fillna(0.5)), ("log(1 + % ADV already executed in episode)", np.log1p(g["ep_done"].fillna(0)))]
        terms += [("log(1 + day participation %)", np.log1p(g["p_gross"].fillna(0))), ("quoted ½-spread (bp)", g["QUOTED_HS"].fillna(g["QUOTED_HS"].median())), ("daily σ (%)", g["DAILY_VOL"].fillna(g["DAILY_VOL"].median()) / 100),
                  ("ADR", g["SEC_TYPE"].astype(str).str.upper().str.contains("ADR").astype(float))]
        if "eff_bps" in g: terms.insert(2, ("effective ½-spread (bp)", g["eff_bps"]))
        X = np.column_stack([np.ones(len(g))] + [np.asarray(v, float) for _, v in terms]); names = ["const"] + [n for n, _ in terms]
        b, se, t_, n, G = ols_cluster(g[y], X, g[w], (g["TICKER"].astype(str) + "|" + g["TRADE_DT"].astype(str)))
        return pd.DataFrame({"term": names, "coef": b, "se": se, "t": t_, "n": n, "clusters": G})

    # ---- item 3: mark-outs against the platform's own persistent flow
    horizons = [int(h) for h in args.markout_horizons.split(",")]
    mo_where, mo_params = where_extra, list(params_extra)
    if args.markout_names and d["TICKER"].nunique() > args.markout_names:
        top = d.groupby("TICKER")["notional"].sum().sort_values(ascending=False).index[: args.markout_names].tolist()
        mo_where = (mo_where + " AND " if mo_where else "") + f"{S['o_TICKER']} IN ({', '.join('?' for _ in top)})"; mo_params += top
        LOG(f"mark-outs restricted to the {len(top)} largest names by notional (--markout-names)")
    mo = load_markouts(S, args, horizons, mo_where, mo_params) if not args.no_markouts else pd.DataFrame()
    mk_summary = {}
    if not mo.empty:
        mo = mo.merge(cov.drop(columns=["q", "notional"]), on="ORDER_ID", how="inner"); mo["reversion_bps"] = mo["realized_bps"] - mo["eff_bps"]
        mo = mo[mo["covered_dol"] > 0].copy(); mo.to_csv(out / "tables" / "markouts_orders.csv", index=False)
        bins = wtab(mo, ["H", "F_bin"], "reversion_bps", "covered_dol").rename(columns={"mean": "reversion_bps", "se": "reversion_se"}); bins.to_csv(out / "tables" / "markouts_by_flow_bin.csv", index=False)
        chart_bins(bins.assign(H=bins["H"].map(lambda h: f"+{h} min")), out / "charts" / "markouts_vs_flow.png", "F_bin", "reversion_bps", "H", f"Reversion after the platform's fills by its own same-direction flow over the preceding {args.lookback_days} days", "realized − effective ½-spread, bp", err="reversion_se")
        pos = wtab(mo[mo["ep_position"].isin(["first day", "middle", "last day"])], ["H", "ep_position"], "reversion_bps", "covered_dol").rename(columns={"mean": "reversion_bps", "se": "reversion_se"}); pos.to_csv(out / "tables" / "markouts_by_episode_position.csv", index=False)
        mo["part_tercile"] = mo.groupby("H")["p_gross"].transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=["low participation", "mid", "high participation"]))
        pos2 = wtab(mo[mo["ep_position"].isin(["first day", "last day"])], ["H", "part_tercile", "ep_position"], "reversion_bps", "covered_dol").rename(columns={"mean": "reversion_bps", "se": "reversion_se"}); pos2.to_csv(out / "tables" / "markouts_first_vs_last_by_participation.csv", index=False)
        chart_bins(pos2[pos2["H"] == horizons[0]].assign(grp=lambda x: x["part_tercile"].astype(str)), out / "charts" / "markouts_first_vs_last.png", "grp", "reversion_bps", "ep_position", f"First against last day of an episode at matched participation (+{horizons[0]} min)", "reversion, bp", err="reversion_se")
        share = wtab(mo, ["H", "platform_share", "F_bin"], "reversion_bps", "covered_dol").rename(columns={"mean": "reversion_bps", "se": "reversion_se"}); share.to_csv(out / "tables" / "markouts_by_platform_share.csv", index=False)
        reg_rows = []
        for H in horizons:
            g = mo[mo["H"] == H]
            if len(g) < 200: continue
            for sample, m in (("all", np.ones(len(g), bool)), ("high-share names", (g["platform_share"] == "high-share names").to_numpy()), ("low-share names", (g["platform_share"] == "low-share names").to_numpy())):
                if m.sum() < 200: continue
                reg_rows.append(regress(g[m], "reversion_bps", "covered_dol").assign(H=H, sample=sample))
        reg = pd.concat(reg_rows, ignore_index=True) if reg_rows else pd.DataFrame(columns=["term", "coef", "se", "t", "n", "clusters", "H", "sample"]); reg.to_csv(out / "tables" / "markouts_regression.csv", index=False)
        key = reg[(reg["sample"] == "all") & (reg["term"].str.startswith("same-direction"))]
        mk_summary = {f"flow_coef_H{int(r['H'])}": float(r["coef"]) for _, r in key.iterrows()} | {f"flow_t_H{int(r['H'])}": float(r["t"]) for _, r in key.iterrows()}
        mk_summary["orders_with_markouts"] = int(mo[mo["H"] == horizons[0]].shape[0])
        LOG("mark-outs: " + "; ".join(f"+{int(r['H'])}m: β(flow)={r['coef']:+.3f} bp per %ADV (t={r['t']:.1f}, n={int(r['n']):,}, clusters={int(r['clusters']):,})" for _, r in key.iterrows()))
        if mo["broker"].notna().any():                                               # by executing broker: level of the reversion and its slope on persistent flow
            rows = []
            for (H, B), g in mo[mo["broker"].notna()].groupby(["H", "broker"]):
                if len(g) < args.min_broker_orders: continue
                r = regress(g, "reversion_bps", "covered_dol", reduced=True); r.insert(0, "broker", B); r.insert(0, "H", H); rows.append(r)
            if rows:
                rb = pd.concat(rows, ignore_index=True); rb.to_csv(out / "tables" / "markouts_regression_by_broker.csv", index=False)
                lv = wtab(mo[mo["broker"].notna()], ["H", "broker", "F_bin"], "reversion_bps", "covered_dol").rename(columns={"mean": "reversion_bps", "se": "reversion_se"}); lv.to_csv(out / "tables" / "markouts_by_broker.csv", index=False)
    else:
        LOG("fills not available: item 3 skipped")

    # ---- item 4: anticipation (pre-arrival drift), the previous day's broker, post-episode returns
    an_summary = {}; ant_reg = pd.DataFrame(); ant_pos = pd.DataFrame(); ant_brk = pd.DataFrame(); post = pd.DataFrame()
    has_drift = cov["ARRIVAL_PRC"].notna().any() and cov["PREV_CLOSE_PRC"].notna().any()
    if has_drift:
        a = cov[(cov["ARRIVAL_PRC"] > 0) & (cov["PREV_CLOSE_PRC"] > 0)].copy()
        a["r_pre"] = (a["ARRIVAL_PRC"] / a["PREV_CLOSE_PRC"] - 1) * 1e4
        ndr = a.groupby(["TICKER", "TRADE_DT"])["r_pre"].median().reset_index(); mkt = ndr.groupby("TRADE_DT")["r_pre"].median()    # the day's cross-section of names: the market proxy
        a["drift_x"] = (a["side"] * (a["r_pre"] - a["TRADE_DT"].map(mkt))).clip(-500, 500); a["w"] = a["notional"].fillna(a["q"] * a["ARRIVAL_PRC"])
        a["own_today"] = a["side"] * a["sign"].fillna(0) * a["p_net"].fillna(0)
        a.to_csv(out / "tables" / "anticipation_orders.csv", index=False)
        ant_bins = wtab(a, ["F_bin"], "drift_x", "w").rename(columns={"mean": "drift_bps", "se": "drift_se"}); ant_bins.to_csv(out / "tables" / "anticipation_by_flow_bin.csv", index=False)
        ant_pos = wtab(a, ["ep_position"], "drift_x", "w").rename(columns={"mean": "drift_bps", "se": "drift_se"}); ant_pos.to_csv(out / "tables" / "anticipation_by_episode_position.csv", index=False)
        ant_pb = wtab(a[a["ep_position"].isin(["first day", "middle", "last day"])], ["ep_position", "F_bin"], "drift_x", "w").rename(columns={"mean": "drift_bps", "se": "drift_se"}); ant_pb.to_csv(out / "tables" / "anticipation_by_position_and_flow.csv", index=False)
        chart_bins(ant_pb, out / "charts" / "anticipation_vs_flow.png", "F_bin", "drift_bps", "ep_position", f"Pre-arrival drift (prev. close → arrival, excess, platform's direction) by own flow over the preceding {args.lookback_days} days", "pre-arrival drift, bp", err="drift_se")
        rows = []
        for sample, m in (("all", np.ones(len(a), bool)), ("first day of an episode (decision day)", (a["ep_position"] == "first day").to_numpy()), ("continuation days (middle + last)", a["ep_position"].isin(["middle", "last day"]).to_numpy()), ("outside episodes", (a["ep_position"] == "outside episode").to_numpy())):
            if m.sum() < 200: continue
            rows.append(regress(a[m], "drift_x", "w", extra_terms=[("own flow today (% ADV, in the order's direction)", a.loc[m, "own_today"].clip(-20, 20))]).assign(sample=sample))
        ant_reg = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(); ant_reg.to_csv(out / "tables" / "anticipation_regression.csv", index=False)
        k1 = ant_reg[ant_reg["term"].str.startswith("same-direction")]
        an_summary = {f"drift_flow_coef[{r['sample']}]": float(r["coef"]) for _, r in k1.iterrows()} | {f"drift_flow_t[{r['sample']}]": float(r["t"]) for _, r in k1.iterrows()}
        LOG("anticipation: " + "; ".join(f"{r['sample']}: β(flow)={r['coef']:+.3f} bp per %ADV (t={r['t']:.1f}, n={int(r['n']):,})" for _, r in k1.iterrows()))
        if a["broker_prev"].notna().any():                                            # by the broker that executed the previous day's flow, on continuation days
            cont = a[a["ep_position"].isin(["middle", "last day"])]; rows = []
            for B, g in cont.groupby("broker_prev"):
                if len(g) < args.min_broker_orders: continue
                r = regress(g, "drift_x", "w", extra_terms=[("own flow today (% ADV, in the order's direction)", g["own_today"].clip(-20, 20))], flow_col="F1", flow_label="previous day", reduced=True); r.insert(0, "broker_prev", B); rows.append(r)
            if rows:
                ant_brk = pd.concat(rows, ignore_index=True); ant_brk.to_csv(out / "tables" / "anticipation_regression_by_prev_broker.csv", index=False)
                lvb = wtab(cont, ["broker_prev", "F1_bin"], "drift_x", "w").rename(columns={"mean": "drift_bps", "se": "drift_se"}); lvb.to_csv(out / "tables" / "anticipation_by_prev_broker.csv", index=False)
                kb = ant_brk[ant_brk["term"].str.startswith("same-direction")]
                fig, ax = plt.subplots(figsize=(7, 4.2)); ax.errorbar(np.arange(len(kb)), kb["coef"], yerr=1.96 * kb["se"], fmt="o", capsize=3); ax.axhline(0, color="grey", lw=.6, ls=":")
                ax.set_xticks(np.arange(len(kb))); ax.set_xticklabels([str(b)[:28] for b in kb["broker_prev"]], rotation=20, fontsize=8); ax.set_ylabel("bp per % ADV of yesterday's same-direction flow")
                ax.set_title("Pre-arrival drift on continuation days by the broker that executed the previous day's flow (±1.96 s.e.)", fontsize=9); fig.tight_layout(); fig.savefig(out / "charts" / "anticipation_by_prev_broker.png", dpi=130); plt.close(fig)
                LOG("anticipation by previous day's broker: " + "; ".join(f"{r['broker_prev']}: {r['coef']:+.2f} bp per %ADV (t={r['t']:.1f})" for _, r in kb.iterrows()))
        # post-episode returns from the closes (where the name traded on the following days)
        if cov["CLOSE_PRC"].notna().any() and len(ep):
            cl = cov[cov["CLOSE_PRC"] > 0].groupby(["TICKER", "t"])["CLOSE_PRC"].median(); pc = cov[cov["PREV_CLOSE_PRC"] > 0].groupby(["TICKER", "t"])["PREV_CLOSE_PRC"].median()
            cl_grid = cl.unstack("t").reindex(columns=range(T)); pc_grid = pc.unstack("t").reindex(columns=range(T))
            ret = np.log(cl_grid).diff(axis=1); mret = ret.median(axis=0)                                   # cross-sectional median log return per day
            rows = []
            for _, e in ep.iterrows():
                tk, e0, sg = e["TICKER"], int(e["end_t"]), int(e["sign"])
                if tk not in cl_grid.index: continue
                c0 = cl_grid.at[tk, e0] if e0 < T else np.nan; p0 = pc_grid.at[tk, int(e["start_t"])]
                def exr(t1):
                    if t1 >= T or not np.isfinite(c0): return np.nan
                    c1 = cl_grid.at[tk, t1]
                    if not np.isfinite(c1): return np.nan
                    return sg * ((np.log(c1 / c0) - mret.iloc[e0 + 1:t1 + 1].sum()) * 1e4)
                rows.append({"episode": e["episode"], "TICKER": tk, "sign": sg, "size_adv_pct": e["size_adv_pct"], "n_days": e["n_days"],
                             "during_bps": sg * ((np.log(c0 / p0) - mret.iloc[int(e["start_t"]):e0 + 1].sum()) * 1e4) if np.isfinite(c0) and np.isfinite(p0) and p0 > 0 else np.nan,
                             "post1_bps": exr(e0 + 1), "post5_bps": exr(e0 + 5), "post20_bps": exr(e0 + 20)})
            pe = pd.DataFrame(rows); pe.to_csv(out / "tables" / "post_episode_returns_episodes.csv", index=False)
            pe["size_bin"] = pd.cut(pe["size_adv_pct"], [0, 0.5, 2, 5, 20, np.inf], labels=["< 0.5", "0.5–2", "2–5", "5–20", "> 20"])
            prow = []
            for b_, g in pe.groupby("size_bin", observed=True):
                row = {"size_bin": b_, "episodes": len(g)}
                for c in ("during_bps", "post1_bps", "post5_bps", "post20_bps"):
                    v = g[c].dropna(); row[c] = float(v.mean()) if len(v) else np.nan; row[c + "_se"] = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 5 else np.nan; row[c + "_n"] = int(len(v))
                prow.append(row)
            post = pd.DataFrame(prow); post.to_csv(out / "tables" / "post_episode_returns.csv", index=False)
            fig, ax = plt.subplots(figsize=(7, 4.2)); xs = np.arange(len(post))
            for k_, c, lbl in ((0, "during_bps", "during the episode"), (1, "post1_bps", "next day"), (2, "post5_bps", "next 5 days"), (3, "post20_bps", "next 20 days")):
                ax.errorbar(xs + (k_ - 1.5) * 0.15, post[c], yerr=1.96 * post[c + "_se"], fmt="o", capsize=2, ms=4, label=lbl)
            ax.axhline(0, color="grey", lw=.6, ls=":"); ax.set_xticks(xs); ax.set_xticklabels(post["size_bin"].astype(str)); ax.set_xlabel("episode size, % ADV"); ax.set_ylabel("excess return in the platform's direction, bp")
            ax.set_title("Price path during and after the platform's episodes (close to close, excess over the cross-section)", fontsize=9); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out / "charts" / "post_episode_returns.png", dpi=130); plt.close(fig)
    else:
        LOG("arrival price and/or previous close not on the orders: item 4 skipped (pass --col ARRIVAL_PRC=… --col PREV_CLOSE_PRC=…)")

    # ---- report
    summary = {"version": VERSION, "generated": dt.datetime.now().isoformat(timespec="seconds"), "demo": bool(args.demo), "start": str(args.start), "end": str(args.end), "sec_types": sec_label, "orders": int(len(d)), "names": int(d["TICKER"].nunique()),
               "trading_days": T, "name_days": int(len(nd)), "active_name_days": int(nd["active"].sum()), "episodes": int(len(ep)), "quiet_pct": args.quiet_pct, "gap_days": args.gap_days,
               "tail_fit": main_fit, "duration_fit": dur_fit, "gamma": gam.to_dict("records") if len(gam) else [], "episode_sign_dependence": dep.to_dict("records"), "origin_kind": origin_kind,
               "brokers": brk_summary, "markouts": mk_summary, "anticipation": an_summary, "schema": {k: v for k, v in S.items() if k != "columns"}, "args": {k: (str(v) if isinstance(v, (dt.date, Path)) else v) for k, v in vars(args).items()}}
    (out / "run.json").write_text(json.dumps(summary, indent=2, default=str))
    write_report(out, summary, ep, tails, gam, lmf, dep, sens, org, brk_summary, mo if not mo.empty else pd.DataFrame(), horizons, args, ant_reg, ant_pos, ant_brk, post)
    LOG(f"done in {(dt.datetime.now() - t0).seconds}s → {out / 'report.md'}")
    return out


def fmt(x, nd=2):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else (f"{x:,.{nd}f}" if isinstance(x, (int, float, np.floating)) else str(x))


def write_report(out: Path, s: dict, ep, tails, gam, lmf, dep, sens, org, brk, mo, horizons, args, ant_reg=None, ant_pos=None, ant_brk=None, post=None) -> None:
    L = [f"# Order-flow persistence of the platform's own flow — v{VERSION}", "", f"Window {s['start']} → {s['end']}, {s['sec_types']}: {s['orders']:,} orders in {s['names']:,} names over {s['trading_days']} trading days; "
         f"{s['name_days']:,} name-days of which {s['active_name_days']:,} active (net participation ≥ {args.quiet_pct:g} % ADV); {s['episodes']:,} directional episodes (same-sign runs of active days, gaps of up to {args.gap_days} days allowed)."
         + (" **Synthetic demo store**: the true CCDF exponent of metaorder sizes is α = %.2f and the built-in reversion slope is %.2f bp per %% ADV of preceding same-direction flow." % (args.demo_alpha, args.demo_lean) if s["demo"] else ""), ""]
    L += ["## 1. The splitting mechanism, observed", ""]
    if len(ep):
        L += [f"Episodes: median size {fmt(ep['size_adv_pct'].median())} % ADV, 90th percentile {fmt(ep['size_adv_pct'].quantile(.9))}, maximum {fmt(ep['size_adv_pct'].max(), 1)}; median duration {fmt(ep['n_days'].median(), 0)} active days, 90th percentile {fmt(ep['n_days'].quantile(.9), 0)}, maximum {int(ep['n_days'].max())}; "
              f"median {fmt(ep['n_orders'].median(), 0)} account orders per episode; median peak day {fmt(ep['max_day_pct'].median())} % ADV.", ""]
    if len(tails):
        L += ["Power-law tails (Clauset–Shalizi–Newman; α is the exponent of P(X > x) ∝ x^(−α), the density exponent is α + 1; bootstrap 95 % CI; gof_p is the KS bootstrap p-value, LR > 0 with a small LR_p favours the power law over a truncated lognormal):", "",
              "| variable | group | n | α (CCDF) | 95 % CI | x_min | tail share | KS | gof p | LR vs lognormal (p) | γ predicted by LMF |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in tails.iterrows():
            L.append(f"| {r['label']} | {r['group']} | {int(r['n']):,} | {fmt(r['alpha_ccdf'])} | [{fmt(r['alpha_lo'])}, {fmt(r['alpha_hi'])}] | {fmt(r['xmin'], 3)} | {fmt(100 * r['tail_share'], 0)} % | {fmt(r['ks'], 3)} | {fmt(r['gof_p'])} | {fmt(r['LR_vs_lognormal'], 1)} ({fmt(r['LR_p'], 3)}) | {fmt(r['gamma_LMF_pred'])} |")
        L.append("")
    if len(gam):
        L += ["Sign autocorrelation C(τ) ∝ τ^(−γ) of the platform's own flow (γ by log–log fit over log-spaced lag bins from lag 1 to the first non-positive bin; 95 % CI from a bootstrap over names (or over day blocks with --boot-mode day); DFA: γ = 2 − 2H from the median Hurst exponent of the names with at least --dfa-min-events events):", "",
              "| level | names | events | γ | 95 % CI | γ from lag 2 | γ lags 1–100 | γ lags 1–1000 | fit up to lag | R² of fit | C(1) | C(10) | C(100) | H (DFA) | γ (DFA) |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in gam.iterrows():
            L.append(f"| {r['description']} | {int(r['names']):,} | {int(r['events']):,} | {fmt(r['gamma'])} | [{fmt(r['gamma_lo'])}, {fmt(r['gamma_hi'])}] | {fmt(r['gamma_from_lag2'])} | {fmt(r['gamma_lags_1_100'])} | {fmt(r['gamma_lags_1_1000'])} | {fmt(r['fit_lag_hi'], 0)} | {fmt(r['fit_r2'])} | {fmt(r['C1'], 3)} | {fmt(r['C10'], 3)} | {fmt(r['C100'], 3)} | {fmt(r['dfa_H_median'])} | {fmt(r['gamma_dfa'])} |")
        L += ["", "Estimator note: with heavy-tailed run lengths the sample autocorrelation at long lags rests on a few long runs and sits below its expectation in a typical sample, which biases a log–log fit over a wide lag range upwards; the restricted ranges, the bootstrap interval and the DFA exponent bracket the uncertainty. Report γ with its interval, never alone.", ""]
    if len(lmf):
        L += ["Lillo–Mike–Farmer test, γ = α − 1 (α from the episode-size tail in % ADV and from the duration tail in days; γ over the full lag range with its interval, and over lags 1–100 and 1–1000):", "", "| level | γ (full range) | γ lags 1–100 | γ lags 1–1000 | γ predicted from α(size) | gap | γ predicted from α(days) | gap |", "|---|---|---|---|---|---|---|---|"]
        for _, r in lmf.iterrows():
            L.append(f"| {r['level']} | {fmt(r['gamma'])} [{fmt(r['gamma_lo'])}, {fmt(r['gamma_hi'])}] | {fmt(r['gamma_lags_1_100'])} | {fmt(r['gamma_lags_1_1000'])} | {fmt(r['gamma_pred_from_size'])} [{fmt(r['alpha_lo'] - 1)}, {fmt(r['alpha_hi'] - 1)}] | {fmt(r['gap_size'])} | {fmt(r['gamma_pred_from_days'])} | {fmt(r['gap_days'])} |")
        L += ["", "Reading: the account-order sequence is persistent by construction (hundreds of synchronised orders); the burst, placement and daily-sign levels are the ones to compare with the prediction. A gap near zero on the daily-sign level means the persistence of the platform's flow is the one its own metaorder sizes imply; a γ well below α − 1 means extra persistence beyond splitting (correlated episodes across names or across time); a γ above α − 1 means the episodes are shorter-memory than their sizes suggest (participation caps that make durations less heavy-tailed than sizes).", ""]
    if len(dep):
        L += ["Dependence between the signs of successive episodes in the same name (0.5 = independent; below 0.5 = the model reverses direction, above = continuation after a pause):", "", "| subset | pairs | P(next same sign) | s.e. | lag-1 sign correlation |", "|---|---|---|---|---|"]
        for _, r in dep.iterrows(): L.append(f"| {r['subset']} | {int(r['pairs']):,} | {fmt(r['P(next episode same sign)'], 3)} | {fmt(r['se'], 3)} | {fmt(r['lag1_sign_corr'], 3)} |")
        L.append("")
    if len(sens):
        L += ["Sensitivity to the quiet threshold (a day below it does not break or start an episode):", "", "| quiet threshold (% ADV) | episodes | active name-days | median size | α(size) | γ (daily sign) |", "|---|---|---|---|---|---|"]
        for _, r in sens.iterrows(): L.append(f"| {r['quiet_pct']:g} | {int(r['episodes']):,} | {int(r['active_name_days']):,} | {fmt(r['median_size_pct'])} | {fmt(r['alpha_size'])} | {fmt(r['gamma_daily_sign'])} |")
        L.append("")
    if len(org):
        origin_txt = "the store's origin code" if s["origin_kind"] == "code" else "account-day proxy: same-side account days with ≥ 3 names = flow-like, mixed account days = rebalance-like"
        L += [f"Persistence by origin of the flow ({origin_txt}):", "",
              "| origin | orders | share of shares | episodes | median days | p90 days | α(size) | γ (daily sign) | C(1) | C(5) |", "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in org.iterrows(): L.append(f"| {r['origin']} | {int(r['orders']):,} | {fmt(100 * r['share_of_shares'], 1)} % | {int(r['episodes']):,} | {fmt(r['median_episode_days'], 0)} | {fmt(r['p90_episode_days'], 0)} | {fmt(r['alpha_size'])} | {fmt(r['gamma_daily_sign'])} | {fmt(r['C1_daily'], 3)} | {fmt(r['C5_daily'], 3)} |")
        L.append("")
    L += ["## 2. The brokerage distortion, measured", ""]
    if brk:
        L += [f"{brk['placements']:,} placements to {brk['brokers']} brokers. Brokers per episode: median {fmt(brk['brokers_per_episode_median'], 0)}, 90th percentile {fmt(brk['brokers_per_episode_p90'], 0)}; brokers per name-day: mean {fmt(brk['brokers_per_name_day_mean'])}. "
              f"Probability that two consecutive child orders of the same episode go to the same broker: {fmt(brk['p_same_consecutive_wmean'], 3)} (two random child orders of the episode: {fmt(brk['p_same_random_pair_wmean'], 3)}). "
              f"Along the platform's placement sequence in a name, P_same(τ) = {fmt(brk['p_same_lag1'], 3)} at lag 1, {fmt(brk['p_same_lag10'], 3)} at lag 10, {fmt(brk['p_same_lag50'], 3)} at lag 50; the largest broker carries {fmt(100 * brk['top_broker_share_overall'], 0)} % of the routed quantity. "
              f"A member-code decomposition of this flow would classify {fmt(100 * brk['herding_share_of_own_acf_50_lags'], 0)} % of the platform's own autocorrelation over the first 50 lags as herding (Σ C(τ)(1 − P_same(τ)) / Σ C(τ)). "
              "Tables: broker_by_episode, broker_psame_by_lag, broker_share, broker_share_by_name, broker_share_by_algo.", ""]
    else:
        L += ["Placements not available in this run; item 2 skipped.", ""]
    L += ["## 3. The liquidity providers' response to persistent flow", ""]
    if len(mo):
        L += [f"Fill mark-outs at {', '.join(f'+{h}' for h in horizons)} minutes on the desk's own quote tape (realized ½-spread = fill against the later mid; reversion = realized − effective ½-spread, positive when the price comes back after the fill), per order, dollar-weighted over covered fills; "
              f"{int(mo[mo['H'] == horizons[0]].shape[0]):,} orders with a covered fill at +{horizons[0]} min. Preceding flow = the platform's net flow in the name over the previous {args.lookback_days} trading days, in % ADV, signed in the order's direction.", ""]
        bins = pd.read_csv(out / "tables" / "markouts_by_flow_bin.csv")
        L += ["| horizon | preceding same-direction flow (% ADV) | orders | realized ½-spread bp | effective ½-spread bp | reversion bp | s.e. (day clusters) | quoted ½-spread bp | participation % |", "|---|---|---|---|---|---|---|---|---|"]
        for _, r in bins.iterrows(): L.append(f"| +{int(r['H'])} min | {r['F_bin']} | {int(r['orders']):,} | {fmt(r['realized_bps'])} | {fmt(r['eff_bps'])} | {fmt(r['reversion_bps'])} | {fmt(r['reversion_se'])} | {fmt(r['quoted_hs_bps'])} | {fmt(r['participation_pct'])} |")
        L.append("")
        reg = pd.read_csv(out / "tables" / "markouts_regression.csv")
        if len(reg):
            L += ["Regression of the reversion (bp) per order, weighted by covered dollars, standard errors clustered by name-day:", "", "| horizon | sample | n | clusters | term | coef | s.e. | t |", "|---|---|---|---|---|---|---|---|"]
            for _, r in reg[reg["term"] != "const"].iterrows(): L.append(f"| +{int(r['H'])} min | {r['sample']} | {int(r['n']):,} | {int(r['clusters']):,} | {r['term']} | {fmt(r['coef'], 3)} | {fmt(r['se'], 3)} | {fmt(r['t'], 1)} |")
            L.append("")
        pos = pd.read_csv(out / "tables" / "markouts_by_episode_position.csv")
        if len(pos):
            L += ["Reversion by the position of the day inside its episode (episodes of ≥ 3 active days):", "", "| horizon | position | orders | reversion bp | s.e. | participation % | preceding flow % ADV |", "|---|---|---|---|---|---|---|"]
            for _, r in pos.iterrows(): L.append(f"| +{int(r['H'])} min | {r['ep_position']} | {int(r['orders']):,} | {fmt(r['reversion_bps'])} | {fmt(r['reversion_se'])} | {fmt(r['participation_pct'])} | {fmt(r['flow_pct'])} |")
            L.append("")
        L += ["Reading: a positive, significant coefficient on the preceding same-direction flow means that the price reverts more after the platform's fills when the platform has been trading the same way for days, i.e. liquidity providers lean against the persistent flow (the anti-herding term of Tóth et al., seen from the metaorder's side). "
              "The coefficient on the forward own flow controls for the platform's own continuation (later same-direction trading pushes the price on and lowers the measured reversion). The first-day against last-day comparison at matched participation separates the response to persistence from the response to size. "
              "Caveat: the tape is the desk's own fill records, so the mark-out exists only where the desk had another fill in the name within the tolerance; coverage is in markouts_orders.csv.", ""]
    else:
        L += ["Fills not available in this run; item 3 skipped.", ""]
    L += ["## 4. Anticipation and counterparties", ""]
    if ant_reg is not None and len(ant_reg):
        L += ["Pre-arrival drift = the price move from the previous close to the arrival of the day's orders, in excess of the cross-section of names that day, signed in the order's direction (positive = the price moved the platform's way before it traded), winsorised at ±500 bp, dollar-weighted, name-day clusters.", ""]
        if ant_pos is not None and len(ant_pos):
            L += ["| position of the day in its episode | orders | drift bp | s.e. | participation % | preceding flow % ADV |", "|---|---|---|---|---|---|"]
            for _, r in ant_pos.iterrows(): L.append(f"| {r['ep_position']} | {int(r['orders']):,} | {fmt(r['drift_bps'])} | {fmt(r['drift_se'])} | {fmt(r['participation_pct'])} | {fmt(r['flow_pct'])} |")
            L.append("")
        L += ["| sample | n | term | coef | s.e. | t |", "|---|---|---|---|---|---|"]
        for _, r in ant_reg[ant_reg["term"].str.contains("flow")].iterrows(): L.append(f"| {r['sample']} | {int(r['n']):,} | {r['term']} | {fmt(r['coef'], 3)} | {fmt(r['se'], 3)} | {fmt(r['t'], 1)} |")
        L += ["", "Reading: on the first day of an episode the model has just decided to trade, so a drift in its direction can be the model's own signal (it buys what has started to rise). On the continuation days the quantity is fixed by the schedule, so a drift in the platform's direction that grows with the flow already executed is the signature of anticipation by other participants, whoever they are. The 'own flow today' term measures how much the day's own decision is correlated with the pre-arrival move; the 'forward own flow' term tests whether the price moves ahead of flow that has not yet been sent.", ""]
        if ant_brk is not None and len(ant_brk):
            kb = ant_brk[ant_brk["term"].str.startswith("same-direction")]
            L += ["By the broker that executed the previous day's flow (continuation days only; the coefficient is bp of pre-arrival drift per % ADV of the previous day's same-direction flow):", "", "| previous day's broker | n | coef | s.e. | t |", "|---|---|---|---|---|"]
            for _, r in kb.iterrows(): L.append(f"| {r['broker_prev']} | {int(r['n']):,} | {fmt(r['coef'], 3)} | {fmt(r['se'], 3)} | {fmt(r['t'], 1)} |")
            L += ["", "Reading: the broker that handled yesterday's child orders is the only outside party that saw them. A coefficient that is positive for one broker and zero for the others, at the same participation, spread and volatility, is evidence consistent with that broker's flow being traded ahead of; it is not proof, because the desk does not route at random (harder names and days may go to particular brokers). A routing experiment with comparable child orders alternated across brokers settles it.", ""]
    else:
        L += ["Arrival price and previous close not available on the orders in this run; item 4 skipped.", ""]
    if post is not None and len(post):
        L += ["Price path during and after the episodes (close-to-close log returns in excess of the day's cross-section, signed in the episode's direction, episodes as observations):", "", "| episode size, % ADV | episodes | during | next day | next 5 days | next 20 days |", "|---|---|---|---|---|---|"]
        for _, r in post.iterrows(): L.append(f"| {r['size_bin']} | {int(r['episodes']):,} | {fmt(r['during_bps'], 1)} ± {fmt(r['during_bps_se'], 1)} | {fmt(r['post1_bps'], 1)} ± {fmt(r['post1_bps_se'], 1)} | {fmt(r['post5_bps'], 1)} ± {fmt(r['post5_bps_se'], 1)} | {fmt(r['post20_bps'], 1)} ± {fmt(r['post20_bps_se'], 1)} |")
        L += ["", "Reading: after the last day of an episode the platform sends no more flow in that name. Partial reversion of the move made during the episode is what impact decay produces on its own (the literature finds roughly a third of the peak impact reverting). Continuation after the end means other participants positioned along the platform's flow and are holding; reversion well beyond the usual decay means positions taken against the platform are being unwound. Only names that keep trading on the following days have closes, so large names are over-represented.", ""]
    L += ["## Files", "", "tables/: name_days, episodes, tail_fit, acf, gamma, lmf_test, episode_sign_dependence, quiet_threshold_sensitivity, origin_persistence, broker_*, markouts_*, anticipation_*, post_episode_returns*; charts/: ccdf_episode_size, ccdf_episode_days, acf_loglog, broker_psame, markouts_vs_flow, markouts_first_vs_last, anticipation_vs_flow, anticipation_by_prev_broker, post_episode_returns; run.json.", ""]
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=dt.date.fromisoformat, default=None, help="first trade date (default: first in the store)")
    ap.add_argument("--end", type=dt.date.fromisoformat, default=None, help="last trade date (default: last in the store)")
    ap.add_argument("--sec-type", action="append", help="security type, repeatable (default ADR and Common Stock via tca_report's aliases; raw codes otherwise)")
    ap.add_argument("--orders-view", default="tca"); ap.add_argument("--placements-view", default=None); ap.add_argument("--fills-view", default=None)
    ap.add_argument("--col", action="append", help="override a column mapping, ROLE=COLUMN (roles: " + ", ".join(CANDIDATES) + ")")
    ap.add_argument("--print-schema", action="store_true", help="print the resolved column mapping and stop")
    ap.add_argument("--buy-codes", default="B,CS,BUY,BC", help="side codes that mean buy (everything else is a sell)")
    ap.add_argument("--pct-adv-is-fraction", action="store_true", help="PCT_ADV is stored as a fraction (0.01 = 1 %%) rather than in percent")
    ap.add_argument("--quiet-pct", type=float, default=0.02, help="a day with net participation below this (%% ADV) is quiet: it neither starts nor breaks an episode (default 0.02)")
    ap.add_argument("--quiet-sensitivity", default="0.01,0.05,0.1", help="other quiet thresholds for the sensitivity table")
    ap.add_argument("--gap-days", type=int, default=1, help="trading days without active flow allowed inside an episode (default 1)")
    ap.add_argument("--burst-minutes", type=int, default=1, help="orders of the same name, side and minute collapse into one burst event")
    ap.add_argument("--max-lag", type=int, default=5000, help="largest lag for the event-time autocorrelations"); ap.add_argument("--max-lag-days", type=int, default=60)
    ap.add_argument("--bootstrap", type=int, default=200, help="bootstrap replications for γ (default 200)"); ap.add_argument("--boot-mode", choices=["name", "day"], default="name", help="resample names (default) or the calendar in day blocks")
    ap.add_argument("--block-days", type=int, default=20, help="block length for --boot-mode day"); ap.add_argument("--dfa-min-events", type=int, default=20000, help="DFA only on sequences with at least this many events; it needs sequences much longer than the longest runs (default 20000)")
    ap.add_argument("--boot-names", type=int, default=5000, help="cap on the number of names in the γ fit and its bootstrap, largest first (default 5000)")
    ap.add_argument("--tail-boot", type=int, default=200, help="bootstrap replications for the tail exponent"); ap.add_argument("--gof-reps", type=int, default=100, help="KS bootstrap replications for the power-law goodness of fit (0 = skip)")
    ap.add_argument("--markout-horizons", default="5,30,60"); ap.add_argument("--lookback-days", type=int, default=5, help="days of preceding own flow in the mark-out analysis (default 5)")
    ap.add_argument("--markout-names", type=int, default=400, help="run the fill mark-outs on the N largest names by notional (0 = all; default 400)")
    ap.add_argument("--no-placements", action="store_true", help="skip item 2 (placements)"); ap.add_argument("--no-markouts", action="store_true", help="skip item 3 (fill mark-outs)")
    ap.add_argument("--min-broker-orders", type=int, default=2000, help="orders a broker needs for its own regression in items 3 and 4 (default 2000)")
    ap.add_argument("--max-names", type=int, default=0, help="keep only the N most traded names (0 = all)")
    ap.add_argument("--out", default=None); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--demo", action="store_true", help="run on a synthetic store with known parameters (self-test)")
    ap.add_argument("--demo-names", type=int, default=30); ap.add_argument("--demo-days", type=int, default=400); ap.add_argument("--demo-alpha", type=float, default=1.5); ap.add_argument("--demo-lean", type=float, default=0.6); ap.add_argument("--demo-leak", type=float, default=12.0)
    args = ap.parse_args()
    if args.sec_type is None and not args.demo: args.sec_type = ["adr", "common stock"]
    build(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
