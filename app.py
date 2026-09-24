
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import NormalDist
from difflib import get_close_matches
from io import StringIO
from urllib.parse import urlencode
import json
import math
import re
import unicodedata
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import streamlit as st

from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.calibration import calibration_curve


# ============================================================
# APP / STORAGE
# ============================================================

APP_DIR = Path(__file__).resolve().parent
BUILD_ID = "v37-live-sportsbook-props-verified-market-scan"
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

NBA_HIST_FILE = DATA_DIR / "game.csv"
WATCHLIST_FILE = DATA_DIR / "watchlist.csv"
PREDICTIONS_FILE = DATA_DIR / "prediction_history.csv"
SETTINGS_FILE = DATA_DIR / "settings.json"
KALSHI_HISTORY_FILE = DATA_DIR / "kalshi_price_history.csv"
MODEL_MARKET_HISTORY_FILE = DATA_DIR / "model_market_history.csv"
GAME_SNAPSHOT_FILE = DATA_DIR / "game_snapshots.json"
BET_SLIP_HISTORY_FILE = DATA_DIR / "bet_slip_history.csv"
PLAYER_LOG_DB_FILE = DATA_DIR / "player_game_logs.csv"
DATA_UPDATE_LOG_FILE = DATA_DIR / "data_update_log.csv"
SEED_DIR = APP_DIR / "seed_data"
BASKETBALL_SEED_FILE = SEED_DIR / "basketball_player_logs.csv.gz"
NFL_PLAYER_SEED_FILE = SEED_DIR / "nfl_player_weekly.csv.gz"
NFL_SNAPS_SEED_FILE = SEED_DIR / "nfl_snap_counts.csv.gz"
NFL_ROSTERS_SEED_FILE = SEED_DIR / "nfl_rosters.csv.gz"
NFL_GAMES_SEED_FILE = SEED_DIR / "nfl_games.csv.gz"

st.set_page_config(
    page_title="Sports Research Lab",
    page_icon="📊",
    layout="wide",
)

LEAGUES = {
    "NBA": {"sport": "basketball", "league": "nba"},
    "WNBA": {"sport": "basketball", "league": "wnba"},
    "NFL": {"sport": "football", "league": "nfl"},
}

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36",
    "Accept": "application/json,text/plain,*/*",
    "Referer": "https://www.espn.com/",
    "Origin": "https://www.espn.com",
}


# ============================================================
# HELPERS
# ============================================================

def fmt_pct(x):
    if x is None or pd.isna(x):
        return "—"
    return f"{float(x)*100:.1f}%"

def fmt_num(x, digits=1):
    if x is None or pd.isna(x):
        return "—"
    return f"{float(x):.{digits}f}"

def fmt_money(x):
    if x is None or pd.isna(x):
        return "—"
    return f"${float(x):,.2f}"

def safe_float(v):
    try:
        if v is None:
            return np.nan
        # ESPN sometimes returns scores/stat values as nested objects on one
        # endpoint and plain strings/numbers on another.
        if isinstance(v,dict):
            for key in ["value","displayValue","score","amount"]:
                if key in v and v.get(key) is not None:
                    return safe_float(v.get(key))
            return np.nan
        if isinstance(v,(list,tuple)):
            return safe_float(v[0]) if v else np.nan
        s = str(v).strip().replace("%","").replace("$","").replace(",","")
        if s in {"","-","--","—","None","nan","NaN"}:
            return np.nan
        if re.fullmatch(r"-?\d+(\.\d+)?-\d+(\.\d+)?", s):
            return float(s.split("-")[0])
        return float(s)
    except Exception:
        return np.nan

def request_json(url, params=None, timeout=20):
    errs=[]
    for headers in [None, BROWSER_HEADERS]:
        try:
            r=requests.get(url,params=params,headers=headers,timeout=timeout)
            if r.status_code < 400:
                return r.json()
            errs.append(f"HTTP {r.status_code}")
        except Exception as e:
            errs.append(str(e))
    raise RuntimeError(" | ".join(errs))

def load_csv(path):
    p=Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

def append_csv(path,row):
    old=load_csv(path)
    new=pd.DataFrame([row])
    out=pd.concat([old,new],ignore_index=True) if not old.empty else new
    out.to_csv(path,index=False)

def load_json(path,default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default

def save_json(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2),encoding="utf-8")

def _first_existing_col(df,candidates):
    low={str(c).strip().lower():c for c in df.columns}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    return None

def _opponent_from_matchup_text(value):
    text=str(value or "").upper()
    toks=re.findall(r"[A-Z]{2,4}",text)
    return toks[-1] if toks else ""

def _team_from_matchup_text(value):
    text=str(value or "").upper()
    toks=re.findall(r"[A-Z]{2,4}",text)
    return toks[0] if toks else ""

def normalize_player_log_rows(df,league=None,player=None,team=None,source="manual/import"):
    """Normalize an arbitrary player-game table into the app's persistent player log database."""
    if df is None or df.empty:
        return pd.DataFrame()
    out=df.copy()
    pcol=_first_existing_col(out,["PLAYER","player","player_name","name","athlete","athlete_name"])
    dcol=_first_existing_col(out,["DATE","date","GAME_DATE","game_date","gameday","game_date_est"])
    lcol=_first_existing_col(out,["LEAGUE","league","sport"])
    tcol=_first_existing_col(out,["TEAM","team","team_abbr","team_abbreviation"])
    ocol=_first_existing_col(out,["OPP","opponent","opp","opponent_abbr"])
    mcol=_first_existing_col(out,["MATCHUP","matchup"])
    scol=_first_existing_col(out,["SEASON","season","season_year"])

    if player is not None:
        out["PLAYER"]=str(player)
    elif pcol is not None:
        out["PLAYER"]=out[pcol].astype(str)
    else:
        return pd.DataFrame()

    if league is not None:
        out["LEAGUE"]=str(league).upper()
    elif lcol is not None:
        out["LEAGUE"]=out[lcol].astype(str).str.upper()
    else:
        return pd.DataFrame()

    if dcol is not None:
        out["DATE"]=out[dcol]
    elif "DATE" not in out.columns:
        return pd.DataFrame()
    dt=pd.to_datetime(out["DATE"],errors="coerce",utc=True)
    out=out[dt.notna()].copy()
    if out.empty:
        return out
    dt=pd.to_datetime(out["DATE"],errors="coerce",utc=True)
    out["DATE"]=dt.dt.strftime("%Y-%m-%d")

    # Preserve the team actually recorded for each historical game.
    # A current-roster team is only a fallback; it must never overwrite a
    # historical team after a trade or roster move.
    if tcol is not None:
        out["TEAM"]=out[tcol].astype(str)
    elif "TEAM" not in out.columns:
        if mcol is not None:
            out["TEAM"]=out[mcol].map(_team_from_matchup_text)
        else:
            out["TEAM"]=str(team or "")
    elif team is not None:
        blank=out["TEAM"].isna() | out["TEAM"].astype(str).str.strip().isin({"","nan","None"})
        out.loc[blank,"TEAM"]=str(team)

    if ocol is not None:
        out["OPP"]=out[ocol].astype(str)
        if mcol is not None:
            blank=out["OPP"].isna() | out["OPP"].astype(str).str.strip().isin({"","nan","None"})
            out.loc[blank,"OPP"]=out.loc[blank,mcol].map(_opponent_from_matchup_text)
    elif "OPP" not in out.columns:
        if mcol is not None:
            out["OPP"]=out[mcol].map(_opponent_from_matchup_text)
        else:
            out["OPP"]=""

    if scol is not None:
        out["SEASON"]=out[scol].astype(str)
    elif "SEASON" not in out.columns:
        out["SEASON"]=""

    # Common stat aliases for manual files. Keep original columns too.
    aliases={
        "PTS":["points","pts"],"REB":["rebounds","reb"],"AST":["assists","ast"],
        "MIN":["minutes","min"],"FG3M":["3pm","three_pointers_made","threes_made","fg3m"],
        "FG3A":["3pa","three_point_attempts","fg3a"],"STL":["steals","stl"],
        "BLK":["blocks","blk"],"TOV":["turnovers","tov"],
    }
    low={str(c).strip().lower():c for c in out.columns}
    for target,cands in aliases.items():
        if target in out.columns:
            continue
        src=next((low[c] for c in cands if c in low),None)
        if src is not None:
            out[target]=out[src]

    out["PLAYER_KEY"]=out["PLAYER"].map(_name_key if "_name_key" in globals() else lambda x: re.sub(r"[^a-z0-9]","",str(x).lower()))
    out["SOURCE_DB"]=str(source)
    out["INGESTED_AT"]=datetime.now().isoformat(timespec="seconds")
    if "EVENT_ID" not in out.columns:
        out["EVENT_ID"]=""
    out["ROW_KEY"]=(
        out["LEAGUE"].astype(str)+"|"+out["PLAYER_KEY"].astype(str)+"|"+
        out["DATE"].astype(str)+"|"+out["EVENT_ID"].astype(str)+"|"+
        out["TEAM"].astype(str)+"|"+out["OPP"].astype(str)
    )
    return out

def upsert_player_log_database(df,league=None,player=None,team=None,source="automatic"):
    norm=normalize_player_log_rows(df,league=league,player=player,team=team,source=source)
    if norm.empty:
        return 0
    old=load_csv(PLAYER_LOG_DB_FILE)
    old_keys=set(old.get("ROW_KEY",pd.Series(dtype=str)).astype(str)) if not old.empty else set()
    added=int((~norm["ROW_KEY"].astype(str).isin(old_keys)).sum())
    merged=pd.concat([old,norm],ignore_index=True,sort=False) if not old.empty else norm
    merged=merged.drop_duplicates("ROW_KEY",keep="last")
    merged.to_csv(PLAYER_LOG_DB_FILE,index=False)
    append_csv(DATA_UPDATE_LOG_FILE,{
        "timestamp":datetime.now().isoformat(timespec="seconds"),
        "kind":"player_logs","league":str(league or "mixed"),
        "player":str(player or "bulk"),"rows_received":len(norm),"rows_added":added,
        "source":source,
    })
    return added

def load_player_log_database(league,player):
    frames=[]
    db=load_csv(PLAYER_LOG_DB_FILE)
    key=re.sub(r"[^a-z0-9]","",str(player).lower())
    if not db.empty:
        if "PLAYER_KEY" not in db.columns:
            db["PLAYER_KEY"]=db.get("PLAYER",pd.Series("",index=db.index)).map(lambda x: re.sub(r"[^a-z0-9]","",str(x).lower()))
        x=db[(db.get("LEAGUE","").astype(str).str.upper()==str(league).upper()) & (db["PLAYER_KEY"].astype(str)==key)].copy()
        if not x.empty: frames.append(x)
    if str(league).upper() in {"NBA","WNBA"}:
        seed=local_basketball_player_log_v33(str(league).upper(),player)
        if not seed.empty: frames.append(seed)
    elif str(league).upper()=="NFL":
        seed=local_nfl_player_log_v33(player)
        if not seed.empty: frames.append(seed)
    if not frames:
        return pd.DataFrame()
    out=pd.concat(frames,ignore_index=True,sort=False)
    # Use the same completeness-aware merge used by live player research so a
    # saved schedule shell can never outrank a stat-bearing seed row.
    return merge_live_and_saved_player_logs(pd.DataFrame(),out,league)

def _canonical_id_v35(value):
    text=str(value or "").strip()
    if not text or text.lower() in {"nan","none"}:
        return ""
    if re.fullmatch(r"\d+\.0",text):
        text=text[:-2]
    return text

def _player_stat_columns_v35(league):
    if str(league).upper() in {"NBA","WNBA"}:
        return ["MIN","PTS","REB","AST","FG3M","FG3A","STL","BLK","TOV","FGA","FGM","FTA","FTM"]
    return [
        "PASSING YARDS","PASSING TDS","PASSING INT","PASSING ATT","COMPLETIONS",
        "RUSHING YARDS","RUSHING ATT","RUSHING TDS","RECEIVING YARDS",
        "RECEPTIONS","TARGETS","RECEIVING TDS","SNAP PCT","SNAPS",
    ]

def _infer_player_log_league_v35(df,league=None):
    if league:
        return str(league).upper()
    if df is not None and not df.empty and "LEAGUE" in df.columns:
        vals=df["LEAGUE"].dropna().astype(str).str.upper()
        if not vals.empty:
            return vals.iloc[0]
    nfl_markers={"PASSING YARDS","RUSHING YARDS","RECEIVING YARDS","SNAP PCT"}
    return "NFL" if df is not None and nfl_markers.intersection(set(df.columns)) else "NBA"

def _prepare_player_log_v35(df,league=None):
    """Keep only game rows that contain actual player statistics and normalize game identity."""
    if df is None or df.empty:
        return pd.DataFrame()
    out=df.copy()
    lg=_infer_player_log_league_v35(out,league)
    if "DATE" not in out.columns and "GAME_DATE" in out.columns:
        out["DATE"]=out["GAME_DATE"]
    if "DATE" in out.columns:
        dt=pd.to_datetime(out["DATE"],errors="coerce",utc=True)
        valid=dt.notna()
        out=out[valid].copy()
        dt=pd.to_datetime(out["DATE"],errors="coerce",utc=True)
        out["DATE"]=dt.dt.strftime("%Y-%m-%d")
    if "OPP" not in out.columns and "MATCHUP" in out.columns:
        out["OPP"]=out["MATCHUP"].map(_opponent_from_matchup_text)
    elif "OPP" in out.columns and "MATCHUP" in out.columns:
        blank=out["OPP"].isna() | out["OPP"].astype(str).str.strip().isin({"","nan","None"})
        out.loc[blank,"OPP"]=out.loc[blank,"MATCHUP"].map(_opponent_from_matchup_text)
    if "TEAM" not in out.columns and "MATCHUP" in out.columns:
        out["TEAM"]=out["MATCHUP"].map(_team_from_matchup_text)
    elif "TEAM" in out.columns and "MATCHUP" in out.columns:
        blank=out["TEAM"].isna() | out["TEAM"].astype(str).str.strip().isin({"","nan","None"})
        out.loc[blank,"TEAM"]=out.loc[blank,"MATCHUP"].map(_team_from_matchup_text)
    if "MATCHUP" not in out.columns:
        team=out.get("TEAM",pd.Series("",index=out.index)).astype(str)
        opp=out.get("OPP",pd.Series("",index=out.index)).astype(str)
        out["MATCHUP"]=np.where(team.str.strip().ne("") & opp.str.strip().ne(""),team+" vs "+opp,"")

    stat_cols=[c for c in _player_stat_columns_v35(lg) if c in out.columns]
    if not stat_cols:
        return pd.DataFrame(columns=out.columns)
    for c in stat_cols:
        out[c]=pd.to_numeric(out[c],errors="coerce")
    completeness=out[stat_cols].notna().sum(axis=1)
    # ESPN sometimes returns schedule/event shells with date/opponent but no player stats.
    # Those rows must never become a fake loss, a fake miss, or push verified games out of Last 20.
    out=out[completeness>0].copy()
    if out.empty:
        return out
    out["_STAT_COMPLETENESS"]=completeness.loc[out.index].astype(int)
    source=out.get("DATA_SOURCE",out.get("SOURCE_DB",pd.Series("",index=out.index))).astype(str).str.lower()
    out["_SOURCE_PRIORITY"]=np.select(
        [
            source.str.contains("official",na=False),
            source.str.contains("sportsdataverse|nflverse|local seed|boxscore",regex=True,na=False),
            source.str.contains("espn athlete",na=False),
        ],
        [4,3,2],
        default=1,
    )
    return out

def merge_live_and_saved_player_logs(live,saved,league=None):
    """Merge sources by game, preferring the row with the most verified stats."""
    parts=[]
    for df in [live,saved]:
        clean=_prepare_player_log_v35(df,league)
        if not clean.empty:
            parts.append(clean)
    if not parts:
        return pd.DataFrame()
    out=pd.concat(parts,ignore_index=True,sort=False)
    if "OPP" not in out.columns:
        out["OPP"]=""
    if "DATE" not in out.columns:
        out["DATE"]=""
    out["_OPP_KEY"]=out["OPP"].map(normalize_team_code)
    out["_DATE_KEY"]=out["DATE"].astype(str)
    out["_FALLBACK_GAME_KEY"]=out.get("EVENT_ID",pd.Series("",index=out.index)).map(_canonical_id_v35)
    if "GAME_ID" in out.columns:
        blank=out["_FALLBACK_GAME_KEY"].astype(str).eq("")
        out.loc[blank,"_FALLBACK_GAME_KEY"]=out.loc[blank,"GAME_ID"].map(_canonical_id_v35)
    out["_GAME_KEY"]=np.where(
        out["_DATE_KEY"].ne("") & out["_OPP_KEY"].ne(""),
        out["_DATE_KEY"]+"|"+out["_OPP_KEY"],
        out["_FALLBACK_GAME_KEY"],
    )
    out=out.sort_values(
        ["_STAT_COMPLETENESS","_SOURCE_PRIORITY"],ascending=[False,False],kind="stable"
    )
    nonblank=out["_GAME_KEY"].astype(str).ne("")
    keep_nonblank=out[nonblank].drop_duplicates("_GAME_KEY",keep="first")
    keep_blank=out[~nonblank]
    out=pd.concat([keep_nonblank,keep_blank],ignore_index=True,sort=False)
    if "DATE" in out.columns:
        out["_date_sort"]=pd.to_datetime(out["DATE"],errors="coerce",utc=True)
        out=out.sort_values("_date_sort",ascending=False,kind="stable")
    return out.drop(columns=[
        "_STAT_COMPLETENESS","_SOURCE_PRIORITY","_OPP_KEY","_DATE_KEY",
        "_FALLBACK_GAME_KEY","_GAME_KEY","_date_sort"
    ],errors="ignore").reset_index(drop=True)

def nba_season_string(d=None):
    d=d or date.today()
    start=d.year if d.month>=7 else d.year-1
    return f"{start}-{str(start+1)[-2:]}"

def league_season_year(league,d=None):
    d=d or date.today()
    if league=="WNBA":
        return d.year
    if league=="NFL":
        return d.year if d.month>=7 else d.year-1
    return d.year if d.month>=7 else d.year-1


@st.cache_data(show_spinner=False)
def load_basketball_seed_v33():
    if not BASKETBALL_SEED_FILE.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(BASKETBALL_SEED_FILE,compression="gzip",low_memory=False)
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_nfl_player_seed_v33():
    if not NFL_PLAYER_SEED_FILE.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(NFL_PLAYER_SEED_FILE,compression="gzip",low_memory=False)
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_nfl_snaps_seed_v33():
    if not NFL_SNAPS_SEED_FILE.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(NFL_SNAPS_SEED_FILE,compression="gzip",low_memory=False)
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_nfl_rosters_seed_v33():
    if not NFL_ROSTERS_SEED_FILE.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(NFL_ROSTERS_SEED_FILE,compression="gzip",low_memory=False)
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_nfl_games_seed_v33():
    if not NFL_GAMES_SEED_FILE.exists():
        return pd.DataFrame()
    try:
        df=pd.read_csv(NFL_GAMES_SEED_FILE,compression="gzip",low_memory=False)
        if "gameday" in df.columns:
            df["_date"]=pd.to_datetime(df["gameday"],errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()

def _name_key_v33(v):
    return re.sub(r"[^a-z0-9]","",str(v or "").lower())

def local_basketball_player_log_v33(league,player):
    df=load_basketball_seed_v33()
    if df.empty: return pd.DataFrame()
    key=_name_key_v33(player)
    names=df.get("PLAYER",pd.Series("",index=df.index)).map(_name_key_v33)
    x=df[(df.get("LEAGUE","").astype(str).str.upper()==str(league).upper()) & (names==key)].copy()
    if x.empty: return x
    x["_date_sort"]=pd.to_datetime(x.get("DATE"),errors="coerce",utc=True)
    return x.sort_values("_date_sort",ascending=False).drop(columns="_date_sort",errors="ignore")

def local_nfl_roster_v33(team,selected_date=None):
    df=load_nfl_rosters_seed_v33()
    if df.empty: return pd.DataFrame()
    season=league_season_year("NFL",selected_date or date.today())
    x=df[(pd.to_numeric(df.get("season"),errors="coerce")==int(season)) & (df.get("team","").astype(str).str.upper()==str(team).upper())].copy()
    if x.empty: return x
    if "game_type" in x.columns:
        reg=x[x["game_type"].astype(str).str.upper()=="REG"]
        if not reg.empty: x=reg
    wk=pd.to_numeric(x.get("week"),errors="coerce")
    if wk.notna().any():
        x=x[wk==wk.max()].copy()
    out=pd.DataFrame({
        "athlete_id":x.get("espn_id",pd.Series("",index=x.index)).fillna("").astype(str).replace("nan",""),
        "player_id":x.get("gsis_id",pd.Series("",index=x.index)).fillna("").astype(str).replace("nan",""),
        "player":x.get("full_name",pd.Series("",index=x.index)).astype(str),
        "position":x.get("position",pd.Series("",index=x.index)).astype(str),
        "jersey":x.get("jersey_number",pd.Series("",index=x.index)).fillna("").astype(str).replace("nan",""),
        "roster_status":x.get("status",pd.Series("",index=x.index)).astype(str),
        "team_abbr":str(team).upper(),
        "team_id":str(team).upper(),
    })
    return out[out["player"].str.strip().ne("")].drop_duplicates(subset=["player"],keep="first")

def local_nfl_schedule_v33(selected_date):
    df=load_nfl_games_seed_v33()
    if df.empty: return []
    d=pd.Timestamp(selected_date).normalize()
    x=df[df.get("_date",pd.Series(pd.NaT,index=df.index)).dt.normalize()==d].copy()
    rows=[]
    for _,r in x.iterrows():
        home=str(r.get("home_team","")).upper(); away=str(r.get("away_team","")).upper()
        hs=safe_float(r.get("home_score")); as_=safe_float(r.get("away_score"))
        state="post" if not pd.isna(hs) and not pd.isna(as_) else "pre"
        rows.append({
            "event_id":str(r.get("game_id","")),"league":"NFL","date":str(r.get("gameday","")),
            "status":"Final" if state=="post" else f"Week {r.get('week','')}","state":state,
            "home_id":home,"home_name":home,"home_abbr":home,"away_id":away,"away_name":away,"away_abbr":away,
            "home_score":hs,"away_score":as_,"venue":str(r.get("stadium","") or ""),"venue_city":"","venue_state":"","venue_country":"US",
            "indoor":str(r.get("roof","")).lower() in {"dome","closed"},"neutral_site":False,
            "home_records":{},"away_records":{},"home_stats":{},"away_stats":{},"home_leaders":{},"away_leaders":{},"odds":[],
            "nfl_week":safe_float(r.get("week")),"home_rest":safe_float(r.get("home_rest")),"away_rest":safe_float(r.get("away_rest")),
            "roof":r.get("roof"),"surface":r.get("surface"),"temp":safe_float(r.get("temp")),"wind":safe_float(r.get("wind")),
            "spread_line":safe_float(r.get("spread_line")),"total_line":safe_float(r.get("total_line")),
            "home_moneyline":safe_float(r.get("home_moneyline")),"away_moneyline":safe_float(r.get("away_moneyline")),
            "home_qb_name":r.get("home_qb_name"),"away_qb_name":r.get("away_qb_name"),
        })
    return rows

def local_nfl_player_log_v33(player,team="",player_id=""):
    stats=load_nfl_player_seed_v33()
    if stats.empty: return pd.DataFrame()
    key=_name_key_v33(player)
    pid=_canonical_id_v35(player_id) if "_canonical_id_v35" in globals() else str(player_id or "").strip()
    name_col="player_display_name" if "player_display_name" in stats.columns else "player_name"
    if pid and "player_id" in stats.columns:
        mask=stats["player_id"].map(lambda v:_canonical_id_v35(v) if "_canonical_id_v35" in globals() else str(v))==pid
        x=stats[mask].copy()
    else:
        mask=stats.get(name_col,pd.Series("",index=stats.index)).map(_name_key_v33)==key
        x=stats[mask].copy()
        # If the exact name maps to one nflverse player ID, keep that player's
        # full history across trades. Use team only when the name is genuinely ambiguous.
        ids=x.get("player_id",pd.Series(dtype=str)).dropna().astype(str).unique() if not x.empty else []
        if team and not x.empty and len(ids)>1:
            same=x[x.get("team","").astype(str).str.upper()==str(team).upper()]
            if not same.empty: x=same
    if x.empty: return x
    games=load_nfl_games_seed_v33()
    if not games.empty and "game_id" in games.columns:
        keep=[c for c in ["game_id","gameday","home_team","away_team","home_score","away_score","home_rest","away_rest","temp","wind","roof","surface"] if c in games.columns]
        x=x.merge(games[keep].drop_duplicates("game_id"),on="game_id",how="left")
    snaps=load_nfl_snaps_seed_v33()
    if not snaps.empty:
        sn=snaps.copy(); sn["_name_key"]=sn.get("player",pd.Series("",index=sn.index)).map(_name_key_v33)
        sn=sn[sn["_name_key"]==key]
        if not sn.empty:
            sn=sn[[c for c in ["game_id","team","offense_snaps","offense_pct","defense_snaps","defense_pct"] if c in sn.columns]].drop_duplicates(["game_id","team"])
            x=x.merge(sn,on=["game_id","team"],how="left")
    x["DATE"]=pd.to_datetime(x.get("gameday"),errors="coerce").dt.strftime("%Y-%m-%d")
    x["SEASON"]=x.get("season","").astype(str); x["WEEK"]=x.get("week",np.nan)
    x["TEAM"]=x.get("team",""); x["OPP"]=x.get("opponent_team","")
    x["POSITION"]=x.get("position",""); x["PLAYER"]=x.get(name_col,player)
    x["MATCHUP"]=x["TEAM"].astype(str)+" vs "+x["OPP"].astype(str)
    if "home_team" in x.columns:
        x["HOME_AWAY"]=np.where(x["TEAM"].astype(str)==x["home_team"].astype(str),"home","away")
        mine=np.where(x["HOME_AWAY"]=="home",pd.to_numeric(x.get("home_score"),errors="coerce"),pd.to_numeric(x.get("away_score"),errors="coerce"))
        opp=np.where(x["HOME_AWAY"]=="home",pd.to_numeric(x.get("away_score"),errors="coerce"),pd.to_numeric(x.get("home_score"),errors="coerce"))
        x["RESULT"]=np.where(pd.isna(mine)|pd.isna(opp),"",np.where(mine>opp,"W","L"))
    aliases={
        "PASSING YARDS":"passing_yards","PASSING TDS":"passing_tds","PASSING INT":"passing_interceptions","PASSING ATT":"attempts","COMPLETIONS":"completions",
        "PASSING EPA":"passing_epa","CPOE":"passing_cpoe","SACKS SUFFERED":"sacks_suffered","RUSHING YARDS":"rushing_yards","RUSHING ATT":"carries",
        "RUSHING TDS":"rushing_tds","RUSHING EPA":"rushing_epa","RECEPTIONS":"receptions","TARGETS":"targets","RECEIVING YARDS":"receiving_yards",
        "RECEIVING TDS":"receiving_tds","RECEIVING AIR YARDS":"receiving_air_yards","RECEIVING YAC":"receiving_yards_after_catch","RECEIVING EPA":"receiving_epa",
        "TARGET SHARE":"target_share","AIR YARDS SHARE":"air_yards_share","SNAP PCT":"offense_pct","SNAPS":"offense_snaps"
    }
    for outc,src in aliases.items():
        if src in x.columns: x[outc]=pd.to_numeric(x[src],errors="coerce")
    x["DATA_SOURCE"]="nflverse local seed"
    x["_date_sort"]=pd.to_datetime(x["DATE"],errors="coerce",utc=True)
    return x.sort_values(["_date_sort","WEEK"],ascending=[False,False]).drop(columns="_date_sort",errors="ignore")

@st.cache_data(show_spinner=False)
def local_nfl_pbp_features_v33(season):
    path=SEED_DIR / f"play_by_play_{int(season)}.parquet"
    if not path.exists(): return pd.DataFrame()
    cols=["game_id","season","week","posteam","defteam","epa","success","pass","rush","sack","qb_hit","interception","fumble_lost","yards_gained"]
    try:
        df=pd.read_parquet(path,columns=cols)
    except Exception:
        return pd.DataFrame()
    if df.empty: return df
    for c in ["epa","success","pass","rush","sack","qb_hit","interception","fumble_lost","yards_gained"]:
        df[c]=pd.to_numeric(df.get(c),errors="coerce")
    # Exclude plays without an offensive team; group to team-game features.
    df=df[df["posteam"].notna()].copy()
    df["turnover"]=((df["interception"].fillna(0)>0)|(df["fumble_lost"].fillna(0)>0)).astype(float)
    df["pressure_event"]=((df["qb_hit"].fillna(0)>0)|(df["sack"].fillna(0)>0)).astype(float)
    df["explosive"]=(df["yards_gained"].fillna(0)>=20).astype(float)
    base=df.groupby(["season","week","game_id","posteam","defteam"],dropna=False).agg(
        epa_per_play=("epa","mean"),success_rate=("success","mean"),turnover_play_rate=("turnover","mean"),explosive_play_rate=("explosive","mean"),pressure_event_rate=("pressure_event","mean")
    ).reset_index().rename(columns={"posteam":"team","defteam":"opponent_team"})
    p=df[df["pass"].fillna(0)>0].groupby(["game_id","posteam"]).agg(pass_success_rate=("success","mean"),pass_epa_per_play=("epa","mean"),pressure_rate=("pressure_event","mean"),sack_play_rate=("sack","mean")).reset_index().rename(columns={"posteam":"team"})
    r=df[df["rush"].fillna(0)>0].groupby(["game_id","posteam"]).agg(rush_success_rate=("success","mean"),rush_epa_per_play=("epa","mean")).reset_index().rename(columns={"posteam":"team"})
    return base.merge(p,on=["game_id","team"],how="left").merge(r,on=["game_id","team"],how="left")

@st.cache_data(show_spinner=False)
def local_nfl_team_game_table_v33():
    p=load_nfl_player_seed_v33()
    if p.empty: return pd.DataFrame()
    for c in ["passing_yards","rushing_yards","passing_epa","rushing_epa","passing_20","rushing_10","sacks_suffered","passing_interceptions","fumbles_lost_total","attempts","carries"]:
        if c not in p.columns: p[c]=0
        p[c]=pd.to_numeric(p[c],errors="coerce").fillna(0)
    g=p.groupby(["season","week","game_id","team","opponent_team"],dropna=False).agg(
        passing_yards=("passing_yards","sum"),rushing_yards=("rushing_yards","sum"),passing_epa=("passing_epa","sum"),rushing_epa=("rushing_epa","sum"),
        explosive_passes=("passing_20","sum"),explosive_rushes=("rushing_10","sum"),sacks_suffered=("sacks_suffered","sum"),
        interceptions=("passing_interceptions","sum"),fumbles_lost=("fumbles_lost_total","sum"),attempts=("attempts","sum"),carries=("carries","sum")
    ).reset_index()
    g["off_epa"]=g["passing_epa"]+g["rushing_epa"]
    g["turnovers"]=g["interceptions"]+g["fumbles_lost"]
    g["dropbacks"]=g["attempts"]+g["sacks_suffered"]
    g["sack_rate"]=np.where(g["dropbacks"]>0,g["sacks_suffered"]/g["dropbacks"],np.nan)
    g["turnover_rate"]=np.where((g["attempts"]+g["carries"])>0,g["turnovers"]/(g["attempts"]+g["carries"]),np.nan)
    pbp_frames=[]
    for yr in sorted(pd.to_numeric(g["season"],errors="coerce").dropna().astype(int).unique()):
        q=local_nfl_pbp_features_v33(int(yr))
        if not q.empty: pbp_frames.append(q)
    if pbp_frames:
        pbp=pd.concat(pbp_frames,ignore_index=True,sort=False)
        g=g.merge(pbp.drop(columns=["season","week","opponent_team"],errors="ignore"),on=["game_id","team"],how="left")
    opp=g[["game_id","team","off_epa","passing_epa","rushing_epa","passing_yards","rushing_yards","sack_rate","turnover_rate"]].copy()
    opp=opp.rename(columns={"team":"opp_lookup","off_epa":"def_epa_allowed","passing_epa":"def_pass_epa_allowed","rushing_epa":"def_rush_epa_allowed","passing_yards":"pass_yards_allowed","rushing_yards":"rush_yards_allowed","sack_rate":"opp_sack_rate","turnover_rate":"takeaway_opportunity_rate"})
    g=g.merge(opp,left_on=["game_id","opponent_team"],right_on=["game_id","opp_lookup"],how="left")
    games=load_nfl_games_seed_v33()
    if not games.empty:
        cols=[c for c in ["game_id","gameday","home_team","away_team","home_rest","away_rest","temp","wind","roof","surface"] if c in games.columns]
        g=g.merge(games[cols].drop_duplicates("game_id"),on="game_id",how="left")
    return g

def nfl_team_profile_v33(team,n=5):
    df=local_nfl_team_game_table_v33()
    if df.empty: return {}
    x=df[df["team"].astype(str).str.upper()==str(team).upper()].copy()
    if x.empty: return {}
    x["_d"]=pd.to_datetime(x.get("gameday"),errors="coerce")
    x=x.sort_values(["_d","season","week"],ascending=False).head(n)
    fields=["off_epa","passing_epa","rushing_epa","def_epa_allowed","def_pass_epa_allowed","def_rush_epa_allowed","passing_yards","rushing_yards","pass_yards_allowed","rush_yards_allowed","sack_rate","turnover_rate","explosive_passes","explosive_rushes","epa_per_play","success_rate","pass_success_rate","rush_success_rate","pressure_rate","sack_play_rate","explosive_play_rate"]
    return {f:pd.to_numeric(x.get(f),errors="coerce").mean() if f in x.columns else np.nan for f in fields} | {"games":len(x)}

def nfl_matchup_bullets_v33(team,opp):
    t=nfl_team_profile_v33(team,5); o=nfl_team_profile_v33(opp,5)
    strengths=[]; concerns=[]
    if not t: return strengths,concerns
    pe=safe_float(t.get("passing_epa")); re_=safe_float(t.get("rushing_epa")); de=safe_float(t.get("def_epa_allowed")); dpass=safe_float(t.get("def_pass_epa_allowed"));
    if not pd.isna(pe): (strengths if pe>0 else concerns).append(f"Pass EPA {'strong' if pe>0 else 'below zero'} recently")
    if not pd.isna(re_): (strengths if re_>0 else concerns).append(f"Rush EPA {'positive' if re_>0 else 'below zero'} recently")
    if not pd.isna(de): (strengths if de<0 else concerns).append(f"Defense EPA allowed {'strong' if de<0 else 'elevated'}")
    if o:
        odp=safe_float(o.get("def_pass_epa_allowed")); odr=safe_float(o.get("def_rush_epa_allowed"))
        if not pd.isna(odp) and odp>0: strengths.append("Opponent vulnerable vs pass")
        if not pd.isna(odr) and odr>0: strengths.append("Opponent vulnerable vs run")
        if not pd.isna(odp) and odp<0: concerns.append("Opponent strong vs pass")
    sr=safe_float(t.get("sack_rate")); tr=safe_float(t.get("turnover_rate"))
    if not pd.isna(sr) and sr>.08: concerns.append("High recent sack rate")
    if not pd.isna(tr) and tr>.035: concerns.append("Turnovers elevated")
    return strengths[:4],concerns[:4]

def render_nfl_depth_context_v33(game):
    st.markdown("## 🧾 QB / offensive personnel context")
    roster=st.session_state.get("roster",pd.DataFrame())
    left,right=st.columns(2)
    for side,col in [("away",left),("home",right)]:
        team=game.get(f"{side}_abbr","")
        qb=game.get(f"{side}_qb_name") or "Not confirmed in schedule seed"
        with col:
            st.markdown(f"### {team}")
            st.write(f"**Listed/expected QB:** {qb}")
            if roster is not None and not roster.empty and "team_abbr" in roster.columns:
                x=roster[roster["team_abbr"].astype(str).str.upper()==str(team).upper()].copy()
                if not x.empty:
                    skill=x[x.get("position",pd.Series("",index=x.index)).astype(str).str.upper().isin(["QB","RB","FB","WR","TE"])].copy()
                    show=[c for c in ["player","position","jersey","roster_status"] if c in skill.columns]
                    if show and not skill.empty: st.dataframe(skill[show].head(18),use_container_width=True,hide_index=True)
            st.caption("This is roster/depth context, not a claim that every listed player is a confirmed starter. Game-day inactive status is checked separately.")

def render_nfl_matchup_dashboard_v33(game):
    st.markdown("## 🏈 NFL matchup intelligence")
    left,right=st.columns(2)
    for side,col in [("away",left),("home",right)]:
        team=game.get(f"{side}_abbr",""); opp=game.get("home_abbr" if side=="away" else "away_abbr","")
        prof=nfl_team_profile_v33(team,5); strengths,concerns=nfl_matchup_bullets_v33(team,opp)
        with col:
            st.markdown(f"### {team}")
            if prof:
                a,b,c,d=st.columns(4)
                a.metric("Pass EPA",fmt_num(prof.get("passing_epa"),2)); b.metric("Rush EPA",fmt_num(prof.get("rushing_epa"),2)); c.metric("Def EPA allowed",fmt_num(prof.get("def_epa_allowed"),2)); d.metric("Success rate",fmt_pct(prof.get("success_rate")))
                st.markdown("**Strengths**")
                for x in strengths or ["No clear positive signal yet"]: st.write("• "+x)
                st.markdown("**Concerns**")
                for x in concerns or ["No major statistical concern flagged"]: st.write("• "+x)
            else: st.caption("Local NFL team history is not available for this team yet.")
    env=[]
    if not pd.isna(safe_float(game.get("away_rest"))): env.append(f"{game.get('away_abbr')} rest {safe_float(game.get('away_rest')):.0f}d")
    if not pd.isna(safe_float(game.get("home_rest"))): env.append(f"{game.get('home_abbr')} rest {safe_float(game.get('home_rest')):.0f}d")
    if game.get("roof"): env.append(str(game.get("roof")))
    if not pd.isna(safe_float(game.get("temp"))): env.append(f"{safe_float(game.get('temp')):.0f}°F")
    if not pd.isna(safe_float(game.get("wind"))): env.append(f"wind {safe_float(game.get('wind')):.0f} mph")
    if env: st.caption("Environment: "+" • ".join(env))

def render_nfl_player_snapshot_v33(log,player_name):
    if log is None or log.empty: return
    pos=str(log.get("POSITION",pd.Series([""])).dropna().iloc[0] if "POSITION" in log.columns and log["POSITION"].notna().any() else "").upper()
    recent=log.head(10)
    st.markdown("## 🏈 Recent NFL workload / production")
    metrics=[]
    if pos=="QB" or pd.to_numeric(recent.get("PASSING ATT",pd.Series(dtype=float)),errors="coerce").fillna(0).sum()>0:
        metrics=[("Pass Yds","PASSING YARDS"),("Pass TD","PASSING TDS"),("INT","PASSING INT"),("Pass EPA","PASSING EPA"),("CPOE","CPOE"),("Sacks","SACKS SUFFERED")]
    elif pos in {"WR","TE"}:
        metrics=[("Rec Yds","RECEIVING YARDS"),("Targets","TARGETS"),("Receptions","RECEPTIONS"),("Target share","TARGET SHARE"),("Air yds","RECEIVING AIR YARDS"),("Snap %","SNAP PCT")]
    else:
        metrics=[("Rush Yds","RUSHING YARDS"),("Carries","RUSHING ATT"),("Rec Yds","RECEIVING YARDS"),("Targets","TARGETS"),("Rush EPA","RUSHING EPA"),("Snap %","SNAP PCT")]
    cols=st.columns(min(6,len(metrics)))
    for i,(lab,c) in enumerate(metrics):
        vals=pd.to_numeric(recent.get(c,pd.Series(dtype=float)),errors="coerce").dropna()
        val=vals.mean() if not vals.empty else np.nan
        if c in {"TARGET SHARE","SNAP PCT"}: txt="—" if pd.isna(val) else f"{val*100:.0f}%"
        else: txt=fmt_num(val,2 if "EPA" in c or c=="CPOE" else 1)
        cols[i].metric(lab,txt)
    show=[c for c in ["DATE","WEEK","TEAM","OPP","HOME_AWAY","SNAP PCT","PASSING YARDS","PASSING TDS","PASSING INT","RUSHING YARDS","RUSHING ATT","RECEIVING YARDS","RECEPTIONS","TARGETS","RESULT"] if c in log.columns]
    if show: st.dataframe(log[show].head(20),use_container_width=True,hide_index=True)

def recency_training_weights(dates,half_life_days=420,min_weight=.12):
    """Keep older games in training, but make newer seasons matter more."""
    d=pd.to_datetime(pd.Series(dates),errors="coerce",utc=True)
    valid=d.dropna()
    if valid.empty:
        return np.ones(len(d),dtype=float)
    newest=valid.max()
    age=(newest-d).dt.days.fillna(half_life_days*4).clip(lower=0).astype(float)
    decay=np.power(.5,age/float(half_life_days))
    return np.asarray(min_weight+(1-min_weight)*decay,dtype=float)


# ============================================================
# SCHEDULES / TEAMS / ROSTERS / RECENT FORM
# ============================================================

@st.cache_data(ttl=180, show_spinner=False)
def _scoreboard_record_map(competitor):
    out={}
    for rec in competitor.get("records",[]) or []:
        typ=str(rec.get("type") or rec.get("name") or "").lower()
        summary=str(rec.get("summary") or "")
        if typ in {"total","overall"} or "overall" in typ:
            out["overall"]=summary
        elif "home" in typ:
            out["home"]=summary
        elif "road" in typ or "away" in typ:
            out["road"]=summary
    return out

def _scoreboard_stat_map(competitor):
    out={}
    for stat in competitor.get("statistics",[]) or []:
        name=str(stat.get("name") or "")
        if not name:
            continue
        out[name]={
            "value":stat.get("displayValue"),
            "rank":stat.get("rankDisplayValue"),
            "abbr":stat.get("abbreviation"),
        }
    return out

def _scoreboard_leaders_map(competitor):
    out={}
    for group in competitor.get("leaders",[]) or []:
        leaders=group.get("leaders") or []
        if not leaders:
            continue
        lead=leaders[0]
        athlete=lead.get("athlete") or {}
        key=str(group.get("name") or group.get("displayName") or "")
        out[key]={
            "name":athlete.get("displayName") or athlete.get("fullName") or "",
            "value":lead.get("displayValue") or lead.get("value"),
            "athlete_id":str(athlete.get("id","")),
        }
    return out

def _scoreboard_odds(comp):
    rows=comp.get("odds") or []
    if not rows:
        return {}
    o=rows[0]
    ml=o.get("moneyline") or {}
    ps=o.get("pointSpread") or {}
    total=o.get("total") or {}

    def nested(obj,*keys):
        cur=obj
        for key in keys:
            if not isinstance(cur,dict):
                return None
            cur=cur.get(key)
        return cur

    return {
        "provider":((o.get("provider") or {}).get("name") or ""),
        "details":o.get("details") or "",
        "spread":safe_float(o.get("spread")),
        "total":safe_float(o.get("overUnder")),

        "home_ml_current":nested(ml,"home","close","odds"),
        "away_ml_current":nested(ml,"away","close","odds"),
        "home_ml_open":nested(ml,"home","open","odds"),
        "away_ml_open":nested(ml,"away","open","odds"),

        "home_spread_current":nested(ps,"home","close","line"),
        "away_spread_current":nested(ps,"away","close","line"),
        "home_spread_open":nested(ps,"home","open","line"),
        "away_spread_open":nested(ps,"away","open","line"),

        "over_current":nested(total,"over","close","line"),
        "under_current":nested(total,"under","close","line"),
        "over_open":nested(total,"over","open","line"),
        "under_open":nested(total,"under","open","line"),
    }

def _record_pct(summary):
    try:
        parts=str(summary).split("-")
        w,l=int(parts[0]),int(parts[1])
        return w/(w+l) if (w+l)>0 else np.nan
    except Exception:
        return np.nan

def _scoreboard_stat_text(stats,name):
    row=(stats or {}).get(name) or {}
    value=row.get("value")
    rank=row.get("rank")
    if value in [None,""]:
        return "—"
    return f"{value}" + (f" • {rank}" if rank else "")

@st.cache_data(ttl=180, show_spinner=False)
def fetch_schedule(league_name, selected_date):
    cfg=LEAGUES[league_name]
    url=f"https://site.api.espn.com/apis/site/v2/sports/{cfg['sport']}/{cfg['league']}/scoreboard"
    try:
        payload=request_json(
            url,
            {"dates":selected_date.strftime("%Y%m%d"),"limit":100},
        )
    except Exception as e:
        if league_name=="NFL":
            local=local_nfl_schedule_v33(selected_date)
            if local:
                return local,f"ESPN schedule unavailable; using local nflverse schedule seed: {e}"
        return [],f"Schedule request failed: {e}"

    rows=[]
    for event in payload.get("events",[]) or []:
        comps=event.get("competitions") or []
        if not comps:
            continue
        comp=comps[0]
        cs=comp.get("competitors") or []
        home=next((x for x in cs if x.get("homeAway")=="home"),{})
        away=next((x for x in cs if x.get("homeAway")=="away"),{})
        ht,at=home.get("team") or {},away.get("team") or {}
        status=((event.get("status") or {}).get("type") or {})
        venue=comp.get("venue") or {}
        address=venue.get("address") or {}

        rows.append({
            "event_id":str(event.get("id","")),
            "league":league_name,
            "date":event.get("date",""),
            "status":status.get("shortDetail") or status.get("detail") or status.get("description",""),
            "state":status.get("state",""),

            "home_id":str(ht.get("id","")),
            "home_name":ht.get("displayName",""),
            "home_abbr":ht.get("abbreviation",""),
            "away_id":str(at.get("id","")),
            "away_name":at.get("displayName",""),
            "away_abbr":at.get("abbreviation",""),

            "home_score":safe_float(home.get("score")),
            "away_score":safe_float(away.get("score")),

            "venue":venue.get("fullName",""),
            "venue_city":address.get("city",""),
            "venue_state":address.get("state",""),
            "venue_country":address.get("country",""),
            "indoor":venue.get("indoor"),
            "neutral_site":comp.get("neutralSite",False),

            # Rich current-game context directly from scoreboard:
            "home_records":_scoreboard_record_map(home),
            "away_records":_scoreboard_record_map(away),
            "home_stats":_scoreboard_stat_map(home),
            "away_stats":_scoreboard_stat_map(away),
            "home_leaders":_scoreboard_leaders_map(home),
            "away_leaders":_scoreboard_leaders_map(away),
            "odds":_scoreboard_odds(comp),
        })
    if league_name=="NFL" and not rows:
        local=local_nfl_schedule_v33(selected_date)
        if local:
            return local,"ESPN returned no NFL games; using local nflverse schedule seed."
    return rows,None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all_teams(league):
    cfg=LEAGUES[league]
    url=f"https://site.api.espn.com/apis/site/v2/sports/{cfg['sport']}/{cfg['league']}/teams"
    try:
        payload=request_json(url,{"limit":100})
    except Exception as e:
        return pd.DataFrame(),str(e)

    rows=[]
    sports=payload.get("sports") or []
    for sport in sports:
        for lg in sport.get("leagues",[]) or []:
            for t in lg.get("teams",[]) or []:
                team=t.get("team") or t
                rows.append({
                    "team_id":str(team.get("id","")),
                    "team_name":team.get("displayName",""),
                    "abbr":team.get("abbreviation",""),
                })
    return pd.DataFrame(rows).drop_duplicates("team_id"),None

def flatten_roster(payload):
    rows=[]
    for group in payload.get("athletes",[]) or []:
        items=group.get("items") if isinstance(group,dict) and isinstance(group.get("items"),list) else [group] if isinstance(group,dict) else []
        for a in items:
            pos=a.get("position") or {}
            status=a.get("status") or {}
            rows.append({
                "athlete_id":str(a.get("id","")),
                "player":a.get("displayName") or a.get("fullName") or "",
                "position":pos.get("abbreviation") or pos.get("displayName") or "",
                "jersey":a.get("jersey",""),
                "roster_status":status.get("name") or status.get("type") or "",
            })
    return pd.DataFrame(rows).drop_duplicates("athlete_id")

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_roster(league,team_id):
    cfg=LEAGUES[league]
    url=f"https://site.api.espn.com/apis/site/v2/sports/{cfg['sport']}/{cfg['league']}/teams/{team_id}/roster"
    try:
        return flatten_roster(request_json(url,timeout=8)),None
    except Exception as e:
        return pd.DataFrame(),f"Roster request failed: {e}"

@st.cache_data(ttl=600, show_spinner=False)
def fetch_team_recent(league,team_id,season):
    cfg=LEAGUES[league]
    url=f"https://site.api.espn.com/apis/site/v2/sports/{cfg['sport']}/{cfg['league']}/teams/{team_id}/schedule"
    try:
        payload=request_json(url,{"season":season,"seasontype":2})
    except Exception as e:
        return pd.DataFrame(),str(e)

    rows=[]
    for event in payload.get("events",[]) or []:
        comps=event.get("competitions") or []
        if not comps:
            continue
        comp=comps[0]
        status=((event.get("status") or {}).get("type") or {})
        if status.get("state")!="post":
            continue
        cs=comp.get("competitors") or []
        mine=next((c for c in cs if str((c.get("team") or {}).get("id",""))==str(team_id)),None)
        opp=next((c for c in cs if str((c.get("team") or {}).get("id",""))!=str(team_id)),None)
        if not mine or not opp:
            continue
        pf,pa=safe_float(mine.get("score")),safe_float(opp.get("score"))
        won=mine.get("winner")
        if won is None and not pd.isna(pf) and not pd.isna(pa):
            won=pf>pa
        rows.append({
            "event_id":str(event.get("id","")),
            "date":event.get("date",""),
            "opponent":(opp.get("team") or {}).get("abbreviation",""),
            "home_away":mine.get("homeAway",""),
            "result":"W" if won else "L",
            "points_for":pf,
            "points_against":pa,
        })
    df=pd.DataFrame(rows)
    if not df.empty:
        df["_d"]=pd.to_datetime(df["date"],errors="coerce",utc=True)
        df=df.sort_values("_d",ascending=False).drop(columns="_d")
    return df,None



# ============================================================
# MATCHUP / HOME-AWAY / WEATHER
# ============================================================

@st.cache_data(ttl=900, show_spinner=False)
def fetch_team_history_multi(league, team_id, current_season, seasons_back=3):
    frames=[]
    errors=[]
    for yr in range(int(current_season), int(current_season)-seasons_back, -1):
        df,err=fetch_team_recent(league,team_id,yr)
        if err:
            errors.append(err)
        if not df.empty:
            tmp=df.copy()
            tmp["season"]=yr
            frames.append(tmp)
    if not frames:
        return pd.DataFrame(), " | ".join(dict.fromkeys(errors)) if errors else None
    out=pd.concat(frames,ignore_index=True)
    out["_date"]=pd.to_datetime(out["date"],errors="coerce",utc=True)
    out=out.sort_values("_date",ascending=False).drop(columns="_date")
    return out.drop_duplicates(subset=["date","opponent","points_for","points_against"]),None

def split_summary(df, where=None, n=None):
    if df is None or df.empty:
        return {}
    x=df.copy()
    if where:
        x=x[x["home_away"].astype(str).str.lower()==where.lower()]
    if n:
        x=x.head(n)
    if x.empty:
        return {}
    wins=int((x["result"].astype(str).str.upper()=="W").sum())
    games=len(x)
    pf=pd.to_numeric(x["points_for"],errors="coerce")
    pa=pd.to_numeric(x["points_against"],errors="coerce")
    valid=pf.notna() & pa.notna()
    score_games=int(valid.sum())
    return {
        "games":games,
        "wins":wins,
        "losses":games-wins,
        "win_pct":wins/games if games else np.nan,
        "score_games":score_games,
        "avg_for":pf[valid].mean() if score_games else np.nan,
        "avg_against":pa[valid].mean() if score_games else np.nan,
        "margin":(pf[valid]-pa[valid]).mean() if score_games else np.nan,
    }

def head_to_head_summary(team_history, opponent_abbr, n=10):
    if team_history is None or team_history.empty:
        return pd.DataFrame(),{}
    target=normalize_team_code(opponent_abbr)
    work=team_history.copy()
    work["_opp_norm"]=work["opponent"].map(normalize_team_code)
    h2h=work[work["_opp_norm"]==target].head(n).drop(columns=["_opp_norm"],errors="ignore").copy()
    return h2h,split_summary(h2h)

def rest_days_before_game(team_history, game_date):
    if team_history is None or team_history.empty:
        return np.nan
    gd=pd.to_datetime(game_date,errors="coerce",utc=True)
    if pd.isna(gd):
        return np.nan
    dates=pd.to_datetime(team_history["date"],errors="coerce",utc=True).dropna()
    dates=dates[dates<gd]
    if dates.empty:
        return np.nan
    return max(0,int((gd-dates.max()).total_seconds()//86400))

def clean_injury_rows(df):
    """
    ESPN can return the same athlete multiple times through nested/current injury
    objects. Count each athlete once, using the newest dated row when possible.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    x=df.copy()
    if "Player" not in x.columns:
        return x
    x["Player"]=x["Player"].astype(str).str.strip()
    x=x[x["Player"]!=""].copy()
    if x.empty:
        return x
    x["_player_key"]=x["Player"].map(_name_key)
    if "Date" in x.columns:
        x["_inj_date"]=pd.to_datetime(x["Date"],errors="coerce",utc=True)
        x=x.sort_values(
            ["_player_key","_inj_date"],
            ascending=[True,False],
            na_position="last",
        )
    x=x.drop_duplicates(subset=["_player_key"],keep="first")
    return x.drop(columns=["_player_key","_inj_date"],errors="ignore").reset_index(drop=True)


def injury_counts(df):
    x=clean_injury_rows(df)
    if x.empty:
        return {"total":0,"out":0,"questionable":0}
    status=x["Status"].astype(str).str.lower() if "Status" in x.columns else pd.Series("",index=x.index)
    out=int(status.str.contains("out|injured reserve|\\bir\\b|doubtful",regex=True).sum())
    questionable=int(status.str.contains("questionable|game-time|day-to-day|day to day",regex=True).sum())
    return {"total":len(x),"out":out,"questionable":questionable}

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_basketball_advanced(league, season_date):
    """
    Official NBA/WNBA Stats advanced-team endpoint.

    The public stats endpoint can intermittently time out or return an empty frame.
    This function makes two official-source attempts with slightly different
    parameter sets before giving the UI a clean failure signal. The UI then uses
    verified schedule/team-history fallbacks instead of showing a dead section.
    """
    errors=[]
    try:
        from nba_api.stats.endpoints import leaguedashteamstats
    except Exception as e:
        return pd.DataFrame(),f"nba_api unavailable: {e}"

    league_id="00" if league=="NBA" else "10"
    season=nba_season_string(season_date) if league=="NBA" else str(season_date.year)

    attempts=[
        {
            "season":season,
            "season_type_all_star":"Regular Season",
            "league_id_nullable":league_id,
            "measure_type_detailed_defense":"Advanced",
            "per_mode_detailed":"PerGame",
            "timeout":8,
        },
    ]

    for kwargs in attempts:
        try:
            obj=leaguedashteamstats.LeagueDashTeamStats(**kwargs)
            frames=obj.get_data_frames()
            if frames and not frames[0].empty:
                return frames[0].copy(),None
            errors.append("official advanced endpoint returned an empty table")
        except Exception as e:
            errors.append(str(e))

    return pd.DataFrame()," | ".join(dict.fromkeys(errors))

@st.cache_data(ttl=900, show_spinner=False)
def fetch_game_day_weather(city, state, game_date):
    if not city:
        return None,"Venue city unavailable."
    try:
        query=f"{city}, {state}" if state else city
        geo=request_json(
            "https://geocoding-api.open-meteo.com/v1/search",
            {"name":query,"count":1,"language":"en","format":"json"},
            timeout=15,
        )
        results=geo.get("results") or []
        if not results:
            geo=request_json(
                "https://geocoding-api.open-meteo.com/v1/search",
                {"name":city,"count":1,"language":"en","format":"json"},
                timeout=15,
            )
            results=geo.get("results") or []
        if not results:
            return None,"Could not locate venue city."

        loc=results[0]
        gd=pd.to_datetime(game_date,errors="coerce")
        if pd.isna(gd):
            return None,"Game date unavailable."
        d=gd.date().isoformat()

        wx=request_json(
            "https://api.open-meteo.com/v1/forecast",
            {
                "latitude":loc["latitude"],
                "longitude":loc["longitude"],
                "daily":"temperature_2m_max,temperature_2m_min,precipitation_probability_max,wind_speed_10m_max",
                "temperature_unit":"fahrenheit",
                "wind_speed_unit":"mph",
                "timezone":"auto",
                "start_date":d,
                "end_date":d,
            },
            timeout=15,
        )
        daily=wx.get("daily") or {}
        if not daily.get("time"):
            return None,"No forecast returned for game day."

        return {
            "high_f":safe_float((daily.get("temperature_2m_max") or [np.nan])[0]),
            "low_f":safe_float((daily.get("temperature_2m_min") or [np.nan])[0]),
            "rain_pct":safe_float((daily.get("precipitation_probability_max") or [np.nan])[0]),
            "wind_mph":safe_float((daily.get("wind_speed_10m_max") or [np.nan])[0]),
            "location":f"{loc.get('name',city)}, {loc.get('admin1',state)}".strip(", "),
        },None
    except Exception as e:
        return None,str(e)

def build_matchup_context(game,league):
    season=league_season_year(
        league,
        st.session_state.get("selected_date",date.today())
    )
    context={}
    for side in ["home","away"]:
        team_id=game[f"{side}_id"]
        abbr=game[f"{side}_abbr"]
        opp=game["away_abbr"] if side=="home" else game["home_abbr"]

        history,_=fetch_team_history_multi(league,team_id,season,3)
        injuries,injury_error=fetch_injuries(league,team_id)
        h2h,h2h_summary=head_to_head_summary(history,opp,10)

        context[side]={
            "abbr":abbr,
            "history":history,
            "recent5":split_summary(history,n=5),
            "recent10":split_summary(history,n=10),
            "season":split_summary(history),
            "home_split":split_summary(history,where="home"),
            "away_split":split_summary(history,where="away"),
            "h2h":h2h,
            "h2h_summary":h2h_summary,
            "rest":rest_days_before_game(history,game.get("date")),
            "injury_df":injuries,
            "injuries":injury_counts(injuries),
            "injury_error":injury_error,
            "injury_verified":injury_error is None,
        }
    return context


def empty_matchup_context(game,error_message=""):
    """Minimal context used when a live source fails. Never invents data."""
    context={}
    for side in ["home","away"]:
        context[side]={
            "abbr":game.get(f"{side}_abbr",""),
            "history":pd.DataFrame(),
            "recent5":{},
            "recent10":{},
            "season":{},
            "home_split":{},
            "away_split":{},
            "h2h":pd.DataFrame(),
            "h2h_summary":{},
            "rest":np.nan,
            "injury_df":pd.DataFrame(),
            "injuries":{"out":0,"questionable":0},
            "injury_error":error_message or "Matchup context source unavailable.",
            "injury_verified":False,
        }
    return context


def safe_build_matchup_context(game,league):
    try:
        return build_matchup_context(game,league),None
    except Exception as e:
        return empty_matchup_context(game,str(e)),str(e)


def plain_matchup_factors(game,league,ctx,advanced_rows=None,weather=None):
    home=ctx["home"]
    away=ctx["away"]
    factors=[]

    if game.get("neutral_site"):
        factors.append("Neutral site: normal home-court/home-field advantage is reduced.")
    else:
        # Use the CURRENT-SEASON scoreboard venue records first. The older
        # version mixed a multi-season history split into a current-game report.
        hrec=(game.get("home_records") or {}).get("home")
        arec=(game.get("away_records") or {}).get("road")
        hpct=_record_pct(hrec) if hrec else np.nan
        apct=_record_pct(arec) if arec else np.nan
        if not pd.isna(hpct) and not pd.isna(apct):
            diff=hpct-apct
            if abs(diff)>=.08:
                lean=home["abbr"] if diff>0 else away["abbr"]
                factors.append(
                    f"Current-season venue split: {home['abbr']} is {hrec} at home "
                    f"({hpct*100:.0f}%) and {away['abbr']} is {arec} on the road "
                    f"({apct*100:.0f}%); this split leans {lean}."
                )
            else:
                factors.append(
                    f"Current-season venue split is fairly close: {home['abbr']} {hrec} at home "
                    f"({hpct*100:.0f}%) vs {away['abbr']} {arec} on the road ({apct*100:.0f}%)."
                )

    hh=home.get("h2h_summary",{})
    if hh and hh.get("games",0):
        factors.append(
            f"Head-to-head: {home['abbr']} is {hh['wins']}-{hh['losses']} against {away['abbr']} "
            f"in the last {hh['games']} meetings found."
        )

    h10=home.get("recent10",{})
    a10=away.get("recent10",{})
    if h10 and a10:
        hmargin=h10.get("margin",np.nan)
        amargin=a10.get("margin",np.nan)
        htxt=f"{home['abbr']} {h10.get('wins',0)}-{h10.get('losses',0)}"
        atxt=f"{away['abbr']} {a10.get('wins',0)}-{a10.get('losses',0)}"
        if not pd.isna(hmargin):
            htxt+=f" ({hmargin:+.1f} avg margin)"
        if not pd.isna(amargin):
            atxt+=f" ({amargin:+.1f} avg margin)"
        factors.append("Recent form: "+htxt+"; "+atxt+".")

    hr,ar=home.get("rest",np.nan),away.get("rest",np.nan)
    if not pd.isna(hr) and not pd.isna(ar):
        if hr!=ar:
            rested=home["abbr"] if hr>ar else away["abbr"]
            factors.append(
                f"Rest advantage: {rested}. {home['abbr']} has {int(hr)} day(s); "
                f"{away['abbr']} has {int(ar)}."
            )
        else:
            factors.append(f"Rest is even: both teams have about {int(hr)} day(s).")

    hi,ai=home["injuries"],away["injuries"]
    if home.get("injury_verified") and away.get("injury_verified"):
        factors.append(
            f"Structured ESPN injury feed currently lists: {home['abbr']} "
            f"{hi.get('out',0)} out/doubtful and {hi.get('questionable',0)} questionable; "
            f"{away['abbr']} {ai.get('out',0)} out/doubtful and "
            f"{ai.get('questionable',0)} questionable. "
            "Use the public-source research above as the final current cross-check."
        )
    else:
        missing=[]
        if not home.get("injury_verified"):
            missing.append(home["abbr"])
        if not away.get("injury_verified"):
            missing.append(away["abbr"])
        factors.append(
            "Injury report was not fully verified for: " + ", ".join(missing)
            + ". Missing rows are NOT being treated as proof of zero injuries."
        )

    if advanced_rows:
        for row in advanced_rows:
            factors.append(
                f"{row['Team']} advanced: offense {row.get('Offensive rating','—')}, "
                f"defense {row.get('Defensive rating','—')}, pace {row.get('Pace','—')}."
            )

    if weather:
        factors.append(
            f"Outdoor weather: about {weather['low_f']:.0f}–{weather['high_f']:.0f}°F, "
            f"rain chance {weather['rain_pct']:.0f}%, wind up to {weather['wind_mph']:.0f} mph."
        )

    return factors


# ============================================================
# INJURIES
# ============================================================

def _walk_injuries(obj,found):
    """Recursively extract injury rows while preserving athlete/team identity."""
    if isinstance(obj,dict):
        athlete=obj.get("athlete")
        status=obj.get("status")
        if isinstance(athlete,dict) and (status is not None or "detail" in obj or "shortComment" in obj):
            if isinstance(status,dict):
                status=status.get("name") or status.get("description") or status.get("type")
            team=athlete.get("team") or {}
            details=obj.get("details") or {}
            found.append({
                "Player":athlete.get("displayName") or athlete.get("fullName") or "",
                "Status":status or "",
                "Detail":obj.get("detail") or obj.get("shortComment") or "",
                "LongComment":obj.get("longComment") or "",
                "Date":obj.get("date") or "",
                "Injury":details.get("type") or "",
                "Side":details.get("side") or "",
                "ReturnDate":details.get("returnDate") or "",
                "AthleteID":str(athlete.get("id","") or ""),
                "TeamID":str(team.get("id","") or ""),
                "Team":team.get("displayName") or team.get("name") or "",
            })
        for v in obj.values():
            _walk_injuries(v,found)
    elif isinstance(obj,list):
        for v in obj:
            _walk_injuries(v,found)


def _filter_injuries_to_team(df,league,team_id):
    """
    Prevent cross-team injury leakage from ESPN's nested injury payloads.
    Prefer exact TeamID; if TeamID is missing, verify the athlete against the
    selected team's roster. If rows cannot be verified, do not show them under
    the wrong team.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    x=clean_injury_rows(df)
    target=str(team_id or "").strip()

    # Strongest check: ESPN athlete team id.
    if "TeamID" in x.columns:
        ids=x["TeamID"].fillna("").astype(str).str.strip()
        has_ids=ids.ne("")
        if has_ids.any():
            matched=x[ids.eq(target)].copy()
            if not matched.empty:
                return clean_injury_rows(matched)
            # All identifiable rows belong to another team: reject them.
            if has_ids.all():
                return pd.DataFrame()

    # Second check: current team roster (normally already cached from selection).
    try:
        roster,_=fetch_roster(league,target)
    except Exception:
        roster=pd.DataFrame()
    if roster is not None and not roster.empty:
        athlete_ids=set(roster.get("athlete_id",pd.Series(dtype=str)).fillna("").astype(str).str.strip())
        athlete_ids.discard("")
        roster_names=set(roster.get("player",pd.Series(dtype=str)).map(_name_key))

        keep=pd.Series(False,index=x.index)
        if "AthleteID" in x.columns and athlete_ids:
            keep=keep | x["AthleteID"].fillna("").astype(str).str.strip().isin(athlete_ids)
        if "Player" in x.columns and roster_names:
            keep=keep | x["Player"].map(_name_key).isin(roster_names)
        verified=x[keep].copy()
        if not verified.empty:
            return clean_injury_rows(verified)
        return pd.DataFrame()

    # No identity source was available. Only keep rows that explicitly carry the
    # requested team id; otherwise fail closed rather than mixing teams.
    if "TeamID" in x.columns:
        return clean_injury_rows(x[x["TeamID"].fillna("").astype(str).str.strip().eq(target)])
    return pd.DataFrame()

def _injury_rows_from_items(items):
    rows=[]
    for obj in items or []:
        if not isinstance(obj,dict):
            continue
        athlete=obj.get("athlete") or {}
        status=obj.get("status") or ""
        if isinstance(status,dict):
            status=(status.get("name") or status.get("description") or status.get("type") or "")
        details=obj.get("details") or {}
        team=athlete.get("team") or {}
        rows.append({
            "Player":athlete.get("displayName") or athlete.get("fullName") or "",
            "Status":status,
            "Detail":obj.get("shortComment") or obj.get("detail") or "",
            "LongComment":obj.get("longComment") or "",
            "Date":obj.get("date") or "",
            "Injury":details.get("type") or "",
            "Side":details.get("side") or "",
            "ReturnDate":details.get("returnDate") or "",
            "AthleteID":str(athlete.get("id","")),
            "TeamID":str(team.get("id","")),
            "Team":team.get("displayName") or "",
        })
    return pd.DataFrame(rows)

@st.cache_data(ttl=180, show_spinner=False)
def fetch_injuries(league,team_id):
    """
    1) team-specific ESPN feed
    2) league-wide ESPN feed filtered to the exact team ID

    Empty + None = source verified and no listed injuries.
    Empty + error = injury report NOT verified.
    """
    cfg=LEAGUES[league]
    team_id=str(team_id)
    team_url=(
        f"https://site.api.espn.com/apis/site/v2/sports/"
        f"{cfg['sport']}/{cfg['league']}/teams/{team_id}/injuries"
    )
    team_error=None
    try:
        payload=request_json(team_url)
        if isinstance(payload,dict):
            direct=payload.get("injuries")
            if isinstance(direct,list):
                raw=_injury_rows_from_items(direct)
                df=_filter_injuries_to_team(raw,league,team_id)
                if not df.empty:
                    return df,None
            found=[]
            _walk_injuries(payload,found)
            raw=pd.DataFrame(found)
            df=_filter_injuries_to_team(raw,league,team_id)
            if not df.empty:
                return df,None
    except Exception as e:
        team_error=str(e)

    league_url=(
        f"https://site.api.espn.com/apis/site/v2/sports/"
        f"{cfg['sport']}/{cfg['league']}/injuries"
    )
    try:
        payload=request_json(league_url)
        groups=payload.get("injuries") or []
        matched=None
        for group in groups:
            if isinstance(group,dict) and str(group.get("id",""))==team_id:
                matched=group
                break
        if matched is not None:
            raw=_injury_rows_from_items(matched.get("injuries") or [])
            if raw.empty:
                return pd.DataFrame(),None
            df=_filter_injuries_to_team(raw,league,team_id)
            if not df.empty:
                return df,None
            return pd.DataFrame(),"Injury rows were returned but could not be verified against this team's identity/roster, so they were hidden instead of risking a cross-team mix-up."
        return pd.DataFrame(),"Could not verify this team's injury report from the available ESPN feeds."
    except Exception as e:
        msg=f"League injury feed failed: {e}"
        if team_error:
            msg=f"Team feed failed: {team_error} | {msg}"
        return pd.DataFrame(),msg


# ============================================================
# NBA PLAYER / TEAM DATA
# ============================================================

@st.cache_data(ttl=1800, show_spinner=False)
def nba_player_names():
    try:
        from nba_api.stats.static import players
        return players.get_players()
    except Exception:
        return []

def _name_key(value):
    s=unicodedata.normalize("NFKD",str(value or ""))
    s="".join(ch for ch in s if not unicodedata.combining(ch))
    s=s.lower().replace("’","'").replace("."," ")
    s=re.sub(r"\b(jr|sr|ii|iii|iv|v)\b"," ",s)
    return re.sub(r"[^a-z0-9]","",s)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_nba_player_directory_live(season):
    try:
        from nba_api.stats.endpoints import commonallplayers
        obj=commonallplayers.CommonAllPlayers(
            is_only_current_season=0,
            league_id="00",
            season=str(season),
        )
        return obj.get_data_frames()[0].copy(),None
    except Exception as e:
        return pd.DataFrame(),str(e)

def _resolve_nba_player(name,season):
    """Resolve only an exact normalized player name; do not silently substitute a fuzzy match."""
    target=_name_key(name)
    directory,err=fetch_nba_player_directory_live(season)
    if not directory.empty and "DISPLAY_FIRST_LAST" in directory.columns:
        work=directory.copy()
        work["_key"]=work["DISPLAY_FIRST_LAST"].map(_name_key)
        exact=work[work["_key"]==target]
        if not exact.empty:
            r=exact.iloc[0]
            return {
                "id":int(r["PERSON_ID"]),
                "full_name":str(r["DISPLAY_FIRST_LAST"]),
                "is_active":True,
                "source":"live NBA directory exact match",
            },None
    try:
        from nba_api.stats.static import players
        allp=players.get_players()
        exact=[p for p in allp if _name_key(p.get("full_name"))==target]
        if exact:
            active=[p for p in exact if p.get("is_active")]
            return (active[0] if active else exact[0]),None
    except Exception as e:
        return None,str(e)
    return None,err or f"No exact NBA player match found for {name}."


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_nba_player_by_name(name,season):
    try:
        from nba_api.stats.endpoints import playergamelog
    except Exception as e:
        return pd.DataFrame(),None,f"nba_api unavailable: {e}"

    player,resolve_err=_resolve_nba_player(name,season)
    if not player:
        return pd.DataFrame(),None,(
            f"No NBA player match found for '{name}'. "
            "The app checked the live NBA player directory and the local nba_api directory."
        )

    try:
        gl=playergamelog.PlayerGameLog(
            player_id=player["id"],
            season=season,
            season_type_all_star="Regular Season",
            timeout=8,
        )
        df=gl.get_data_frames()[0].copy()
        if df.empty:
            return df,player,"Player matched, but no games were returned for this season."
        df=df.rename(columns={"GAME_DATE":"DATE","WL":"RESULT"})
        if "MATCHUP" in df.columns:
            df["OPP"]=df["MATCHUP"].astype(str).str.split().str[-1]
            df["HOME_AWAY"]=np.where(
                df["MATCHUP"].astype(str).str.contains("@"),
                "away",
                "home",
            )
        return df,player,None
    except Exception as e:
        return pd.DataFrame(),player,f"NBA Stats request failed: {e}"

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_nba_advanced_teams(season):
    try:
        from nba_api.stats.endpoints import leaguedashteamstats
        obj=leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star="Regular Season",
            measure_type_detailed_defense="Advanced",
            timeout=8,
        )
        return obj.get_data_frames()[0],None
    except Exception as e:
        return pd.DataFrame(),f"NBA advanced stats failed: {e}"



# ============================================================
# WNBA OFFICIAL STATS (LeagueID 10)
# ============================================================

def _normalize_person_name(name):
    return re.sub(r"[^a-z0-9]", "", str(name or "").lower())

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_wnba_player_directory(season_year):
    """
    WNBA shares the NBA Stats infrastructure. LeagueID 10 is the WNBA.
    Returns official player IDs/names for the requested WNBA season.
    """
    try:
        from nba_api.stats.endpoints import commonallplayers
        obj = commonallplayers.CommonAllPlayers(
            is_only_current_season=0,
            league_id="10",
            season=str(season_year),
        )
        df = obj.get_data_frames()[0].copy()
        return df, None
    except Exception as e:
        return pd.DataFrame(), f"WNBA player directory failed: {e}"

def match_wnba_player(name, directory):
    """Exact normalized identity match only. Never fuzzy-match a different player."""
    if directory is None or directory.empty:
        return None
    name_col = "DISPLAY_FIRST_LAST" if "DISPLAY_FIRST_LAST" in directory.columns else None
    if not name_col:
        return None
    work=directory.copy()
    target=_normalize_person_name(name)
    work["_norm"]=work[name_col].map(_normalize_person_name)
    exact=work[work["_norm"]==target]
    return exact.iloc[0] if not exact.empty else None


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_wnba_player_by_name(name, season_year):
    """
    Pull a WNBA player's official game log through the NBA/WNBA Stats endpoint.
    Falls back to an error message rather than inventing missing values.
    """
    directory, derr = fetch_wnba_player_directory(season_year)
    if derr and directory.empty:
        return pd.DataFrame(), None, derr

    row = match_wnba_player(name, directory)
    if row is None:
        return pd.DataFrame(), None, f"Could not match {name} in the WNBA player directory."

    player_id = str(row.get("PERSON_ID", ""))
    display_name = str(row.get("DISPLAY_FIRST_LAST", name))
    try:
        from nba_api.stats.endpoints import playergamelog
        obj = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=str(season_year),
            season_type_all_star="Regular Season",
            league_id_nullable="10",
            timeout=8,
        )
        df = obj.get_data_frames()[0].copy()
        if df.empty:
            return df, {"id": player_id, "full_name": display_name}, "No WNBA games returned for this player."
        df = df.rename(columns={"GAME_DATE": "DATE", "WL": "RESULT"})
        if "MATCHUP" in df.columns:
            df["OPP"] = df["MATCHUP"].astype(str).str.split().str[-1]
            df["HOME_AWAY"] = np.where(
                df["MATCHUP"].astype(str).str.contains("@"), "away", "home"
            )
        return df, {"id": player_id, "full_name": display_name}, None
    except Exception as e:
        return (
            pd.DataFrame(),
            {"id": player_id, "full_name": display_name},
            f"WNBA official game-log request failed: {e}",
        )

# ============================================================
# ESPN PLAYER GAME LOG PARSER (WNBA/NFL + FALLBACK)
# ============================================================

def _event_id(ev,fallback=""):
    if not isinstance(ev,dict):
        return str(fallback)
    return str(ev.get("eventId") or ev.get("event_id") or ev.get("id") or fallback)

def _base_event_rows(payload):
    rows={}
    events=payload.get("events") or payload.get("games") or {}
    iterable=list(events.items()) if isinstance(events,dict) else [(str(i),e) for i,e in enumerate(events)] if isinstance(events,list) else []
    for key,ev in iterable:
        if not isinstance(ev,dict):
            continue
        eid=_event_id(ev,key)
        opp=ev.get("opponent") or {}
        opp_name=opp.get("abbreviation") or opp.get("displayName") if isinstance(opp,dict) else str(opp)
        rows[eid]={
            "EVENT_ID":eid,
            "DATE":ev.get("date") or ev.get("gameDate") or "",
            "OPP":opp_name or ev.get("opponentAbbreviation") or "",
            "RESULT":ev.get("gameResult") or ev.get("result") or "",
            "HOME_AWAY":ev.get("homeAway") or ev.get("atVs") or "",
        }
    return rows

def _stat_col(category,label):
    cat=str(category or "").strip().upper()
    lab=str(label or "").strip().upper()
    if lab in {"MIN","PTS","REB","AST","STL","BLK","TO","TOV","FG","3PT","FT","+/-"} and cat in {"","GENERAL","BASIC","GAME","REGULAR SEASON"}:
        return lab
    return f"{cat} {lab}".strip() if cat and cat not in {"GENERAL","BASIC","GAME","REGULAR SEASON"} else lab

def _merge_block(rows,category,labels,events):
    if not labels:
        return
    iterable=list(events.items()) if isinstance(events,dict) else [(str(i),e) for i,e in enumerate(events)] if isinstance(events,list) else []
    for key,ev in iterable:
        if not isinstance(ev,dict):
            continue
        stats=ev.get("stats") or ev.get("statistics")
        if not isinstance(stats,list):
            continue
        eid=_event_id(ev,key)
        rows.setdefault(eid,{"EVENT_ID":eid})
        use_labels=labels[-len(stats):] if len(labels)>=len(stats) else labels
        for label,value in zip(use_labels,stats):
            col=_stat_col(category,label)
            if col in rows[eid] and rows[eid][col]!=value:
                col=f"{str(category).upper()} {str(label).upper()}".strip()
            rows[eid][col]=value

def _walk_blocks(node,rows,parent=""):
    if isinstance(node,dict):
        cat=node.get("displayName") or node.get("name") or parent or ""
        labels=node.get("labels") or node.get("statNames") or node.get("names")
        events=node.get("events")
        if isinstance(labels,list) and events is not None and str(node.get("type","")).lower()!="total":
            _merge_block(rows,cat,labels,events)
        for k,v in node.items():
            if k=="events" and isinstance(labels,list):
                continue
            _walk_blocks(v,rows,cat)
    elif isinstance(node,list):
        for v in node:
            _walk_blocks(v,rows,parent)

def _derive_stats(df):
    if df.empty:
        return df
    out=df.copy()
    aliases={
        "PTS":["PTS","POINTS"],
        "REB":["REB","REBOUNDS"],
        "AST":["AST","ASSISTS"],
        "STL":["STL","STEALS"],
        "BLK":["BLK","BLOCKS"],
        "TOV":["TOV","TO","TURNOVERS"],
        "MIN":["MIN","MINUTES"],
    }
    upper={str(c).upper():c for c in out.columns}
    for alias,names in aliases.items():
        if alias not in out.columns:
            for n in names:
                if n in upper:
                    out[alias]=out[upper[n]]
                    break
    for c in list(out.columns):
        cu=str(c).upper()
        if cu=="3PT" or cu.endswith(" 3PT"):
            s=out[c].astype(str)
            out["FG3M"]=s.str.split("-").str[0].map(safe_float)
            out["FG3A"]=s.str.split("-").str[1].map(safe_float)
    return out

def parse_espn_gamelog(payload):
    rows=_base_event_rows(payload)
    labels=payload.get("labels") or payload.get("statNames") or payload.get("names")
    if isinstance(labels,list) and payload.get("events") is not None:
        _merge_block(rows,"",labels,payload.get("events"))
    _walk_blocks(payload.get("seasonTypes") or [],rows)
    _walk_blocks(payload.get("categories") or [],rows)
    if not rows:
        return pd.DataFrame()
    df=_derive_stats(pd.DataFrame(list(rows.values())))
    if "DATE" in df.columns:
        df["_d"]=pd.to_datetime(df["DATE"],errors="coerce",utc=True)
        df=df.sort_values("_d",ascending=False).drop(columns="_d")
    return df.reset_index(drop=True)

@st.cache_data(ttl=1200, show_spinner=False)
def fetch_espn_player_gamelog(league,athlete_id,season):
    cfg=LEAGUES[league]
    url=f"https://site.web.api.espn.com/apis/common/v3/sports/{cfg['sport']}/{cfg['league']}/athletes/{athlete_id}/gamelog"
    try:
        df=parse_espn_gamelog(request_json(url,{"season":season}))
        if df.empty:
            return df,"No game rows could be parsed."
        return df,None
    except Exception as e:
        return pd.DataFrame(),f"Player game log failed: {e}"



@st.cache_data(ttl=21600, show_spinner=False)
def fetch_espn_event_player_boxscore_v35(league,event_id,athlete_id="",player_name=""):
    """Fetch one player's completed box-score row by exact athlete ID/name."""
    if not event_id:
        return {},"No event ID"
    cfg=LEAGUES[league]
    url=f"https://site.api.espn.com/apis/site/v2/sports/{cfg['sport']}/{cfg['league']}/summary"
    try:
        payload=request_json(url,{"event":str(event_id)},timeout=10)
    except Exception as e:
        return {},str(e)
    target_id=_canonical_id_v35(athlete_id)
    target_name=_name_key(player_name)
    for team_block in ((payload.get("boxscore") or {}).get("players") or []):
        team=(team_block.get("team") or {})
        team_abbr=str(team.get("abbreviation") or "").upper()
        for stat_group in team_block.get("statistics",[]) or []:
            cat=str(stat_group.get("name") or stat_group.get("displayName") or "").lower()
            labels=stat_group.get("labels") or stat_group.get("names") or stat_group.get("statLabels") or []
            for item in stat_group.get("athletes",[]) or []:
                athlete=item.get("athlete") or {}
                aid=_canonical_id_v35(athlete.get("id"))
                nm=str(athlete.get("displayName") or athlete.get("fullName") or "")
                if target_id:
                    if aid!=target_id:
                        continue
                elif target_name and _name_key(nm)!=target_name:
                    continue
                else:
                    continue
                stats=item.get("stats") or item.get("statistics") or []
                if not isinstance(stats,list):
                    continue
                row={"TEAM":team_abbr,"DATA_SOURCE":"ESPN event boxscore verified"}
                for lab,val in zip(labels,stats):
                    labu=str(lab).strip().upper()
                    if league in {"NBA","WNBA"}:
                        if labu in {"MIN","PTS","REB","AST","STL","BLK","TO","TOV"}:
                            row["TOV" if labu=="TO" else labu]=safe_float(val)
                        elif labu in {"FG","3PT","FT"}:
                            bits=str(val).split("-")
                            if len(bits)==2:
                                made=safe_float(bits[0]); att=safe_float(bits[1])
                                if labu=="FG": row["FGM"],row["FGA"]=made,att
                                elif labu=="3PT": row["FG3M"],row["FG3A"]=made,att
                                else: row["FTM"],row["FTA"]=made,att
                    else:
                        if "pass" in cat:
                            if labu in {"C/ATT","CMP/ATT"}:
                                bits=str(val).split("/")
                                if len(bits)==2:
                                    row["COMPLETIONS"]=safe_float(bits[0]); row["PASSING ATT"]=safe_float(bits[1])
                            elif labu in {"YDS","YARDS"}: row["PASSING YARDS"]=safe_float(val)
                            elif labu=="TD": row["PASSING TDS"]=safe_float(val)
                            elif labu=="INT": row["PASSING INT"]=safe_float(val)
                        elif "rush" in cat:
                            if labu in {"CAR","ATT"}: row["RUSHING ATT"]=safe_float(val)
                            elif labu in {"YDS","YARDS"}: row["RUSHING YARDS"]=safe_float(val)
                            elif labu=="TD": row["RUSHING TDS"]=safe_float(val)
                        elif "receiv" in cat:
                            if labu=="REC": row["RECEPTIONS"]=safe_float(val)
                            elif labu in {"TGTS","TGT","TARGETS"}: row["TARGETS"]=safe_float(val)
                            elif labu in {"YDS","YARDS"}: row["RECEIVING YARDS"]=safe_float(val)
                            elif labu=="TD": row["RECEIVING TDS"]=safe_float(val)
                if _prepare_player_log_v35(pd.DataFrame([dict(row,DATE="2000-01-01")]),league).empty:
                    continue
                return row,None
    return {},"Player was not found in the event boxscore."

def enrich_schedule_only_rows_v35(df,league,athlete_id="",player_name="",max_events=8):
    """Fill a small number of newest schedule-only ESPN rows from exact event boxscores."""
    if df is None or df.empty or "EVENT_ID" not in df.columns:
        return df,0,0
    work=df.copy()
    stat_cols=[c for c in _player_stat_columns_v35(league) if c in work.columns]
    if stat_cols:
        missing=work[stat_cols].apply(pd.to_numeric,errors="coerce").notna().sum(axis=1).eq(0)
    else:
        missing=pd.Series(True,index=work.index)
    targets=work[missing & work["EVENT_ID"].astype(str).str.strip().ne("")].head(max_events)
    if targets.empty:
        return work,0,0
    filled=0
    attempted=len(targets)
    def one(item):
        ix,row=item
        stats,err=fetch_espn_event_player_boxscore_v35(
            league,str(row.get("EVENT_ID","")),athlete_id=athlete_id,player_name=player_name
        )
        return ix,stats
    results=[]
    with ThreadPoolExecutor(max_workers=min(4,attempted)) as ex:
        futs=[ex.submit(one,item) for item in targets.iterrows()]
        for fut in as_completed(futs):
            try: results.append(fut.result())
            except Exception: pass
    for ix,stats in results:
        if not stats:
            continue
        for k,v in stats.items():
            if k not in work.columns:
                work[k]=np.nan if k not in {"TEAM","DATA_SOURCE"} else ""
            work.at[ix,k]=v
        filled+=1
    return work,filled,attempted


# ============================================================
# PLAYER VS SPECIFIC OPPONENT MATCHUP DATA
# ============================================================

def basketball_seasons_for_date(league, d, count=3):
    d=d or date.today()
    if league=="WNBA":
        return [str(d.year-i) for i in range(count)]
    start=d.year if d.month>=7 else d.year-1
    return [f"{start-i}-{str(start-i+1)[-2:]}" for i in range(count)]

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_basketball_player_matchup_logs(
    league,
    player_id,
    opponent_team_id,
    selected_date,
    seasons_back=3,
):
    """
    Official NBA/WNBA Stats player logs filtered to the exact opponent.
    """
    try:
        from nba_api.stats.endpoints import playergamelogs
    except Exception as e:
        return pd.DataFrame(), f"nba_api unavailable: {e}"

    league_id="00" if league=="NBA" else "10"
    frames=[]
    errors=[]

    for season in basketball_seasons_for_date(league, selected_date, seasons_back):
        try:
            obj=playergamelogs.PlayerGameLogs(
                player_id_nullable=str(player_id),
                opp_team_id_nullable=str(opponent_team_id),
                league_id_nullable=league_id,
                season_nullable=season,
                season_type_nullable="Regular Season",
                per_mode_simple_nullable="Totals",
                timeout=8,
            )
            df=obj.get_data_frames()[0].copy()
            if not df.empty:
                df["SOURCE_SEASON"]=season
                frames.append(df)
        except Exception as e:
            errors.append(f"{season}: {e}")

    if not frames:
        return pd.DataFrame(), " | ".join(errors) if errors else "No games against this opponent were returned."

    out=pd.concat(frames,ignore_index=True)
    if "GAME_DATE" in out.columns:
        out["GAME_DATE"]=pd.to_datetime(out["GAME_DATE"],errors="coerce")
        out=out.sort_values("GAME_DATE",ascending=False)
        out["DATE"]=out["GAME_DATE"].dt.strftime("%Y-%m-%d")
    if "WL" in out.columns and "RESULT" not in out.columns:
        out["RESULT"]=out["WL"]
    if "MATCHUP" in out.columns:
        out["OPP"]=out["MATCHUP"].astype(str).str.split().str[-1]
        out["HOME_AWAY"]=np.where(
            out["MATCHUP"].astype(str).str.contains("@"),
            "away",
            "home",
        )
    return out.reset_index(drop=True),None

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_espn_player_gamelog_multi(league, athlete_id, current_season, seasons_back=3):
    frames=[]
    errors=[]
    for yr in range(int(current_season), int(current_season)-seasons_back, -1):
        df,err=fetch_espn_player_gamelog(league,athlete_id,yr)
        if err:
            errors.append(f"{yr}: {err}")
        if not df.empty:
            tmp=df.copy()
            tmp["SOURCE_SEASON"]=yr
            frames.append(tmp)
    if not frames:
        return pd.DataFrame(), " | ".join(errors) if errors else "No player game logs returned."
    out=pd.concat(frames,ignore_index=True)
    if "DATE" in out.columns:
        out["_d"]=pd.to_datetime(out["DATE"],errors="coerce",utc=True)
        out=out.sort_values("_d",ascending=False).drop(columns="_d")
    return out.reset_index(drop=True),None


def exact_opponent_rows_from_player_log(log,opponent_abbr):
    if log is None or log.empty or "OPP" not in log.columns:
        return pd.DataFrame()
    target=normalize_team_code(opponent_abbr)
    work=log.copy()
    work["_opp_norm"]=work["OPP"].map(normalize_team_code)
    out=work[work["_opp_norm"]==target].drop(
        columns=["_opp_norm"],errors="ignore"
    ).copy()
    return out.reset_index(drop=True)


def fetch_wnba_player_best_available(name,athlete_id,current_season):
    """
    WNBA priority:
    1) ESPN athlete game logs when an ESPN athlete ID is already known from roster.
    2) NBA/WNBA Stats infrastructure as a fallback.

    This avoids making stats.nba.com a single point of failure on Streamlit Cloud.
    """
    errors=[]

    athlete_id=str(athlete_id or "").strip()
    if athlete_id:
        espn_log,espn_err=fetch_espn_player_gamelog_multi(
            "WNBA",athlete_id,current_season,3
        )
        if not espn_log.empty:
            return (
                espn_log,
                {
                    "id":"",
                    "full_name":str(name),
                    "athlete_id":athlete_id,
                    "source":"ESPN player game log",
                },
                None,
            )
        if espn_err:
            errors.append("ESPN: "+str(espn_err))

    nba_log,player,nba_err=fetch_wnba_player_by_name(name,current_season)
    if not nba_log.empty:
        if player is None:
            player={"id":"","full_name":str(name)}
        player=dict(player)
        player["athlete_id"]=athlete_id
        player["source"]="WNBA/NBA Stats game log"
        return nba_log,player,None

    if nba_err:
        errors.append("WNBA Stats: "+str(nba_err))

    return (
        pd.DataFrame(),
        {
            "id":str((player or {}).get("id","")) if isinstance(player,dict) else "",
            "full_name":str(name),
            "athlete_id":athlete_id,
            "source":"",
        },
        " | ".join(errors) if errors else "No WNBA player game log returned.",
    )


def guessed_player_name_from_bet(raw):
    """Best-effort name extraction for text such as 'Alysha Clark over 10 points'."""
    s=str(raw or "").strip()
    if not s:
        return ""
    # Cut before the first betting/stat cue or number.
    cut=re.split(
        r"\b(?:over|under|more than|less than|at least|at most|points?|pts|rebounds?|rebs?|assists?|asts?|threes?|3-pointers?|3pm|steals?|blocks?|turnovers?|passing|rushing|receiving|receptions?|targets?|moneyline|ml)\b|(?=\d)",
        s,
        maxsplit=1,
        flags=re.I,
    )[0]
    return re.sub(r"\s+"," ",cut).strip(" -–—")


def ensure_player_in_roster_for_bet(raw,league,roster):
    """
    Type-a-Bet fallback: if the selected-game roster is empty or the player is
    missing from it, resolve the typed player through the league roster index.
    """
    base=roster.copy() if roster is not None else pd.DataFrame()
    guessed=guessed_player_name_from_bet(raw)
    if not guessed:
        return base,None

    target=_name_key(guessed)
    if not base.empty and "player" in base.columns:
        keys=base["player"].map(_name_key)
        if (keys==target).any():
            return base,None

    found,err=find_player_anywhere_v23(league,guessed)
    if found is not None:
        add=pd.DataFrame([found.to_dict() if hasattr(found,"to_dict") else dict(found)])
        if base.empty:
            base=add
        else:
            base=pd.concat([base,add],ignore_index=True)
        if "player" in base.columns:
            base=base.drop_duplicates(subset=["player"],keep="first")
        return base,err

    return base,err or f"Could not resolve player name '{guessed}'."


def typed_bet_web_queries(raw,game,league):
    d=str(st.session_state.get("selected_date",date.today()))
    return [
        f'"{raw}" {league} {d} player stats injury matchup',
        f'"{raw}" {game.get("away_name","")} {game.get("home_name","")} {d}',
    ]


def typed_bet_web_question(raw,game,league):
    return f"""
Research this exact user-entered bet using only returned public evidence:

{raw}

Selected game: {game.get('away_name')} at {game.get('home_name')} ({league})

Give:
### Player / team identity
### Current availability and role
### Recent relevant production
### Exact-opponent history
### Matchup evidence
### What supports the line
### What argues against the line
### What is still not verified

Do not invent a model probability. If the structured dashboard model cannot
calculate one, say that clearly while still giving useful source-backed research.
Do not recommend or guarantee the wager.
"""


def summarize_player_matchup(df):
    if df is None or df.empty:
        return {}
    result={"games":len(df)}
    for c in ["MIN","PTS","REB","AST","FG3M","STL","BLK","TOV"]:
        if c in df.columns:
            x=pd.to_numeric(df[c],errors="coerce")
            if x.notna().any():
                result[c]=float(x.mean())
    if all(c in df.columns for c in ["PTS","REB","AST"]):
        pra=(
            pd.to_numeric(df["PTS"],errors="coerce")
            + pd.to_numeric(df["REB"],errors="coerce")
            + pd.to_numeric(df["AST"],errors="coerce")
        )
        result["PRA"]=float(pra.mean())
    return result

def player_home_away_summary(df, target_side):
    if df is None or df.empty or "HOME_AWAY" not in df.columns:
        return {}
    subset=df[df["HOME_AWAY"].astype(str).str.lower()==target_side.lower()]
    return summarize_player_matchup(subset)

def comparison_delta(overall_df, matchup_df, stat):
    if (
        overall_df is None or overall_df.empty
        or matchup_df is None or matchup_df.empty
        or stat not in overall_df.columns
        or stat not in matchup_df.columns
    ):
        return np.nan
    overall=pd.to_numeric(overall_df[stat],errors="coerce").mean()
    matchup=pd.to_numeric(matchup_df[stat],errors="coerce").mean()
    if pd.isna(overall) or pd.isna(matchup):
        return np.nan
    return matchup-overall

def build_player_matchup_takeaways(overall_df, matchup_df, today_side, opponent, injury_df=None):
    notes=[]
    if matchup_df is not None and not matchup_df.empty:
        summary=summarize_player_matchup(matchup_df)
        games=summary.get("games",0)
        if games:
            notes.append(f"{games} prior game(s) against {opponent} were found.")

        for stat,label in [("PTS","points"),("REB","rebounds"),("AST","assists"),("FG3M","made threes")]:
            delta=comparison_delta(overall_df,matchup_df,stat)
            if not pd.isna(delta) and abs(delta)>=1.0:
                direction="higher" if delta>0 else "lower"
                notes.append(
                    f"Against {opponent}, {label} average is {abs(delta):.1f} {direction} than the player's overall average."
                )

        if all(c in overall_df.columns for c in ["PTS","REB","AST"]) and all(c in matchup_df.columns for c in ["PTS","REB","AST"]):
            overall_pra=(
                pd.to_numeric(overall_df["PTS"],errors="coerce")
                + pd.to_numeric(overall_df["REB"],errors="coerce")
                + pd.to_numeric(overall_df["AST"],errors="coerce")
            ).mean()
            matchup_pra=(
                pd.to_numeric(matchup_df["PTS"],errors="coerce")
                + pd.to_numeric(matchup_df["REB"],errors="coerce")
                + pd.to_numeric(matchup_df["AST"],errors="coerce")
            ).mean()
            delta=matchup_pra-overall_pra
            if abs(delta)>=2.0:
                direction="higher" if delta>0 else "lower"
                notes.append(
                    f"PRA versus {opponent} is {abs(delta):.1f} {direction} than the player's overall PRA average."
                )

    split=player_home_away_summary(overall_df,today_side)
    if split and split.get("games",0)>=3:
        parts=[]
        for stat,label in [("PTS","PTS"),("REB","REB"),("AST","AST")]:
            if stat in split:
                parts.append(f"{split[stat]:.1f} {label}")
        if parts:
            notes.append(
                f"Today's {today_side} split: " + ", ".join(parts) + f" across {split['games']} games in this log."
            )

    if injury_df is not None and not injury_df.empty:
        notes.append(
            "Team injury/availability context is shown because teammate absences can change minutes and usage."
        )

    return notes

# ============================================================
# PROP DEFINITIONS / MODEL
# ============================================================

BASKETBALL_PROPS={
    "Points":"PTS",
    "Rebounds":"REB",
    "Assists":"AST",
    "3-Pointers Made":"FG3M",
    "3-Point Attempts":"FG3A",
    "Steals":"STL",
    "Blocks":"BLK",
    "Turnovers":"TOV",
    "Minutes":"MIN",
}

NFL_PATTERNS={
    "Passing Yards":[r"PASSING YDS",r"PASSING YARDS"],
    "Passing TDs":[r"PASSING TD"],
    "Interceptions Thrown":[r"PASSING INT"],
    "Rushing Yards":[r"RUSHING YDS",r"RUSHING YARDS"],
    "Rushing Attempts":[r"RUSHING CAR",r"RUSHING ATT"],
    "Rushing TDs":[r"RUSHING TD"],
    "Receiving Yards":[r"RECEIVING YDS",r"RECEIVING YARDS"],
    "Receptions":[r"RECEPTIONS$",r"RECEIVING REC"],
    "Targets":[r"TARGETS$",r"RECEIVING TGT",r"RECEIVING TAR"],
    "Receiving TDs":[r"RECEIVING TD"],
}

def find_pattern_col(df,patterns):
    for c in df.columns:
        cu=re.sub(r"[_\-]+"," ",str(c).upper())
        if any(re.search(p,cu) for p in patterns):
            return c
    return None

def available_props(df,league):
    props={}
    if league in {"NBA","WNBA"}:
        for label,col in BASKETBALL_PROPS.items():
            if col in df.columns:
                props[label]=("col",col)
        for label,cols in {
            "PRA":["PTS","REB","AST"],
            "PR":["PTS","REB"],
            "PA":["PTS","AST"],
            "RA":["REB","AST"],
        }.items():
            if all(c in df.columns for c in cols):
                props[label]=("sum",cols)
    else:
        for label,pats in NFL_PATTERNS.items():
            c=find_pattern_col(df,pats)
            if c:
                props[label]=("col",c)
    return props

def extract_values(df,spec):
    kind,payload=spec
    if kind=="col":
        return df[payload].map(safe_float)
    total=pd.Series(0.0,index=df.index)
    for c in payload:
        total += df[c].map(safe_float)
    return total

def hit_mask(values,side,line):
    x=pd.to_numeric(values,errors="coerce")
    if side=="Over":
        return x>line
    if side=="Under":
        return x<line
    if side=="At least (X+)":
        return x>=line
    return x<=line

def hit_rate(values,side,line):
    x=pd.to_numeric(values,errors="coerce").dropna()
    if x.empty:
        return np.nan
    return float(hit_mask(x,side,line).mean())

def weighted_mean_std(values,decay=.10):
    x=pd.to_numeric(values,errors="coerce").dropna().to_numpy(dtype=float)
    if len(x)==0:
        return np.nan,np.nan
    # Recency dominates, but older games never become literally zero-weight.
    w=.04 + .96*np.exp(-decay*np.arange(len(x)))
    mu=float(np.average(x,weights=w))
    var=float(np.average((x-mu)**2,weights=w))
    return mu,max(math.sqrt(var),1e-6)

def normal_prob(mu,sigma,side,line):
    if pd.isna(mu) or pd.isna(sigma) or sigma<=0:
        return np.nan
    threshold=line-.5 if side=="At least (X+)" and float(line).is_integer() else line
    cdf=NormalDist(mu=mu,sigma=sigma).cdf(threshold)
    return 1-cdf if side in {"Over","At least (X+)"} else cdf

def player_projection(df,values,league,current_side=None,opponent=None):
    vals=pd.to_numeric(values,errors="coerce")
    clean=vals.dropna()
    if len(clean)<5:
        return {"mu":np.nan,"sigma":np.nan,"expected_minutes":np.nan,"reasons":[]}

    mu,sigma=weighted_mean_std(clean,.10)
    reasons=[]
    expected_minutes=np.nan

    if league in {"NBA","WNBA"} and "MIN" in df.columns:
        mins=pd.to_numeric(df["MIN"],errors="coerce")
        pair=pd.DataFrame({"v":vals,"m":mins}).dropna()
        pair=pair[pair["m"]>0]
        if len(pair)>=5:
            expected_minutes,_=weighted_mean_std(pair["m"].head(10),.15)
            rate=pair["v"]/pair["m"]
            rate_mu,_=weighted_mean_std(rate,.10)
            minutes_projection=rate_mu*expected_minutes
            mu=.60*minutes_projection+.40*mu
            reasons.append(f"expected minutes ≈ {expected_minutes:.1f}")

    if current_side and "HOME_AWAY" in df.columns:
        split=df["HOME_AWAY"].astype(str).str.lower()
        mask=split.str.contains("home") if current_side=="home" else split.str.contains("away")
        subset=vals[mask].dropna()
        if len(subset)>=5:
            delta=subset.mean()-clean.mean()
            mu += .20*delta
            reasons.append(f"{current_side} split adjustment {0.20*delta:+.2f}")

    if opponent and "OPP" in df.columns:
        mask=df["OPP"].astype(str).str.upper().str.contains(str(opponent).upper(),regex=False)
        opp=vals[mask].dropna()
        if len(opp)>=2:
            delta=opp.mean()-clean.mean()
            weight=min(.20,.05*len(opp))
            mu += weight*delta
            reasons.append(f"vs {opponent}: {len(opp)} prior games, adjustment {weight*delta:+.2f}")

    return {"mu":float(mu),"sigma":float(sigma),"expected_minutes":expected_minutes,"reasons":reasons}

def rolling_prop_backtest(df,values,league,side,line,min_train=10):
    tmp=pd.DataFrame({"value":pd.to_numeric(values,errors="coerce")})
    if "MIN" in df.columns:
        tmp["MIN"]=pd.to_numeric(df["MIN"],errors="coerce")
    tmp=tmp.dropna(subset=["value"]).iloc[::-1].reset_index(drop=True)

    preds=[]
    ys=[]
    for i in range(min_train,len(tmp)):
        hist=tmp.iloc[:i].iloc[::-1].copy()
        proj=player_projection(hist,hist["value"],league)
        p=normal_prob(proj["mu"],proj["sigma"],side,line)
        if pd.isna(p):
            continue
        y=int(hit_mask(pd.Series([tmp.iloc[i]["value"]]),side,line).iloc[0])
        preds.append(float(np.clip(p,.001,.999)))
        ys.append(y)
    return np.array(preds),np.array(ys)

def calibrate_current(raw_p,bt_p,y):
    out={
        "probability":raw_p,
        "calibrated":False,
        "samples":len(y),
        "accuracy":np.nan,
        "brier":np.nan,
        "logloss":np.nan,
    }
    if len(y)==0:
        return out
    out["accuracy"]=accuracy_score(y,bt_p>=.5)
    out["brier"]=brier_score_loss(y,bt_p)
    try:
        out["logloss"]=log_loss(y,np.clip(bt_p,.001,.999),labels=[0,1])
    except Exception:
        pass
    if len(y)>=20 and len(np.unique(y))==2:
        try:
            iso=IsotonicRegression(out_of_bounds="clip")
            iso.fit(bt_p,y)
            out["probability"]=float(np.clip(iso.predict([raw_p])[0],.001,.999))
            out["calibrated"]=True
        except Exception:
            pass
    return out

def confidence_label(cal):
    n=cal["samples"]
    b=cal["brier"]
    if n>=30 and not pd.isna(b) and b<=.22:
        return "Higher"
    if n>=15:
        return "Medium"
    return "Low"



# ============================================================
# AUTOMATIC RESEARCH IDEAS
# ============================================================

def generate_research_ideas(gamelog, league, opponent=""):
    """
    Generate lines worth investigating from the player's own historical distribution.
    These are NOT automatic bets: market price still determines whether an apparent
    probability advantage has positive EV.
    """
    props = available_props(gamelog, league)
    supported = []
    volatile = []

    for prop_name, spec in props.items():
        if prop_name == "Minutes":
            continue

        vals = pd.to_numeric(extract_values(gamelog, spec), errors="coerce").dropna()
        if len(vals) < 8:
            continue

        # Integer X+ thresholds are easier to understand and map well to common prop markets.
        low_line = max(1, int(math.floor(vals.quantile(.30))))
        high_line = max(1, int(math.ceil(vals.quantile(.75))))

        for threshold, bucket in [(low_line, "supported"), (high_line, "volatile")]:
            side = "At least (X+)"
            proj = player_projection(gamelog, extract_values(gamelog, spec), league, opponent=opponent)
            raw_p = normal_prob(proj["mu"], proj["sigma"], side, threshold)
            if pd.isna(raw_p):
                continue

            bt_p, bt_y = rolling_prop_backtest(
                gamelog, extract_values(gamelog, spec), league, side, threshold
            )
            cal = calibrate_current(raw_p, bt_p, bt_y)
            p = cal["probability"]
            last10 = hit_rate(extract_values(gamelog, spec).head(10), side, threshold)

            opp_rate = np.nan
            if opponent and "OPP" in gamelog.columns:
                mask = gamelog["OPP"].astype(str).str.upper().str.contains(
                    opponent.upper(), regex=False
                )
                opp_vals = extract_values(gamelog, spec)[mask]
                if len(pd.to_numeric(opp_vals, errors="coerce").dropna()) >= 2:
                    opp_rate = hit_rate(opp_vals, side, threshold)

            row = {
                "Stat": prop_name,
                "Line": f"{threshold}+",
                "Model probability": p,
                "Last 10": last10,
                "Vs opponent": opp_rate,
                "Backtest N": cal["samples"],
                "Brier": cal["brier"],
                "Fair price before fees": p * 100 if not pd.isna(p) else np.nan,
            }

            if bucket == "supported":
                if (
                    p >= .58
                    and cal["samples"] >= 8
                    and (pd.isna(last10) or last10 >= .50)
                ):
                    supported.append(row)
            else:
                # Higher threshold, lower probability: potentially higher payout,
                # but only interesting if market price is sufficiently low.
                if .18 <= p <= .55:
                    volatile.append(row)

    supported = sorted(
        supported,
        key=lambda r: (
            -float(r["Model probability"] if not pd.isna(r["Model probability"]) else 0),
            -int(r["Backtest N"]),
        ),
    )[:5]
    volatile = sorted(
        volatile,
        key=lambda r: (
            -float(r["Model probability"] if not pd.isna(r["Model probability"]) else 0),
            -int(r["Backtest N"]),
        ),
    )[:5]

    return supported, volatile

def ideas_to_display(rows):
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).copy()
    for c in ["Model probability", "Last 10", "Vs opponent"]:
        if c in df.columns:
            df[c] = df[c].map(lambda x: "—" if pd.isna(x) else f"{x*100:.1f}%")
    if "Brier" in df.columns:
        df["Brier"] = df["Brier"].map(lambda x: "—" if pd.isna(x) else f"{x:.3f}")
    if "Fair price before fees" in df.columns:
        df["Fair price before fees"] = df["Fair price before fees"].map(
            lambda x: "—" if pd.isna(x) else f"{x:.0f}¢"
        )
    return df

# ============================================================
# KALSHI / EV / RISK
# ============================================================

@st.cache_data(ttl=30, show_spinner=False)
def fetch_kalshi_market(ticker):
    ticker=ticker.strip().upper()
    if not ticker:
        return None,"Enter a ticker."
    try:
        payload=request_json(f"https://external-api.kalshi.com/trade-api/v2/markets/{ticker}",timeout=15)
        return payload.get("market") or {},None
    except Exception as e:
        return None,str(e)

def kalshi_implied(m):
    vals={}
    for key in ["yes_bid","yes_ask","last_price"]:
        if m.get(key) is not None:
            vals[key]=safe_float(m[key])/100
    for key,out in [("yes_bid_dollars","yes_bid"),("yes_ask_dollars","yes_ask"),("last_price_dollars","last_price")]:
        if out not in vals and m.get(key) not in (None,""):
            vals[out]=safe_float(m[key])
    if "yes_bid" in vals and "yes_ask" in vals:
        return (vals["yes_bid"]+vals["yes_ask"])/2
    return vals.get("last_price",vals.get("yes_ask",vals.get("yes_bid",np.nan)))

def contract_ev(model_p,price_cents,fee=0):
    price=price_cents/100
    return model_p*(1-price)-(1-model_p)*price-fee

def fractional_kelly(model_p,price_cents,fraction=.25):
    price=price_cents/100
    if price<=0 or price>=1:
        return 0
    b=(1-price)/price
    q=1-model_p
    full=(b*model_p-q)/b
    return max(0,float(full)*fraction)



# ============================================================
# KALSHI SPORTS MISPRICING SCANNER
# ============================================================

KALSHI_SPORT_SERIES = {
    "NBA": {
        "Game winner": "KXNBAGAME",
        "Spread": "KXNBASPREAD",
        "Total": "KXNBATOTAL",
    },
    "WNBA": {
        "Game winner": "KXWNBAGAME",
        "Spread": "KXWNBASPREAD",
        "Total": "KXWNBATOTAL",
    },
    "NFL": {
        "Game winner": "KXNFLGAME",
        "Spread": "KXNFLSPREAD",
        "Total": "KXNFLTOTAL",
    },
}

TEAM_CODE_MAP = {
    "NY": "NYK", "GS": "GSW", "SA": "SAS", "NO": "NOP", "PHO": "PHX",
    "CONN": "CON", "PDX": "POR",
}

def normalize_team_code(code):
    code = re.sub(r"[^A-Z]", "", str(code or "").upper())
    return TEAM_CODE_MAP.get(code, code)

def market_decimal(market, dollar_key, cents_key):
    if market.get(dollar_key) not in (None, ""):
        v = safe_float(market.get(dollar_key))
        return v if not pd.isna(v) else np.nan
    if market.get(cents_key) not in (None, ""):
        v = safe_float(market.get(cents_key))
        return v / 100 if not pd.isna(v) else np.nan
    return np.nan

def kalshi_yes_ask(market):
    ask = market_decimal(market, "yes_ask_dollars", "yes_ask")
    if not pd.isna(ask):
        return ask
    return market_decimal(market, "last_price_dollars", "last_price")

def kalshi_yes_bid(market):
    return market_decimal(market, "yes_bid_dollars", "yes_bid")

def kalshi_market_spread(market):
    bid, ask = kalshi_yes_bid(market), kalshi_yes_ask(market)
    if pd.isna(bid) or pd.isna(ask):
        return np.nan
    return max(0.0, ask - bid)

def kalshi_market_outcome_code(market):
    ticker = str(market.get("ticker", "")).upper()
    if not ticker:
        return ""
    return normalize_team_code(ticker.split("-")[-1])

def kalshi_event_date(market):
    raw = str(market.get("event_ticker") or market.get("ticker") or "").upper()
    m = re.search(r"-(\d{2})(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d{2})", raw)
    if not m:
        return None
    yy, mon, dd = m.groups()
    months = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,"JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}
    try:
        return date(2000 + int(yy), months[mon], int(dd))
    except Exception:
        return None

@st.cache_data(ttl=30, show_spinner=False)
def fetch_kalshi_series_markets(series_ticker, max_pages=5):
    url = "https://external-api.kalshi.com/trade-api/v2/markets"
    out, cursor = [], ""
    try:
        for _ in range(max_pages):
            params = {"limit": 200, "status": "open", "series_ticker": series_ticker}
            if cursor:
                params["cursor"] = cursor
            payload = request_json(url, params=params, timeout=20)
            out.extend(payload.get("markets") or [])
            cursor = str(payload.get("cursor") or "")
            if not cursor:
                break
        return out, None
    except Exception as e:
        return out, f"{series_ticker}: {e}"

def american_to_implied(odds):
    o = safe_float(odds)
    if pd.isna(o) or o == 0:
        return np.nan
    if o < 0:
        return (-o) / ((-o) + 100.0)
    return 100.0 / (o + 100.0)


# ============================================================
# V37: THE ODDS API — LIVE PLAYER PROP MARKETS
# ============================================================

ODDS_API_SPORT_KEYS={
    "NBA":"basketball_nba",
    "WNBA":"basketball_wnba",
    "NFL":"americanfootball_nfl",
}

# Keep this to <=10 bookmakers so The Odds API treats it as one bookmaker group.
# This includes major US books plus Hard Rock Bet Florida when that book publishes
# the requested market. Availability still varies by game/market.
ODDS_API_BOOKMAKERS=[
    "draftkings","fanduel","betmgm","betrivers",
    "betonlineag","bovada","hardrockbet_fl","espnbet",
]

ODDS_API_MARKETS={
    "NBA":[
        "player_points","player_rebounds","player_assists","player_threes",
        "player_points_rebounds_assists","player_points_rebounds",
        "player_points_assists","player_rebounds_assists",
    ],
    "WNBA":[
        "player_points","player_rebounds","player_assists","player_threes",
        "player_points_rebounds_assists","player_points_rebounds",
        "player_points_assists","player_rebounds_assists",
    ],
    "NFL":[
        "player_pass_yds","player_pass_tds","player_pass_interceptions",
        "player_rush_yds","player_rush_attempts",
        "player_reception_yds","player_receptions","player_reception_tds",
    ],
}

ODDS_API_MARKET_TO_PROP={
    "player_points":"Points",
    "player_rebounds":"Rebounds",
    "player_assists":"Assists",
    "player_threes":"3-Pointers Made",
    "player_points_rebounds_assists":"PRA",
    "player_points_rebounds":"PR",
    "player_points_assists":"PA",
    "player_rebounds_assists":"RA",
    "player_pass_yds":"Passing Yards",
    "player_pass_tds":"Passing TDs",
    "player_pass_interceptions":"Interceptions Thrown",
    "player_rush_yds":"Rushing Yards",
    "player_rush_attempts":"Rushing Attempts",
    "player_reception_yds":"Receiving Yards",
    "player_receptions":"Receptions",
    "player_reception_tds":"Receiving TDs",
}


def get_the_odds_api_key():
    key=str(os.getenv("THE_ODDS_API_KEY","") or "").strip()
    if not key:
        try:
            key=str(st.secrets.get("THE_ODDS_API_KEY","") or "").strip()
        except Exception:
            key=""
    return key


def _odds_team_key_v37(value):
    return re.sub(r"[^a-z0-9]","",unicodedata.normalize("NFKD",str(value or "")).encode("ascii","ignore").decode("ascii").lower())


def _odds_team_tokens_v37(value):
    s=unicodedata.normalize("NFKD",str(value or "")).encode("ascii","ignore").decode("ascii").lower()
    return [x for x in re.findall(r"[a-z0-9]+",s) if x]


def _odds_team_match_score_v37(a,b):
    ka,kb=_odds_team_key_v37(a),_odds_team_key_v37(b)
    if ka and ka==kb:
        return 1.0
    ta,tb=_odds_team_tokens_v37(a),_odds_team_tokens_v37(b)
    if not ta or not tb:
        return 0.0
    # Mascot/team nickname is usually the final token and is highly stable
    # across ESPN vs sportsbook naming conventions (e.g. LA vs Los Angeles).
    if ta[-1]==tb[-1]:
        return 0.90
    sa,sb=set(ta),set(tb)
    overlap=len(sa & sb)/max(1,len(sa | sb))
    if overlap>=.66:
        return .82
    if ka in kb or kb in ka:
        return .72
    return overlap*.65


def _odds_api_get_v37(url,params,timeout=10):
    """Single-request Odds API helper. Never retries an HTTP error and burns credits twice."""
    try:
        r=requests.get(url,params=params,timeout=timeout)
    except Exception as e:
        return None,{},f"The Odds API request failed: {e}"
    meta={
        "remaining":r.headers.get("x-requests-remaining"),
        "used":r.headers.get("x-requests-used"),
        "last":r.headers.get("x-requests-last"),
    }
    if r.status_code>=400:
        detail=""
        try:
            payload=r.json()
            detail=str(payload.get("message") or payload.get("error") or payload)
        except Exception:
            detail=r.text[:300]
        return None,meta,f"The Odds API HTTP {r.status_code}: {detail}"
    try:
        return r.json(),meta,None
    except Exception as e:
        return None,meta,f"The Odds API returned invalid JSON: {e}"


@st.cache_data(ttl=300,show_spinner=False)
def fetch_the_odds_events_v37(league,api_key):
    sport=ODDS_API_SPORT_KEYS.get(str(league).upper())
    if not sport:
        return [],{},f"Unsupported league for live props: {league}"
    if not api_key:
        return [],{},"THE_ODDS_API_KEY is not configured in Streamlit Secrets."
    url=f"https://api.the-odds-api.com/v4/sports/{sport}/events"
    payload,meta,err=_odds_api_get_v37(url,{"apiKey":api_key,"dateFormat":"iso"},timeout=8)
    if err:
        return [],meta,err
    return payload if isinstance(payload,list) else [],meta,None


def match_the_odds_event_v37(game,events):
    if not events:
        return None,"The Odds API returned no live/upcoming events for this league."
    gh=str(game.get("home_name") or game.get("home_abbr") or "")
    ga=str(game.get("away_name") or game.get("away_abbr") or "")
    selected=st.session_state.get("selected_date",date.today())
    sd=pd.to_datetime(selected,errors="coerce",utc=True)
    best=None; best_score=-1.0
    for ev in events:
        hs=_odds_team_match_score_v37(gh,ev.get("home_team",""))
        as_=_odds_team_match_score_v37(ga,ev.get("away_team",""))
        if hs<.70 or as_<.70:
            continue
        score=hs+as_
        et=pd.to_datetime(ev.get("commence_time"),errors="coerce",utc=True)
        if not pd.isna(sd) and not pd.isna(et):
            day_gap=abs((et.normalize()-sd.normalize()).days)
            score-=min(day_gap,7)*.06
        if score>best_score:
            best_score=score; best=ev
    if best is None:
        return None,(
            "Could not safely match the selected ESPN game to a live Odds API event. "
            "The app will not guess a different game."
        )
    return best,None


@st.cache_data(ttl=300,show_spinner=False)
def fetch_the_odds_event_props_v37(league,event_id,api_key):
    sport=ODDS_API_SPORT_KEYS.get(str(league).upper())
    markets=ODDS_API_MARKETS.get(str(league).upper(),[])
    if not sport or not markets:
        return {},{},f"No live prop market map is configured for {league}."
    if not api_key:
        return {},{},"THE_ODDS_API_KEY is not configured in Streamlit Secrets."
    url=f"https://api.the-odds-api.com/v4/sports/{sport}/events/{event_id}/odds"
    params={
        "apiKey":api_key,
        "bookmakers":",".join(ODDS_API_BOOKMAKERS),
        "markets":",".join(markets),
        "oddsFormat":"american",
        "dateFormat":"iso",
    }
    payload,meta,err=_odds_api_get_v37(url,params,timeout=12)
    if err:
        return {},meta,err
    return payload if isinstance(payload,dict) else {},meta,None


def _best_american_price_v37(rows):
    valid=[r for r in rows if not pd.isna(safe_float(r.get("price")))]
    if not valid:
        return np.nan,""
    best=max(valid,key=lambda r:float(safe_float(r.get("price"))))
    return float(safe_float(best.get("price"))),str(best.get("book") or "")


def parse_the_odds_props_v37(payload,league,roster):
    """Normalize sportsbook props and require an exact selected-game roster identity match."""
    if not payload or roster is None or roster.empty:
        return [],[{"Status":"No live props","Detail":"No sportsbook payload or selected-game roster was available."}]
    r=roster.copy()
    r["_player_key_v37"]=r.get("player",pd.Series("",index=r.index)).map(_name_key)
    # Duplicate normalized names in one selected-game roster are ambiguous; do not guess.
    counts=r["_player_key_v37"].value_counts()
    unique_keys=set(counts[counts==1].index)
    rmap={k:r[r["_player_key_v37"]==k].iloc[0] for k in unique_keys if k}
    entries=[]; diagnostics=[]
    supported=set(ODDS_API_MARKETS.get(str(league).upper(),[]))
    for book in payload.get("bookmakers",[]) or []:
        bkey=str(book.get("key") or "")
        btitle=str(book.get("title") or bkey or "Book")
        for market in book.get("markets",[]) or []:
            mkey=str(market.get("key") or "")
            if mkey not in supported or mkey not in ODDS_API_MARKET_TO_PROP:
                continue
            prop=ODDS_API_MARKET_TO_PROP[mkey]
            for outcome in market.get("outcomes",[]) or []:
                side=str(outcome.get("name") or "").title()
                if side not in {"Over","Under"}:
                    continue
                pname=str(outcome.get("description") or "").strip()
                pkey=_name_key(pname)
                if not pname or pkey not in rmap:
                    diagnostics.append({
                        "Status":"Unmatched sportsbook player",
                        "Detail":f"{pname or 'Unnamed player'} ({prop}) was not an exact unique match in the selected-game roster; skipped.",
                    })
                    continue
                point=safe_float(outcome.get("point")); price=safe_float(outcome.get("price"))
                if pd.isna(point) or pd.isna(price):
                    continue
                rr=rmap[pkey]
                entries.append({
                    "player":str(rr.get("player",pname)),"player_key":pkey,
                    "team":str(rr.get("team_abbr","")).upper(),
                    "team_id":str(rr.get("team_id","")),"position":str(rr.get("position","")),
                    "athlete_id":str(rr.get("athlete_id","")),"player_id":str(rr.get("player_id","")),
                    "market_key":mkey,"stat":prop,"side":side,"line":float(point),"price":float(price),
                    "book":btitle,"book_key":bkey,"last_update":market.get("last_update") or book.get("last_update") or "",
                })
    return entries,diagnostics


def consensus_live_prop_markets_v37(entries):
    """Collapse books to the most commonly posted line per player/stat, with de-vig consensus."""
    if not entries:
        return []
    df=pd.DataFrame(entries)
    out=[]
    for (pkey,stat),g in df.groupby(["player_key","stat"],dropna=False):
        # Prefer the line posted by the largest number of distinct books.
        line_counts=(g.groupby("line")["book_key"].nunique().sort_values(ascending=False))
        if line_counts.empty:
            continue
        maxn=line_counts.iloc[0]
        candidates=sorted([float(x) for x,v in line_counts.items() if v==maxn])
        line=float(candidates[len(candidates)//2])
        z=g[np.isclose(pd.to_numeric(g["line"],errors="coerce"),line)].copy()
        paired=[]
        for bkey,bg in z.groupby("book_key"):
            ovs=bg[bg["side"]=="Over"]; uns=bg[bg["side"]=="Under"]
            if ovs.empty or uns.empty:
                continue
            op=safe_float(ovs.iloc[0].get("price")); up=safe_float(uns.iloc[0].get("price"))
            oi,ui=american_to_implied(op),american_to_implied(up)
            if pd.isna(oi) or pd.isna(ui) or oi+ui<=0:
                continue
            paired.append({"over":oi/(oi+ui),"under":ui/(oi+ui),"book":str(bg.iloc[0].get("book",""))})
        first=z.iloc[0]
        for side in ["Over","Under"]:
            sr=z[z["side"]==side]
            if sr.empty:
                continue
            best_price,best_book=_best_american_price_v37(sr.to_dict("records"))
            raw_probs=[american_to_implied(x) for x in pd.to_numeric(sr["price"],errors="coerce").dropna().tolist()]
            raw_probs=[x for x in raw_probs if not pd.isna(x)]
            if paired:
                fair=float(np.mean([x[side.lower()] for x in paired]))
            else:
                fair=float(np.median(raw_probs)) if raw_probs else np.nan
            out.append({
                "Player":str(first.get("player","")),"Team":str(first.get("team","")),
                "Team ID":str(first.get("team_id","")),"Position":str(first.get("position","")),
                "Athlete ID":str(first.get("athlete_id","")),"Player ID":str(first.get("player_id","")),
                "Stat":stat,"Market key":str(first.get("market_key","")),"Side":side,"Line value":line,
                "Best odds":best_price,"Best book":best_book,"Market probability":fair,
                "Break-even":american_to_implied(best_price),"Books":int(sr["book_key"].nunique()),
                "Consensus books":len(paired),
                "Last update":str(sr["last_update"].dropna().max() if "last_update" in sr.columns and not sr.empty else ""),
            })
    return out


def scan_live_sportsbook_props_v37(game,league,roster,ctx,progress_cb=None):
    key=get_the_odds_api_key()
    if not key:
        return [],[],{},"THE_ODDS_API_KEY is missing from Streamlit Secrets."
    if progress_cb:
        progress_cb(0,4,"","matching selected game")
    events,emeta,eerr=fetch_the_odds_events_v37(league,key)
    if eerr:
        return [],[],emeta,eerr
    event,merr=match_the_odds_event_v37(game,events)
    if merr:
        return [],[],emeta,merr
    if progress_cb:
        progress_cb(1,4,"","fetching current sportsbook player props")
    payload,pmeta,perr=fetch_the_odds_event_props_v37(league,str(event.get("id","")),key)
    meta={**emeta,**{k:v for k,v in pmeta.items() if v is not None},"event":event}
    if perr:
        return [],[],meta,perr
    entries,diagnostics=parse_the_odds_props_v37(payload,league,roster)
    markets=consensus_live_prop_markets_v37(entries)
    if not markets:
        return [],diagnostics,meta,"No verified current player over/under props were returned for this game."

    # Fetch each team injury feed once. Historical evaluation stays local for speed.
    injury_cache={}
    for team_id in roster.get("team_id",pd.Series(dtype=str)).astype(str).unique():
        if team_id:
            injury_cache[team_id]=fetch_injuries(league,team_id,game.get("event_id",""))

    ideas=[]
    player_cache={}
    total=len(markets)
    for i,mkt in enumerate(markets,1):
        pname=str(mkt["Player"]); team=str(mkt["Team"]); side=str(mkt["Side"])
        if progress_cb:
            progress_cb(2+i/max(total,1),4,pname,f"checking {mkt['Stat']} {side.lower()} {mkt['Line value']:g}")
        cache_key=(_name_key(pname),team.upper())
        base=player_cache.get(cache_key)
        if base is None:
            match=roster[(roster["player"].map(_name_key)==_name_key(pname)) & (roster["team_abbr"].astype(str).str.upper()==team.upper())]
            if len(match)!=1:
                diagnostics.append({"Player":pname,"Status":"Identity check failed","Detail":"Current sportsbook player did not resolve to exactly one selected-game roster row."})
                player_cache[cache_key]={"error":True}
                continue
            row=match.iloc[0]
            d=_full_player_game_data(game,league,row,fast_scan=True)
            if d.get("error") or d["log"].empty:
                diagnostics.append({"Player":pname,"Status":"No verified history","Detail":str(d.get("error") or "No local stat-bearing player history.")})
                player_cache[cache_key]={"error":True}
                continue
            props=available_props(d["log"],league)
            today_side="home" if team==game.get("home_abbr") else "away"
            injuries,inj_err=injury_cache.get(str(row.get("team_id","")),(pd.DataFrame(),"No injury source"))
            team_ctx=(ctx or {}).get(today_side,{})
            if league=="NFL":
                workload=nfl_workload_info_v33(d["log"],injuries,pname,injury_verified=team_ctx.get("injury_verified",inj_err is None))
                workload_text=nfl_workload_summary_text_v33(workload)
            else:
                workload=estimate_player_availability_minutes(d["log"],injuries,pname,injury_verified=team_ctx.get("injury_verified",inj_err is None))
                workload_text=minutes_summary_text(workload)
            base={"error":False,"row":row,"d":d,"props":props,"today_side":today_side,"team_ctx":team_ctx,"workload":workload,"workload_text":workload_text}
            player_cache[cache_key]=base
        if base.get("error"):
            continue
        row=base["row"]; d=base["d"]; props=base["props"]; today_side=base["today_side"]; team_ctx=base["team_ctx"]; workload=base["workload"]; workload_text=base["workload_text"]
        stat=str(mkt["Stat"])
        if stat not in props:
            diagnostics.append({"Player":pname,"Status":"Unsupported stat","Detail":f"Verified local history did not contain the columns needed for {stat}."})
            continue
        analysis=easy_prop_analysis(
            d["log"],d["matchup"],league,props[stat],side,float(mkt["Line value"]),
            market_probability=mkt.get("Market probability"),current_side=today_side,opponent=d["opponent"],
        )
        fit=player_vs_team_fit(game,league,d["log"],pname,d["position"],d["opponent"],stat)
        quality=research_data_quality(d["log"],d["matchup"],team_ctx.get("injury_verified",False),workload,analysis)
        flags=automatic_red_flags(d["log"],d["matchup"],workload,analysis)
        model_p=analysis.get("probability",np.nan)
        fair=mkt.get("Market probability",np.nan); be=mkt.get("Break-even",np.nan)
        ideas.append({
            "Player":pname,"Team":team,"Position":d["position"],"Opponent":d["opponent"],
            "Stat":stat,"Line":f"{side} {float(mkt['Line value']):g}","Line value":float(mkt["Line value"]),"Market side":side,
            "Model probability":model_p,"Last 10":analysis.get("last10",np.nan),"Last 20":analysis.get("last20",np.nan),
            "Vs opponent":analysis.get("vs_opponent",np.nan),"Backtest N":analysis.get("backtest_n",0),"Brier":analysis.get("brier",np.nan),
            "Minutes":workload_text,"Data quality":quality["label"],"Data quality score":quality["score"],
            "Helps":fit["helps"],"Hurts":fit["hurts"],"Player strengths":fit["player_strengths"],"Player weaknesses":fit["player_weaknesses"],
            "Opponent weaknesses":fit["opponent_weaknesses"],"Opponent strengths":fit["opponent_strengths"],"Red flags":flags,
            "Volatility":"Supported","Live market":True,"Market probability":fair,"Break-even":be,
            "Market edge":(model_p-fair if not pd.isna(model_p) and not pd.isna(fair) else np.nan),
            "Price edge":(model_p-be if not pd.isna(model_p) and not pd.isna(be) else np.nan),
            "Best odds":mkt.get("Best odds",np.nan),"Best book":mkt.get("Best book",""),"Books":mkt.get("Books",0),
            "Consensus books":mkt.get("Consensus books",0),"Market key":mkt.get("Market key",""),"Odds event ID":str(event.get("id","")),
            "Odds event":f"{event.get('away_team','')} at {event.get('home_team','')}","Odds last update":mkt.get("Last update",""),
            "Market verdict":analysis.get("verdict",""),"Market verdict text":analysis.get("verdict_text",""),
        })
    if progress_cb:
        progress_cb(4,4,"","ranking real sportsbook lines")
    # Market edge first, then evidence quality. Negative/unknown edges stay at the bottom.
    ideas=sorted(ideas,key=lambda r:(
        -float(r.get("Market edge",-9) if not pd.isna(r.get("Market edge",np.nan)) else -9),
        -int(r.get("Data quality score",0) or 0),
        -float(r.get("Model probability",0) if not pd.isna(r.get("Model probability",np.nan)) else 0),
    ))
    return ideas,diagnostics,meta,None

@st.cache_data(ttl=180, show_spinner=False)
def fetch_espn_moneyline_consensus(league, event_id):
    cfg = LEAGUES[league]
    url = (
        f"https://sports.core.api.espn.com/v2/sports/{cfg['sport']}"
        f"/leagues/{cfg['league']}/events/{event_id}/competitions/{event_id}/odds"
    )
    try:
        payload = request_json(url, {"limit": 100}, timeout=20)
    except Exception as e:
        return None, f"Sportsbook odds request failed: {e}"

    books = []
    for item in payload.get("items", []) or []:
        home = item.get("homeTeamOdds") or {}
        away = item.get("awayTeamOdds") or {}
        ph = american_to_implied(home.get("moneyLine"))
        pa = american_to_implied(away.get("moneyLine"))
        if pd.isna(ph) or pd.isna(pa) or ph + pa <= 0:
            continue
        total = ph + pa
        provider = (item.get("provider") or {}).get("name") or "Book"
        books.append({
            "provider": provider,
            "home_p": ph / total,
            "away_p": pa / total,
            "home_ml":safe_float(home.get("moneyLine")),
            "away_ml":safe_float(away.get("moneyLine")),
        })

    if not books:
        return None, "No usable two-sided sportsbook moneylines were returned."

    df = pd.DataFrame(books)
    return {
        "home_probability": float(df["home_p"].mean()),
        "away_probability": float(df["away_p"].mean()),
        "books": len(df),
        "providers": ", ".join(df["provider"].astype(str).drop_duplicates().tolist()),
        "quotes":books,
    }, None

def find_espn_game_for_market(league, market, schedule_cache):
    d = kalshi_event_date(market)
    outcome = kalshi_market_outcome_code(market)
    if d is None or not outcome:
        return None, None
    key = (league, d.isoformat())
    if key not in schedule_cache:
        schedule_cache[key] = fetch_schedule(league, d)
    games, err = schedule_cache[key]
    if err:
        return None, err
    for game in games:
        if outcome in {normalize_team_code(game.get("home_abbr")), normalize_team_code(game.get("away_abbr"))}:
            return game, None
    return None, None

def current_nba_team_probability(game):
    if not NBA_HIST_FILE.exists():
        return np.nan, "Historical NBA model not loaded."
    try:
        pack, _, err = build_nba_model(NBA_HIST_FILE.stat().st_mtime)
        if err or not pack:
            return np.nan, err or "NBA model unavailable."
        model, feats, _ = pack
        season = league_season_year("NBA", date.today())
        hdf, herr = fetch_team_recent("NBA", game["home_id"], season)
        adf, aerr = fetch_team_recent("NBA", game["away_id"], season)
        if herr or aerr or hdf.empty or adf.empty:
            return np.nan, "Not enough current team form."
        h, a = hdf.head(10), adf.head(10)
        if len(h) < 5 or len(a) < 5:
            return np.nan, "Not enough current games."
        game_dt = pd.to_datetime(game.get("date"), errors="coerce", utc=True)
        h_last = pd.to_datetime(h.iloc[0]["date"], errors="coerce", utc=True)
        a_last = pd.to_datetime(a.iloc[0]["date"], errors="coerce", utc=True)
        h_rest = (game_dt - h_last).days if not pd.isna(game_dt) and not pd.isna(h_last) else 2
        a_rest = (game_dt - a_last).days if not pd.isna(game_dt) and not pd.isna(a_last) else 2
        row = pd.DataFrame([{
            "offense_diff": h["points_for"].mean() - a["points_for"].mean(),
            "defense_diff": a["points_against"].mean() - h["points_against"].mean(),
            "form_diff": (h["result"] == "W").mean() - (a["result"] == "W").mean(),
            "rest_diff": np.clip(h_rest, 0, 14) - np.clip(a_rest, 0, 14),
        }])
        p_home = float(model.predict_proba(row[feats])[:, 1][0])
        return p_home, None
    except Exception as e:
        return np.nan, str(e)

def scan_kalshi_sports_markets(leagues, discrepancy_threshold=.05, fee_buffer=.02, min_books=2):
    schedule_cache, odds_cache = {}, {}
    results, inventory, internal, errors = [], [], [], []
    event_groups = {}

    for league in leagues:
        for market_type, series in KALSHI_SPORT_SERIES.get(league, {}).items():
            markets, err = fetch_kalshi_series_markets(series)
            if err:
                errors.append(err)
            for market in markets:
                ask, bid = kalshi_yes_ask(market), kalshi_yes_bid(market)
                spread = kalshi_market_spread(market)
                inventory.append({
                    "League": league,
                    "Type": market_type,
                    "Ticker": market.get("ticker", ""),
                    "Title": market.get("title") or market.get("subtitle") or market.get("yes_sub_title") or "",
                    "YES bid": bid,
                    "YES ask": ask,
                    "Spread": spread,
                    "Volume 24h": safe_float(market.get("volume_24h_fp") or market.get("volume_24h")),
                })
                if market_type != "Game winner" or pd.isna(ask):
                    continue

                event_ticker = str(market.get("event_ticker") or "")
                event_groups.setdefault((league, event_ticker), []).append(market)
                game, match_err = find_espn_game_for_market(league, market, schedule_cache)
                if match_err:
                    errors.append(match_err)
                if not game:
                    continue

                key = (league, game["event_id"])
                if key not in odds_cache:
                    odds_cache[key] = fetch_espn_moneyline_consensus(league, game["event_id"])
                consensus, cerr = odds_cache[key]
                if cerr or not consensus:
                    if cerr:
                        errors.append(f"{league} {game['away_abbr']}@{game['home_abbr']}: {cerr}")
                    continue

                outcome = kalshi_market_outcome_code(market)
                home_code = normalize_team_code(game["home_abbr"])
                away_code = normalize_team_code(game["away_abbr"])
                if outcome == home_code:
                    ref_p = consensus["home_probability"]
                    side = game["home_name"]
                elif outcome == away_code:
                    ref_p = consensus["away_probability"]
                    side = game["away_name"]
                else:
                    continue

                model_p = np.nan
                if league == "NBA":
                    p_home, _ = current_nba_team_probability(game)
                    if not pd.isna(p_home):
                        model_p = p_home if outcome == home_code else 1 - p_home

                raw_gap = ref_p - ask
                adj_gap = ref_p - (ask + fee_buffer)
                if consensus["books"] >= min_books and adj_gap >= discrepancy_threshold:
                    flag = "Potentially underpriced"
                elif consensus["books"] >= min_books and adj_gap > 0:
                    flag = "Small positive discrepancy"
                elif consensus["books"] >= min_books and adj_gap <= -discrepancy_threshold:
                    flag = "Kalshi above consensus"
                else:
                    flag = "No strong discrepancy"

                agreement = ""
                if not pd.isna(model_p):
                    if model_p > ask + fee_buffer and ref_p > ask + fee_buffer:
                        agreement = "Sportsbook consensus and the NBA model both price this side above Kalshi."
                    elif model_p < ask and ref_p < ask:
                        agreement = "Sportsbook consensus and the NBA model both price this side below Kalshi."
                    else:
                        agreement = "The NBA model and sportsbook consensus disagree, so confidence should be lower."

                reasons = (
                    f"Kalshi YES ask is {ask*100:.1f}¢ (~{ask*100:.1f}%). "
                    f"The de-vig sportsbook consensus is {ref_p*100:.1f}% across {consensus['books']} books. "
                    f"That is a raw difference of {raw_gap*100:+.1f} percentage points and "
                    f"{adj_gap*100:+.1f} points after a {fee_buffer*100:.1f}¢ fee/slippage buffer."
                )
                if not pd.isna(model_p):
                    reasons += f" The dashboard's historical NBA model gives a {model_p*100:.1f}% second opinion."
                if agreement:
                    reasons += " " + agreement
                if not pd.isna(spread):
                    reasons += f" Kalshi's visible YES bid/ask spread is about {spread*100:.1f}¢."

                results.append({
                    "League": league,
                    "Game": f"{game['away_abbr']} at {game['home_abbr']}",
                    "Outcome": side,
                    "Ticker": market.get("ticker", ""),
                    "Kalshi ask": ask,
                    "Consensus probability": ref_p,
                    "NBA model": model_p,
                    "Raw gap": raw_gap,
                    "Adjusted gap": adj_gap,
                    "Books": consensus["books"],
                    "Providers": consensus["providers"],
                    "Kalshi spread": spread,
                    "Flag": flag,
                    "Why": reasons,
                })

    # Two-outcome internal price mismatch check.
    for (league, event_ticker), markets in event_groups.items():
        priced = [(m, kalshi_yes_ask(m)) for m in markets]
        priced = [(m, p) for m, p in priced if not pd.isna(p)]
        if len(priced) != 2:
            continue
        total_ask = priced[0][1] + priced[1][1]
        cushion = 1.0 - total_ask - (2 * fee_buffer)
        if cushion > 0:
            internal.append({
                "League": league,
                "Event": event_ticker,
                "Side 1": priced[0][0].get("yes_sub_title") or kalshi_market_outcome_code(priced[0][0]),
                "Side 2": priced[1][0].get("yes_sub_title") or kalshi_market_outcome_code(priced[1][0]),
                "Combined asks": total_ask,
                "After buffer": cushion,
                "Why": (
                    f"Both YES asks total {total_ask*100:.1f}¢. After a {fee_buffer*100:.1f}¢ buffer on each side, "
                    f"the theoretical cushion is {cushion*100:.1f}¢. Verify the two contracts are exhaustive/opposite outcomes "
                    "and both prices are actually fillable before treating this as real."
                ),
            })

    return (
        sorted(results, key=lambda x: x["Adjusted gap"], reverse=True),
        inventory,
        sorted(internal, key=lambda x: x["After buffer"], reverse=True),
        errors,
    )

def scanner_display_df(rows):
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).copy()
    for c in ["Kalshi ask", "Consensus probability", "NBA model"]:
        if c in df.columns:
            df[c] = df[c].map(lambda x: "—" if pd.isna(x) else f"{x*100:.1f}%")
    for c in ["Raw gap", "Adjusted gap"]:
        if c in df.columns:
            df[c] = df[c].map(lambda x: "—" if pd.isna(x) else f"{x*100:+.1f} pts")
    if "Kalshi spread" in df.columns:
        df["Kalshi spread"] = df["Kalshi spread"].map(lambda x: "—" if pd.isna(x) else f"{x*100:.1f}¢")
    return df

def inventory_display_df(rows):
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).copy()
    for c in ["YES bid", "YES ask", "Spread"]:
        if c in df.columns:
            df[c] = df[c].map(lambda x: "—" if pd.isna(x) else f"{x*100:.1f}¢")
    return df


def contract_scenario(price_probability, reference_probability, stake_dollars):
    if pd.isna(price_probability) or price_probability<=0 or price_probability>=1:
        return {}
    contracts=int(stake_dollars//price_probability)
    if contracts<1:
        return {}
    actual_cost=contracts*price_probability
    payout=contracts*1.0
    return {
        "contracts":contracts,
        "cost":actual_cost,
        "payout":payout,
        "profit_if_win":payout-actual_cost,
        "max_loss":actual_cost,
        "expected_profit":contracts*(reference_probability-price_probability) if not pd.isna(reference_probability) else np.nan,
    }


# ============================================================
# SIMPLE HISTORICAL NBA TEAM MODEL
# ============================================================

def guess_col(cols,candidates):
    cols=list(cols)
    low={c.lower():c for c in cols}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    for c in cols:
        if any(cand.lower() in c.lower() for cand in candidates):
            return c
    return None

@st.cache_resource(show_spinner=False)
def build_nba_model(mtime):
    games=pd.read_csv(NBA_HIST_FILE)
    d=guess_col(games.columns,["game_date","date"])
    hteam=guess_col(games.columns,["team_abbreviation_home","home_team"])
    ateam=guess_col(games.columns,["team_abbreviation_away","away_team"])
    hpts=guess_col(games.columns,["pts_home","home_pts","home_score"])
    apts=guess_col(games.columns,["pts_away","away_pts","away_score"])
    if any(c is None for c in [d,hteam,ateam,hpts,apts]):
        return None,None,"Historical NBA columns not recognized."
    g=games[[d,hteam,ateam,hpts,apts]].copy()
    g[d]=pd.to_datetime(g[d],errors="coerce")
    g[hpts]=pd.to_numeric(g[hpts],errors="coerce")
    g[apts]=pd.to_numeric(g[apts],errors="coerce")
    g=g.dropna().sort_values(d).reset_index(drop=True)

    home=pd.DataFrame({"idx":g.index,"date":g[d],"team":g[hteam].astype(str),"pf":g[hpts],"pa":g[apts],"home":1})
    away=pd.DataFrame({"idx":g.index,"date":g[d],"team":g[ateam].astype(str),"pf":g[apts],"pa":g[hpts],"home":0})
    tg=pd.concat([home,away],ignore_index=True).sort_values(["team","date","idx"])
    tg["won"]=(tg["pf"]>tg["pa"]).astype(int)
    grp=tg.groupby("team",group_keys=False)
    tg["off"]=grp["pf"].transform(lambda s:s.shift(1).rolling(10,min_periods=5).mean())
    tg["def"]=grp["pa"].transform(lambda s:s.shift(1).rolling(10,min_periods=5).mean())
    tg["form"]=grp["won"].transform(lambda s:s.shift(1).rolling(10,min_periods=5).mean())
    tg["prev"]=grp["date"].shift(1)
    tg["rest"]=(tg["date"]-tg["prev"]).dt.days.clip(0,14)
    hh=tg[tg["home"]==1].set_index("idx")
    aa=tg[tg["home"]==0].set_index("idx")
    m=pd.DataFrame({
        "date":g[d],
        "home_team":g[hteam].astype(str),
        "away_team":g[ateam].astype(str),
        "offense_diff":hh["off"]-aa["off"],
        "defense_diff":aa["def"]-hh["def"],
        "form_diff":hh["form"]-aa["form"],
        "rest_diff":hh["rest"]-aa["rest"],
        "home_won":(g[hpts]>g[apts]).astype(int),
    }).dropna()
    if len(m)<100:
        return None,m,"Not enough usable NBA games."
    feats=["offense_diff","defense_diff","form_diff","rest_diff"]
    cut=int(len(m)*.8)
    train=m.iloc[:cut]
    test=m.iloc[cut:].copy()
    model=Pipeline([("scale",StandardScaler()),("lr",LogisticRegression(max_iter=2000))])
    train_w=recency_training_weights(train["date"],half_life_days=420)
    model.fit(train[feats],train["home_won"],lr__sample_weight=train_w)
    p=model.predict_proba(test[feats])[:,1]
    test["model_p"]=p
    metrics={
        "accuracy":accuracy_score(test["home_won"],p>=.5),
        "brier":brier_score_loss(test["home_won"],p),
        "logloss":log_loss(test["home_won"],np.clip(p,.001,.999)),
    }
    return (model,feats,metrics),test,None


# ============================================================
# RESEARCH PANEL
# ============================================================

def render_player_research(gamelog,player_name,league,player_team="",opponent="",game=None,injury_df=None):
    if gamelog is None or gamelog.empty:
        st.warning("No player game log loaded.")
        return

    props=available_props(gamelog,league)
    if not props:
        st.warning("Could not recognize usable stat columns for this player.")
        with st.expander("Raw columns"):
            st.write(gamelog.columns.tolist())
        return

    st.markdown(f"## {player_name}")
    if player_team or opponent:
        st.caption(f"{player_team or 'Team'} • upcoming opponent: {opponent or 'not selected'}")

    c1,c2,c3=st.columns(3)
    with c1:
        prop=st.selectbox("Stat / prop",list(props.keys()),key=f"prop_{player_name}_{league}")
    with c2:
        side=st.selectbox("Bet type",["Over","Under","At least (X+)","At most"],key=f"side_{player_name}_{league}")
    with c3:
        default=5.0 if side=="At least (X+)" else 20.5
        line=st.number_input("Line / threshold",value=float(default),step=.5,key=f"line_{player_name}_{league}")

    vals=extract_values(gamelog,props[prop])
    current_side=None
    if game and player_team:
        if player_team==game.get("home_abbr"):
            current_side="home"
        elif player_team==game.get("away_abbr"):
            current_side="away"

    proj=player_projection(gamelog,vals,league,current_side=current_side,opponent=opponent)
    raw_p=normal_prob(proj["mu"],proj["sigma"],side,line)
    bt_p,bt_y=rolling_prop_backtest(gamelog,vals,league,side,line)
    cal=calibrate_current(raw_p,bt_p,bt_y) if not pd.isna(raw_p) else {
        "probability":np.nan,"calibrated":False,"samples":0,"accuracy":np.nan,"brier":np.nan,"logloss":np.nan
    }
    model_p=cal["probability"]
    confidence=confidence_label(cal)

    opp_mask=pd.Series(False,index=gamelog.index)
    if opponent and "OPP" in gamelog.columns:
        opp_mask=gamelog["OPP"].astype(str).str.upper().str.contains(opponent.upper(),regex=False)
    opp_vals=vals[opp_mask] if opp_mask.any() else pd.Series(dtype=float)

    h1,h2,h3,h4,h5,h6=st.columns(6)
    h1.metric("Last 5 hit",fmt_pct(hit_rate(vals.head(5),side,line)))
    h2.metric("Last 10",fmt_pct(hit_rate(vals.head(10),side,line)))
    h3.metric("Last 20",fmt_pct(hit_rate(vals.head(20),side,line)))
    h4.metric("Season",fmt_pct(hit_rate(vals,side,line)))
    h5.metric(f"Vs {opponent}" if opponent else "Vs opponent","—" if opp_vals.empty else fmt_pct(hit_rate(opp_vals,side,line)))
    h6.metric("Model probability",fmt_pct(model_p))

    a1,a2,a3,a4,a5=st.columns(5)
    a1.metric("Projection",fmt_num(proj["mu"]))
    a2.metric("Expected minutes",fmt_num(proj["expected_minutes"]))
    a3.metric("Last 10 avg",fmt_num(vals.head(10).mean()))
    a4.metric("Season avg",fmt_num(vals.mean()))
    a5.metric("Model confidence",confidence)

    if not opp_vals.empty:
        st.caption(
            f"Against {opponent}: {len(opp_vals)} prior games in this log • "
            f"average {opp_vals.mean():.1f} • range {opp_vals.min():.0f}–{opp_vals.max():.0f}"
        )

    if injury_df is not None and not injury_df.empty:
        match=injury_df[injury_df["Player"].astype(str).str.lower()==player_name.lower()]
        if not match.empty:
            st.error("Availability flag: " + " | ".join(
                f"{r.Status}: {r.Detail}" for r in match.itertuples()
            ))

    if proj["reasons"]:
        st.markdown("**What moved the projection:** " + "; ".join(proj["reasons"]))

    st.markdown("### Out-of-sample model check")
    b1,b2,b3,b4=st.columns(4)
    b1.metric("Backtest samples",cal["samples"])
    b2.metric("Backtest accuracy",fmt_pct(cal["accuracy"]))
    b3.metric("Brier score","—" if pd.isna(cal["brier"]) else f"{cal['brier']:.3f}")
    b4.metric("Calibrated?", "Yes" if cal["calibrated"] else "Not enough data")
    st.caption("The rolling backtest only uses games that occurred before each tested game.")

    table=pd.DataFrame({
        "DATE":gamelog["DATE"] if "DATE" in gamelog.columns else gamelog.index,
        "MATCHUP":gamelog["MATCHUP"] if "MATCHUP" in gamelog.columns else gamelog["OPP"] if "OPP" in gamelog.columns else "",
        "VALUE":pd.to_numeric(vals,errors="coerce"),
        "MIN":pd.to_numeric(gamelog["MIN"],errors="coerce") if "MIN" in gamelog.columns else np.nan,
    }).dropna(subset=["VALUE"])
    table["HIT"]=np.where(hit_mask(table["VALUE"],side,line),"✅","—")
    st.markdown("### Recent games")
    st.dataframe(table.head(20),use_container_width=True,hide_index=True)

    st.markdown("### Market comparison")
    m1,m2=st.columns(2)
    with m1:
        ticker=st.text_input("Exact Kalshi ticker (optional)",key=f"ticker_{player_name}_{league}")
    market=None
    market_p=np.nan
    if ticker:
        market,err=fetch_kalshi_market(ticker)
        if err:
            st.caption(f"Kalshi lookup: {err}")
        else:
            market_p=kalshi_implied(market)
    with m2:
        default_price=50.0 if pd.isna(market_p) else float(market_p*100)
        price=st.number_input("Current YES / over-side price (cents)",1.0,99.0,default_price,step=1.0,key=f"price_{player_name}_{league}")

    fee=st.number_input("Estimated fee per contract ($)",0.0,1.0,0.00,step=.01,key=f"fee_{player_name}_{league}")
    market_p=price/100
    edge=model_p-market_p if not pd.isna(model_p) else np.nan
    ev=contract_ev(model_p,price,fee) if not pd.isna(model_p) else np.nan

    e1,e2,e3,e4=st.columns(4)
    e1.metric("Market implied",fmt_pct(market_p))
    e2.metric("Model probability",fmt_pct(model_p))
    e3.metric("Edge","—" if pd.isna(edge) else f"{edge*100:+.1f} pts")
    e4.metric("EV / contract","—" if pd.isna(ev) else f"${ev:+.3f}")

    # Research-signal classification: informative, not directive.
    recent10=hit_rate(vals.head(10),side,line)
    reasons=[]
    injury_flag=False
    if injury_df is not None and not injury_df.empty:
        injury_flag=not injury_df[injury_df["Player"].astype(str).str.lower()==player_name.lower()].empty

    if not pd.isna(edge):
        if edge>=.07 and confidence in {"Medium","Higher"} and (pd.isna(recent10) or recent10>=.50) and not injury_flag:
            signal="Stronger data support"
            reasons.append(f"model edge {edge*100:.1f} percentage points")
            reasons.append(f"{cal['samples']} rolling out-of-sample tests")
            if not pd.isna(recent10):
                reasons.append(f"last-10 hit rate {recent10*100:.0f}%")
        elif edge>0:
            signal="Higher-risk possibility"
            reasons.append(f"positive estimated edge {edge*100:.1f} points")
            if confidence=="Low":
                reasons.append("limited backtest sample")
            if injury_flag:
                reasons.append("availability/injury uncertainty")
        else:
            signal="No positive model edge"
            reasons.append("market price is at or above the model estimate")
    else:
        signal="Insufficient data"
        reasons.append("model probability could not be estimated")

    st.markdown("### Research signal")
    if signal=="Stronger data support":
        st.success(signal + " — " + "; ".join(reasons))
    elif signal=="Higher-risk possibility":
        st.warning(signal + " — " + "; ".join(reasons))
    else:
        st.info(signal + " — " + "; ".join(reasons))

    bankroll=load_json(SETTINGS_FILE,{"bankroll":1000.0,"kelly_fraction":.25,"max_pct":.02})
    kelly=fractional_kelly(model_p,price,bankroll["kelly_fraction"]) if not pd.isna(model_p) else 0
    cap=min(kelly,bankroll["max_pct"]) if not pd.isna(ev) and ev>0 else 0
    dollars=bankroll["bankroll"]*cap
    st.caption(
        f"Risk module: current saved bankroll {fmt_money(bankroll['bankroll'])}; "
        f"conservative cap for these inputs {fmt_money(dollars)}. "
        "This is a risk-control output, not an instruction to wager."
    )

    selection=f"{player_name} — {side} {line} {prop}"
    st.session_state["latest_selection"]=selection
    st.session_state["latest_model_p"]=model_p
    st.session_state["latest_market_price"]=price
    st.session_state["latest_edge"]=edge
    st.session_state["latest_ev"]=ev
    st.session_state["latest_confidence"]=confidence

    s1,s2=st.columns(2)
    with s1:
        if st.button("Save to My Bets",key=f"watch_{player_name}_{league}"):
            append_csv(WATCHLIST_FILE,{
                "saved_at":datetime.now().isoformat(timespec="seconds"),
                "league":league,
                "selection":selection,
                "opponent":opponent,
                "model_probability":model_p,
                "market_price_cents":price,
                "edge":edge,
                "estimated_ev":ev,
                "confidence":confidence,
                "status":"Watching",
                "result":"",
            })
            st.success("Saved.")
    with s2:
        if st.button("Record prediction",key=f"record_{player_name}_{league}"):
            append_csv(PREDICTIONS_FILE,{
                "timestamp":datetime.now().isoformat(timespec="seconds"),
                "league":league,
                "selection":selection,
                "opponent":opponent,
                "model_probability":model_p,
                "market_price_cents":price,
                "edge":edge,
                "estimated_ev":ev,
                "confidence":confidence,
                "outcome":"",
                "profit_loss":"",
            })
            st.success("Prediction recorded.")




# ============================================================
# V10: WEEKLY LIVE SCHEDULE / EASY RESEARCH / SOURCE LINKS
# ============================================================

@st.cache_data(ttl=120, show_spinner=False)
def fetch_week_schedule(league, start_date, days=7):
    rows=[]
    errors=[]
    for offset in range(days):
        d=start_date + timedelta(days=offset)
        games,err=fetch_schedule(league,d)
        if err:
            errors.append(f"{d.isoformat()}: {err}")
        for g in games:
            x=dict(g)
            x["schedule_date"]=d
            rows.append(x)
    return rows,errors

def weekly_game_label(game):
    d=game.get("schedule_date")
    if isinstance(d,date):
        day=d.strftime("%a %b %d")
    else:
        day=str(d or "")
    return f"{day} • {game['away_abbr']} @ {game['home_abbr']} • {game.get('status','')}"

def espn_game_page_url(league,event_id):
    slug={"NBA":"nba","WNBA":"wnba","NFL":"nfl"}.get(league,league.lower())
    return f"https://www.espn.com/{slug}/game/_/gameId/{event_id}"

def espn_scoreboard_source_url(league,selected_date):
    cfg=LEAGUES[league]
    return (
        f"https://site.api.espn.com/apis/site/v2/sports/{cfg['sport']}/{cfg['league']}/scoreboard"
        f"?dates={selected_date.strftime('%Y%m%d')}&limit=100"
    )

def espn_injury_source_url(league,team_id):
    cfg=LEAGUES[league]
    return (
        f"https://site.api.espn.com/apis/site/v2/sports/{cfg['sport']}/{cfg['league']}"
        f"/teams/{team_id}/injuries"
    )

def espn_odds_source_url(league,event_id):
    cfg=LEAGUES[league]
    return (
        f"https://sports.core.api.espn.com/v2/sports/{cfg['sport']}/leagues/{cfg['league']}"
        f"/events/{event_id}/competitions/{event_id}/odds?limit=100"
    )

def kalshi_source_url(ticker):
    return f"https://external-api.kalshi.com/trade-api/v2/markets/{ticker}"

def official_player_page_url(league,player_id):
    if not player_id:
        return None
    if league=="NBA":
        return f"https://www.nba.com/player/{player_id}"
    if league=="WNBA":
        return f"https://www.wnba.com/player/{player_id}"
    return f"https://www.espn.com/nfl/player/_/id/{player_id}"

def basketball_matchup_source_url(league,player_id,opponent_team_id,selected_date):
    if not player_id or not opponent_team_id:
        return None
    league_id="00" if league=="NBA" else "10"
    season=basketball_seasons_for_date(league,selected_date,1)[0]
    params={
        "LeagueID":league_id,
        "PlayerID":str(player_id),
        "OpponentTeamID":str(opponent_team_id),
        "Season":season,
        "SeasonType":"Regular Season",
        "PerMode":"Totals",
    }
    return "https://stats.nba.com/stats/playergamelogs?" + urlencode(params)

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_basketball_player_advanced(league,selected_date):
    try:
        from nba_api.stats.endpoints import leaguedashplayerstats
        league_id="00" if league=="NBA" else "10"
        season=nba_season_string(selected_date) if league=="NBA" else str(selected_date.year)
        obj=leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star="Regular Season",
            league_id_nullable=league_id,
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="PerGame",
            timeout=8,
        )
        return obj.get_data_frames()[0].copy(),None
    except Exception as e:
        return pd.DataFrame(),str(e)

def opponent_environment_summary(game,league,ctx,selected_date):
    notes=[]
    opponent_side=None
    player_team=st.session_state.get("research_team","")
    if player_team:
        opponent_side="home" if player_team==game.get("away_abbr") else "away"
    if opponent_side and ctx:
        opp=ctx.get(opponent_side,{})
        recent=opp.get("recent10",{})
        if recent:
            if recent.get("margin",0) < -3:
                notes.append(
                    f"{opp.get('abbr','Opponent')} has been getting outscored by {abs(recent['margin']):.1f} points per game over its recent sample."
                )
            elif recent.get("margin",0) > 3:
                notes.append(
                    f"{opp.get('abbr','Opponent')} has been strong recently with a {recent['margin']:+.1f} scoring margin."
                )

    if league in {"NBA","WNBA"}:
        adv,_=fetch_basketball_advanced(league,selected_date)
        if not adv.empty and "TEAM_ABBREVIATION" in adv.columns:
            opponent=st.session_state.get("research_opponent","")
            row=adv[adv["TEAM_ABBREVIATION"].astype(str).str.upper()==str(opponent).upper()]
            if not row.empty:
                r=row.iloc[0]
                dr=safe_float(r.get("DEF_RATING"))
                pace=safe_float(r.get("PACE"))
                league_dr=pd.to_numeric(adv.get("DEF_RATING"),errors="coerce").median() if "DEF_RATING" in adv.columns else np.nan
                league_pace=pd.to_numeric(adv.get("PACE"),errors="coerce").median() if "PACE" in adv.columns else np.nan
                if not pd.isna(dr) and not pd.isna(league_dr):
                    if dr > league_dr + 1.5:
                        notes.append(
                            f"{opponent}'s defensive rating is worse than the league median, which can help offensive counting stats."
                        )
                    elif dr < league_dr - 1.5:
                        notes.append(
                            f"{opponent}'s defensive rating is better than the league median, which can make scoring props tougher."
                        )
                if not pd.isna(pace) and not pd.isna(league_pace):
                    if pace > league_pace + 1.0:
                        notes.append(
                            f"{opponent} plays faster than the league median, creating more possessions and more chances for counting stats."
                        )
                    elif pace < league_pace - 1.0:
                        notes.append(
                            f"{opponent} plays slower than the league median, which can reduce total possessions."
                        )
    return notes

def easy_prop_analysis(gamelog,matchup_log,league,prop_spec,side,line,market_probability=None,current_side=None,opponent=""):
    vals=extract_values(gamelog,prop_spec)
    proj=player_projection(
        gamelog,
        vals,
        league,
        current_side=current_side,
        opponent=opponent,
    )
    raw_p=normal_prob(proj["mu"],proj["sigma"],side,line)
    bt_p,bt_y=rolling_prop_backtest(gamelog,vals,league,side,line)
    cal=calibrate_current(raw_p,bt_p,bt_y) if not pd.isna(raw_p) else {
        "probability":np.nan,"calibrated":False,"samples":0,
        "accuracy":np.nan,"brier":np.nan,"logloss":np.nan,
    }
    p=cal["probability"]

    opp_rate=np.nan
    if matchup_log is not None and not matchup_log.empty:
        try:
            mvals=extract_values(matchup_log,prop_spec)
            opp_rate=hit_rate(mvals,side,line)
        except Exception:
            pass

    valid_vals=pd.to_numeric(vals,errors="coerce").dropna()
    last5=hit_rate(valid_vals.head(5),side,line)
    last10=hit_rate(valid_vals.head(10),side,line)
    last20=hit_rate(valid_vals.head(20),side,line)
    season=hit_rate(valid_vals,side,line)

    edge=np.nan
    if market_probability is not None and not pd.isna(market_probability) and not pd.isna(p):
        edge=p-market_probability

    confidence=confidence_label(cal)

    positives=[]
    cautions=[]
    if not pd.isna(p):
        positives.append(f"model estimates about {p*100:.1f}%")
    if not pd.isna(last10):
        if last10>=.70:
            positives.append(f"hit in {last10*100:.0f}% of the last 10")
        elif last10<.40:
            cautions.append(f"only hit in {last10*100:.0f}% of the last 10")
    if not pd.isna(opp_rate):
        if opp_rate>=.65:
            positives.append(f"hit in {opp_rate*100:.0f}% of the prior games vs {opponent}")
        elif opp_rate<.40:
            cautions.append(f"only hit in {opp_rate*100:.0f}% of prior games vs {opponent}")
    if not pd.isna(proj.get("expected_minutes",np.nan)):
        positives.append(f"expected minutes around {proj['expected_minutes']:.1f}")
    if cal["samples"]<10:
        cautions.append("small out-of-sample test sample")
    if not pd.isna(cal["brier"]) and cal["brier"]>.25:
        cautions.append("historical probability error is still fairly high")
    if confidence=="Low":
        cautions.append("model confidence is low")

    if market_probability is None or pd.isna(market_probability):
        if pd.isna(p):
            verdict="NOT ENOUGH DATA"
            verdict_text="There is not enough usable data for a reliable probability."
        elif p>=.65 and confidence in {"Medium","Higher"} and (pd.isna(last10) or last10>=.50):
            verdict="STRONGER DATA SUPPORT"
            verdict_text=(
                f"The available stats estimate about a {p*100:.1f}% chance for this line, "
                f"with {confidence.lower()} model confidence."
            )
        elif p>=.55:
            verdict="SOME SUPPORT / MORE RISK"
            verdict_text=(
                f"The available stats estimate about a {p*100:.1f}% chance. "
                "There is some support, but the margin is not large enough to treat it as a strong signal."
            )
        else:
            verdict="WEAK SUPPORT / PASS ON DATA"
            verdict_text=(
                f"The available stats estimate only about a {p*100:.1f}% chance for this line. "
                "The data does not make this one stand out."
            )
    elif pd.isna(edge):
        verdict="NOT ENOUGH DATA"
        verdict_text="The app could not estimate a reliable probability."
    elif edge>=.07 and confidence in {"Medium","Higher"} and (pd.isna(last10) or last10>=.50):
        verdict="STRONGER SUPPORT"
        verdict_text=(
            f"The model is {edge*100:.1f} percentage points above the market price, with {confidence.lower()} confidence."
        )
    elif edge>=.025:
        verdict="SOME SUPPORT / MORE RISK"
        verdict_text=(
            f"The model is only {edge*100:.1f} points above the market, so the margin for error is smaller."
        )
    elif edge<=0:
        verdict="PASS AT THIS PRICE"
        verdict_text=(
            f"The market is pricing this at or above the model estimate ({edge*100:+.1f} point edge)."
        )
    else:
        verdict="TOO CLOSE"
        verdict_text="The model and market are too close to call this a meaningful edge."

    return {
        "probability":p,
        "projection":proj["mu"],
        "expected_minutes":proj.get("expected_minutes",np.nan),
        "last5":last5,
        "last10":last10,
        "last20":last20,
        "season":season,
        "vs_opponent":opp_rate,
        "backtest_n":cal["samples"],
        "accuracy":cal["accuracy"],
        "brier":cal["brier"],
        "calibrated":cal["calibrated"],
        "confidence":confidence,
        "edge":edge,
        "verdict":verdict,
        "verdict_text":verdict_text,
        "positives":positives,
        "cautions":list(dict.fromkeys(cautions)),
    }

def top_idea_explanation(idea,player_name,opponent):
    p=idea.get("Model probability",np.nan)
    last10=idea.get("Last 10",np.nan)
    vs=idea.get("Vs opponent",np.nan)
    fair=max(0.0,(p-.05)) if not pd.isna(p) else np.nan
    why=[]
    risk=[]
    if not pd.isna(p):
        why.append(f"model probability {p*100:.1f}%")
    if not pd.isna(last10):
        why.append(f"last-10 hit rate {last10*100:.0f}%")
    if not pd.isna(vs):
        why.append(f"vs {opponent} hit rate {vs*100:.0f}%")
    if idea.get("Backtest N",0)<15:
        risk.append("the out-of-sample sample is still limited")
    if pd.isna(vs):
        risk.append("there is not enough exact-opponent history")
    if not pd.isna(idea.get("Brier",np.nan)) and idea["Brier"]>.25:
        risk.append("the probability model has meaningful historical error")
    return {
        "title":f"{player_name} {idea['Line']} {idea['Stat']}",
        "why":why,
        "risk":risk,
        "max_price":fair,
    }

def previous_kalshi_price_map(tickers):
    hist=load_csv(KALSHI_HISTORY_FILE)
    if hist.empty or "ticker" not in hist.columns:
        return {}
    out={}
    for ticker in tickers:
        rows=hist[hist["ticker"].astype(str)==str(ticker)]
        if not rows.empty:
            rows=rows.sort_values("timestamp")
            val=pd.to_numeric(rows.iloc[-1].get("yes_ask"),errors="coerce")
            if not pd.isna(val):
                out[str(ticker)]=float(val)
    return out

def record_kalshi_snapshots(inventory):
    if not inventory:
        return
    ts=datetime.now().isoformat(timespec="seconds")
    rows=[]
    for r in inventory:
        ask=r.get("YES ask",np.nan)
        bid=r.get("YES bid",np.nan)
        if pd.isna(ask):
            continue
        rows.append({
            "timestamp":ts,
            "ticker":r.get("Ticker",""),
            "league":r.get("League",""),
            "type":r.get("Type",""),
            "title":r.get("Title",""),
            "yes_ask":ask,
            "yes_bid":bid,
            "volume_24h":r.get("Volume 24h",np.nan),
        })
    if not rows:
        return
    old=load_csv(KALSHI_HISTORY_FILE)
    new=pd.DataFrame(rows)
    out=pd.concat([old,new],ignore_index=True) if not old.empty else new
    out.to_csv(KALSHI_HISTORY_FILE,index=False)

def research_source_links(game,league,player_id="",athlete_id="",team_id="",opponent_team_id="",kalshi_ticker=""):
    selected_date=st.session_state.get("selected_date",date.today())
    links=[]
    if game:
        links.append(("ESPN game page",espn_game_page_url(league,game.get("event_id",""))))
        links.append(("ESPN schedule feed",espn_scoreboard_source_url(league,selected_date)))
        if team_id:
            links.append(("Team injury feed",espn_injury_source_url(league,team_id)))
    pid=player_id or athlete_id
    purl=official_player_page_url(league,pid)
    if purl:
        links.append(("Official player page",purl))
    if league in {"NBA","WNBA"} and player_id and opponent_team_id:
        url=basketball_matchup_source_url(
            league,player_id,opponent_team_id,selected_date
        )
        if url:
            links.append(("Official player-vs-opponent stats query",url))
    if kalshi_ticker:
        links.append(("Kalshi market API",kalshi_source_url(kalshi_ticker)))
    return links


# ============================================================
# V11 HELPERS
# ============================================================

def statmuse_query_url(league,query):
    return None

def statmuse_player_links(league,player_name,opponent,today_side):
    return []

def checkmark_prop_table(gamelog,prop_spec,side,line,opponent="",n=20):
    if gamelog is None or gamelog.empty:
        return pd.DataFrame()
    vals=pd.to_numeric(extract_values(gamelog,prop_spec),errors="coerce")
    valid_idx=vals[vals.notna()].index[:int(n)]
    if len(valid_idx)==0:
        return pd.DataFrame()
    date_col=(gamelog.loc[valid_idx,"DATE"].astype(str) if "DATE" in gamelog.columns else pd.Series(valid_idx.astype(str),index=valid_idx))
    matchup_col=(
        gamelog.loc[valid_idx,"MATCHUP"].astype(str)
        if "MATCHUP" in gamelog.columns
        else gamelog.loc[valid_idx,"OPP"].astype(str)
        if "OPP" in gamelog.columns else pd.Series("",index=valid_idx)
    )
    out=pd.DataFrame({
        "Date":date_col.values,
        "Matchup":matchup_col.values,
        "Stat":vals.loc[valid_idx].values,
        "Line":float(line),
    })
    out["Hit?"]=np.where(hit_mask(pd.Series(out["Stat"]),side,line),"✅","❌")
    if "DATA_SOURCE" in gamelog.columns:
        out["Source"]=gamelog.loc[valid_idx,"DATA_SOURCE"].astype(str).values
    if opponent:
        out["Vs today's opponent"]=np.where(
            out["Matchup"].astype(str).str.upper().str.contains(str(opponent).upper(),regex=False),"🎯",""
        )
    return out

def render_last20_prop_chart(gamelog,prop_spec,side,line,opponent="",title="Last 20 games"):
    if gamelog is None or gamelog.empty:
        st.caption("No verified game log is available for this prop.")
        return
    vals=pd.to_numeric(extract_values(gamelog,prop_spec),errors="coerce")
    valid_idx=vals[vals.notna()].index[:20]
    if len(valid_idx)==0:
        st.caption("The selected stat was not available in the verified recent game log.")
        return
    dates=(gamelog.loc[valid_idx,"DATE"].astype(str) if "DATE" in gamelog.columns else pd.Series(range(len(valid_idx)),index=valid_idx).astype(str))
    chart=pd.DataFrame({"Game":dates.values,"Stat":vals.loc[valid_idx].values,"Line":float(line)})
    chart=chart.iloc[::-1].copy()
    st.markdown(f"### {title}")
    st.line_chart(chart.set_index("Game")[["Stat","Line"]],use_container_width=True)
    valid_vals=vals.loc[valid_idx]
    c1,c2,c3=st.columns(3)
    c1.metric("Last 5",fmt_pct(hit_rate(valid_vals.head(5),side,line)))
    c2.metric("Last 10",fmt_pct(hit_rate(valid_vals.head(10),side,line)))
    c3.metric("Last 20",fmt_pct(hit_rate(valid_vals.head(20),side,line)))
    st.markdown("#### Last 10 — game-by-game hit check")
    check10=checkmark_prop_table(gamelog,prop_spec,side,line,opponent,n=10)
    if not check10.empty:
        st.dataframe(check10,use_container_width=True,hide_index=True)
    check20=checkmark_prop_table(gamelog,prop_spec,side,line,opponent,n=20)
    if len(check20)>10:
        with st.expander("Show games 11–20"):
            st.dataframe(check20.iloc[10:20],use_container_width=True,hide_index=True)

def render_short_player_matchup_notes(game,league,log,player_name,position,opponent,prop_name=None):
    try:
        fit=player_vs_team_fit(game,league,log,player_name,position,opponent,prop_stat=prop_name)
    except Exception as e:
        st.caption("Matchup notes unavailable: "+str(e))
        return
    left,right=st.columns(2)
    with left:
        st.markdown("**Strengths / matchup help**")
        notes=(fit.get("player_strengths",[])+fit.get("helps",[]))
        for note in list(dict.fromkeys(notes))[:4]:
            st.write("• "+str(note))
    with right:
        st.markdown("**Weaknesses / concerns**")
        notes=(fit.get("player_weaknesses",[])+fit.get("hurts",[]))
        for note in list(dict.fromkeys(notes))[:4]:
            st.write("• "+str(note))

def render_typed_bet_detail(typed,game,league,roster,ctx):
    try:
        resolved,_=ensure_player_in_roster_for_bet(typed,league,roster)
        parsed=parse_freeform_bet(typed,game,league,resolved)
        if parsed.get("type")!="player_prop":
            return
        row=parsed["row"]
        research=fetch_player_research_v23(row,game,league)
        log=research.get("log",pd.DataFrame())
        if log.empty:
            return
        props=available_props(log,league)
        if parsed.get("prop") not in props:
            return
        pname=str(parsed.get("player","")); opponent=research.get("opponent","")
        position=str(row.get("position",""))
        render_last20_prop_chart(log,props[parsed["prop"]],parsed["side"],parsed["line"],opponent,title=f"{pname} — last 20")
        render_short_player_matchup_notes(game,league,log,pname,position,opponent,parsed.get("prop"))
    except Exception as e:
        st.caption("Detailed chart unavailable: "+str(e))

def render_matchup_breakdown_v11(game,league,ctx):
    st.subheader(f"{game['away_name']} @ {game['home_name']}")
    venue=game.get("venue") or "Venue not listed"
    st.caption(f"{game.get('status','')} • {venue}")

    home=ctx["home"]
    away=ctx["away"]
    home_records=game.get("home_records") or {}
    away_records=game.get("away_records") or {}
    home_stats=game.get("home_stats") or {}
    away_stats=game.get("away_stats") or {}

    # ---------------- Home / away ----------------
    st.markdown("## 🏠 Home vs ✈️ away")
    c1,c2=st.columns(2)

    with c1:
        st.markdown(f"### {game['home_abbr']} at home")
        hs=home.get("home_split",{})
        if hs:
            a,b,c=st.columns(3)
            a.metric("Home record",f"{hs['wins']}-{hs['losses']}")
            b.metric("Home win rate",fmt_pct(hs["win_pct"]))
            c.metric("Avg margin",f"{hs['margin']:+.1f}")
        elif home_records.get("home"):
            a,b=st.columns(2)
            a.metric("Home record",home_records["home"])
            b.metric("Home win rate",fmt_pct(_record_pct(home_records["home"])))
        else:
            st.caption("Home split not available from either team-history or game feed.")

        if home_records.get("overall"):
            st.write(f"**Overall:** {home_records['overall']}")

    with c2:
        st.markdown(f"### {game['away_abbr']} on the road")
        aws=away.get("away_split",{})
        if aws:
            a,b,c=st.columns(3)
            a.metric("Road record",f"{aws['wins']}-{aws['losses']}")
            b.metric("Road win rate",fmt_pct(aws["win_pct"]))
            c.metric("Avg margin",f"{aws['margin']:+.1f}")
        elif away_records.get("road"):
            a,b=st.columns(2)
            a.metric("Road record",away_records["road"])
            b.metric("Road win rate",fmt_pct(_record_pct(away_records["road"])))
        else:
            st.caption("Road split not available from either team-history or game feed.")

        if away_records.get("overall"):
            st.write(f"**Overall:** {away_records['overall']}")

    # ---------------- Head to head ----------------
    st.markdown("## 🤝 How these teams have played against each other")
    h2h=home.get("h2h",pd.DataFrame())
    hh=home.get("h2h_summary",{})
    if hh and hh.get("games",0):
        a,b,c=st.columns(3)
        a.metric(
            f"{game['home_abbr']} record vs {game['away_abbr']}",
            f"{hh['wins']}-{hh['losses']}"
        )
        b.metric("Avg scored",fmt_num(hh.get("avg_for")))
        c.metric("Avg margin",f"{hh.get('margin',0):+.1f}")
        show=[
            c for c in
            ["date","home_away","result","points_for","points_against"]
            if c in h2h.columns
        ]
        st.dataframe(h2h[show].head(10),use_container_width=True,hide_index=True)
    else:
        st.info(
            "The automatic team-history feed did not return a usable head-to-head sample. "
            "Use the links below to cross-check instead of treating this as 'they never played.'"
        )
        st.caption("Use the team-history table, league game pages, or the local historical database for the cross-check; no StatMuse dependency is used.")

    # ---------------- Season statistical profile ----------------
    st.markdown("## 📈 Current team profile")
    profile=pd.DataFrame([
        {
            "Team":game["away_abbr"],
            "Overall record":away_records.get("overall","—"),
            "PPG":_scoreboard_stat_text(away_stats,"avgPoints"),
            "RPG":_scoreboard_stat_text(away_stats,"avgRebounds"),
            "APG":_scoreboard_stat_text(away_stats,"avgAssists"),
            "FG%":_scoreboard_stat_text(away_stats,"fieldGoalPct"),
            "3P%":_scoreboard_stat_text(
                away_stats,
                "threePointPct" if "threePointPct" in away_stats else "threePointFieldGoalPct"
            ),
        },
        {
            "Team":game["home_abbr"],
            "Overall record":home_records.get("overall","—"),
            "PPG":_scoreboard_stat_text(home_stats,"avgPoints"),
            "RPG":_scoreboard_stat_text(home_stats,"avgRebounds"),
            "APG":_scoreboard_stat_text(home_stats,"avgAssists"),
            "FG%":_scoreboard_stat_text(home_stats,"fieldGoalPct"),
            "3P%":_scoreboard_stat_text(
                home_stats,
                "threePointPct" if "threePointPct" in home_stats else "threePointFieldGoalPct"
            ),
        },
    ])
    st.dataframe(profile,use_container_width=True,hide_index=True)

    # ---------------- Team leaders ----------------
    st.markdown("## ⭐ Team leaders")
    lead_rows=[]
    for side in ["away","home"]:
        abbr=game[f"{side}_abbr"]
        leaders=game.get(f"{side}_leaders") or {}
        lead_rows.append({
            "Team":abbr,
            "Scoring":(
                f"{leaders.get('pointsPerGame',{}).get('name','—')} "
                f"{leaders.get('pointsPerGame',{}).get('value','')}"
            ).strip(),
            "Rebounds":(
                f"{leaders.get('reboundsPerGame',{}).get('name','—')} "
                f"{leaders.get('reboundsPerGame',{}).get('value','')}"
            ).strip(),
            "Assists":(
                f"{leaders.get('assistsPerGame',{}).get('name','—')} "
                f"{leaders.get('assistsPerGame',{}).get('value','')}"
            ).strip(),
        })
    st.dataframe(pd.DataFrame(lead_rows),use_container_width=True,hide_index=True)

    # ---------------- Recent form/rest ----------------
    st.markdown("## 🔥 Recent form / rest")
    l,r=st.columns(2)
    for side,col in [("away",l),("home",r)]:
        t=ctx[side]
        with col:
            st.markdown(f"### {t['abbr']}")
            recent=t.get("recent10",{})
            if recent:
                st.write(
                    f"**Last 10:** {recent['wins']}-{recent['losses']} • "
                    f"{recent['avg_for']:.1f} scored • {recent['avg_against']:.1f} allowed • "
                    f"{recent['margin']:+.1f} margin"
                )
            else:
                rec=game.get(f"{side}_records") or {}
                st.write(
                    f"**Season record:** {rec.get('overall','not returned')} "
                    f"• venue split: "
                    f"{rec.get('road','—') if side=='away' else rec.get('home','—')}"
                )
                st.caption(
                    "Last-10 game sequence was not returned, so the app is showing the verified season profile instead."
                )

            rest=t.get("rest",np.nan)
            if pd.isna(rest):
                st.caption("Rest-day calculation unavailable from the recent-game feed.")
            else:
                st.write(f"**Rest:** {int(rest)} day(s)")

    # ---------------- Pace / efficiency ----------------
    advanced_rows=[]
    if league in {"NBA","WNBA"}:
        st.markdown("## ⚡ Pace / offense / defense")
        advanced_rows,_=render_basketball_advanced_or_fallback(
            game,league,ctx
        )

    # ---------------- Injuries ----------------
    st.markdown("## 🚑 Injuries / availability")
    l,r=st.columns(2)
    for side,col in [("away",l),("home",r)]:
        t=ctx[side]
        with col:
            st.markdown(f"### {t['abbr']}")
            counts=t.get("injuries",{})
            verified=t.get("injury_verified",False)
            injury_error=t.get("injury_error")
            inj=t.get("injury_df",pd.DataFrame())
            if verified:
                st.write(
                    f"**{counts.get('out',0)} out/doubtful • "
                    f"{counts.get('questionable',0)} questionable**"
                )
                if inj.empty:
                    st.success(
                        "The injury feed was checked and did not list a player."
                    )
            else:
                st.warning(
                    "Injury report NOT verified — this is not the same as zero injuries."
                )
                if injury_error:
                    st.caption(injury_error)

            if not inj.empty:
                show=[
                    c for c in
                    ["Player","Status","Injury","Detail","ReturnDate"]
                    if c in inj.columns
                ]
                st.dataframe(inj[show].head(15),use_container_width=True,hide_index=True)

    # ---------------- Market context / line movement ----------------
    st.markdown("## 📉 Current outside line / movement")
    odds=game.get("odds") or {}
    if odds:
        a,b,c=st.columns(3)
        a.metric("Current line",odds.get("details") or "—")
        b.metric(
            "Game total",
            "—" if pd.isna(odds.get("total",np.nan))
            else f"{odds['total']:.1f}"
        )
        a_open=odds.get("away_ml_open")
        a_cur=odds.get("away_ml_current")
        h_open=odds.get("home_ml_open")
        h_cur=odds.get("home_ml_current")
        ml_text=(
            f"{game['away_abbr']} {a_cur or '—'} • "
            f"{game['home_abbr']} {h_cur or '—'}"
        )
        c.metric("Current moneyline",ml_text)

        if any([a_open,h_open,odds.get("home_spread_open"),odds.get("away_spread_open")]):
            st.write(
                f"**Opening moneyline:** {game['away_abbr']} {a_open or '—'} • "
                f"{game['home_abbr']} {h_open or '—'}"
            )
            if odds.get("home_spread_open") or odds.get("home_spread_current"):
                st.write(
                    f"**Spread movement:** opened {game['home_abbr']} "
                    f"{odds.get('home_spread_open') or '—'} → current "
                    f"{odds.get('home_spread_current') or '—'}"
                )
        if odds.get("provider"):
            st.caption(f"Odds shown in the game feed: {odds['provider']}.")
    else:
        st.caption("No outside betting line was included in this game's feed.")

    # ---------------- Weather ----------------
    weather=None
    if league=="NFL":
        st.markdown("## 🌦️ Weather")
        if game.get("indoor") is True:
            st.success("Indoor stadium — outdoor weather should have little direct impact.")
        else:
            weather,err=fetch_game_day_weather(
                game.get("venue_city",""),
                game.get("venue_state",""),
                game.get("date",""),
            )
            if weather:
                a,b,c,d=st.columns(4)
                a.metric("High",f"{weather['high_f']:.0f}°F")
                b.metric("Low",f"{weather['low_f']:.0f}°F")
                c.metric("Rain",f"{weather['rain_pct']:.0f}%")
                d.metric("Wind",f"{weather['wind_mph']:.0f} mph")
            else:
                st.caption(f"Weather unavailable: {err}")

    # ---------------- Plain English ----------------
    st.divider()
    st.markdown("## 🧠 What stands out")
    takeaways=matchup_scoreboard_takeaways(game)
    takeaways+=plain_matchup_factors(game,league,ctx,advanced_rows,weather)
    for note in list(dict.fromkeys(takeaways)):
        st.write("• "+note)


def selected_game_kalshi_prices(game,league):
    markets,err=fetch_kalshi_series_markets(
        KALSHI_SPORT_SERIES[league]["Game winner"]
    )
    if err:
        return [],err

    rows=[]
    schedule_cache={}
    for m in markets:
        matched,_=find_espn_game_for_market(
            league,m,schedule_cache
        )
        if not matched or matched.get("event_id")!=game.get("event_id"):
            continue

        outcome=kalshi_market_outcome_code(m)
        home=normalize_team_code(game["home_abbr"])
        away=normalize_team_code(game["away_abbr"])
        if outcome==home:
            name=game["home_name"]
        elif outcome==away:
            name=game["away_name"]
        else:
            continue

        rows.append({
            "Team":name,
            "Abbr":game["home_abbr"] if outcome==home else game["away_abbr"],
            "Kalshi probability":kalshi_yes_ask(m),
            "Ticker":m.get("ticker",""),
            "Bid":kalshi_yes_bid(m),
            "Spread":kalshi_market_spread(m),
        })
    return rows,None

def selected_game_other_kalshi_markets(game,league):
    rows=[]
    target_date=pd.to_datetime(
        game.get("date"),errors="coerce",utc=True
    )
    target_date=target_date.date() if not pd.isna(target_date) else None

    a=normalize_team_code(game["away_abbr"])
    h=normalize_team_code(game["home_abbr"])

    for typ in ["Spread","Total"]:
        series=KALSHI_SPORT_SERIES[league].get(typ)
        if not series:
            continue
        markets,_=fetch_kalshi_series_markets(series)
        for m in markets:
            d=kalshi_event_date(m)
            raw=(
                str(m.get("event_ticker",""))+" "+
                str(m.get("ticker",""))+" "+
                str(m.get("title",""))+" "+
                str(m.get("subtitle",""))+" "+
                str(m.get("yes_sub_title",""))+" "+
                str(m.get("no_sub_title",""))
            ).upper()
            if target_date and d and d!=target_date:
                continue
            if a in raw and h in raw:
                rows.append({
                    "Type":typ,
                    "Market":m.get("title") or m.get("subtitle") or m.get("ticker"),
                    "Title":m.get("title",""),
                    "Subtitle":m.get("subtitle",""),
                    "YES subtitle":m.get("yes_sub_title",""),
                    "NO subtitle":m.get("no_sub_title",""),
                    "YES probability":kalshi_yes_ask(m),
                    "YES bid":kalshi_yes_bid(m),
                    "Spread":kalshi_market_spread(m),
                    "Ticker":m.get("ticker",""),
                    "Event ticker":m.get("event_ticker",""),
                    "Raw market":m,
                })
    return rows[:80]


def team_win_research(game,league):
    consensus,cons_err=fetch_espn_moneyline_consensus(
        league,game["event_id"]
    )

    out={
        "consensus":consensus,
        "consensus_error":cons_err,
        "nba_home_model":np.nan,
        "nba_model_error":"",
    }

    if league=="NBA":
        p,err=current_nba_team_probability(game)
        out["nba_home_model"]=p
        out["nba_model_error"]=err or ""
    return out

# ============================================================
# V12: AVAILABILITY / MINUTES / FREEFORM BET / SPORTS AI
# ============================================================

def official_injury_report_url(league):
    if league=="WNBA":
        return "https://www.wnba.com/wnba-injury-report"
    if league=="NBA":
        return "https://official.nba.com/nba-injury-report-2025-26-season/"
    return "https://www.espn.com/nfl/injuries"

def _injury_row_for_player(injury_df,player_name):
    if injury_df is None or injury_df.empty or "Player" not in injury_df.columns:
        return None
    target=_name_key(player_name)
    work=injury_df.copy()
    work["_key"]=work["Player"].map(_name_key)
    exact=work[work["_key"]==target]
    if not exact.empty:
        return exact.iloc[0]
    close=get_close_matches(target,work["_key"].astype(str).tolist(),n=1,cutoff=.72)
    if close:
        row=work[work["_key"]==close[0]]
        if not row.empty:
            return row.iloc[0]
    return None

def _extract_minute_restriction(value):
    s=str(value or "").lower()
    terms=["minute restriction","minutes restriction","restricted","limited to","minute limit","minutes limit","limit of"]
    if not any(t in s for t in terms):
        return None
    m=re.search(r"(\d{1,2})\s*(?:-|–|to)\s*(\d{1,2})\s*(?:min|minute)",s)
    if m:
        lo,hi=map(int,m.groups())
        return {"low":lo,"high":hi,"label":f"{lo}-{hi} minutes"}
    m=re.search(r"(\d{1,2})\s*(?:min|minute)",s)
    if m:
        val=int(m.group(1))
        return {"low":max(0,val-2),"high":val+2,"label":f"about {val} minutes"}
    return {"low":None,"high":None,"label":"restriction reported; exact minutes not stated"}

def estimate_player_availability_minutes(gamelog,injury_df,player_name,injury_verified=True):
    row=_injury_row_for_player(injury_df,player_name)
    status=("No injury designation found on verified feed" if injury_verified else "INJURY STATUS NOT VERIFIED")
    source_note=("The verified feed did not list this player." if injury_verified else "The team injury feed could not be verified.")
    confidence="Medium"
    restriction=None
    out=False
    if row is not None:
        status_raw=str(row.get("Status","") or "")
        detail=str(row.get("Detail","") or "")
        long_comment=str(row.get("LongComment","") or "")
        injury_type=str(row.get("Injury","") or "")
        combined=f"{status_raw} {detail} {long_comment}".lower()
        source_note=detail or long_comment or status_raw
        if injury_type:
            source_note=f"{injury_type}: {source_note}"
        restriction=_extract_minute_restriction(combined)
        if "out" in combined or "inactive" in combined:
            status="OUT / NOT EXPECTED TO PLAY"; out=True; confidence="High"
        elif "doubtful" in combined:
            status="DOUBTFUL"; confidence="Low"
        elif "questionable" in combined or "day-to-day" in combined:
            status="QUESTIONABLE / DAY-TO-DAY"; confidence="Low"
        elif "probable" in combined:
            status="PROBABLE"
        elif status_raw:
            status=status_raw.upper()
    if out:
        return {"status":status,"likely":0.0,"low":0.0,"high":0.0,"restriction":restriction,"confidence":confidence,"source_note":source_note,"basis":"Current injury designation says out/inactive."}
    mins=pd.Series(dtype=float)
    if gamelog is not None and not gamelog.empty and "MIN" in gamelog.columns:
        mins=pd.to_numeric(gamelog["MIN"],errors="coerce").dropna()
        mins=mins[mins>0]
    if mins.empty:
        return {"status":status,"likely":np.nan,"low":np.nan,"high":np.nan,"restriction":restriction,"confidence":"Low","source_note":source_note,"basis":"Recent minutes were unavailable."}
    last5=mins.head(5); last10=mins.head(10)
    avg5=float(last5.mean()) if len(last5) else np.nan
    avg10=float(last10.mean()) if len(last10) else float(mins.mean())
    med10=float(last10.median()) if len(last10) else float(mins.median())
    likely=(0.55*avg5+0.25*avg10+0.20*med10 if not pd.isna(avg5) else 0.65*avg10+0.35*med10)
    sd=float(last10.std(ddof=0)) if len(last10)>=3 else 4.0
    width=max(2.0,min(7.0,sd)); low=max(0.0,likely-width); high=likely+width
    if restriction:
        if restriction.get("low") is not None: low=max(low,float(restriction["low"]))
        if restriction.get("high") is not None:
            high=min(high,float(restriction["high"])); likely=min(likely,float(restriction["high"]))
        confidence="High" if restriction.get("high") is not None else confidence
    basis=(f"Recent minutes: last 5 avg {'—' if pd.isna(avg5) else f'{avg5:.1f}'}, last 10 avg {avg10:.1f}.")
    if status in {"QUESTIONABLE / DAY-TO-DAY","DOUBTFUL"}: basis+=" Minute estimate is IF ACTIVE."
    return {"status":status,"likely":likely,"low":low,"high":high,"restriction":restriction,"confidence":confidence,"source_note":source_note,"basis":basis}

def minutes_summary_text(info):
    if not info: return "Minutes/availability unavailable."
    if info["status"].startswith("OUT"): return "OUT — projected 0 minutes."
    parts=[info["status"]]
    if not pd.isna(info.get("likely",np.nan)):
        suffix=" if active" if info["status"] in {"QUESTIONABLE / DAY-TO-DAY","DOUBTFUL"} else ""
        parts.append(f"likely {info['likely']:.1f} minutes{suffix}")
    if not pd.isna(info.get("low",np.nan)) and not pd.isna(info.get("high",np.nan)):
        parts.append(f"range {info['low']:.0f}-{info['high']:.0f}")
    if info.get("restriction"): parts.append(f"restriction: {info['restriction']['label']}")
    return " • ".join(parts)

def nfl_workload_info_v33(log,injuries,player_name,injury_verified=False):
    status="INJURY STATUS NOT VERIFIED" if not injury_verified else "NO CURRENT DESIGNATION FOUND"
    inj=_injury_row_for_player(injuries,player_name)
    if inj is not None:
        raw=str(inj.get("Status","") or inj.get("Detail","") or "").upper()
        if "OUT" in raw or "IR" in raw: status="OUT"
        elif "DOUBT" in raw: status="DOUBTFUL"
        elif "QUESTION" in raw or "DAY-TO-DAY" in raw: status="QUESTIONABLE / DAY-TO-DAY"
        elif raw: status=raw
    snaps=pd.to_numeric(log.get("SNAP PCT",pd.Series(dtype=float)),errors="coerce").dropna().head(10)
    likely=snaps.head(5).mean()*100 if len(snaps) else np.nan
    low=snaps.head(5).min()*100 if len(snaps) else np.nan
    high=snaps.head(5).max()*100 if len(snaps) else np.nan
    conf="High" if len(snaps)>=5 else "Medium" if len(snaps)>=3 else "Low"
    return {"status":status,"likely":likely,"low":low,"high":high,"restriction":None,"confidence":conf,"basis":"Recent offensive snap share","source_note":"NFL workload uses offensive snap percentage, not basketball minutes."}

def nfl_workload_summary_text_v33(info):
    if not info: return "Workload/availability unavailable."
    if str(info.get("status","")).startswith("OUT"): return "OUT — expected offensive workload near zero."
    parts=[str(info.get("status",""))]
    if not pd.isna(info.get("likely",np.nan)): parts.append(f"recent snap share ~{info['likely']:.0f}%")
    if not pd.isna(info.get("low",np.nan)) and not pd.isna(info.get("high",np.nan)): parts.append(f"recent range {info['low']:.0f}-{info['high']:.0f}%")
    return " • ".join(parts)

def parse_freeform_bet(text_value,game,league,roster):
    raw=str(text_value or "").strip(); low=raw.lower(); compact=_name_key(raw)
    if game and any(k in low for k in [" to win","moneyline"," money line"," ml","winner","beat "]):
        for side in ["home","away"]:
            name=str(game[f"{side}_name"]); abbr=str(game[f"{side}_abbr"])
            if _name_key(name) in compact or _name_key(abbr) in compact:
                return {"type":"team_win","team_side":side,"team_name":name,"team_abbr":abbr,"raw":raw}
    if roster is None or roster.empty:
        return {"type":"unknown","raw":raw,"error":"No selected-game roster is loaded."}
    matched=None; candidates=[]
    for _,row in roster.iterrows():
        key=_name_key(row.get("player",""))
        if key and key in compact: candidates.append((len(key),row))
    if candidates: matched=max(candidates,key=lambda x:x[0])[1]
    if matched is None:
        # Use first several words as fuzzy name query, not the entire bet sentence.
        words=raw.split(); query=" ".join(words[:3])
        names=roster["player"].astype(str).tolist()
        matches=get_close_matches(query,names,n=1,cutoff=.35)
        if matches: matched=roster[roster["player"].astype(str)==matches[0]].iloc[0]
    if matched is None:
        return {"type":"unknown","raw":raw,"error":"I could not match a player or team from the selected game."}
    aliases={
        "Points":[" points"," point"," pts","pt "],"Rebounds":[" rebounds"," rebound"," rebs"," reb"],
        "Assists":[" assists"," assist"," asts"," ast"],"3-Pointers Made":[" threes"," three pointers"," 3 pointers"," 3-pointers"," 3pm"," 3s"],
        "Steals":[" steals"," steal"],"Blocks":[" blocks"," block"],"Turnovers":[" turnovers"," turnover"],
        "PRA":[" pra"," points rebounds assists"],"PR":[" pr "," points rebounds"],"PA":[" pa "," points assists"],"RA":[" ra "," rebounds assists"],
        "Passing Yards":[" passing yards"," pass yards"],"Passing TDs":[" passing tds"," passing touchdowns"," pass tds"],
        "Rushing Yards":[" rushing yards"," rush yards"],"Receiving Yards":[" receiving yards"," rec yards"],"Receptions":[" receptions"," catches"],"Targets":[" targets"]}
    prop=None; padded=" "+low+" "
    for label,terms in aliases.items():
        if any(term in padded for term in terms): prop=label; break
    line=None; side=None
    plus=re.search(r"(\d+(?:\.\d+)?)\s*\+",low)
    if plus: line=float(plus.group(1)); side="At least (X+)"
    if line is None:
        patterns=[("Over",r"\b(?:over|more than)\s*(\d+(?:\.\d+)?)"),("Under",r"\b(?:under|less than)\s*(\d+(?:\.\d+)?)"),("At least (X+)",r"\bat least\s*(\d+(?:\.\d+)?)"),("At most",r"\bat most\s*(\d+(?:\.\d+)?)")]
        for side_name,pat in patterns:
            mm=re.search(pat,low)
            if mm: line=float(mm.group(1)); side=side_name; break
    if line is None:
        nums=re.findall(r"(?<![A-Za-z])(\d+(?:\.\d+)?)(?![A-Za-z])",low)
        if nums: line=float(nums[-1]); side="At least (X+)"
    if not prop or line is None:
        return {"type":"player_prop_incomplete","row":matched,"player":str(matched["player"]),"prop":prop,"line":line,"side":side,"raw":raw,"error":"I matched the player but could not fully identify the stat and line."}
    return {"type":"player_prop","row":matched,"player":str(matched["player"]),"prop":prop,"line":line,"side":side or "At least (X+)","raw":raw}

def build_ai_dashboard_context():
    game=st.session_state.get("matchup")
    ctx={"current_date":date.today().isoformat(),"selected_league":st.session_state.get("league"),"selected_game":game,"selected_player":st.session_state.get("research_name"),"selected_opponent":st.session_state.get("research_opponent"),"player_minutes":st.session_state.get("research_minutes_info"),"latest_player_bet":st.session_state.get("latest_easy_selection"),"latest_player_analysis":st.session_state.get("latest_easy_analysis")}
    matchup=st.session_state.get("matchup_context")
    if matchup:
        compact={}
        for side in ["home","away"]:
            t=matchup.get(side,{})
            compact[side]={"abbr":t.get("abbr"),"recent10":t.get("recent10"),"home_split":t.get("home_split"),"away_split":t.get("away_split"),"rest":t.get("rest"),"injuries":t.get("injuries"),"injury_verified":t.get("injury_verified"),"injury_error":t.get("injury_error")}
        ctx["selected_matchup_context"]=compact
    return ctx


# ============================================================
# V25: ZERO-COST RESEARCH ENGINE
# Tavily free web-search credits + local Ollama AI
# ============================================================

def _free_env_value(key):
    """Read a local .env file without adding another Python dependency."""
    env_path=APP_DIR/".env"
    if not env_path.exists():
        return ""
    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line=raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k,v=line.split("=",1)
            if k.strip()==key:
                return v.strip().strip('"').strip("'")
    except Exception:
        return ""
    return ""


def get_tavily_key():
    key=str(os.getenv("TAVILY_API_KEY","") or "").strip()
    if not key:
        key=_free_env_value("TAVILY_API_KEY")
    if not key:
        try:
            key=str(st.secrets.get("TAVILY_API_KEY","") or "").strip()
        except Exception:
            key=""
    if not key:
        key=str(st.session_state.get("v25_tavily_key","") or "").strip()
    return key


def tavily_search_free(query,api_key,max_results=7):
    """Search current public web results through Tavily's free-tier API."""
    if not api_key:
        return [],"Tavily API key is not configured."

    body={
        "query":str(query),
        "search_depth":"basic",
        "topic":"general",
        "max_results":int(max_results),
        "include_answer":False,
        "include_raw_content":False,
        "include_images":False,
    }

    # Tavily supports bearer auth. A second legacy body-key attempt is kept
    # only as a compatibility fallback.
    attempts=[
        {
            "headers":{
                "Authorization":f"Bearer {api_key}",
                "Content-Type":"application/json",
            },
            "json":body,
        },
        {
            "headers":{"Content-Type":"application/json"},
            "json":dict(body,api_key=api_key),
        },
    ]

    last_error=""
    for attempt in attempts:
        try:
            r=requests.post(
                "https://api.tavily.com/search",
                headers=attempt["headers"],
                json=attempt["json"],
                timeout=45,
            )
            if r.status_code>=400:
                last_error=f"Tavily returned HTTP {r.status_code}: {r.text[:400]}"
                continue

            payload=r.json()
            rows=[]
            for item in payload.get("results",[]) or []:
                url=str(item.get("url") or "").strip()
                title=str(item.get("title") or url or "Source").strip()
                content=str(item.get("content") or "").strip()
                if not url:
                    continue
                rows.append({
                    "title":title,
                    "url":url,
                    "content":content,
                    "score":safe_float(item.get("score")),
                })
            return rows,None
        except Exception as e:
            last_error=f"Tavily request failed: {e}"

    return [],last_error or "Tavily search failed."



def get_groq_key():
    key=str(os.getenv("GROQ_API_KEY","") or "").strip()
    if not key:
        key=_free_env_value("GROQ_API_KEY")
    if not key:
        try:
            key=str(st.secrets.get("GROQ_API_KEY","") or "").strip()
        except Exception:
            key=""
    return key


def get_groq_model():
    model=str(os.getenv("GROQ_MODEL","") or "").strip()
    if not model:
        model=_free_env_value("GROQ_MODEL")
    if not model:
        try:
            model=str(st.secrets.get("GROQ_MODEL","") or "").strip()
        except Exception:
            model=""
    return model or "openai/gpt-oss-20b"


def groq_free_available():
    return bool(get_groq_key())


def call_groq_free(question,sources,context=None):
    source_text=[]
    for i,s in enumerate(sources,1):
        source_text.append(
            f"[{i}] {s.get('title','')}\\nURL: {s.get('url','')}\\n"
            f"EVIDENCE: {s.get('content','')}"
        )

    system_prompt="""
You are the evidence summarizer inside a sports-research dashboard for NBA, WNBA, and NFL.

STRICT RULES:
- Use ONLY the supplied source evidence and dashboard context.
- Never invent injuries, starters, player stats, matchup history, odds, probabilities, weaknesses, restrictions, or citations.
- Cite factual claims using [1], [2], etc. matching the supplied source list.
- If the source snippets do not prove a detail, write "not verified from the returned sources."
- Distinguish confirmed starter, probable/expected starter, and recent starter.
- Empty injury data never means healthy.
- A weakness must say exactly WHAT is weak and the evidence for it.
- Separate offensive weakness from defensive weakness.
- Exact-opponent history means the player actually faced that exact team.
- If sources conflict, state the conflict.
- For prices, compare the exact same market only.
- Market-implied probability is a price signal, not automatically the true probability.
- Do not promise wins or describe any wager as guaranteed.
- Write for a normal bettor: concise, plain English, no filler.
"""

    user_prompt=(
        "DASHBOARD CONTEXT:\\n"
        +json.dumps(context or {},default=str,indent=2)
        +"\\n\\nREQUEST:\\n"
        +str(question)
        +"\\n\\nRETURNED WEB EVIDENCE:\\n"
        +"\\n\\n".join(source_text)
    )

    key=get_groq_key()
    if not key:
        return "","Groq Free API key is not configured on the server."

    body={
        "model":get_groq_model(),
        "messages":[
            {"role":"system","content":system_prompt},
            {"role":"user","content":user_prompt},
        ],
        "temperature":0.1,
    }

    try:
        r=requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization":f"Bearer {key}",
                "Content-Type":"application/json",
            },
            json=body,
            timeout=90,
        )
        if r.status_code>=400:
            return "",f"Groq Free returned HTTP {r.status_code}: {r.text[:500]}"
        payload=r.json()
        choices=payload.get("choices") or []
        out=""
        if choices:
            out=str(((choices[0].get("message") or {}).get("content")) or "").strip()
        return out or "No summary text was returned.",None
    except Exception as e:
        return "",f"Groq request failed: {e}"


def dedupe_research_sources(rows):
    seen=set()
    out=[]
    for r in rows:
        url=str(r.get("url") or "")
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(r)
    return out


def run_free_research(queries,question,context=None):
    """
    Search the public web with Tavily, then summarize returned evidence with
    the owner-configured Groq Free API when available.
    """
    key=get_tavily_key()
    if not key:
        return {
            "sources":[],
            "summary":"",
            "errors":["Tavily is not connected yet."],
            "groq_used":False,
        }

    sources=[]
    errors=[]
    for q in queries[:4]:
        rows,err=tavily_search_free(q,key,max_results=7)
        sources.extend(rows)
        if err:
            errors.append(err)

    sources=dedupe_research_sources(sources)[:22]

    summary=""
    groq_used=False
    if groq_free_available() and sources:
        summary,serr=call_groq_free(
            question,
            sources,
            context=context,
        )
        if serr:
            errors.append(serr)
        else:
            groq_used=True
    elif sources:
        errors.append(
            "Groq Free AI is not configured on the server. Public search evidence is still shown below."
        )

    return {
        "sources":sources,
        "summary":summary,
        "errors":list(dict.fromkeys(errors)),
        "groq_used":groq_used,
        "checked_at":datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def game_free_research_queries(game,league):
    away=game.get("away_name","")
    home=game.get("home_name","")
    d=str(st.session_state.get("selected_date",date.today()))
    return [
        f'{away} {home} {d} official injury report probable starters lineup {league}',
        f'{away} {home} recent games head to head {league} {d} stats',
        f'{away} team strengths weaknesses offense defense {league} {home} matchup stats',
        f'{home} team strengths weaknesses offense defense {league} {away} matchup stats',
    ]


def game_free_research_question(game,league):
    return f"""
Research {game.get('away_name')} at {game.get('home_name')} for the selected {league} game.

Use the returned public sources to give:
### Availability / injuries
### Confirmed or possible starters
### Recent form and rest
### Exact head-to-head history
### {game.get('away_abbr')} strengths
### {game.get('away_abbr')} weak points
### {game.get('home_abbr')} strengths
### {game.get('home_abbr')} weak points
### Matchup-specific evidence
### Important stats for betting research
### What is still not verified

For basketball, distinguish shooting accuracy from shot volume and offense from defense.
For NFL, include QB/OL/skill-player/secondary availability and EPA/efficiency when the returned sources support it.
Do not recommend a bet.
"""


def player_free_research_queries(player_name,game,league):
    opponent=(
        game.get("home_name")
        if st.session_state.get("research_team","")==game.get("away_abbr")
        else game.get("away_name")
    )
    d=str(st.session_state.get("selected_date",date.today()))
    return [
        f'"{player_name}" {league} {d} injury status role minutes starts official',
        f'"{player_name}" vs "{opponent}" stats game log',
        f'"{player_name}" last 10 games stats {league}',
    ]


def player_free_research_question(player_name,game,league):
    opponent=st.session_state.get("research_opponent") or (
        game.get("home_abbr")
        if st.session_state.get("research_team","")==game.get("away_abbr")
        else game.get("away_abbr")
    )
    return f"""
Research {player_name} for {game.get('away_name')} at {game.get('home_name')} ({league}).

Give:
### Current status and role
### Recent production
### Exact history vs {opponent}
List each exact-opponent game you can verify with date and stat line.
### Research-backed strengths
### Research-backed weak points
Weak points must be supported by returned evidence, not generic labels.
### Matchup-specific evidence
### Prop-relevant context
### What is still not verified

If exact-opponent history is not proven by the sources, say so.
Do not recommend a wager.
"""


def render_targeted_player_status_search_v33(player_name,game,league):
    if not get_tavily_key():
        return
    st.markdown("## 📰 Current restriction / role check")
    st.caption("Targeted public-source search only — no chat-style AI lookup. Use it when injury status, a return from injury, minute/snap restriction, or role change could make historical stats misleading.")
    if st.button("Check current player news",key="v33_status_"+_name_key_v33(player_name),use_container_width=True):
        d=str(st.session_state.get("selected_date",date.today()))
        terms=("minutes restriction expected minutes return injury role starter" if league in {"NBA","WNBA"} else "injury inactive snap count limitation workload starter role")
        queries=[
            f'"{player_name}" {league} {terms} {d}',
            f'"{player_name}" {game.get("away_name","")} {game.get("home_name","")} injury status {d}',
        ]
        found=[]; errors=[]
        for q in queries:
            rows,err=tavily_search_free(q,get_tavily_key(),max_results=5)
            if err: errors.append(err)
            found.extend(rows or [])
        seen=set(); clean=[]
        for r in found:
            u=str(r.get("url","") or "")
            if u and u not in seen:
                seen.add(u); clean.append(r)
        st.session_state["v33_status_results_"+_name_key_v33(player_name)]=clean[:8]
        if errors: st.session_state["v33_status_errors_"+_name_key_v33(player_name)]=errors
    rows=st.session_state.get("v33_status_results_"+_name_key_v33(player_name),[])
    for r in rows[:6]:
        title=str(r.get("title") or "Source")
        url=str(r.get("url") or "")
        snippet=str(r.get("content") or r.get("snippet") or "")[:500]
        if url: st.markdown(f"**[{title}]({url})**")
        else: st.write("**"+title+"**")
        if snippet: st.caption(snippet)

def market_free_research_queries(game,league):
    away=game.get("away_name","")
    home=game.get("home_name","")
    d=str(st.session_state.get("selected_date",date.today()))
    return [
        f'{away} {home} {d} odds DraftKings FanDuel BetMGM Caesars moneyline spread total',
        f'{away} {home} {d} odds comparison moneyline spread total',
        f'Kalshi {away} {home} {d} {league}',
    ]


def market_free_research_question(game,league):
    return f"""
Research CURRENT publicly visible prices for {game.get('away_name')} at {game.get('home_name')} ({league}).

Use only prices explicitly present in returned sources and identify the source.
### Winner / moneyline prices
### Spread prices
### Total prices
### Kalshi prices
### Same-market comparison
For the exact same outcome, explain which quoted price would produce the larger profit on $100 if correct.
### Not verified

Do not treat a higher implied probability as a better payout.
Do not invent a sportsbook quote that is absent from the returned evidence.
Do not recommend which wager to place.
"""


def render_free_research_block(state_key,title,queries,question,button_key,context=None):
    st.markdown(f"## 🔎 {title}")

    if not get_tavily_key():
        st.info(
            "Free web research is not connected yet. Tavily's free account currently includes "
            "monthly search credits and does not require a credit card. Add a TAVILY_API_KEY in the sidebar."
        )
        return None

    if groq_free_available():
        st.caption(
            f"Web search: Tavily free tier • AI summary: Groq Free ({get_groq_model()}) • server-side"
        )
    else:
        st.caption(
            "Web search is connected, but the free hosted AI summarizer is not configured on the server."
        )

    if st.button(
        "Research / refresh",
        key=button_key,
        type="primary",
        use_container_width=True,
    ):
        with st.spinner("Searching public sources..."):
            result=run_free_research(
                queries,
                question,
                context=context or build_ai_dashboard_context(),
            )
        st.session_state[state_key]=result

    result=st.session_state.get(state_key)
    if not result:
        return None

    st.caption(
        "Last checked: "
        +str(result.get("checked_at",""))
        +f" • {len(result.get('sources',[]))} public source result(s)"
    )

    if result.get("summary"):
        st.markdown(result["summary"])
    elif result.get("sources"):
        st.info(
            "The web search worked, but the free hosted AI summarizer is unavailable or has reached a free-tier limit. "
            "The source evidence is still shown below."
        )

    if result.get("errors"):
        with st.expander("Research messages"):
            for err in result["errors"]:
                st.write("• "+str(err))

    if result.get("sources"):
        with st.expander("Public sources / evidence",expanded=True):
            for i,s in enumerate(result["sources"],1):
                st.markdown(f"**[{i}] [{s.get('title','Source')}]({s.get('url','')})**")
                if s.get("content"):
                    st.write(s["content"])

    return result


def v25_live_reliability_table(game,ctx):
    gid=str(game.get("event_id",""))
    game_web=st.session_state.get(f"v25_game_research::{gid}")
    market_web=st.session_state.get(f"v25_market_research::{gid}")
    model_rows=st.session_state.get("v21_complete_market_rows") or []

    web_status="✅ researched" if game_web and game_web.get("sources") else "⚪ not run"
    market_status="✅ researched" if market_web and market_web.get("sources") else "⚪ not run"

    recent_ok=all(
        not ctx.get(side,{}).get("history",pd.DataFrame()).empty
        for side in ["away","home"]
    )
    injury_ok=all(
        bool(ctx.get(side,{}).get("injury_verified"))
        for side in ["away","home"]
    )
    h2h_ok=any(
        not ctx.get(side,{}).get("h2h",pd.DataFrame()).empty
        for side in ["away","home"]
    )
    rest_ok=all(
        not pd.isna(ctx.get(side,{}).get("rest",np.nan))
        for side in ["away","home"]
    )

    rows=[
        ["Recent team history","✅" if recent_ok else "⚠️ incomplete",web_status],
        ["Injuries / availability","✅" if injury_ok else "⚠️ incomplete",web_status],
        ["Head-to-head","✅" if h2h_ok else "⚠️ none returned",web_status],
        ["Rest","✅" if rest_ok else "⚠️ incomplete",web_status],
        ["Public market-price research","✅" if model_rows else "⚠️ automatic rows incomplete",market_status],
        ["Free public-source citations","—",web_status],
    ]
    return pd.DataFrame(
        rows,
        columns=["Check","Structured feeds","Free web research"],
    )


# ============================================================
# V13: FREE SPORTS RESEARCH ASSISTANT
# ============================================================

def _assistant_sources_for_current_context():
    game=st.session_state.get("matchup")
    league=st.session_state.get("league")
    sources=[]

    if game and league:
        sources.append((
            "ESPN game page",
            espn_game_page_url(league,game.get("event_id","")),
        ))
        sources.append((
            "ESPN schedule feed",
            espn_scoreboard_source_url(
                league,
                st.session_state.get("selected_date",date.today()),
            ),
        ))

    player=st.session_state.get("research_name")
    opponent=st.session_state.get("research_opponent")
    today_side=st.session_state.get("research_today_side","")
    player_id=st.session_state.get("research_player_id","")
    athlete_id=st.session_state.get("research_athlete_id","")
    team_id=st.session_state.get("research_team_id","")
    opp_team_id=st.session_state.get("research_opponent_team_id","")

    if league and player:
        for label,url in research_source_links(
            game,
            league,
            player_id=player_id,
            athlete_id=athlete_id,
            team_id=team_id,
            opponent_team_id=opp_team_id,
        ):
            sources.append((label,url))

        sources.append((
            "Official / league injury report",
            official_injury_report_url(league),
        ))

    # preserve order, drop duplicate URLs
    seen=set()
    out=[]
    for title,url in sources:
        if url and url not in seen:
            seen.add(url)
            out.append((title,url))
    return out

def _assistant_player_research_text():
    player=st.session_state.get("research_name")
    if not player:
        return (
            "Choose a game and click a player first. Then I can summarize that player's "
            "recent form, exact-opponent history, home/away split, likely minutes, injury status, "
            "and current prop research."
        )

    opponent=st.session_state.get("research_opponent","")
    log=st.session_state.get("research_log",pd.DataFrame())
    matchup_log=st.session_state.get("research_matchup_log",pd.DataFrame())
    minutes=st.session_state.get("research_minutes_info")
    latest=st.session_state.get("latest_easy_analysis")
    selection=st.session_state.get("latest_easy_selection")

    lines=[f"### {player}" + (f" vs {opponent}" if opponent else "")]

    if minutes:
        lines.append(f"**Availability/minutes:** {minutes_summary_text(minutes)}")
        if minutes.get("source_note"):
            lines.append(f"Source note: {minutes['source_note']}")

    if not matchup_log.empty:
        s=summarize_player_matchup(matchup_log)
        bits=[]
        for key,label in [("PTS","PTS"),("REB","REB"),("AST","AST"),("PRA","PRA")]:
            if key in s:
                bits.append(f"{s[key]:.1f} {label}")
        if bits:
            lines.append(
                f"**Against {opponent}:** {s.get('games',len(matchup_log))} prior games • "
                + " • ".join(bits)
            )
    elif opponent:
        lines.append(
            f"**Against {opponent}:** no usable exact-opponent game sample was returned, "
            "so I would put less weight on opponent-specific history."
        )

    if latest and selection:
        lines.append(f"**Latest researched bet:** {selection}")
        lines.append(
            f"Estimated probability: {fmt_pct(latest.get('probability'))} • "
            f"projection: {fmt_num(latest.get('projection'))} • "
            f"last-10 hit rate: {fmt_pct(latest.get('last10'))} • "
            f"vs opponent: {fmt_pct(latest.get('vs_opponent'))}"
        )
        lines.append(f"**Data verdict:** {latest.get('verdict','—')}")
        if latest.get("positives"):
            lines.append("**Supports it:** " + "; ".join(latest["positives"]))
        if latest.get("cautions"):
            lines.append("**Cautions:** " + "; ".join(latest["cautions"]))

    return "\n\n".join(lines)

def _assistant_matchup_text():
    game=st.session_state.get("matchup")
    league=st.session_state.get("league")
    ctx=st.session_state.get("matchup_context")
    if not game or not league:
        return "Choose a game in **This Week** first."

    if not ctx:
        ctx=build_matchup_context(game,league)
        st.session_state["matchup_context"]=ctx

    factors=plain_matchup_factors(game,league,ctx,[],None)
    lines=[
        f"### {game['away_name']} @ {game['home_name']}",
        "**Main matchup factors:**",
    ]
    if factors:
        lines.extend([f"- {x}" for x in factors])
    else:
        lines.append("- Not enough verified matchup data was returned.")

    if league=="NFL":
        if game.get("indoor") is True:
            lines.append("- Indoor venue: outdoor weather should have little direct field impact.")
        else:
            weather,err=fetch_game_day_weather(
                game.get("venue_city",""),
                game.get("venue_state",""),
                game.get("date",""),
            )
            if weather:
                lines.append(
                    f"- Weather: {weather['low_f']:.0f}-{weather['high_f']:.0f}°F, "
                    f"{weather['rain_pct']:.0f}% rain chance, wind up to {weather['wind_mph']:.0f} mph."
                )
            elif err:
                lines.append(f"- Weather could not be verified: {err}")

    return "\n".join(lines)

def _assistant_team_win_text():
    game=st.session_state.get("matchup")
    league=st.session_state.get("league")
    if not game or not league:
        return "Choose a game in **This Week** first."

    research=team_win_research(game,league)
    consensus=research.get("consensus")
    if not consensus:
        return (
            "I could not verify a usable outside moneyline consensus for this game. "
            + str(research.get("consensus_error") or "")
        )

    hp=consensus["home_probability"]
    ap=consensus["away_probability"]
    lines=[
        f"### {game['away_name']} @ {game['home_name']}",
        f"**De-vigged outside win likelihood:**",
        f"- {game['away_name']}: **{ap*100:.1f}%**",
        f"- {game['home_name']}: **{hp*100:.1f}%**",
        f"- Outside books used: **{consensus['books']}**",
    ]

    if league=="NBA" and not pd.isna(research.get("nba_home_model",np.nan)):
        mh=research["nba_home_model"]
        lines.append(
            f"**Historical NBA model second opinion:** "
            f"{game['away_abbr']} {(1-mh)*100:.1f}% • {game['home_abbr']} {mh*100:.1f}%"
        )
        same=((mh>=.5)==(hp>=.5))
        lines.append(
            "The historical model and outside consensus **agree on the more likely side**."
            if same else
            "The historical model and outside consensus **disagree**, which is a caution flag."
        )

    prices,err=selected_game_kalshi_prices(game,league)
    if prices:
        lines.append("**Matching Kalshi prices:**")
        for r in prices:
            fair=hp if r["Abbr"]==game["home_abbr"] else ap
            gap=fair-r["Kalshi probability"]
            lines.append(
                f"- {r['Team']}: Kalshi {r['Kalshi probability']*100:.1f}% vs "
                f"outside fair {fair*100:.1f}% ({gap*100:+.1f} point difference)"
            )
    elif err:
        lines.append(f"Kalshi comparison unavailable: {err}")

    return "\n".join(lines)

def _assistant_kalshi_text():
    game=st.session_state.get("matchup")
    league=st.session_state.get("league")
    if game and league:
        base=_assistant_team_win_text()
        return (
            base
            + "\n\n**How to read Kalshi:** a 35% YES price is roughly $0.35 for a $1 contract. "
            "If it resolves YES, the contract pays $1; if NO, it pays $0. "
            "A high probability does not automatically mean good value—the price must be lower than "
            "your best-supported fair probability by enough to cover fees, uncertainty, and model error."
        )

    return (
        "**Kalshi basics:** a YES contract priced at 35% costs about $0.35 and settles at $1 if YES wins. "
        "The dashboard looks for situations where independent, de-vigged outside odds imply a materially "
        "higher probability than Kalshi's current buy price. That is a possible discrepancy, not proof Kalshi is wrong."
    )

def _assistant_concept_text(q):
    low=q.lower()

    if "expected value" in low or re.search(r"\bev\b",low):
        return (
            "**Expected value (EV)** is the average profit/loss you would expect over many identical situations. "
            "For a $1 YES contract, a simple pre-fee estimate is roughly: model probability minus contract price. "
            "Positive EV does not mean the next single bet will win."
        )

    if "implied probability" in low:
        return (
            "**Implied probability** is the probability represented by a price or odds quote. "
            "On Kalshi, a $0.60 YES contract is approximately a 60% market-implied probability before fees/spread."
        )

    if "calibration" in low:
        return (
            "**Calibration** asks whether the model's percentages behave like real probabilities. "
            "If it calls many outcomes 60%, roughly 60% of those outcomes should happen over a large sample."
        )

    if "backtest" in low:
        return (
            "**Backtesting** means training on older games and then testing predictions on later games the model "
            "was not allowed to see during training. It is a basic defense against fooling ourselves with training data."
        )

    if "epa" in low:
        return (
            "**EPA (Expected Points Added)** measures how much a football play changes the offense's expected points. "
            "It is much more informative than raw yards for many NFL analyses because down, distance, field position, "
            "and game situation matter."
        )

    if "bankroll" in low:
        return (
            "**Bankroll** just means the amount of money someone intentionally sets aside for sports markets. "
            "It is not their bank-account balance. The point of a risk rule is to keep one uncertain idea from using too much."
        )

    if "sharp money" in low:
        return (
            "A price moving does **not** prove 'sharp money.' To identify professional-vs-public order flow reliably, "
            "you need actual betting/order-flow information. This dashboard only labels the observed price movement."
        )

    return None

def free_sports_assistant_answer(question):
    """
    A no-fee, no-LLM assistant. It routes common sports-research questions into the
    dashboard's live data functions and returns source links where possible.
    """
    q=str(question or "").strip()
    low=q.lower()
    sources=_assistant_sources_for_current_context()

    # Educational concepts first when clearly asked.
    concept=_assistant_concept_text(q)
    if concept:
        return concept,sources

    game=st.session_state.get("matchup")
    league=st.session_state.get("league")
    roster=st.session_state.get("roster",pd.DataFrame())

    # A typed bet gets the same research path as Type a Bet.
    if game and not roster.empty:
        parsed=parse_freeform_bet(q,game,league,roster)

        if parsed.get("type")=="team_win":
            return _assistant_team_win_text(),sources

        if parsed.get("type")=="player_prop":
            try:
                load_player_v11(parsed["row"],game,league)

                log=st.session_state.get("research_log",pd.DataFrame())
                matchup_log=st.session_state.get("research_matchup_log",pd.DataFrame())
                player=st.session_state.get("research_name",parsed["player"])
                opponent=st.session_state.get("research_opponent","")
                today_side=st.session_state.get("research_today_side","")
                team_id=st.session_state.get("research_team_id","")
                injuries,inj_err=(
                    fetch_injuries(league,team_id)
                    if team_id else (pd.DataFrame(),"No team ID")
                )
                ctx=st.session_state.get("matchup_context")
                player_team=st.session_state.get("research_team","")
                side_key="home" if player_team==game["home_abbr"] else "away"
                team_ctx=ctx.get(side_key,{}) if ctx else {}

                minutes=estimate_player_availability_minutes(
                    log,
                    injuries,
                    player,
                    injury_verified=team_ctx.get("injury_verified",inj_err is None),
                )
                st.session_state["research_minutes_info"]=minutes

                props=available_props(log,league)
                if parsed["prop"] not in props:
                    return (
                        f"I matched **{player}**, but the current data feed did not return a usable "
                        f"**{parsed['prop']}** stat for this player."
                    ),_assistant_sources_for_current_context()

                analysis=easy_prop_analysis(
                    log,
                    matchup_log,
                    league,
                    props[parsed["prop"]],
                    parsed["side"],
                    parsed["line"],
                    market_probability=None,
                    current_side=today_side,
                    opponent=opponent,
                )
                st.session_state["latest_easy_analysis"]=analysis
                st.session_state["latest_easy_selection"]=(
                    f"{player} — {parsed['side']} {parsed['line']} {parsed['prop']}"
                )

                lines=[
                    f"### {player} — {parsed['side']} {parsed['line']} {parsed['prop']}",
                    f"**Estimated probability:** {fmt_pct(analysis['probability'])}",
                    f"**Projection:** {fmt_num(analysis['projection'])}",
                    f"**Last 10 hit rate:** {fmt_pct(analysis['last10'])}",
                    f"**Vs {opponent}:** {fmt_pct(analysis['vs_opponent'])}",
                    f"**Availability/minutes:** {minutes_summary_text(minutes)}",
                    f"**Data verdict:** {analysis['verdict']}",
                    analysis["verdict_text"],
                ]

                if analysis["positives"]:
                    lines.append("**Why it could hit:**")
                    lines.extend([f"- ✅ {x}" for x in analysis["positives"]])

                caution=list(analysis["cautions"])
                if minutes["status"] in {"QUESTIONABLE / DAY-TO-DAY","DOUBTFUL"}:
                    caution.append("availability is uncertain; minute estimate is IF ACTIVE")
                if minutes["status"].startswith("OUT"):
                    caution.append("player is listed out")

                if caution:
                    lines.append("**Why I would be careful / pass:**")
                    lines.extend([f"- ⚠️ {x}" for x in list(dict.fromkeys(caution))])

                ctx=st.session_state.get("matchup_context")
                if ctx:
                    env=opponent_environment_summary(
                        game,league,ctx,st.session_state.get("selected_date",date.today())
                    )
                    if env:
                        lines.append("**Matchup context:**")
                        lines.extend([f"- {x}" for x in env])

                return "\n".join(lines),_assistant_sources_for_current_context()
            except Exception as e:
                return (
                    "I understood that as a player bet, but the live research path hit an error: "
                    + str(e)
                ),sources

    # Intent-based current-context answers.
    if any(x in low for x in [
        "injur","questionable","doubtful","out tonight",
        "minute restriction","minutes restriction","likely minutes",
        "how many minutes","is he playing","is she playing",
        "expected to play","active tonight",
    ]):
        player=st.session_state.get("research_name")
        if not player:
            return (
                "Click the player in **Player Research** first. Then I can answer from that player's "
                "current injury feed, likely-minute estimate, restriction language, and recent minutes."
            ),sources

        minutes=st.session_state.get("research_minutes_info")
        if not minutes:
            log=st.session_state.get("research_log",pd.DataFrame())
            team_id=st.session_state.get("research_team_id","")
            injuries,err=(
                fetch_injuries(league,team_id)
                if league and team_id else (pd.DataFrame(),"No team selected")
            )
            game=st.session_state.get("matchup")
            ctx=st.session_state.get("matchup_context")
            pteam=st.session_state.get("research_team")
            side="home" if game and pteam==game["home_abbr"] else "away"
            team_ctx=ctx.get(side,{}) if ctx else {}
            minutes=estimate_player_availability_minutes(
                log,
                injuries,
                player,
                injury_verified=team_ctx.get("injury_verified",err is None),
            )
            st.session_state["research_minutes_info"]=minutes

        answer=(
            f"### {player}\n\n"
            f"**{minutes_summary_text(minutes)}**\n\n"
            f"{minutes['basis']}\n\n"
            f"Availability note: {minutes['source_note']}"
        )
        if minutes.get("restriction"):
            answer+=f"\n\n**Reported restriction:** {minutes['restriction']['label']}"
        return answer,_assistant_sources_for_current_context()

    if "weather" in low or "wind" in low or "rain" in low:
        game=st.session_state.get("matchup")
        league=st.session_state.get("league")
        if not game:
            return "Choose an NFL game in **This Week** first.",sources
        if league!="NFL":
            return (
                "NBA/WNBA games are generally indoors, so outdoor weather is not normally a direct game factor."
            ),sources
        if game.get("indoor") is True:
            return "This game is listed at an indoor venue, so outdoor weather should have little direct field impact.",sources
        weather,err=fetch_game_day_weather(
            game.get("venue_city",""),
            game.get("venue_state",""),
            game.get("date",""),
        )
        if weather:
            return (
                f"Game-day weather near {weather['location']}: "
                f"{weather['low_f']:.0f}-{weather['high_f']:.0f}°F, "
                f"{weather['rain_pct']:.0f}% rain chance, wind up to "
                f"{weather['wind_mph']:.0f} mph."
            ),sources
        return f"Weather could not be verified: {err}",sources

    if "kalshi" in low or "contract" in low or "misprice" in low or "underpriced" in low:
        return _assistant_kalshi_text(),sources

    if any(x in low for x in [
        "who win","who wins","win probability","win likelihood",
        "moneyline","team to win","team bet",
    ]):
        return _assistant_team_win_text(),sources

    if any(x in low for x in [
        "matchup","home advantage","away","road","rest",
        "pace","defense","offense","head to head",
    ]):
        return _assistant_matchup_text(),sources

    if any(x in low for x in [
        "player","prop","points","rebounds","assists","threes",
        "pra","best bet","best idea","research this",
    ]):
        return _assistant_player_research_text(),sources

    return (
        "I can answer this for free from the dashboard's built-in sports research. "
        "For the best result, select the relevant game/player first.\n\n"
        "Try asking things like:\n"
        "- **Is this player expected to play and how many minutes?**\n"
        "- **Research Rachel Banham 3+ points.**\n"
        "- **How does this player do against Toronto?**\n"
        "- **Who has the higher win likelihood in this game?**\n"
        "- **Explain the Kalshi price for this game.**\n"
        "- **What does the weather mean for this NFL game?**\n"
        "- **What are the biggest matchup advantages/disadvantages?**"
    ),sources


# ============================================================
# V14: RICH MATCHUP FALLBACK + SIMPLE KALSHI LANGUAGE
# ============================================================

def matchup_scoreboard_takeaways(game):
    home_rec=game.get("home_records") or {}
    away_rec=game.get("away_records") or {}
    home_stats=game.get("home_stats") or {}
    away_stats=game.get("away_stats") or {}
    notes=[]

    hp=_record_pct(home_rec.get("home",""))
    ap=_record_pct(away_rec.get("road",""))

    if not pd.isna(hp) and not pd.isna(ap):
        diff=hp-ap
        if diff>=.12:
            notes.append(
                f"Strong venue edge for {game['home_abbr']}: "
                f"{home_rec.get('home')} at home vs {away_rec.get('road')} for "
                f"{game['away_abbr']} on the road."
            )
        elif diff<=-.12:
            notes.append(
                f"The road split actually compares well for {game['away_abbr']}: "
                f"{away_rec.get('road')} away vs {home_rec.get('home')} for "
                f"{game['home_abbr']} at home."
            )
        else:
            notes.append(
                f"Home/road records are fairly close: {game['home_abbr']} "
                f"{home_rec.get('home')} at home; {game['away_abbr']} "
                f"{away_rec.get('road')} away."
            )

    # Scoring
    hpts=safe_float(((home_stats.get("avgPoints") or {}).get("value")))
    apts=safe_float(((away_stats.get("avgPoints") or {}).get("value")))
    if not pd.isna(hpts) and not pd.isna(apts):
        better=game["home_abbr"] if hpts>=apts else game["away_abbr"]
        notes.append(
            f"Scoring profile: {game['home_abbr']} {hpts:.1f} PPG vs "
            f"{game['away_abbr']} {apts:.1f} PPG; {better} has the higher scoring average."
        )

    # Shooting
    hfg=safe_float(((home_stats.get("fieldGoalPct") or {}).get("value")))
    afg=safe_float(((away_stats.get("fieldGoalPct") or {}).get("value")))
    h3=safe_float(((home_stats.get("threePointPct") or home_stats.get("threePointFieldGoalPct") or {}).get("value")))
    a3=safe_float(((away_stats.get("threePointPct") or away_stats.get("threePointFieldGoalPct") or {}).get("value")))

    if not pd.isna(hfg) and not pd.isna(afg):
        notes.append(
            f"Field-goal shooting: {game['home_abbr']} {hfg:.1f}% vs "
            f"{game['away_abbr']} {afg:.1f}%."
        )
    if not pd.isna(h3) and not pd.isna(a3):
        notes.append(
            f"Three-point shooting: {game['home_abbr']} {h3:.1f}% vs "
            f"{game['away_abbr']} {a3:.1f}%."
        )

    odds=game.get("odds") or {}
    if odds.get("details"):
        notes.append(
            f"Current outside line shown by the game feed: **{odds['details']}** "
            + (f"with total **{odds['total']:.1f}**." if not pd.isna(odds.get("total",np.nan)) else "")
        )

    return notes

def simple_kalshi_status(row,threshold=.05):
    gap=row.get("Adjusted gap",np.nan)
    if pd.isna(gap):
        return "Could not verify"
    if gap>=threshold:
        return "Large positive difference"
    if gap>0:
        return "Small difference — probably not enough"
    if gap<=-threshold:
        return "Kalshi is more expensive than outside estimate"
    return "Very close — no clear pricing edge"

def kalshi_plain_explanation(row,stake=500):
    price=row["Kalshi ask"]
    fair=row["Consensus probability"]
    gap=row["Adjusted gap"]
    scenario=contract_scenario(price,fair,stake)

    lines=[
        f"**Kalshi probability:** {price*100:.1f}%",
        f"**Outside fair estimate:** {fair*100:.1f}%",
        f"**Difference after safety margin:** {gap*100:+.1f} percentage points",
    ]

    if gap>=.05:
        lines.append(
            "**Meaning:** Kalshi looks meaningfully cheaper than the outside estimate. "
            "This is worth investigating further."
        )
    elif gap>0:
        lines.append(
            "**Meaning:** Kalshi is a little cheaper, but the difference is small. "
            "It is not large enough to call a strong pricing opportunity."
        )
    else:
        lines.append(
            "**Meaning:** Kalshi is not cheaper than the outside estimate after the safety margin. "
            "The scanner is not seeing a pricing bargain here."
        )

    if scenario:
        lines.append(
            f"For about **{fmt_money(stake)}**, the maximum loss is about "
            f"**{fmt_money(scenario['max_loss'])}** and gross profit if it wins is about "
            f"**{fmt_money(scenario['profit_if_win'])}**."
        )
    return "\n\n".join(lines)


# ============================================================
# V15: GAME INTELLIGENCE / STYLE / ALERTS / BET SLIP
# ============================================================

def _median_numeric(df,col):
    if df is None or df.empty or col not in df.columns:
        return np.nan
    return pd.to_numeric(df[col],errors="coerce").median()

@st.cache_data(show_spinner=False)
def local_basketball_opponent_profile_v34(league, selected_date):
    """Build opponent-allowed team stats instantly from the bundled player-game logs."""
    seed=load_basketball_seed_v33()
    if seed is None or seed.empty:
        return pd.DataFrame()
    x=seed[seed.get("LEAGUE","").astype(str).str.upper()==str(league).upper()].copy()
    if x.empty:
        return pd.DataFrame()

    expected=(nba_season_string(selected_date) if league=="NBA" else str(selected_date.year))
    seasons=[str(v) for v in x.get("SEASON",pd.Series(dtype=str)).dropna().astype(str).unique()]
    use=expected if expected in seasons else (sorted(seasons)[-1] if seasons else "")
    if use:
        x=x[x["SEASON"].astype(str)==use].copy()
    if x.empty:
        return pd.DataFrame()

    stats=[c for c in ["PTS","FG3M","FG3A","REB","AST"] if c in x.columns]
    if not stats:
        return pd.DataFrame()
    for c in stats:
        x[c]=pd.to_numeric(x[c],errors="coerce")

    keys=[c for c in ["GAME_ID","DATE","TEAM","OPP"] if c in x.columns]
    if "TEAM" not in keys or "OPP" not in keys:
        return pd.DataFrame()
    team_games=x.groupby(keys,dropna=False)[stats].sum(min_count=1).reset_index()
    if team_games.empty:
        return pd.DataFrame()

    agg=team_games.groupby("OPP",dropna=False)[stats].mean().reset_index()
    counts=team_games.groupby("OPP",dropna=False).size().rename("SAMPLE_GAMES").reset_index()
    agg=agg.merge(counts,on="OPP",how="left")
    agg=agg.rename(columns={
        "OPP":"TEAM_ABBREVIATION",
        "PTS":"OPP_PTS",
        "FG3M":"OPP_FG3M",
        "FG3A":"OPP_FG3A",
        "REB":"OPP_REB",
        "AST":"OPP_AST",
    })
    agg["SOURCE_SEASON"]=use
    agg["SOURCE_NAME"]="Bundled player-game logs"
    return agg


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_basketball_opponent_profile(league, selected_date):
    """
    Fast opponent-allowed profile. Use the bundled player-game database first so
    the Game Report never waits 20-30 seconds for a flaky NBA Stats request.
    The remote endpoint remains a fallback when no bundled profile exists.
    """
    local=local_basketball_opponent_profile_v34(league,selected_date)
    if local is not None and not local.empty:
        return local,None
    try:
        from nba_api.stats.endpoints import leaguedashteamstats
        league_id="00" if league=="NBA" else "10"
        season=nba_season_string(selected_date) if league=="NBA" else str(selected_date.year)
        obj=leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star="Regular Season",
            league_id_nullable=league_id,
            measure_type_detailed_defense="Opponent",
            per_mode_detailed="PerGame",
            timeout=8,
        )
        return obj.get_data_frames()[0].copy(),None
    except Exception as e:
        return pd.DataFrame(),str(e)

def classify_basketball_role(position):
    p=str(position or "").upper()
    if any(x in p for x in ["PG","SG","G"]):
        return "guard"
    if any(x in p for x in ["PF","C","F-C","C-F"]):
        return "big"
    if any(x in p for x in ["SF","F","G-F","F-G"]):
        return "wing"
    return "player"

def basketball_style_matchup_notes(game,league,opponent_abbr,player_position="",player_log=None):
    selected_date=st.session_state.get("selected_date",date.today())
    notes=[]
    positives=[]
    negatives=[]
    role=classify_basketball_role(player_position)

    adv,adv_err=fetch_basketball_advanced(league,selected_date)
    opp,opp_err=fetch_basketball_opponent_profile(league,selected_date)

    # Advanced pace/defensive rating.
    if not adv.empty and "TEAM_ABBREVIATION" in adv.columns:
        row=adv[
            adv["TEAM_ABBREVIATION"].astype(str).str.upper()==str(opponent_abbr).upper()
        ]
        if not row.empty:
            r=row.iloc[0]
            dr=safe_float(r.get("DEF_RATING"))
            pace=safe_float(r.get("PACE"))
            med_dr=_median_numeric(adv,"DEF_RATING")
            med_pace=_median_numeric(adv,"PACE")

            if not pd.isna(dr) and not pd.isna(med_dr):
                if dr>=med_dr+1.5:
                    positives.append(
                        f"{opponent_abbr} has a worse defensive rating than the league median "
                        f"({dr:.1f} vs {med_dr:.1f}); offensive production gets a matchup boost."
                    )
                elif dr<=med_dr-1.5:
                    negatives.append(
                        f"{opponent_abbr} has a better defensive rating than the league median "
                        f"({dr:.1f} vs {med_dr:.1f}); this is a tougher scoring environment."
                    )
                else:
                    notes.append(
                        f"{opponent_abbr}'s defensive rating is close to league-average."
                    )

            if not pd.isna(pace) and not pd.isna(med_pace):
                if pace>=med_pace+1.0:
                    positives.append(
                        f"{opponent_abbr} plays faster than the league median "
                        f"({pace:.1f} pace vs {med_pace:.1f}), creating more possessions."
                    )
                elif pace<=med_pace-1.0:
                    negatives.append(
                        f"{opponent_abbr} plays slower than the league median "
                        f"({pace:.1f} pace vs {med_pace:.1f}), which can reduce counting-stat opportunities."
                    )

    # Opponent-allowed profile.
    if not opp.empty and "TEAM_ABBREVIATION" in opp.columns:
        if "SOURCE_SEASON" in opp.columns:
            src_seasons=opp["SOURCE_SEASON"].dropna().astype(str).unique().tolist()
            if src_seasons:
                notes.append(
                    f"Opponent-allowed indicators use the bundled {src_seasons[0]} game-log sample, which loads instantly while live feeds are checked separately."
                )
        row=opp[
            opp["TEAM_ABBREVIATION"].astype(str).str.upper()==str(opponent_abbr).upper()
        ]
        if not row.empty:
            r=row.iloc[0]

            checks=[
                ("OPP_FG3M","made threes allowed",1.0),
                ("OPP_FG3A","three-point attempts allowed",1.5),
                ("OPP_OREB","offensive rebounds allowed",0.8),
                ("OPP_REB","total rebounds allowed",1.5),
                ("OPP_AST","assists allowed",1.5),
                ("OPP_PTS","points allowed",2.0),
            ]
            values={}
            for col,label,threshold in checks:
                if col in opp.columns:
                    val=safe_float(r.get(col))
                    med=_median_numeric(opp,col)
                    values[col]=(val,med)
                    if not pd.isna(val) and not pd.isna(med):
                        if val>=med+threshold:
                            positives.append(
                                f"{opponent_abbr} allows more {label} than the league median "
                                f"({val:.1f} vs {med:.1f})."
                            )
                        elif val<=med-threshold:
                            negatives.append(
                                f"{opponent_abbr} allows fewer {label} than the league median "
                                f"({val:.1f} vs {med:.1f})."
                            )

            # Position/style translation.
            if role=="guard":
                if "OPP_FG3M" in values and not pd.isna(values["OPP_FG3M"][0]):
                    val,med=values["OPP_FG3M"]
                    if val>med:
                        positives.append(
                            "Guard-style matchup: the opponent's above-median three-point allowance "
                            "can help guards/wings who create value from perimeter shooting."
                        )
                if "OPP_AST" in values and not pd.isna(values["OPP_AST"][0]):
                    val,med=values["OPP_AST"]
                    if val>med:
                        positives.append(
                            "Guard-style matchup: the opponent allows above-median assists, "
                            "which can help primary ball-handlers/playmakers."
                        )

            elif role=="big":
                if "OPP_OREB" in values and not pd.isna(values["OPP_OREB"][0]):
                    val,med=values["OPP_OREB"]
                    if val>med:
                        positives.append(
                            "Big-style matchup: the opponent gives up above-median offensive rebounds, "
                            "which can help rebound/put-back opportunities."
                        )
                if "OPP_REB" in values and not pd.isna(values["OPP_REB"][0]):
                    val,med=values["OPP_REB"]
                    if val>med:
                        positives.append(
                            "Big-style matchup: overall rebounds allowed are above the league median."
                        )

            elif role=="wing":
                if "OPP_FG3M" in values and not pd.isna(values["OPP_FG3M"][0]):
                    val,med=values["OPP_FG3M"]
                    if val>med:
                        positives.append(
                            "Wing-style matchup: perimeter scoring opportunities look more favorable than average."
                        )

    # Player-specific shot/profile hints from the actual log.
    if player_log is not None and not player_log.empty:
        if "FG3A" in player_log.columns:
            threes=pd.to_numeric(player_log["FG3A"],errors="coerce").head(10).mean()
            if not pd.isna(threes) and threes>=4:
                notes.append(
                    f"This player averages about {threes:.1f} three-point attempts over the recent sample, "
                    "so opponent perimeter defense matters more than it would for a low-volume shooter."
                )
        if "OREB" in player_log.columns:
            oreb=pd.to_numeric(player_log["OREB"],errors="coerce").head(10).mean()
            if not pd.isna(oreb) and oreb>=1.5:
                notes.append(
                    f"This player averages about {oreb:.1f} offensive rebounds recently, "
                    "so the opponent's defensive-rebounding weakness/strength is directly relevant."
                )

    if not positives and not negatives and (adv_err or opp_err):
        notes.append(
            "Exact style/allowed-by-opponent metrics were not fully returned, so this section is not inventing a matchup edge."
        )

    return {
        "role":role,
        "positives":list(dict.fromkeys(positives)),
        "negatives":list(dict.fromkeys(negatives)),
        "notes":list(dict.fromkeys(notes)),
        "advanced_error":adv_err,
        "opponent_error":opp_err,
    }

def nfl_style_matchup_notes(game,opponent_side,player_position=""):
    ctx=st.session_state.get("matchup_context") or {}
    opp=ctx.get(opponent_side,{})
    notes=[]
    positives=[]
    negatives=[]
    pos=str(player_position or "").upper()

    recent=opp.get("recent10",{})
    if recent:
        allowed=recent.get("avg_against",np.nan)
        margin=recent.get("margin",np.nan)
        if not pd.isna(allowed):
            if allowed>=27:
                positives.append(
                    f"{opp.get('abbr','Opponent')} has allowed {allowed:.1f} points per game in its recent sample, "
                    "which is a favorable overall offensive environment."
                )
            elif allowed<=19:
                negatives.append(
                    f"{opp.get('abbr','Opponent')} has allowed only {allowed:.1f} points per game recently, "
                    "which is a tougher overall offensive matchup."
                )
        if not pd.isna(margin) and margin<=-5:
            positives.append(
                f"{opp.get('abbr','Opponent')} has a {margin:+.1f} recent scoring margin, showing broad team-level weakness."
            )

    # Do not fake exact pressure/position splits when the feed does not provide them.
    if any(x in pos for x in ["QB","WR","TE"]):
        notes.append(
            "For pass-game props, pressure rate, pass yards allowed, explosive passes, and targets allowed by position "
            "are the key style metrics. The dashboard only labels them when a source actually returns those fields."
        )
    elif "RB" in pos:
        notes.append(
            "For RB props, rush yards allowed, yards per carry allowed, box/run defense, and red-zone rushing defense "
            "are the key style metrics. Missing fields are not filled in with guesses."
        )
    else:
        notes.append(
            "NFL position-style analysis uses verified team/position defensive fields when available; "
            "otherwise it falls back to recent points allowed and matchup context."
        )

    return {
        "role":pos or "NFL player",
        "positives":positives,
        "negatives":negatives,
        "notes":notes,
    }

def player_role_usage_change(log,league):
    """
    Usage/role trend proxy:
    compare the most recent 5 games with the prior 10 games.
    This is not the same as a true on/off teammate split.
    """
    if log is None or log.empty:
        return {"rows":[],"notes":["No player log available."]}

    candidates=(
        ["MIN","FGA","FG3A","AST","REB","PTS"]
        if league in {"NBA","WNBA"}
        else ["MIN","TGT","REC","RUSH_ATT","CAR","ATT","PASS_ATT","PTS"]
    )

    rows=[]
    for c in candidates:
        if c not in log.columns:
            continue
        vals=pd.to_numeric(log[c],errors="coerce")
        recent=vals.head(5).mean()
        prior=vals.iloc[5:15].mean()
        if pd.isna(recent) or pd.isna(prior):
            continue
        delta=recent-prior
        pct=(delta/prior) if prior not in [0,np.nan] and abs(prior)>1e-9 else np.nan
        rows.append({
            "Metric":c,
            "Last 5":recent,
            "Previous 10":prior,
            "Change":delta,
            "Change %":pct,
        })

    notes=[]
    for r in rows:
        pct=r["Change %"]
        if pd.isna(pct):
            continue
        if r["Metric"]=="MIN" and pct>=.12:
            notes.append(
                f"Minutes are up about {pct*100:.0f}% versus the previous sample, suggesting a larger recent role."
            )
        elif r["Metric"]=="MIN" and pct<=-.12:
            notes.append(
                f"Minutes are down about {abs(pct)*100:.0f}% versus the previous sample, a warning for volume-based props."
            )
        elif r["Metric"] in {"FGA","TGT","REC","RUSH_ATT","CAR","PASS_ATT"} and pct>=.18:
            notes.append(
                f"{r['Metric']} volume is up about {pct*100:.0f}% recently, suggesting increased usage."
            )
        elif r["Metric"] in {"FGA","TGT","REC","RUSH_ATT","CAR","PASS_ATT"} and pct<=-.18:
            notes.append(
                f"{r['Metric']} volume is down about {abs(pct)*100:.0f}% recently, suggesting decreased usage."
            )

    if not notes:
        notes.append("Recent role/usage looks relatively stable versus the prior sample.")

    return {"rows":rows,"notes":notes}

def research_data_quality(
    log,
    matchup_log,
    injury_verified,
    minutes_info=None,
    analysis=None,
):
    score=0
    reasons=[]

    n=len(log) if log is not None else 0
    if n>=20:
        score+=25; reasons.append("20+ recent/season games available")
    elif n>=10:
        score+=18; reasons.append("10+ games available")
    elif n>=5:
        score+=10; reasons.append("only 5-9 games available")
    else:
        reasons.append("very small player-game sample")

    m=len(matchup_log) if matchup_log is not None else 0
    if m>=8:
        score+=20; reasons.append("8+ exact-opponent games")
    elif m>=4:
        score+=14; reasons.append("4-7 exact-opponent games")
    elif m>=2:
        score+=7; reasons.append("only 2-3 exact-opponent games")
    else:
        reasons.append("little/no exact-opponent history")

    if injury_verified:
        score+=15; reasons.append("injury feed verified")
    else:
        reasons.append("injury feed not verified")

    if minutes_info:
        conf=minutes_info.get("confidence","Low")
        if conf=="High":
            score+=15; reasons.append("high-confidence minute estimate")
        elif conf=="Medium":
            score+=10; reasons.append("medium-confidence minute estimate")
        else:
            score+=4; reasons.append("low-confidence minute estimate")

    if analysis:
        samples=analysis.get("backtest_n",0)
        if samples>=25:
            score+=15; reasons.append("25+ out-of-sample tests")
        elif samples>=10:
            score+=10; reasons.append("10+ out-of-sample tests")
        elif samples>0:
            score+=4; reasons.append("small out-of-sample test")
        else:
            reasons.append("no out-of-sample test sample")

        brier=analysis.get("brier",np.nan)
        if not pd.isna(brier):
            if brier<=.20:
                score+=10; reasons.append("good probability-error signal")
            elif brier<=.25:
                score+=6; reasons.append("moderate probability-error signal")
            else:
                reasons.append("probability error is relatively high")
    else:
        # Reserve model-quality points when no specific line is selected yet.
        score+=5

    score=max(0,min(100,score))
    label="HIGH" if score>=75 else "MEDIUM" if score>=50 else "LOW"
    return {"score":score,"label":label,"reasons":reasons}

def automatic_red_flags(
    log,
    matchup_log,
    minutes_info,
    analysis=None,
    team_win_probability=None,
):
    flags=[]

    if minutes_info:
        status=minutes_info.get("status","")
        if status.startswith("OUT"):
            flags.append("🚫 Player is listed out / not expected to play.")
        elif status in {"QUESTIONABLE / DAY-TO-DAY","DOUBTFUL"}:
            flags.append("⚠️ Availability is uncertain; workload projection is IF ACTIVE.")
        elif status=="INJURY STATUS NOT VERIFIED":
            flags.append("⚠️ Injury status is not verified.")

        if minutes_info.get("restriction"):
            flags.append(
                "⚠️ A workload/minutes restriction is reported: "
                + minutes_info["restriction"]["label"]
            )

    if matchup_log is None or len(matchup_log)<3:
        flags.append("⚠️ Exact-opponent sample is smaller than 3 games.")

    if log is not None and not log.empty and "MIN" in log.columns:
        mins=pd.to_numeric(log["MIN"],errors="coerce").head(10).dropna()
        if len(mins)>=5 and mins.mean()>0 and mins.std(ddof=0)/mins.mean()>=.20:
            flags.append("⚠️ Recent minutes are volatile, which makes volume props less stable.")

    if analysis:
        p=analysis.get("probability",np.nan)
        if not pd.isna(p) and p>=.95 and analysis.get("backtest_n",0)<15:
            flags.append(
                "⚠️ Extremely high model probability with a limited backtest sample — treat the 95%+ number cautiously."
            )
        last5=analysis.get("last5",np.nan)
        season=analysis.get("season",np.nan)
        if not pd.isna(last5) and not pd.isna(season) and last5-season>=.30:
            flags.append(
                "⚠️ Recent hit rate is much hotter than the full sample; this may be a short-term streak."
            )

    if team_win_probability is not None and not pd.isna(team_win_probability):
        if team_win_probability>=.78:
            flags.append(
                "⚠️ Large favorite / blowout risk can reduce fourth-quarter minutes for starters."
            )

    return list(dict.fromkeys(flags))

def _snapshot_injuries(ctx):
    out={}
    for side in ["home","away"]:
        team=ctx.get(side,{})
        rows=team.get("injury_df",pd.DataFrame())
        vals={}
        if rows is not None and not rows.empty:
            for _,r in rows.iterrows():
                name=str(r.get("Player","")).strip()
                if name:
                    vals[name]=str(r.get("Status","") or r.get("Detail",""))
        out[team.get("abbr",side)]=vals
    return out

def game_snapshot_payload(game,ctx):
    odds=game.get("odds") or {}
    return {
        "timestamp":datetime.now().isoformat(timespec="seconds"),
        "injuries":_snapshot_injuries(ctx or {}),
        "odds":{
            "details":odds.get("details"),
            "total":odds.get("total"),
            "home_ml":odds.get("home_ml_current"),
            "away_ml":odds.get("away_ml_current"),
            "home_spread":odds.get("home_spread_current"),
            "away_spread":odds.get("away_spread_current"),
        },
    }

def compare_game_snapshot(game,ctx,store=True):
    all_snaps=load_json(GAME_SNAPSHOT_FILE,{})
    event_id=str(game.get("event_id",""))
    current=game_snapshot_payload(game,ctx)
    previous=all_snaps.get(event_id)
    changes=[]

    if previous:
        p_inj=previous.get("injuries",{})
        c_inj=current.get("injuries",{})
        teams=set(p_inj)|set(c_inj)
        for team in teams:
            old=p_inj.get(team,{})
            new=c_inj.get(team,{})
            for player,status in new.items():
                if player not in old:
                    changes.append(f"🚨 NEW injury/availability listing: {player} ({team}) — {status}")
                elif old[player]!=status:
                    changes.append(
                        f"🚨 Injury status changed: {player} ({team}) — {old[player]} → {status}"
                    )
            for player,status in old.items():
                if player not in new:
                    changes.append(
                        f"✅ {player} ({team}) is no longer listed on the injury feed."
                    )

        old_o=previous.get("odds",{})
        new_o=current.get("odds",{})
        for key,label in [
            ("home_spread","home spread"),
            ("away_spread","away spread"),
            ("total","game total"),
            ("home_ml","home moneyline"),
            ("away_ml","away moneyline"),
        ]:
            if old_o.get(key) not in [None,""] and new_o.get(key) not in [None,""] and old_o.get(key)!=new_o.get(key):
                changes.append(
                    f"📉 {label} moved: {old_o.get(key)} → {new_o.get(key)}"
                )

    if store:
        all_snaps[event_id]=current
        save_json(GAME_SNAPSHOT_FILE,all_snaps)

    return changes,previous,current

def current_pregame_alerts(game,ctx):
    alerts=[]
    for side in ["away","home"]:
        t=ctx.get(side,{})
        abbr=t.get("abbr",side)
        if not t.get("injury_verified",False):
            alerts.append(
                f"⚠️ {abbr} injury report is not fully verified."
            )
        inj=t.get("injury_df",pd.DataFrame())
        if inj is not None and not inj.empty:
            for _,r in inj.head(12).iterrows():
                player=str(r.get("Player","")).strip()
                status=str(r.get("Status","")).strip()
                if player:
                    alerts.append(f"🚑 {abbr}: {player} — {status or 'injury listing'}")

    odds=game.get("odds") or {}
    if odds.get("details"):
        alerts.append(
            f"📉 Current outside line: {odds.get('details')}"
            + (
                f" • total {odds['total']:.1f}"
                if not pd.isna(odds.get("total",np.nan))
                else ""
            )
        )
    return alerts

def parse_market_price_from_bet_text(raw):
    s=str(raw or "").lower()

    # Explicit probability / Kalshi-style percentage.
    m=re.search(r"(?:@|price|at)\s*(\d{1,2}(?:\.\d+)?)\s*%",s)
    if m:
        p=float(m.group(1))/100
        return {"probability":p,"label":f"{p*100:.1f}%","kind":"probability"}

    # Kalshi dollar/cents price.
    m=re.search(r"(?:@|price|at)\s*\$?0\.(\d{1,2})",s)
    if m:
        p=float("0."+m.group(1))
        return {"probability":p,"label":f"${p:.2f} per $1","kind":"kalshi"}

    m=re.search(r"(?:@|price|at)\s*(\d{1,2})\s*(?:c|¢)\b",s)
    if m:
        p=float(m.group(1))/100
        return {"probability":p,"label":f"{int(p*100)}¢","kind":"kalshi"}

    # American odds.
    m=re.search(r"(?<!\d)([+-]\d{3,4})(?!\d)",s)
    if m:
        odds=int(m.group(1))
        p=american_to_implied(odds)
        return {"probability":p,"label":str(odds),"kind":"american"}

    return None

def break_even_value_text(model_p,market_info):
    if not market_info or pd.isna(model_p):
        return {
            "break_even":np.nan,
            "edge":np.nan,
            "text":"No market price/odds were included, so the app can judge statistical support but not price value."
        }
    be=market_info["probability"]
    edge=model_p-be
    if edge>=.07:
        text=(
            f"Break-even chance is {be*100:.1f}%. Research estimate is {model_p*100:.1f}% "
            f"({edge*100:+.1f} points). That is a meaningful positive gap."
        )
    elif edge>=.025:
        text=(
            f"Break-even chance is {be*100:.1f}%. Research estimate is {model_p*100:.1f}% "
            f"({edge*100:+.1f} points). Small positive gap; there is less room for error."
        )
    elif edge>0:
        text=(
            f"Break-even chance is {be*100:.1f}%. Research estimate is only {edge*100:.1f} points higher. "
            "That is probably too close after uncertainty/fees."
        )
    else:
        text=(
            f"Break-even chance is {be*100:.1f}%, but research estimate is {model_p*100:.1f}%. "
            "At this price the model does not show positive value."
        )
    return {"break_even":be,"edge":edge,"text":text}

def analyze_bet_line_for_slip(raw,game,league,roster,ctx):
    resolved_roster,resolve_note=ensure_player_in_roster_for_bet(
        raw,league,roster
    )
    parsed=parse_freeform_bet(raw,game,league,resolved_roster)
    market=parse_market_price_from_bet_text(raw)

    if parsed.get("type")=="team_win":
        research=team_win_research(game,league)
        cons=research.get("consensus")
        if not cons:
            return {
                "Bet":raw,"Type":"Team win","Probability":np.nan,
                "Break-even":market["probability"] if market else np.nan,
                "Edge":np.nan,"Verdict":"Could not verify team probability",
                "Risk":"Outside consensus unavailable",
            }
        side=parsed["team_side"]
        p=cons["home_probability"] if side=="home" else cons["away_probability"]
        value=break_even_value_text(p,market)
        verdict=(
            "Stronger support" if p>=.60
            else "Some support" if p>=.52
            else "Weak support"
        )
        return {
            "Bet":raw,"Type":"Team win","Probability":p,
            "Break-even":value["break_even"],"Edge":value["edge"],
            "Verdict":verdict,
            "Risk":"Team outcomes still have high one-game variance",
            "Value explanation":value["text"],
        }

    if parsed.get("type")!="player_prop":
        return {
            "Bet":raw,"Type":"Unknown","Probability":np.nan,
            "Break-even":market["probability"] if market else np.nan,
            "Edge":np.nan,"Verdict":parsed.get("error","Could not parse"),
            "Risk":"Rewrite as Player Name + stat + line, or Team to win",
        }

    row=parsed["row"]
    pname=str(row["player"])
    pteam=str(row["team_abbr"])
    opponent=game["home_abbr"] if pteam==game["away_abbr"] else game["away_abbr"]
    opponent_team_id=game["home_id"] if pteam==game["away_abbr"] else game["away_id"]
    today_side="home" if pteam==game["home_abbr"] else "away"
    selected_date=st.session_state.get("selected_date",date.today())
    season=league_season_year(league,selected_date)

    try:
        research=fetch_player_research_v23(row,game,league)
        log=research.get("log",pd.DataFrame())
        matchup=research.get("matchup",pd.DataFrame())
        notes=research.get("notes",[])
        err=None if not log.empty else (" | ".join(notes[-3:]) if notes else "No player game log available")
    except Exception as e:
        log=pd.DataFrame(); matchup=pd.DataFrame(); err=str(e)

    if err and (log is None or log.empty):
        return {
            "Bet":raw,
            "Type":"Player prop",
            "Probability":np.nan,
            "Break-even":market["probability"] if market else np.nan,
            "Edge":np.nan,
            "Verdict":"Structured player stats unavailable — use public-source research below",
            "Risk":str(err),
            "Web fallback needed":True,
            "Resolved player":pname,
        }

    props=available_props(log,league)
    if parsed["prop"] not in props:
        return {
            "Bet":raw,"Type":"Player prop","Probability":np.nan,
            "Break-even":market["probability"] if market else np.nan,
            "Edge":np.nan,"Verdict":"Stat not returned","Risk":parsed["prop"],
        }

    analysis=easy_prop_analysis(
        log,matchup,league,props[parsed["prop"]],
        parsed["side"],parsed["line"],
        market_probability=market["probability"] if market else None,
        current_side=today_side,
        opponent=opponent,
    )

    team_id=str(row.get("team_id",""))
    injuries,inj_err=fetch_injuries(league,team_id,game.get("event_id",""))
    side_key="home" if pteam==game["home_abbr"] else "away"
    team_ctx=ctx.get(side_key,{}) if ctx else {}
    if league=="NFL":
        minutes=nfl_workload_info_v33(log,injuries,pname,injury_verified=team_ctx.get("injury_verified",inj_err is None))
    else:
        minutes=estimate_player_availability_minutes(
            log,injuries,pname,
            injury_verified=team_ctx.get("injury_verified",inj_err is None),
        )
    quality=research_data_quality(
        log,matchup,team_ctx.get("injury_verified",False),
        minutes,analysis,
    )
    flags=automatic_red_flags(log,matchup,minutes,analysis)
    value=break_even_value_text(analysis["probability"],market)

    return {
        "Bet":raw,
        "Type":"Player prop",
        "Probability":analysis["probability"],
        "Break-even":value["break_even"],
        "Edge":value["edge"],
        "Verdict":analysis["verdict"],
        "Data quality":quality["label"],
        "Minutes":nfl_workload_summary_text_v33(minutes) if league=="NFL" else minutes_summary_text(minutes),
        "Risk":"; ".join(flags[:3]) if flags else "No major automatic red flag",
        "Value explanation":value["text"],
    }

def game_wide_player_scan(game,league,roster,limit_per_team=5):
    rows=[]
    selected_date=st.session_state.get("selected_date",date.today())
    season=league_season_year(league,selected_date)

    for abbr in [game["away_abbr"],game["home_abbr"]]:
        frame=roster[roster["team_abbr"]==abbr].head(limit_per_team)
        for _,r in frame.iterrows():
            name=str(r["player"])
            try:
                if league=="NBA":
                    log,p,err=fetch_nba_player_by_name(
                        name,nba_season_string(selected_date)
                    )
                elif league=="WNBA":
                    log,p,err=fetch_wnba_player_best_available(
                        name,str(r.get("athlete_id","") or ""),season
                    )
                else:
                    log,err=fetch_espn_player_gamelog_multi(
                        "NFL",str(r["athlete_id"]),season,3
                    )
                if err or log.empty:
                    continue

                opponent=game["home_abbr"] if abbr==game["away_abbr"] else game["away_abbr"]
                supported,volatile=generate_research_ideas(log,league,opponent)
                if supported:
                    idea=supported[0]
                    rows.append({
                        "Player":name,
                        "Team":abbr,
                        "Idea":f"{idea['Line']} {idea['Stat']}",
                        "Model probability":idea.get("Model probability",np.nan),
                        "Last 10":idea.get("Last 10",np.nan),
                        "Backtest N":idea.get("Backtest N",0),
                        "Brier":idea.get("Brier",np.nan),
                    })
            except Exception:
                continue

    return sorted(
        rows,
        key=lambda x:x.get("Model probability",0) if not pd.isna(x.get("Model probability",np.nan)) else -1,
        reverse=True,
    )

def tracked_bet_history_summary(df):
    if df is None or df.empty:
        return {}
    out={"rows":len(df)}
    result=df.get("result",pd.Series(dtype=str)).astype(str).str.lower()
    wins=result.isin(["win","won","w","hit","yes"])
    losses=result.isin(["loss","lost","l","miss","no"])
    resolved=wins|losses
    out["resolved"]=int(resolved.sum())
    out["wins"]=int(wins.sum())
    out["losses"]=int(losses.sum())
    out["win_rate"]=(wins.sum()/resolved.sum()) if resolved.sum() else np.nan

    pnl=pd.to_numeric(
        df.get("profit_loss",pd.Series(dtype=float)),
        errors="coerce"
    ).dropna()
    out["pnl"]=float(pnl.sum()) if len(pnl) else np.nan
    return out


# ============================================================
# V16: EXPLAINABLE TEAM WEAKNESSES / STYLE MATCHUP
# ============================================================

def _ordinal_rank_number(value):
    if value is None:
        return np.nan
    m=re.search(r"(\d+)",str(value))
    return float(m.group(1)) if m else np.nan

def _team_stat_row(stats,key):
    row=(stats or {}).get(key) or {}
    return {
        "value":safe_float(row.get("value")),
        "rank":_ordinal_rank_number(row.get("rank")),
        "rank_text":row.get("rank") or "",
    }

def _league_rank_tiers(league):
    # Approximate league-size buckets used only for plain-English interpretation.
    # The actual rank itself is still shown to the user.
    if league=="NBA":
        return {"elite":10,"weak":21}
    if league=="WNBA":
        return {"elite":5,"weak":11}
    return {"elite":10,"weak":23}

def scoreboard_team_strengths_weaknesses(game,league,side):
    """
    Build an always-available profile from the ESPN scoreboard statistics.
    This does NOT claim opponent-allowed defense when that data is unavailable.
    """
    stats=game.get(f"{side}_stats") or {}
    records=game.get(f"{side}_records") or {}
    abbr=game.get(f"{side}_abbr","")
    opponent_side="away" if side=="home" else "home"
    opp_stats=game.get(f"{opponent_side}_stats") or {}
    opp_abbr=game.get(f"{opponent_side}_abbr","")
    tiers=_league_rank_tiers(league)

    labels={
        "avgPoints":"scoring",
        "fieldGoalPct":"field-goal shooting",
        "threePointPct":"three-point shooting",
        "threePointFieldGoalPct":"three-point shooting",
        "avgRebounds":"rebounding",
        "avgAssists":"playmaking/assists",
        "freeThrowPct":"free-throw shooting",
        "freeThrowsAttempted":"getting to the free-throw line",
        "threePointFieldGoalsAttempted":"three-point volume",
        "fieldGoalsAttempted":"shot volume",
    }

    # Prefer one 3P key to avoid duplicate bullets.
    keys=[
        "avgPoints","fieldGoalPct","threePointPct",
        "avgRebounds","avgAssists","freeThrowPct",
        "freeThrowsAttempted","threePointFieldGoalsAttempted",
        "fieldGoalsAttempted",
    ]
    if "threePointPct" not in stats and "threePointFieldGoalPct" in stats:
        keys[2]="threePointFieldGoalPct"

    strengths=[]
    weaknesses=[]
    neutral=[]

    for key in keys:
        row=_team_stat_row(stats,key)
        value=row["value"]
        rank=row["rank"]
        if pd.isna(value):
            continue

        label=labels[key]
        suffix=f" ({row['rank_text']})" if row["rank_text"] else ""

        # avgPoints/avgRebounds/avgAssists frequently have no rank in ESPN's feed.
        # Compare them directly to today's opponent so the section still explains something.
        if pd.isna(rank):
            opp=_team_stat_row(opp_stats,key)
            if not pd.isna(opp["value"]):
                diff=value-opp["value"]
                # Meaningful simple comparison thresholds.
                threshold={
                    "avgPoints":2.0,
                    "avgRebounds":1.5,
                    "avgAssists":1.0,
                }.get(key,1.0)
                if diff>=threshold:
                    strengths.append(
                        f"{abbr} has the better {label} number in this matchup "
                        f"({value:.1f} vs {opp['value']:.1f} for {opp_abbr})."
                    )
                elif diff<=-threshold:
                    weaknesses.append(
                        f"{abbr} trails {opp_abbr} in {label} "
                        f"({value:.1f} vs {opp['value']:.1f})."
                    )
                else:
                    neutral.append(
                        f"{label.title()} is close: {abbr} {value:.1f} vs {opp_abbr} {opp['value']:.1f}."
                    )
            continue

        if rank<=tiers["elite"]:
            strengths.append(
                f"{label.title()} is a team strength: {value:.1f}{suffix}."
            )
        elif rank>=tiers["weak"]:
            weaknesses.append(
                f"{label.title()} is a team weakness: {value:.1f}{suffix}."
            )
        else:
            neutral.append(
                f"{label.title()} is around the middle of the league: {value:.1f}{suffix}."
            )

    # Venue record as a very concrete weakness/strength.
    venue_key="home" if side=="home" else "road"
    venue_label="at home" if side=="home" else "on the road"
    rec=records.get(venue_key)
    pct=_record_pct(rec) if rec else np.nan
    if not pd.isna(pct):
        if pct>=.60:
            strengths.insert(
                0,
                f"{abbr} has been strong {venue_label}: {rec} ({pct*100:.1f}% win rate)."
            )
        elif pct<=.40:
            weaknesses.insert(
                0,
                f"{abbr} has struggled {venue_label}: {rec} ({pct*100:.1f}% win rate)."
            )
        else:
            neutral.insert(
                0,
                f"{abbr} is {rec} {venue_label} ({pct*100:.1f}% win rate)."
            )

    return {
        "team":abbr,
        "strengths":list(dict.fromkeys(strengths)),
        "weaknesses":list(dict.fromkeys(weaknesses)),
        "neutral":list(dict.fromkeys(neutral)),
    }

def opponent_style_exploit_report(
    game,
    league,
    target_side,
    player_position="",
    player_log=None,
):
    """
    Explain HOW the opponent might attack the target team.
    Uses true opponent-allowed/advanced data when available, plus scoreboard
    team-profile weaknesses as a fallback so the section is never blank.
    """
    target_abbr=game[f"{target_side}_abbr"]
    attacker_side="away" if target_side=="home" else "home"
    attacker_abbr=game[f"{attacker_side}_abbr"]

    profile=scoreboard_team_strengths_weaknesses(
        game,league,target_side
    )
    attacker=scoreboard_team_strengths_weaknesses(
        game,league,attacker_side
    )

    exact={
        "positives":[],
        "negatives":[],
        "notes":[],
        "role":"team",
    }

    if league in {"NBA","WNBA"}:
        exact=basketball_style_matchup_notes(
            game,
            league,
            target_abbr,
            player_position=player_position,
            player_log=player_log,
        )
    else:
        exact=nfl_style_matchup_notes(
            game,
            target_side,
            player_position=player_position,
        )

    exploit=[]
    caution=[]
    prop_implications=[]

    # True opponent-allowed / pace / defensive-rating signals.
    for x in exact.get("positives",[]):
        exploit.append(x)
    for x in exact.get("negatives",[]):
        caution.append(x)

    # Fallback interpretation from target's own weak areas.
    for weakness in profile.get("weaknesses",[])[:6]:
        exploit.append(
            f"Team-profile weakness relevant to betting: {weakness}"
        )

    # Match attacker strengths to target weaknesses in plain language.
    target_text=" ".join(profile.get("weaknesses",[])).lower()
    attack_text=" ".join(attacker["strengths"]).lower()

    if any(k in target_text for k in ["rebound","rebounding"]):
        prop_implications.append(
            f"If {target_abbr}'s rebounding weakness holds, {attacker_abbr} rebound props—especially for high-minute forwards/centers—deserve extra attention."
        )
    if any(k in target_text for k in ["three-point","3p"]):
        prop_implications.append(
            f"If {target_abbr}'s perimeter weakness is real, {attacker_abbr} shooters with high three-point volume can have a better-than-normal style matchup."
        )
    if any(k in target_text for k in ["assist","playmaking"]):
        prop_implications.append(
            f"Playmaking can matter here: primary {attacker_abbr} ball-handlers may have more assist opportunities if the defense struggles to contain creation."
        )
    if any(k in target_text for k in ["field-goal","scoring"]):
        prop_implications.append(
            f"A weak scoring/shooting profile can also create game-script risk for {target_abbr}; if they fall behind, rotation and late-game minutes can change."
        )

    # Attacker-specific strengths make the explanation more concrete.
    if "three-point" in attack_text:
        exploit.append(
            f"{attacker_abbr}'s own profile shows three-point shooting/volume as a strength, so perimeter matchups matter more in this game."
        )
    if "rebound" in attack_text:
        exploit.append(
            f"{attacker_abbr}'s rebounding is a strength, which can amplify any board weakness on {target_abbr}."
        )
    if "assist" in attack_text or "playmaking" in attack_text:
        exploit.append(
            f"{attacker_abbr}'s playmaking is a strength, so assist/creation matchups are especially relevant."
        )

    # Always include what the target does WELL so the user understands why a bet could fail.
    for strength in profile.get("strengths",[])[:4]:
        caution.append(
            f"What could stop the attack: {strength}"
        )

    if not exploit:
        exploit.append(
            f"No single verified defensive weakness clearly stands out for {target_abbr}. "
            "That means the matchup should be treated as more neutral rather than forcing an edge."
        )
    if not caution:
        caution.append(
            f"No major verified counter-strength was identified for {target_abbr}; injury/rotation and recent-form uncertainty still matter."
        )
    if not prop_implications:
        prop_implications.append(
            "No clean player-prop style edge was verified from the available team profile. "
            "Use the player's own role, minutes, and exact-opponent history before treating this as a prop advantage."
        )

    return {
        "target":target_abbr,
        "attacker":attacker_abbr,
        "weaknesses":profile.get("weaknesses",[]),
        "strengths":profile.get("strengths",[]),
        "exploit":list(dict.fromkeys(exploit)),
        "caution":list(dict.fromkeys(caution)),
        "prop_implications":list(dict.fromkeys(prop_implications)),
        "notes":list(dict.fromkeys(exact.get("notes",[]))),
    }

def render_team_attack_report(game,league,target_side):
    """
    Stable team report:
    1) always show the ESPN scoreboard/team-profile facts when available;
    2) attempt defense/opponent-allowed research separately;
    3) never relabel an offensive weakness as a defensive weakness.
    """
    abbr=game.get(f"{target_side}_abbr","Team")
    opponent_side="away" if target_side=="home" else "home"
    opp_abbr=game.get(f"{opponent_side}_abbr","Opponent")

    try:
        profile=scoreboard_team_strengths_weaknesses(
            game,league,target_side
        )
    except Exception as e:
        profile={
            "strengths":[],
            "weaknesses":[],
            "neutral":[],
        }
        st.warning(f"{abbr} scoreboard profile could not be built: {e}")

    st.markdown(f"### {abbr} team profile")

    left,right=st.columns(2)
    with left:
        st.markdown("#### 🟢 Verified team strengths")
        if profile.get("strengths"):
            for x in profile.get("strengths",[])[:8]:
                st.write("• "+str(x))
        else:
            st.write(
                "No ranking/comparison-based strength was verified from the current scoreboard feed."
            )

    with right:
        st.markdown("#### 🔴 Verified team weak points")
        if profile.get("weaknesses"):
            for x in profile.get("weaknesses",[])[:8]:
                st.write("• "+str(x))
        else:
            st.write(
                "No ranking/comparison-based weak point was verified from the current scoreboard feed."
            )

    if profile.get("neutral"):
        with st.expander("Middle-of-the-pack / close comparisons"):
            for x in profile.get("neutral",[])[:8]:
                st.write("• "+str(x))

    st.markdown(f"#### 🛡️ {abbr} defense / opponent-allowed indicators")
    try:
        if league in {"NBA","WNBA"}:
            defense=basketball_style_matchup_notes(
                game,league,abbr
            )
        else:
            defense=nfl_style_matchup_notes(
                game,target_side
            )

        defensive_weaknesses=defense.get("positives",[]) or []
        defensive_strengths=defense.get("negatives",[]) or []

        d1,d2=st.columns(2)
        with d1:
            st.markdown("**Defensive vulnerabilities actually returned**")
            if defensive_weaknesses:
                for x in defensive_weaknesses[:8]:
                    st.write("• "+str(x))
            else:
                st.write(
                    "No specific defensive vulnerability was verified from the current structured feeds."
                )

        with d2:
            st.markdown("**Defensive strengths / tougher indicators actually returned**")
            if defensive_strengths:
                for x in defensive_strengths[:8]:
                    st.write("• "+str(x))
            else:
                st.write(
                    "No specific defensive strength was verified from the current structured feeds."
                )

        if defense.get("notes"):
            with st.expander("Defense-source notes"):
                for x in defense.get("notes",[])[:8]:
                    st.write("• "+str(x))
    except Exception as e:
        st.warning(
            "The advanced defense/opponent-allowed source did not load. "
            "The offensive/team-profile facts above are still valid; no defensive weakness is being guessed."
        )
        st.caption("Source error: "+str(e))

    st.markdown(f"#### ⚔️ Matchup with {opp_abbr}")
    try:
        opp_profile=scoreboard_team_strengths_weaknesses(
            game,league,opponent_side
        )
        own_strength_text=" ".join(profile.get("strengths",[])).lower()
        opp_weak_text=" ".join(opp_profile.get("weaknesses",[])).lower()

        matched=[]
        keyword_labels=[
            ("three-point","three-point shooting"),
            ("rebound","rebounding"),
            ("assist","playmaking/assists"),
            ("scoring","scoring"),
            ("field-goal","field-goal shooting"),
            ("free-throw","free throws"),
        ]
        for key,label in keyword_labels:
            if key in own_strength_text and key in opp_weak_text:
                matched.append(
                    f"{abbr}'s verified {label} strength lines up with a verified {opp_abbr} weak point in the same area."
                )

        if matched:
            for x in matched:
                st.write("• "+x)
        else:
            st.write(
                "No same-category strength-vs-weakness matchup was verified from the current structured team profile. "
                "Use the public-source research section for additional current context."
            )
    except Exception as e:
        st.caption("Matchup pairing unavailable: "+str(e))



# ============================================================
# V17: EVERY-PLAYER BET IDEAS + PLAYER STRENGTH VS TEAM WEAKNESS
# ============================================================

def _recent_avg(log,col,n=10):
    if log is None or log.empty or col not in log.columns:
        return np.nan
    vals=pd.to_numeric(log[col],errors="coerce").head(n).dropna()
    return float(vals.mean()) if len(vals) else np.nan

def _recent_pct(log,made_col,attempt_col,n=10):
    if log is None or log.empty:
        return np.nan
    if made_col not in log.columns or attempt_col not in log.columns:
        return np.nan
    made=pd.to_numeric(log[made_col],errors="coerce").head(n).sum()
    att=pd.to_numeric(log[attempt_col],errors="coerce").head(n).sum()
    return made/att if att and att>0 else np.nan

def basketball_player_tendency_profile(log):
    """
    Recent player style profile. These are recent-sample tendencies, not permanent labels.
    """
    pts=_recent_avg(log,"PTS")
    reb=_recent_avg(log,"REB")
    ast=_recent_avg(log,"AST")
    fga=_recent_avg(log,"FGA")
    fgm=_recent_avg(log,"FGM")
    fg3a=_recent_avg(log,"FG3A")
    fg3m=_recent_avg(log,"FG3M")
    tov=_recent_avg(log,"TOV")
    mins=_recent_avg(log,"MIN")
    fg_pct=_recent_pct(log,"FGM","FGA")
    fg3_pct=_recent_pct(log,"FG3M","FG3A")

    strengths=[]
    weaknesses=[]
    facts=[]

    if not pd.isna(mins):
        facts.append(f"recent minutes: {mins:.1f}")
    if not pd.isna(pts):
        facts.append(f"{pts:.1f} PPG")
        if pts>=18:
            strengths.append(f"high recent scoring volume ({pts:.1f} PPG)")
        elif pts<7 and not pd.isna(mins) and mins>=20:
            weaknesses.append(f"low scoring output for the recent minutes ({pts:.1f} PPG in {mins:.1f} MPG)")

    if not pd.isna(ast):
        facts.append(f"{ast:.1f} APG")
        if ast>=5:
            strengths.append(f"strong playmaking volume ({ast:.1f} APG)")
        elif ast<2 and not pd.isna(mins) and mins>=24:
            weaknesses.append(f"limited assist production ({ast:.1f} APG)")

    if not pd.isna(reb):
        facts.append(f"{reb:.1f} RPG")
        if reb>=7:
            strengths.append(f"strong rebounding volume ({reb:.1f} RPG)")
        elif reb<3 and not pd.isna(mins) and mins>=24:
            weaknesses.append(f"low rebounding volume ({reb:.1f} RPG)")

    if not pd.isna(fg3a):
        facts.append(f"{fg3a:.1f} 3PA/game")
        if fg3a>=4:
            if not pd.isna(fg3_pct) and fg3_pct>=.36:
                strengths.append(
                    f"high-volume efficient three-point shooting "
                    f"({fg3a:.1f} attempts/game, {fg3_pct*100:.1f}%)"
                )
            elif not pd.isna(fg3_pct) and fg3_pct<=.31:
                weaknesses.append(
                    f"high three-point volume but weak recent efficiency "
                    f"({fg3a:.1f} attempts/game, {fg3_pct*100:.1f}%)"
                )
            else:
                strengths.append(
                    f"high three-point volume ({fg3a:.1f} attempts/game)"
                )
        elif fg3a<=1.5 and not pd.isna(mins) and mins>=20:
            weaknesses.append(
                f"very low three-point volume ({fg3a:.1f} attempts/game)"
            )

    if not pd.isna(fg_pct):
        facts.append(f"{fg_pct*100:.1f}% FG")
        if fga>=8 if not pd.isna(fga) else False:
            if fg_pct>=.50:
                strengths.append(f"efficient overall shooting ({fg_pct*100:.1f}% FG)")
            elif fg_pct<=.40:
                weaknesses.append(f"inefficient recent shooting ({fg_pct*100:.1f}% FG)")

    if not pd.isna(tov) and tov>=3.2:
        weaknesses.append(f"high turnover volume ({tov:.1f} per game)")

    stl=_recent_avg(log,"STL")
    blk=_recent_avg(log,"BLK")
    stocks=(0 if pd.isna(stl) else stl)+(0 if pd.isna(blk) else blk)
    if not pd.isna(stl): facts.append(f"{stl:.1f} steals/game")
    if not pd.isna(blk): facts.append(f"{blk:.1f} blocks/game")
    if stocks>=1.5:
        strengths.append(f"active defensive event production ({stocks:.1f} steals + blocks/game)")
    # Role-adjusted rates help bench players show real skills without pretending
    # their raw totals are starter-level. These are descriptive per-36 rates.
    if not pd.isna(mins) and mins>=8:
        if not pd.isna(reb):
            reb36=reb/mins*36
            if reb36>=7:
                strengths.append(f"strong rebounding rate for the role ({reb36:.1f} per 36 minutes)")
        if not pd.isna(ast):
            ast36=ast/mins*36
            if ast36>=5:
                strengths.append(f"useful playmaking rate for the role ({ast36:.1f} assists per 36)")
        stocks36=stocks/mins*36
        if stocks36>=2.5:
            strengths.append(f"active defensive-event rate ({stocks36:.1f} steals + blocks per 36)")
    if not pd.isna(mins):
        if mins>=28:
            strengths.append(f"stable/high workload ({mins:.1f} MPG recently)")
        elif mins<18:
            weaknesses.append(f"limited recent role ({mins:.1f} MPG), which caps counting-stat volume")

    # Always show useful evidence, but never invent an absolute strength/weakness.
    if not strengths:
        candidates=[]
        for label,val in [("scoring",pts),("rebounding",reb),("playmaking",ast)]:
            if not pd.isna(val): candidates.append((float(val),label,float(val)))
        if candidates:
            _,label,val=max(candidates,key=lambda z:z[0])
            strengths.append(f"best recent counting-stat signal is {label} ({val:.1f} per game)")
        else:
            strengths.append("no verified recent strength could be calculated from the available stat rows")
    if not weaknesses:
        if not pd.isna(mins) and mins<24:
            weaknesses.append(f"role/volume is the main limitation ({mins:.1f} MPG recently)")
        else:
            weaknesses.append("no clear statistical weakness crossed the current evidence threshold")

    return {
        "strengths":strengths,
        "weaknesses":weaknesses,
        "facts":facts,
        "pts":pts,"reb":reb,"ast":ast,
        "fg3a":fg3a,"fg3_pct":fg3_pct,
        "fg_pct":fg_pct,"min":mins,
    }

def _texts_with_keywords(items,keywords):
    out=[]
    for x in items or []:
        lx=str(x).lower()
        if any(k in lx for k in keywords):
            out.append(str(x))
    return out

def basketball_player_vs_team_fit(
    game,
    league,
    log,
    player_name,
    player_position,
    opponent_abbr,
    prop_stat=None,
):
    """
    Match the player's actual recent style to the opponent's verified strengths/weaknesses.
    """
    player=basketball_player_tendency_profile(log)
    opponent_side="home" if opponent_abbr==game["home_abbr"] else "away"
    opp_report=opponent_style_exploit_report(
        game,
        league,
        opponent_side,
        player_position=player_position,
        player_log=log,
    )
    exact=basketball_style_matchup_notes(
        game,
        league,
        opponent_abbr,
        player_position=player_position,
        player_log=log,
    )

    helps=[]
    hurts=[]
    neutral=[]

    positives=(exact.get("positives") or []) + (opp_report.get("exploit") or [])
    negatives=(exact.get("negatives") or []) + (opp_report.get("caution") or [])

    stat=str(prop_stat or "").lower()
    wants_three=("3-point" in stat or "3-pointer" in stat or "three" in stat)
    wants_reb=("rebound" in stat or stat in {"pra","pr","ra"})
    wants_ast=("assist" in stat or stat in {"pra","pa","ra"})
    wants_pts=("point" in stat or stat in {"pra","pr","pa"})

    # 3-point fit.
    three_help=_texts_with_keywords(
        positives,
        ["three-point","three point","threes","perimeter"]
    )
    three_hurt=_texts_with_keywords(
        negatives,
        ["three-point","three point","threes","perimeter"]
    )

    if wants_three or (not pd.isna(player["fg3a"]) and player["fg3a"]>=4):
        if three_help:
            if not pd.isna(player["fg3_pct"]) and player["fg3_pct"]>=.36:
                helps.append(
                    f"{player_name}'s perimeter strength lines up with an opponent weakness: "
                    f"{player['fg3a']:.1f} recent 3PA/game at {player['fg3_pct']*100:.1f}%."
                )
            elif not pd.isna(player["fg3_pct"]) and player["fg3_pct"]<=.31:
                neutral.append(
                    f"The opponent matchup may help perimeter volume, BUT {player_name}'s own recent "
                    f"three-point efficiency is weak ({player['fg3_pct']*100:.1f}%). "
                    "The team weakness does not automatically erase the player's weakness."
                )
            else:
                helps.append(
                    f"{player_name} takes {player['fg3a']:.1f} threes per game recently, "
                    "so the opponent's perimeter weakness is directly relevant."
                )
        if three_hurt:
            if not pd.isna(player["fg3_pct"]) and player["fg3_pct"]<=.31:
                hurts.append(
                    f"Bad strength-vs-weakness combination: {player_name} is already shooting only "
                    f"{player['fg3_pct']*100:.1f}% from three recently, and the opponent's profile "
                    "also points to stronger perimeter defense."
                )
            else:
                hurts.append(
                    f"The opponent's perimeter profile could suppress {player_name}'s three-point production."
                )

    # Rebounding fit.
    reb_help=_texts_with_keywords(positives,["rebound","offensive board"])
    reb_hurt=_texts_with_keywords(negatives,["rebound","offensive board"])
    if wants_reb or (not pd.isna(player["reb"]) and player["reb"]>=6):
        if reb_help:
            helps.append(
                f"{player_name} averages {player['reb']:.1f} rebounds recently, and the opponent's "
                "rebounding weakness creates a style fit."
            )
        if reb_hurt:
            hurts.append(
                f"The opponent is stronger on the glass, which can work against {player_name}'s rebound line."
            )

    # Assists fit.
    ast_help=_texts_with_keywords(positives,["assist","playmaking","ball-handler"])
    ast_hurt=_texts_with_keywords(negatives,["assist","playmaking","ball-handler"])
    if wants_ast or (not pd.isna(player["ast"]) and player["ast"]>=4):
        if ast_help:
            helps.append(
                f"{player_name} averages {player['ast']:.1f} assists recently, and the opponent's "
                "assist/playmaking weakness fits that part of the player's game."
            )
        if ast_hurt:
            hurts.append(
                f"The opponent's playmaking/assist defense is a tougher fit for {player_name}."
            )

    # Scoring / pace / defense.
    score_help=_texts_with_keywords(
        positives,
        ["defensive rating","points allowed","faster","pace","scoring"]
    )
    score_hurt=_texts_with_keywords(
        negatives,
        ["defensive rating","points allowed","slower","pace","scoring"]
    )
    if wants_pts or (not pd.isna(player["pts"]) and player["pts"]>=12):
        if score_help:
            helps.append(
                f"The overall scoring environment is more favorable for {player_name}'s points-based production."
            )
        if score_hurt:
            hurts.append(
                f"The opponent's defense/pace profile can work against {player_name}'s points-based production."
            )

    if not helps:
        helps.append(
            "No clear player-strength/opponent-weakness match was verified for this stat."
        )
    if not hurts:
        hurts.append(
            "No clear opponent-strength/player-weakness clash was verified for this stat."
        )

    return {
        "player_strengths":player["strengths"],
        "player_weaknesses":player["weaknesses"],
        "player_facts":player["facts"],
        "helps":list(dict.fromkeys(helps)),
        "hurts":list(dict.fromkeys(hurts)),
        "opponent_weaknesses":opp_report.get("weaknesses",[]),
        "opponent_strengths":opp_report.get("strengths",[]),
    }

def nfl_player_vs_team_fit(
    game,
    log,
    player_name,
    player_position,
    opponent_abbr,
    prop_stat=None,
):
    opponent_side="home" if opponent_abbr==game["home_abbr"] else "away"
    report=nfl_style_matchup_notes(
        game,opponent_side,player_position=player_position
    )
    helps=list(report.get("positives",[]))
    hurts=list(report.get("negatives",[]))
    notes=list(report.get("notes",[]))

    # Recent prop-specific workload facts where possible.
    profile=[]
    for c,label in [
        ("TGT","targets"),
        ("REC","receptions"),
        ("RUSH_ATT","rush attempts"),
        ("CAR","carries"),
        ("PASS_ATT","pass attempts"),
        ("MIN","minutes"),
    ]:
        if c in log.columns:
            v=_recent_avg(log,c)
            if not pd.isna(v):
                profile.append(f"{v:.1f} {label}/game recently")

    if not helps:
        helps.append("No verified NFL defensive weakness clearly matches this player's prop.")
    if not hurts:
        hurts.append("No verified opponent strength clearly works against this player's prop.")

    return {
        "player_strengths":profile or ["Recent workload available in the player log."],
        "player_weaknesses":["Exact NFL defense-vs-position fields are only shown when a source actually returns them."],
        "player_facts":profile,
        "helps":helps,
        "hurts":hurts+notes,
        "opponent_weaknesses":[],
        "opponent_strengths":[],
    }

def player_vs_team_fit(
    game,league,log,player_name,player_position,opponent_abbr,prop_stat=None
):
    if league in {"NBA","WNBA"}:
        return basketball_player_vs_team_fit(
            game,league,log,player_name,player_position,opponent_abbr,prop_stat
        )
    return nfl_player_vs_team_fit(
        game,log,player_name,player_position,opponent_abbr,prop_stat
    )

def _full_player_game_data(game,league,row,fast_scan=False):
    """Unified player loader. Fast scans use local verified databases; final checks use live refreshes."""
    name=str(row.get("player",""))
    abbr=str(row.get("team_abbr","")).upper()
    opponent=game["home_abbr"] if abbr==game["away_abbr"] else game["away_abbr"]
    if fast_scan:
        if league=="NFL":
            log=local_nfl_player_log_v33(name,abbr,row.get("player_id",""))
        else:
            log=load_player_log_database(league,name)
        log=_prepare_player_log_v35(log,league)
        matchup=_exact_opponent_filter_v23(log,opponent)
        return {
            "name":name,"team":abbr,"opponent":opponent,
            "opponent_team_id":game.get("home_id","") if abbr==game.get("away_abbr") else game.get("away_id",""),
            "position":str(row.get("position","")),"team_id":str(row.get("team_id","")),
            "athlete_id":str(row.get("athlete_id","")),"player_id":"",
            "log":log,"matchup":matchup,"notes":["Fast scan used verified local history; final check refreshes live sources."],
            "error":None if not log.empty else "No verified local game log returned.",
        }
    try:
        result=fetch_player_research_v23(row,game,league)
        log=result.get("log",pd.DataFrame())
        matchup=result.get("matchup",pd.DataFrame())
        notes=result.get("notes",[])
        err=None if not log.empty else (" | ".join(notes[-3:]) if notes else "No usable player game log returned.")
        return {
            "name":name,"team":abbr,"opponent":result.get("opponent",opponent),
            "opponent_team_id":result.get("opponent_team_id",""),
            "position":str(row.get("position","")),"team_id":str(row.get("team_id","")),
            "athlete_id":str(row.get("athlete_id","")),"player_id":str(result.get("player_id","")),
            "log":log,"matchup":matchup,"notes":notes,"error":err,
        }
    except Exception as e:
        return {
            "name":name,"team":abbr,"opponent":opponent,"opponent_team_id":"",
            "position":str(row.get("position","")),"team_id":str(row.get("team_id","")),
            "athlete_id":str(row.get("athlete_id","")),"player_id":"","log":pd.DataFrame(),
            "matchup":pd.DataFrame(),"notes":[],"error":str(e),
        }


def scan_every_player_for_game(game,league,roster,ctx,progress_cb=None):
    """
    Full selected-game scan. Every roster row is attempted.
    Players with source errors/insufficient logs are reported in diagnostics.
    """
    ideas=[]
    diagnostics=[]
    total=len(roster)

    # Fetch team injury feeds once.
    injury_cache={}
    for team_id in roster.get("team_id",pd.Series(dtype=str)).astype(str).unique():
        if team_id:
            injury_cache[team_id]=fetch_injuries(league,team_id)

    for idx,(_,row) in enumerate(roster.reset_index(drop=True).iterrows(),1):
        name=str(row.get("player",""))
        if progress_cb:
            try: progress_cb(idx,total,name,"loading verified history")
            except Exception: pass
        # Current NFL prop engine supports offensive skill/QB markets only. Skip
        # linemen/defenders before any expensive work instead of timing out on a 90+ player roster.
        pos=str(row.get("position","") or "").upper()
        roster_status=str(row.get("roster_status",row.get("status","")) or "").upper()
        if league=="NFL" and roster_status and roster_status not in {"ACT","ACTIVE"}:
            diagnostics.append({"Player":name,"Status":"Skipped","Detail":f"Roster status {roster_status}; not treated as an active prop candidate."})
            continue
        if league=="NFL" and pos not in {"QB","RB","FB","WR","TE"}:
            diagnostics.append({"Player":name,"Status":"Skipped","Detail":f"{pos or 'Position unknown'} has no supported offensive player-prop family in this build."})
            continue
        d=_full_player_game_data(game,league,row,fast_scan=True)
        name=d["name"]

        if d["error"]:
            diagnostics.append({
                "Player":name,
                "Status":"Source error",
                "Detail":str(d["error"]),
            })
            continue
        if d["log"].empty:
            diagnostics.append({
                "Player":name,
                "Status":"No game log",
                "Detail":"No usable player game log returned.",
            })
            continue

        supported,volatile=generate_research_ideas(
            d["log"],league,d["opponent"]
        )

        if not supported and not volatile:
            diagnostics.append({
                "Player":name,
                "Status":"No line passed",
                "Detail":"Player was checked, but no current line passed the research filters.",
            })

        # Availability/minutes.
        injuries,inj_err=injury_cache.get(
            d["team_id"],(pd.DataFrame(),"No injury source")
        )
        side_key="home" if d["team"]==game["home_abbr"] else "away"
        team_ctx=ctx.get(side_key,{}) if ctx else {}
        if league=="NFL":
            minutes=nfl_workload_info_v33(d["log"],injuries,name,injury_verified=team_ctx.get("injury_verified",inj_err is None))
        else:
            minutes=estimate_player_availability_minutes(
                d["log"],
                injuries,
                name,
                injury_verified=team_ctx.get("injury_verified",inj_err is None),
            )

        # Add all supported ideas (not merely one per player).
        for idea in supported:
            stat=idea["Stat"]
            fit=player_vs_team_fit(
                game,league,d["log"],name,d["position"],d["opponent"],stat
            )

            # Build an analysis-shaped dict for data-quality/red-flag logic.
            analysis_proxy={
                "probability":idea.get("Model probability",np.nan),
                "last10":idea.get("Last 10",np.nan),
                "season":np.nan,
                "backtest_n":idea.get("Backtest N",0),
                "brier":idea.get("Brier",np.nan),
            }
            quality=research_data_quality(
                d["log"],
                d["matchup"],
                team_ctx.get("injury_verified",False),
                minutes,
                analysis_proxy,
            )
            flags=automatic_red_flags(
                d["log"],d["matchup"],minutes,analysis_proxy
            )

            ideas.append({
                "Player":name,
                "Team":d["team"],
                "Position":d["position"],
                "Opponent":d["opponent"],
                "Stat":stat,
                "Line":idea["Line"],
                "Model probability":idea.get("Model probability",np.nan),
                "Last 10":idea.get("Last 10",np.nan),
                "Vs opponent":idea.get("Vs opponent",np.nan),
                "Backtest N":idea.get("Backtest N",0),
                "Brier":idea.get("Brier",np.nan),
                "Minutes":nfl_workload_summary_text_v33(minutes) if league=="NFL" else minutes_summary_text(minutes),
                "Data quality":quality["label"],
                "Data quality score":quality["score"],
                "Helps":fit["helps"],
                "Hurts":fit["hurts"],
                "Player strengths":fit["player_strengths"],
                "Player weaknesses":fit["player_weaknesses"],
                "Opponent weaknesses":fit["opponent_weaknesses"],
                "Opponent strengths":fit["opponent_strengths"],
                "Red flags":flags,
                "Volatility":"Supported",
            })

        # Keep a few higher-risk possibilities too.
        for idea in volatile[:2]:
            stat=idea["Stat"]
            fit=player_vs_team_fit(
                game,league,d["log"],name,d["position"],d["opponent"],stat
            )
            ideas.append({
                "Player":name,
                "Team":d["team"],
                "Position":d["position"],
                "Opponent":d["opponent"],
                "Stat":stat,
                "Line":idea["Line"],
                "Model probability":idea.get("Model probability",np.nan),
                "Last 10":idea.get("Last 10",np.nan),
                "Vs opponent":idea.get("Vs opponent",np.nan),
                "Backtest N":idea.get("Backtest N",0),
                "Brier":idea.get("Brier",np.nan),
                "Minutes":nfl_workload_summary_text_v33(minutes) if league=="NFL" else minutes_summary_text(minutes),
                "Data quality":"—",
                "Data quality score":0,
                "Helps":fit["helps"],
                "Hurts":fit["hurts"],
                "Player strengths":fit["player_strengths"],
                "Player weaknesses":fit["player_weaknesses"],
                "Opponent weaknesses":fit["opponent_weaknesses"],
                "Opponent strengths":fit["opponent_strengths"],
                "Red flags":[],
                "Volatility":"Higher-risk",
            })

    if progress_cb:
        try: progress_cb(total,total,"","ranking results")
        except Exception: pass
    ideas=sorted(
        ideas,
        key=lambda r:(
            0 if r["Volatility"]=="Supported" else 1,
            -float(r["Model probability"] if not pd.isna(r["Model probability"]) else 0),
            -int(r["Data quality score"]),
            -int(r["Backtest N"]),
        )
    )
    return ideas,diagnostics,total

def explain_game_idea(row):
    p=row.get("Model probability",np.nan)
    lines=[
        f"**{row['Player']} — {row['Line']} {row['Stat']} vs {row['Opponent']}**",
        f"Estimated probability: **{fmt_pct(p)}** • "
        f"Last 10: **{fmt_pct(row.get('Last 10'))}** • "
        f"Vs opponent: **{fmt_pct(row.get('Vs opponent'))}** • "
        f"Data quality: **{row.get('Data quality','—')}**",
        f"Availability/workload: **{row.get('Minutes','—')}**",
    ]
    return "\n\n".join(lines)


# ============================================================
# V18: SIMPLE "FIND BETS" EXPERIENCE
# ============================================================

def quick_probability_band(probability, quality_label):
    if probability is None or pd.isna(probability):
        return np.nan,np.nan
    spread={
        "HIGH":.035,
        "MEDIUM":.055,
        "LOW":.085,
    }.get(str(quality_label).upper(),.07)
    return max(.01,float(probability)-spread),min(.99,float(probability)+spread)

def quick_idea_agreement(row):
    """
    Eight independent-ish research checks. This is a compact decision aid,
    not eight truly independent statistical models.
    """
    checks=[]

    p=row.get("Model probability",np.nan)
    checks.append(("Model support",not pd.isna(p) and p>=.60))

    l10=row.get("Last 10",np.nan)
    checks.append(("Recent form",not pd.isna(l10) and l10>=.60))

    opp=row.get("Vs opponent",np.nan)
    checks.append(("Exact-opponent history",not pd.isna(opp) and opp>=.50))

    q=str(row.get("Data quality","")).upper()
    checks.append(("Data quality",q in {"HIGH","MEDIUM"}))

    mins=str(row.get("Minutes","")).upper()
    healthy=not any(
        x in mins
        for x in [
            "OUT","DOUBTFUL","QUESTIONABLE",
            "NOT VERIFIED","RESTRICTION"
        ]
    )
    checks.append(("Minutes / availability",healthy))

    helps=row.get("Helps") or []
    hurts=row.get("Hurts") or []
    useful_help=sum(
        1 for x in helps
        if "no clear" not in str(x).lower()
    )
    useful_hurt=sum(
        1 for x in hurts
        if "no clear" not in str(x).lower()
        and "no major" not in str(x).lower()
    )
    checks.append(("Matchup fit",useful_help>useful_hurt))

    bt=int(row.get("Backtest N",0) or 0)
    checks.append(("Backtest sample",bt>=12))

    brier=row.get("Brier",np.nan)
    checks.append((
        "Probability error",
        not pd.isna(brier) and float(brier)<=.25
    ))

    passed=sum(1 for _,ok in checks if ok)
    return passed,checks

def quick_support_score(row):
    p=row.get("Model probability",np.nan)
    p=0 if pd.isna(p) else float(p)

    q=float(row.get("Data quality score",0) or 0)/100
    l10=row.get("Last 10",np.nan)
    l10=.5 if pd.isna(l10) else float(l10)
    opp=row.get("Vs opponent",np.nan)
    opp=.5 if pd.isna(opp) else float(opp)

    agreement,_=quick_idea_agreement(row)
    agree=agreement/8

    red=len(row.get("Red flags") or [])
    helps=sum(
        1 for x in (row.get("Helps") or [])
        if "no clear" not in str(x).lower()
    )
    hurts=sum(
        1 for x in (row.get("Hurts") or [])
        if "no clear" not in str(x).lower()
    )

    market_bonus=0.0
    if row.get("Live market"):
        edge=row.get("Market edge",np.nan)
        price_edge=row.get("Price edge",np.nan)
        if not pd.isna(edge):
            market_bonus += .22*max(-.15,min(.15,float(edge)))/.15
        if not pd.isna(price_edge):
            market_bonus += .08*max(-.15,min(.15,float(price_edge)))/.15

    return (
        .31*p
        + .14*l10
        + .07*opp
        + .14*q
        + .16*agree
        + .025*min(helps,3)
        - .03*min(hurts,3)
        - .035*min(red,3)
        + market_bonus
    )

def rank_quick_ideas(ideas):
    supported=[
        dict(x)
        for x in ideas
        if x.get("Volatility")=="Supported"
    ]
    for row in supported:
        row["_quick_score"]=quick_support_score(row)
        row["_agreement"]=quick_idea_agreement(row)[0]
    return sorted(
        supported,
        key=lambda r:(
            -r["_quick_score"],
            -r["_agreement"],
            -float(
                0 if pd.isna(r.get("Model probability",np.nan))
                else r["Model probability"]
            ),
        ),
    )

def quick_idea_reasons(row):
    reasons=[]

    if row.get("Live market"):
        edge=row.get("Market edge",np.nan)
        pe=row.get("Price edge",np.nan)
        if not pd.isna(edge) and edge>0:
            reasons.append(f"model is {edge*100:.1f} percentage points above the de-vigged sportsbook consensus for this exact line")
        if not pd.isna(pe) and pe>0:
            reasons.append(f"model is {pe*100:.1f} points above the break-even probability at the best current price")

    l10=row.get("Last 10",np.nan)
    if not pd.isna(l10) and l10>=.60:
        reasons.append(
            f"hit this line in {l10*100:.0f}% of the recent 10-game sample"
        )

    opp=row.get("Vs opponent",np.nan)
    if not pd.isna(opp) and opp>=.50:
        reasons.append(
            f"exact-opponent history is supportive ({opp*100:.0f}% hit rate)"
        )

    for x in row.get("Helps") or []:
        lx=str(x).lower()
        if "no clear" not in lx and "no major" not in lx:
            reasons.append(str(x))
        if len(reasons)>=4:
            break

    if len(reasons)<3:
        for x in row.get("Player strengths") or []:
            if "no single" not in str(x).lower():
                reasons.append(str(x))
            if len(reasons)>=4:
                break

    if not reasons:
        reasons.append(
            "the player's own historical distribution passed the stronger-support filter"
        )

    return list(dict.fromkeys(reasons))[:4]

def quick_idea_concern(row):
    red=row.get("Red flags") or []
    if red:
        return str(red[0])

    for x in row.get("Hurts") or []:
        lx=str(x).lower()
        if "no clear" not in lx and "no major" not in lx:
            return str(x)

    q=str(row.get("Data quality","")).upper()
    if q=="LOW":
        return "Data quality is low, so the probability estimate has more uncertainty."

    return (
        "No major specific warning was found, but one-game variance still matters."
    )

def quick_idea_label(row):
    agreement,_=quick_idea_agreement(row)
    p=row.get("Model probability",np.nan)
    q=str(row.get("Data quality","")).upper()
    mins=str(row.get("Minutes","")).upper()
    flags=row.get("Red flags") or []

    if any(
        x in mins
        for x in ["OUT","DOUBTFUL","QUESTIONABLE","NOT VERIFIED"]
    ):
        return "WAIT FOR AVAILABILITY","warning"

    if flags and any(
        "OUT" in str(x).upper()
        or "RESTRICTION" in str(x).upper()
        for x in flags
    ):
        return "HOLD OFF","warning"

    if row.get("Live market"):
        edge=row.get("Market edge",np.nan)
        price_edge=row.get("Price edge",np.nan)
        if (
            not pd.isna(p) and not pd.isna(edge)
            and p>=.55 and float(edge)>=.04
            and agreement>=5 and q in {"HIGH","MEDIUM"}
            and (pd.isna(price_edge) or float(price_edge)>=.015)
        ):
            return "LIVE MARKET EDGE — STRONGER SUPPORT","success"
        if not pd.isna(edge) and float(edge)>=.02 and not pd.isna(p) and p>=.53:
            return "LIVE LINE — SOME SUPPORT","info"
        if not pd.isna(edge) and float(edge)<=0:
            return "MARKET AT/ABOVE MODEL — WEAK / PASS","warning"
        return "LIVE LINE — TOO CLOSE / MORE RISK","warning"

    if (
        not pd.isna(p)
        and p>=.64
        and agreement>=6
        and q in {"HIGH","MEDIUM"}
    ):
        return "STRONGER-SUPPORTED IDEA","success"

    if not pd.isna(p) and p>=.58 and agreement>=4:
        return "SOME SUPPORT","info"

    return "WEAK / PASS","warning"

def _fresh_selected_game(game,league):
    selected_date=game.get("schedule_date") or st.session_state.get(
        "selected_date",date.today()
    )
    if isinstance(selected_date,str):
        try:
            selected_date=datetime.fromisoformat(
                selected_date.replace("Z","+00:00")
            ).date()
        except Exception:
            selected_date=date.today()

    games,err=fetch_schedule(league,selected_date)
    if err:
        return None,err

    event_id=str(game.get("event_id",""))
    fresh=next(
        (
            g for g in games
            if str(g.get("event_id",""))==event_id
        ),
        None,
    )
    if not fresh:
        return None,"The selected game was not returned by the fresh schedule feed."

    fresh["schedule_date"]=selected_date
    return fresh,None

def refresh_live_market_for_idea_v37(game,league,roster,idea):
    if not idea.get("Live market"):
        return None,{},None
    key=get_the_odds_api_key()
    if not key:
        return None,{},"THE_ODDS_API_KEY is not configured."
    event_id=str(idea.get("Odds event ID","") or "")
    meta={}
    if not event_id:
        events,emeta,eerr=fetch_the_odds_events_v37(league,key)
        meta.update(emeta)
        if eerr:
            return None,meta,eerr
        event,merr=match_the_odds_event_v37(game,events)
        if merr:
            return None,meta,merr
        event_id=str(event.get("id","") or "")
        meta["event"]=event
    payload,pmeta,perr=fetch_the_odds_event_props_v37(league,event_id,key)
    meta.update({k:v for k,v in pmeta.items() if v is not None})
    if perr:
        return None,meta,perr
    entries,diagnostics=parse_the_odds_props_v37(payload,league,roster)
    markets=consensus_live_prop_markets_v37(entries)
    player_key=_name_key(idea.get("Player",""))
    stat=str(idea.get("Stat",""))
    side=str(idea.get("Market side","") or "")
    same=[m for m in markets if _name_key(m.get("Player",""))==player_key and str(m.get("Stat",""))==stat and str(m.get("Side",""))==side]
    if not same:
        return None,meta,"The exact player/stat/side is no longer posted by the queried sportsbooks."
    old_line=safe_float(idea.get("Line value"))
    if pd.isna(old_line):
        mm=re.search(r"(\d+(?:\.\d+)?)",str(idea.get("Line","")))
        old_line=float(mm.group(1)) if mm else np.nan
    if not pd.isna(old_line):
        exact=[m for m in same if abs(float(m.get("Line value",np.nan))-float(old_line))<1e-9]
        if exact:
            return exact[0],meta,None
        same=sorted(same,key=lambda m:abs(float(m.get("Line value",0))-float(old_line)))
    return same[0],meta,None


def final_check_game_idea(game,league,roster,idea):
    """Fresh single-idea verification without clearing the heavy local-data caches."""
    # Clear only live endpoints that matter to this exact recheck. Do NOT clear
    # the bundled CSV/parquet caches; that was making the app unnecessarily slow.
    for fn in [
        fetch_schedule,fetch_injuries,fetch_the_odds_events_v37,fetch_the_odds_event_props_v37,
        fetch_nba_player_by_name,fetch_wnba_player_by_name,fetch_espn_player_gamelog_multi,
    ]:
        try:
            fn.clear()
        except Exception:
            pass

    fresh_game,err=_fresh_selected_game(game,league)
    if err:
        return {"status":"COULD NOT VERIFY","kind":"warning","messages":[err]}

    fresh_ctx=build_matchup_context(fresh_game,league)
    player=str(idea["Player"]); team=str(idea["Team"])
    match=roster[
        (roster["player"].map(_name_key)==_name_key(player))
        & (roster["team_abbr"].astype(str).str.upper()==team.upper())
    ]
    if len(match)!=1:
        return {
            "status":"COULD NOT VERIFY","kind":"warning",
            "messages":["Player could not be matched to exactly one loaded roster identity."],
        }

    row=match.iloc[0]
    d=_full_player_game_data(fresh_game,league,row)
    if d.get("error") or d["log"].empty:
        return {
            "status":"COULD NOT VERIFY","kind":"warning",
            "messages":[str(d.get("error") or "Fresh player game log was unavailable.")],
        }

    props=available_props(d["log"],league)
    stat=str(idea["Stat"])
    if stat not in props:
        return {
            "status":"COULD NOT VERIFY","kind":"warning",
            "messages":[f"Fresh player data no longer returned the {stat} field."],
        }

    mm=re.search(r"(\d+(?:\.\d+)?)",str(idea.get("Line","")))
    if not mm:
        return {"status":"COULD NOT VERIFY","kind":"warning","messages":["Could not parse the saved prop line."]}
    old_line=float(mm.group(1))
    side=str(idea.get("Market side") or ("Under" if str(idea.get("Line","")).lower().startswith("under") else "Over" if str(idea.get("Line","")).lower().startswith("over") else "At least (X+)"))
    line=old_line
    market_probability=None
    fresh_market=None
    market_meta={}
    market_err=None
    if idea.get("Live market"):
        fresh_market,market_meta,market_err=refresh_live_market_for_idea_v37(fresh_game,league,roster,idea)
        if fresh_market:
            line=float(fresh_market.get("Line value",old_line))
            market_probability=fresh_market.get("Market probability",np.nan)

    side_key="home" if team==fresh_game["home_abbr"] else "away"
    analysis=easy_prop_analysis(
        d["log"],d["matchup"],league,props[stat],side,line,
        market_probability=market_probability,
        current_side=side_key,opponent=d["opponent"],
    )

    injuries,inj_err=fetch_injuries(league,d["team_id"],fresh_game.get("event_id",""))
    team_ctx=fresh_ctx.get(side_key,{})
    if league=="NFL":
        workload=nfl_workload_info_v33(d["log"],injuries,player,injury_verified=team_ctx.get("injury_verified",inj_err is None))
        workload_text=nfl_workload_summary_text_v33(workload)
    else:
        workload=estimate_player_availability_minutes(d["log"],injuries,player,injury_verified=team_ctx.get("injury_verified",inj_err is None))
        workload_text=minutes_summary_text(workload)

    fresh_fit=player_vs_team_fit(fresh_game,league,d["log"],player,d["position"],d["opponent"],stat)
    quality=research_data_quality(d["log"],d["matchup"],team_ctx.get("injury_verified",False),workload,analysis)

    messages=[]
    old_p=idea.get("Model probability",np.nan); new_p=analysis.get("probability",np.nan)
    messages.append(f"Fresh estimated probability: {fmt_pct(new_p)} (previous scan: {fmt_pct(old_p)}).")
    messages.append(("Fresh workload / availability: " if league=="NFL" else "Fresh minutes / availability: ")+workload_text)
    messages.append(f"Fresh data quality: {quality['label']} ({quality['score']}/100).")

    if idea.get("Live market"):
        if fresh_market:
            best=fresh_market.get("Best odds",np.nan)
            messages.append(
                f"Fresh sportsbook market: {side} {line:g} {stat} at "
                + ("—" if pd.isna(best) else f"{float(best):+.0f}")
                + f" ({fresh_market.get('Best book','—')}); de-vigged consensus {fmt_pct(fresh_market.get('Market probability'))}."
            )
            if abs(line-old_line)>1e-9:
                messages.append(f"Player prop line moved: {old_line:g} → {line:g}.")
            old_best=idea.get("Best odds",np.nan)
            if not pd.isna(old_best) and not pd.isna(best) and float(old_best)!=float(best):
                messages.append(f"Best price moved: {float(old_best):+.0f} → {float(best):+.0f}.")
            edge=analysis.get("edge",np.nan)
            if not pd.isna(edge):
                messages.append(f"Fresh model-vs-market difference: {float(edge)*100:+.1f} percentage points.")
            rem=market_meta.get("remaining")
            if rem not in [None,""]:
                messages.append(f"Odds API credits remaining after this refresh: {rem}.")
        elif market_err:
            messages.append("Live sportsbook recheck: "+str(market_err))

    # Team-line movement is secondary context.
    old_odds=game.get("odds") or {}; new_odds=fresh_game.get("odds") or {}
    old_detail=old_odds.get("details"); new_detail=new_odds.get("details")
    old_total=old_odds.get("total"); new_total=new_odds.get("total")
    if old_detail and new_detail and old_detail!=new_detail:
        messages.append(f"Game line changed: {old_detail} → {new_detail}.")
    if old_total is not None and new_total is not None and not pd.isna(safe_float(old_total)) and not pd.isna(safe_float(new_total)) and float(safe_float(old_total))!=float(safe_float(new_total)):
        messages.append(f"Game total changed: {old_total} → {new_total}.")

    status=str(workload.get("status",""))
    fresh_edge=analysis.get("edge",np.nan)
    if status.startswith("OUT"):
        final="HOLD UP — PLAYER OUT"; kind="error"
    elif status in {"QUESTIONABLE / DAY-TO-DAY","DOUBTFUL","INJURY STATUS NOT VERIFIED"}:
        final="HOLD UP — AVAILABILITY UNCERTAIN"; kind="warning"
    elif workload.get("restriction"):
        final="HOLD UP — WORKLOAD RESTRICTION"; kind="warning"
    elif idea.get("Live market") and fresh_market is None:
        final="LIVE MARKET NO LONGER VERIFIED"; kind="warning"
    elif idea.get("Live market") and not pd.isna(fresh_edge) and float(fresh_edge)<=0:
        final="MARKET EDGE NO LONGER PRESENT"; kind="warning"
    elif not pd.isna(new_p) and new_p<.55:
        final="SUPPORT WEAKENED"; kind="warning"
    elif not pd.isna(old_p) and not pd.isna(new_p) and float(old_p)-float(new_p)>=.08:
        final="SUPPORT DROPPED — REVIEW AGAIN"; kind="warning"
    else:
        final="CURRENT CHECK STILL SUPPORTS THE RESEARCH"; kind="success"

    for x in fresh_fit.get("helps",[]):
        if "no clear" not in str(x).lower():
            messages.append("Matchup still helps: "+str(x)); break
    for x in fresh_fit.get("hurts",[]):
        if "no clear" not in str(x).lower():
            messages.append("Main fresh matchup concern: "+str(x)); break

    return {
        "status":final,"kind":kind,"messages":messages,
        "checked_at":datetime.now().strftime("%I:%M %p"),
        "fresh_game":fresh_game,"fresh_ctx":fresh_ctx,
    }

def save_quick_idea_to_watchlist(row,league):
    append_csv(
        WATCHLIST_FILE,
        {
            "saved_at":datetime.now().isoformat(timespec="seconds"),
            "league":league,
            "selection":f"{row['Player']} — {row['Line']} {row['Stat']}",
            "opponent":row.get("Opponent",""),
            "model_probability":row.get("Model probability",np.nan),
            "confidence":row.get("Data quality",""),
            "data_verdict":quick_idea_label(row)[0],
            "planned_amount":0.0,
            "status":"Researching",
            "result":"",
            "profit_loss":"",
        },
    )



def load_player_v11(row,game,league):
    pname=str(row.get("player",""))
    pteam=str(row.get("team_abbr",""))
    athlete_id=str(row.get("athlete_id","") or "")

    # If a manually resolved player has no team abbreviation, infer the selected
    # game side only when the team_id matches. Otherwise leave it unknown and
    # let public-source research identify the role rather than guessing.
    row_team_id=str(row.get("team_id","") or "")
    if not pteam and row_team_id:
        if row_team_id==str(game.get("away_id","")):
            pteam=game["away_abbr"]
        elif row_team_id==str(game.get("home_id","")):
            pteam=game["home_abbr"]

    if pteam==game.get("away_abbr"):
        opponent=game.get("home_abbr","")
        opponent_team_id=game.get("home_id","")
        today_side="away"
    elif pteam==game.get("home_abbr"):
        opponent=game.get("away_abbr","")
        opponent_team_id=game.get("away_id","")
        today_side="home"
    else:
        opponent=""
        opponent_team_id=""
        today_side=""

    selected_date=st.session_state.get("selected_date",date.today())
    season=league_season_year(league,selected_date)

    # Clear stale values immediately. A failed source must never leave the
    # previous player's matchup sample on screen.
    st.session_state["research_error"]=""
    st.session_state["research_log"]=pd.DataFrame()
    st.session_state["research_matchup_log"]=pd.DataFrame()
    st.session_state["research_name"]=pname
    st.session_state["research_league"]=league
    st.session_state["research_team"]=pteam
    st.session_state["research_opponent"]=opponent
    st.session_state["research_today_side"]=today_side
    st.session_state["research_team_id"]=row_team_id
    st.session_state["research_opponent_team_id"]=str(opponent_team_id or "")
    st.session_state["research_player_id"]=""
    st.session_state["research_athlete_id"]=athlete_id

    with st.spinner(f"Researching {pname}" + (f" vs {opponent}" if opponent else "") + "..."):
        matchup_log=pd.DataFrame()
        player_id=""
        player={"full_name":pname}
        err=None

        if league=="NBA":
            log,player,err=fetch_nba_player_by_name(
                pname,nba_season_string(selected_date)
            )
            if player:
                player_id=str(player.get("id",""))
            if not log.empty and opponent:
                # Exact opponent from returned log is a safe fallback even when
                # the specific matchup endpoint fails.
                matchup_log=exact_opponent_rows_from_player_log(log,opponent)
            if player_id and opponent_team_id:
                exact,exact_err=fetch_basketball_player_matchup_logs(
                    "NBA",player_id,opponent_team_id,selected_date,3
                )
                if not exact.empty:
                    matchup_log=exact

        elif league=="WNBA":
            log,player,err=fetch_wnba_player_best_available(
                pname,athlete_id,season
            )
            player_id=str((player or {}).get("id",""))
            if not log.empty and opponent:
                matchup_log=exact_opponent_rows_from_player_log(log,opponent)

            # Only call the NBA/WNBA exact matchup endpoint when its player ID
            # was successfully resolved; failure here does not erase ESPN data.
            if player_id and opponent_team_id:
                exact,exact_err=fetch_basketball_player_matchup_logs(
                    "WNBA",player_id,opponent_team_id,selected_date,3
                )
                if not exact.empty:
                    matchup_log=exact

        else:
            log,err=fetch_espn_player_gamelog_multi(
                "NFL",athlete_id,season,3
            )
            player={"full_name":pname}
            if not log.empty and opponent:
                matchup_log=exact_opponent_rows_from_player_log(log,opponent)

    if log is None:
        log=pd.DataFrame()

    st.session_state["research_log"]=log
    st.session_state["research_matchup_log"]=matchup_log
    st.session_state["research_name"]=(player or {}).get("full_name",pname)
    st.session_state["research_player_id"]=player_id
    st.session_state["research_athlete_id"]=athlete_id or str((player or {}).get("athlete_id",""))

    if err and log.empty:
        st.session_state["research_error"]=(
            str(err)
            + " Public-source web research is still available below."
        )
    else:
        st.session_state["research_error"]=""





# ============================================================
# V20 UI HELPERS — FULL RESEARCH, LESS REPETITION
# ============================================================

def v20_team_profile_df(game):
    rows=[]
    for side in ["away","home"]:
        stats=game.get(f"{side}_stats") or {}
        records=game.get(f"{side}_records") or {}
        abbr=game[f"{side}_abbr"]
        rows.append({
            "Team":abbr,
            "Overall":records.get("overall","—"),
            "Home/Road":(
                records.get("road","—")
                if side=="away"
                else records.get("home","—")
            ),
            "PPG":_scoreboard_stat_text(stats,"avgPoints"),
            "RPG":_scoreboard_stat_text(stats,"avgRebounds"),
            "APG":_scoreboard_stat_text(stats,"avgAssists"),
            "FG%":_scoreboard_stat_text(stats,"fieldGoalPct"),
            "3P%":_scoreboard_stat_text(
                stats,
                "threePointPct"
                if "threePointPct" in stats
                else "threePointFieldGoalPct"
            ),
        })
    return pd.DataFrame(rows)

def v20_injury_summary(ctx,side):
    t=(ctx or {}).get(side,{})
    counts=t.get("injuries",{})
    if not t.get("injury_verified",False):
        return "NOT VERIFIED"
    return (
        f"{counts.get('out',0)} out/doubtful • "
        f"{counts.get('questionable',0)} questionable"
    )

def v20_top_game_ideas(ideas):
    supported=[dict(x) for x in ideas if x.get("Volatility")=="Supported"]
    if any(x.get("Live market") for x in supported):
        # A live-market Find a Bet result must have an actual positive discrepancy;
        # high historical hit rate by itself is not enough to promote a line.
        supported=[
            x for x in supported
            if not pd.isna(x.get("Market edge",np.nan))
            and float(x.get("Market edge"))>=.02
            and not pd.isna(x.get("Price edge",np.nan))
            and float(x.get("Price edge"))>0
            and str(x.get("Data quality","")).upper() in {"HIGH","MEDIUM"}
        ]
    for row in supported:
        row["_score"]=quick_support_score(row)
        row["_agreement"]=quick_idea_agreement(row)[0]
    return sorted(
        supported,
        key=lambda r:(
            -r["_score"],
            -r["_agreement"],
            -float(0 if pd.isna(r.get("Market edge",np.nan)) else r.get("Market edge",0)),
            -float(0 if pd.isna(r.get("Model probability",np.nan)) else r.get("Model probability",0)),
        ),
    )

def v20_readable_opponent_sample(n):
    if n is None or pd.isna(n):
        return "Opponent history unavailable"
    n=int(n)
    if n==0:
        return "No prior games vs this opponent were found"
    if n==1:
        return "Only 1 prior game vs this opponent — very small sample"
    if n==2:
        return "Only 2 prior games vs this opponent — small sample"
    return f"{n} prior games vs this opponent"

def v20_render_injuries(game,ctx):
    left,right=st.columns(2)
    for side,col in [("away",left),("home",right)]:
        t=ctx.get(side,{})
        with col:
            st.markdown(f"### {t.get('abbr',game[f'{side}_abbr'])}")
            if not t.get("injury_verified",False):
                st.warning(
                    "The structured ESPN injury feed was not verified. Missing rows do NOT mean nobody is injured."
                )
            else:
                counts=t.get("injuries",{})
                st.write(
                    f"**ESPN feed: {counts.get('out',0)} out/doubtful • "
                    f"{counts.get('questionable',0)} questionable**"
                )
                st.caption(
                    "Players are deduplicated before counting. Use the live public-source research above for the final pregame cross-check."
                )
            inj=clean_injury_rows(t.get("injury_df",pd.DataFrame()))
            if inj is not None and not inj.empty:
                show=[
                    c for c in
                    ["Player","Status","Injury","Detail","ReturnDate"]
                    if c in inj.columns
                ]
                st.dataframe(
                    inj[show].head(15),
                    use_container_width=True,
                    hide_index=True,
                )
            elif t.get("injury_verified",False):
                st.success("The structured ESPN feed returned no listed players; final public-source cross-check is still recommended.")

def v20_render_market_snapshot(game,league):
    odds=game.get("odds") or {}
    a,b,c=st.columns(3)
    a.metric("Current spread",odds.get("details") or "—")
    b.metric(
        "Game total",
        "—"
        if pd.isna(odds.get("total",np.nan))
        else f"{odds['total']:.1f}"
    )
    c.metric(
        "Odds provider",
        odds.get("provider") or "—",
    )

    research=team_win_research(game,league)
    cons=research.get("consensus")
    if cons:
        x,y=st.columns(2)
        x.metric(
            f"{game['away_abbr']} outside win estimate",
            fmt_pct(cons["away_probability"]),
        )
        y.metric(
            f"{game['home_abbr']} outside win estimate",
            fmt_pct(cons["home_probability"]),
        )
        st.caption(
            f"De-vigged consensus from {cons['books']} outside sportsbook source(s)."
        )
    elif research.get("consensus_error"):
        st.caption(
            "Outside win estimate unavailable: "
            + str(research["consensus_error"])
        )

def v20_render_player_style(game,league,log,player_name,position,opponent):
    fit=player_vs_team_fit(
        game,
        league,
        log,
        player_name,
        position,
        opponent,
        prop_stat=None,
    )
    left,right=st.columns(2)
    with left:
        st.markdown("### ✅ What helps")
        for x in fit.get("player_strengths",[])[:4]:
            st.write("• "+str(x))
        for x in fit.get("helps",[])[:5]:
            if "no clear" not in str(x).lower():
                st.write("✅ "+str(x))
    with right:
        st.markdown("### ⚠️ What works against him/her")
        for x in fit.get("player_weaknesses",[])[:4]:
            st.write("• "+str(x))
        for x in fit.get("hurts",[])[:5]:
            if "no clear" not in str(x).lower():
                st.write("⚠️ "+str(x))
    return fit



# ============================================================
# V21: BASKETBALL ADVANCED-STATS FALLBACKS
# ============================================================

def basketball_verified_fallback_profile(game,ctx):
    """
    When official pace/ratings are unavailable, summarize only metrics that can
    be supported by the already-loaded scoreboard and team-history feeds.

    IMPORTANT: this does NOT manufacture pace, offensive rating, or defensive
    rating. It labels the substitutes as recent scoring/defensive proxies.
    """
    rows=[]
    for side in ["away","home"]:
        abbr=game[f"{side}_abbr"]
        t=(ctx or {}).get(side,{})
        recent=t.get("recent10",{}) or {}
        season=t.get("season",{}) or {}
        stats=game.get(f"{side}_stats") or {}

        recent_for=recent.get("avg_for",np.nan)
        recent_against=recent.get("avg_against",np.nan)
        recent_margin=recent.get("margin",np.nan)

        season_for=season.get("avg_for",np.nan)
        season_against=season.get("avg_against",np.nan)

        scoreboard_ppg=safe_float(
            (stats.get("avgPoints") or {}).get("value")
        )

        offense=(
            recent_for
            if not pd.isna(recent_for)
            else (
                scoreboard_ppg
                if not pd.isna(scoreboard_ppg)
                else season_for
            )
        )
        defense=(
            recent_against
            if not pd.isna(recent_against)
            else season_against
        )

        scoring_env=(
            recent_for+recent_against
            if not pd.isna(recent_for) and not pd.isna(recent_against)
            else np.nan
        )

        rows.append({
            "Team":abbr,
            "Recent points scored":offense,
            "Recent points allowed":defense,
            "Recent margin":recent_margin,
            "Recent combined scoring":scoring_env,
            "Official pace":np.nan,
            "Official offensive rating":np.nan,
            "Official defensive rating":np.nan,
        })

    return pd.DataFrame(rows)

def basketball_fallback_takeaways(game,ctx):
    df=basketball_verified_fallback_profile(game,ctx)
    if df.empty or len(df)<2:
        return []

    notes=[]
    away=df.iloc[0]
    home=df.iloc[1]

    af=safe_float(away["Recent points scored"])
    hf=safe_float(home["Recent points scored"])
    aa=safe_float(away["Recent points allowed"])
    ha=safe_float(home["Recent points allowed"])
    ae=safe_float(away["Recent combined scoring"])
    he=safe_float(home["Recent combined scoring"])

    if not pd.isna(af) and not pd.isna(hf):
        if af>=hf+2:
            notes.append(
                f"{away['Team']} has the stronger recent scoring output "
                f"({af:.1f} vs {hf:.1f} points per game)."
            )
        elif hf>=af+2:
            notes.append(
                f"{home['Team']} has the stronger recent scoring output "
                f"({hf:.1f} vs {af:.1f} points per game)."
            )
        else:
            notes.append(
                f"Recent scoring output is fairly close: "
                f"{away['Team']} {af:.1f}, {home['Team']} {hf:.1f}."
            )

    if not pd.isna(aa) and not pd.isna(ha):
        if aa<=ha-2:
            notes.append(
                f"{away['Team']} has allowed fewer recent points "
                f"({aa:.1f} vs {ha:.1f}), a better simple defensive-results signal."
            )
        elif ha<=aa-2:
            notes.append(
                f"{home['Team']} has allowed fewer recent points "
                f"({ha:.1f} vs {aa:.1f}), a better simple defensive-results signal."
            )

    if not pd.isna(ae) and not pd.isna(he):
        faster_like=away if ae>he else home
        slower_like=home if ae>he else away
        diff=abs(ae-he)
        if diff>=5:
            notes.append(
                f"{faster_like['Team']}'s recent games have had a higher combined-scoring "
                f"environment than {slower_like['Team']}'s. "
                "That can mean more statistical opportunity, but it is NOT the same thing as verified pace."
            )

    notes.append(
        "Official possession-based pace/offensive-rating/defensive-rating data was unavailable, "
        "so the app is intentionally not inventing those numbers."
    )
    return notes

def render_basketball_advanced_or_fallback(game,league,ctx):
    selected_date=st.session_state.get("selected_date",date.today())
    adv,err=fetch_basketball_advanced(league,selected_date)

    advanced_rows=[]
    if not adv.empty and "TEAM_ABBREVIATION" in adv.columns:
        for abbr in [game["away_abbr"],game["home_abbr"]]:
            row=adv[
                adv["TEAM_ABBREVIATION"].astype(str).str.upper()==abbr.upper()
            ]
            if not row.empty:
                r=row.iloc[0]
                advanced_rows.append({
                    "Team":abbr,
                    "Offensive rating":safe_float(r.get("OFF_RATING")),
                    "Defensive rating":safe_float(r.get("DEF_RATING")),
                    "Net rating":safe_float(r.get("NET_RATING")),
                    "Pace":safe_float(r.get("PACE")),
                })

    if advanced_rows:
        st.success("Official NBA/WNBA advanced team stats loaded.")
        st.dataframe(
            pd.DataFrame(advanced_rows),
            use_container_width=True,
            hide_index=True,
        )
        return advanced_rows,True

    st.warning(
        "Official pace/ratings did not load, so the app is using verified fallback team results instead of leaving this section empty."
    )
    fallback=basketball_verified_fallback_profile(game,ctx).copy()

    display_cols=[
        "Team",
        "Recent points scored",
        "Recent points allowed",
        "Recent margin",
        "Recent combined scoring",
    ]
    for c in display_cols[1:]:
        if c in fallback.columns:
            fallback[c]=fallback[c].map(
                lambda x:"—" if pd.isna(x) else f"{float(x):.1f}"
            )

    st.dataframe(
        fallback[display_cols],
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "These are scoring/defensive-result proxies from verified team history. "
        "They are NOT possession-based pace or efficiency ratings."
    )

    for note in basketball_fallback_takeaways(game,ctx):
        st.write("• "+note)

    if err:
        with st.expander("Why official advanced stats were unavailable"):
            st.code(str(err))

    return [],False


# ============================================================
# V21 FULL CHECKLIST: HISTORICAL MODELS / NFL EPA / MARKET EV / TRACKING
# ============================================================

NFLVERSE_GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"

def _remote_csv(url, timeout=45):
    r=requests.get(url,headers=BROWSER_HEADERS,timeout=timeout)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_nflverse_schedules():
    local=load_nfl_games_seed_v33()
    if not local.empty:
        return local.drop(columns=["_date"],errors="ignore"),None
    try:
        df=_remote_csv(NFLVERSE_GAMES_URL,45)
        return df,None
    except Exception as e:
        return pd.DataFrame(),f"NFL historical schedules failed: {e}"

@st.cache_data(ttl=21600, show_spinner=False)
def fetch_nflverse_team_stats(season):
    season=int(season)
    current=league_season_year("NFL",date.today())
    url=(
        "https://github.com/nflverse/nflverse-data/releases/download/"
        f"stats_team/stats_team_week_{season}.csv"
    )
    live_err=""
    if season>=int(current):
        try:
            live=_remote_csv(url,45)
            if not live.empty: return live,None
        except Exception as e:
            live_err=str(e)
    local=local_nfl_team_game_table_v33()
    if not local.empty:
        x=local[pd.to_numeric(local.get("season"),errors="coerce")==season].copy()
        if not x.empty:
            x=x.rename(columns={"explosive_passes":"passing_20","explosive_rushes":"rushing_10"})
            keep=[c for c in ["season","week","game_id","team","opponent_team","passing_epa","rushing_epa","passing_yards","rushing_yards","passing_20","rushing_10","sacks_suffered"] if c in x.columns]
            return x[keep].copy(),(f"Live current-season team refresh failed; using bundled data: {live_err}" if live_err else None)
    try:
        return _remote_csv(url,45),None
    except Exception as e:
        return pd.DataFrame(),f"NFL team stats {season} failed: {live_err or e}"


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_nflverse_player_stats(season):
    season=int(season)
    current=league_season_year("NFL",date.today())
    url=(
        "https://github.com/nflverse/nflverse-data/releases/download/"
        f"stats_player/stats_player_week_{season}.csv"
    )
    if season>=int(current):
        try:
            live=_remote_csv(url,45)
            if not live.empty: return live,None
        except Exception as e:
            live_err=str(e)
        else:
            live_err="current nflverse player file was empty"
    else:
        live_err=""
    local=load_nfl_player_seed_v33()
    if not local.empty and "season" in local.columns:
        x=local[pd.to_numeric(local["season"],errors="coerce")==season].copy()
        if not x.empty:
            return x,(f"Live current-season refresh failed; using bundled seed: {live_err}" if live_err else None)
    try:
        return _remote_csv(url,45),None
    except Exception as e:
        return pd.DataFrame(),f"NFL player stats {season} failed: {live_err or e}"


def _col_num(df,name,default=np.nan):
    if name not in df.columns:
        return pd.Series(default,index=df.index,dtype=float)
    return pd.to_numeric(df[name],errors="coerce")

def _nfl_stats_with_defense(df):
    if df is None or df.empty:
        return pd.DataFrame()
    x=df.copy()
    for c in ["passing_epa","rushing_epa","passing_yards","rushing_yards",
              "passing_20","rushing_10","sacks_suffered"]:
        if c not in x.columns:
            x[c]=np.nan
        x[c]=pd.to_numeric(x[c],errors="coerce")
    x["off_epa"]=x[["passing_epa","rushing_epa"]].fillna(0).sum(axis=1)

    opp=x[
        ["game_id","team","off_epa","passing_epa","rushing_epa",
         "passing_yards","rushing_yards","passing_20","rushing_10"]
    ].copy()
    opp=opp.rename(columns={
        "team":"opponent_lookup",
        "off_epa":"def_epa_allowed",
        "passing_epa":"def_pass_epa_allowed",
        "rushing_epa":"def_rush_epa_allowed",
        "passing_yards":"pass_yards_allowed",
        "rushing_yards":"rush_yards_allowed",
        "passing_20":"explosive_passes_allowed",
        "rushing_10":"explosive_rushes_allowed",
    })
    x=x.merge(
        opp,
        left_on=["game_id","opponent_team"],
        right_on=["game_id","opponent_lookup"],
        how="left",
    )
    return x

def _rolling_shift_mean(group,col,window=5,min_periods=3):
    return group[col].transform(
        lambda s:s.shift(1).rolling(window,min_periods=min_periods).mean()
    )

@st.cache_resource(show_spinner=False)
def build_nfl_model(current_season):
    """
    Separate NFL model using nflverse historical schedules + weekly team stats.
    Uses chronological train -> calibration -> test splits.
    """
    seasons=list(range(max(2019,int(current_season)-7),int(current_season)+1))
    frames=[]
    errors=[]
    for yr in seasons:
        df,err=fetch_nflverse_team_stats(yr)
        if err:
            errors.append(err)
        if not df.empty:
            frames.append(df)

    schedules,serr=fetch_nflverse_schedules()
    if serr:
        errors.append(serr)

    if not frames or schedules.empty:
        return None,None," | ".join(errors) or "NFL historical data unavailable."

    ts=pd.concat(frames,ignore_index=True)
    ts=_nfl_stats_with_defense(ts)
    if ts.empty:
        return None,None,"NFL team stats were empty."

    need=["season","week","team","game_id","opponent_team"]
    if any(c not in ts.columns for c in need):
        return None,None,"NFL team stats columns were not recognized."

    ts["season"]=pd.to_numeric(ts["season"],errors="coerce")
    ts["week"]=pd.to_numeric(ts["week"],errors="coerce")
    ts=ts.sort_values(["team","season","week","game_id"]).reset_index(drop=True)
    grp=ts.groupby("team",group_keys=False)

    base_cols=[
        "off_epa","passing_epa","rushing_epa","def_epa_allowed",
        "def_pass_epa_allowed","def_rush_epa_allowed",
        "passing_yards","rushing_yards","pass_yards_allowed","rush_yards_allowed",
        "passing_20","rushing_10","explosive_passes_allowed","explosive_rushes_allowed",
        "sacks_suffered",
    ]
    for c in base_cols:
        if c not in ts.columns:
            ts[c]=np.nan
        ts[f"{c}_r5"]=_rolling_shift_mean(grp,c,5,3)

    schedules=schedules.copy()
    for c in ["season","week","home_score","away_score","home_rest","away_rest",
              "spread_line","temp","wind"]:
        if c in schedules.columns:
            schedules[c]=pd.to_numeric(schedules[c],errors="coerce")
    schedules=schedules[
        schedules["game_type"].astype(str).eq("REG")
        if "game_type" in schedules.columns
        else pd.Series(True,index=schedules.index)
    ].copy()

    h=ts.add_prefix("h_")
    a=ts.add_prefix("a_")
    m=schedules.merge(
        h,
        left_on=["game_id","home_team"],
        right_on=["h_game_id","h_team"],
        how="inner",
    ).merge(
        a,
        left_on=["game_id","away_team"],
        right_on=["a_game_id","a_team"],
        how="inner",
    )

    feature_defs={
        "off_epa_diff":("h_off_epa_r5","a_off_epa_r5"),
        "pass_epa_diff":("h_passing_epa_r5","a_passing_epa_r5"),
        "rush_epa_diff":("h_rushing_epa_r5","a_rushing_epa_r5"),
        "def_epa_diff":("a_def_epa_allowed_r5","h_def_epa_allowed_r5"),
        "def_pass_epa_diff":("a_def_pass_epa_allowed_r5","h_def_pass_epa_allowed_r5"),
        "def_rush_epa_diff":("a_def_rush_epa_allowed_r5","h_def_rush_epa_allowed_r5"),
        "pass_yards_diff":("h_passing_yards_r5","a_passing_yards_r5"),
        "rush_yards_diff":("h_rushing_yards_r5","a_rushing_yards_r5"),
        "explosive_pass_diff":("h_passing_20_r5","a_passing_20_r5"),
        "explosive_rush_diff":("h_rushing_10_r5","a_rushing_10_r5"),
    }
    for out,(x1,x2) in feature_defs.items():
        if x1 in m.columns and x2 in m.columns:
            m[out]=pd.to_numeric(m[x1],errors="coerce")-pd.to_numeric(m[x2],errors="coerce")
        else:
            m[out]=np.nan

    m["rest_diff"]=(
        pd.to_numeric(m.get("home_rest"),errors="coerce")
        - pd.to_numeric(m.get("away_rest"),errors="coerce")
    )
    if "temp" not in m.columns:
        m["temp"]=np.nan
    if "wind" not in m.columns:
        m["wind"]=np.nan

    m["home_won"]=(
        pd.to_numeric(m["home_score"],errors="coerce")
        > pd.to_numeric(m["away_score"],errors="coerce")
    ).astype(int)

    date_col="gameday" if "gameday" in m.columns else None
    if date_col:
        m["_date"]=pd.to_datetime(m[date_col],errors="coerce")
    else:
        # Build Python strings explicitly instead of vectorized Arrow-string
        # concatenation. Streamlit Cloud may use pandas' PyArrow-backed string
        # dtype, where scalar "+" can raise ArrowNotImplementedError.
        _season_vals=m["season"].astype("Int64").tolist()
        _week_vals=m["week"].astype("Int64").tolist()
        _date_labels=[
            f"{s}-{w}"
            for s,w in zip(_season_vals,_week_vals)
        ]
        m["_date"]=pd.to_datetime(
            _date_labels,
            errors="coerce",
        )

    feats=[
        "off_epa_diff","pass_epa_diff","rush_epa_diff",
        "def_epa_diff","def_pass_epa_diff","def_rush_epa_diff",
        "pass_yards_diff","rush_yards_diff",
        "explosive_pass_diff","explosive_rush_diff",
        "rest_diff","temp","wind",
    ]

    # Fill weather with historical medians; all other features require actual rolling history.
    medians={}
    for c in feats:
        med=pd.to_numeric(m[c],errors="coerce").median()
        medians[c]=0.0 if pd.isna(med) else float(med)
        m[c]=pd.to_numeric(m[c],errors="coerce").fillna(medians[c])

    m=m.dropna(subset=["_date","home_won"]).sort_values("_date").reset_index(drop=True)
    if len(m)<500:
        return None,m,f"Only {len(m)} usable NFL historical games were built."

    n=len(m)
    cut1=int(n*.60)
    cut2=int(n*.80)
    train=m.iloc[:cut1]
    cal=m.iloc[cut1:cut2]
    test=m.iloc[cut2:].copy()

    base=Pipeline([
        ("scale",StandardScaler()),
        ("lr",LogisticRegression(max_iter=2500)),
    ])
    train_w=recency_training_weights(train["_date"],half_life_days=420)
    base.fit(train[feats],train["home_won"],lr__sample_weight=train_w)

    cal_raw=base.predict_proba(cal[feats])[:,1]
    iso=IsotonicRegression(out_of_bounds="clip")
    cal_w=recency_training_weights(cal["_date"],half_life_days=420)
    iso.fit(cal_raw,cal["home_won"],sample_weight=cal_w)

    raw_test=base.predict_proba(test[feats])[:,1]
    p=np.clip(iso.predict(raw_test),.001,.999)
    test["model_p"]=p

    metrics={
        "accuracy":accuracy_score(test["home_won"],p>=.5),
        "brier":brier_score_loss(test["home_won"],p),
        "logloss":log_loss(test["home_won"],p),
        "test_n":len(test),
        "train_n":len(train),
        "cal_n":len(cal),
        "first_date":str(m["_date"].min().date()),
        "last_date":str(m["_date"].max().date()),
    }
    pack={
        "model":base,
        "calibrator":iso,
        "features":feats,
        "medians":medians,
        "metrics":metrics,
        "seasons":seasons,
    }
    return pack,test,None

def _latest_nfl_team_profile(team_stats,team):
    x=team_stats[team_stats["team"].astype(str)==str(team)].copy()
    if x.empty:
        return {}
    x=x.sort_values(["season","week"],ascending=[False,False]).head(5)
    out={}
    for c in [
        "off_epa","passing_epa","rushing_epa","def_epa_allowed",
        "def_pass_epa_allowed","def_rush_epa_allowed",
        "passing_yards","rushing_yards","pass_yards_allowed","rush_yards_allowed",
        "passing_20","rushing_10","explosive_passes_allowed","explosive_rushes_allowed",
    ]:
        if c in x.columns:
            out[c]=pd.to_numeric(x[c],errors="coerce").mean()
    return out

def current_nfl_team_probability(game):
    season=league_season_year("NFL",st.session_state.get("selected_date",date.today()))
    pack,_,err=build_nfl_model(season)
    if err or not pack:
        return np.nan,err or "NFL model unavailable.",None

    frames=[]
    for yr in [season-1,season]:
        df,_=fetch_nflverse_team_stats(yr)
        if not df.empty:
            frames.append(df)
    if not frames:
        return np.nan,"Current NFL team statistics unavailable.",pack

    ts=_nfl_stats_with_defense(pd.concat(frames,ignore_index=True))
    hp=_latest_nfl_team_profile(ts,game["home_abbr"])
    ap=_latest_nfl_team_profile(ts,game["away_abbr"])
    if not hp or not ap:
        return np.nan,"Not enough current NFL EPA/team history.",pack

    row={}
    row["off_epa_diff"]=hp.get("off_epa",np.nan)-ap.get("off_epa",np.nan)
    row["pass_epa_diff"]=hp.get("passing_epa",np.nan)-ap.get("passing_epa",np.nan)
    row["rush_epa_diff"]=hp.get("rushing_epa",np.nan)-ap.get("rushing_epa",np.nan)
    row["def_epa_diff"]=ap.get("def_epa_allowed",np.nan)-hp.get("def_epa_allowed",np.nan)
    row["def_pass_epa_diff"]=ap.get("def_pass_epa_allowed",np.nan)-hp.get("def_pass_epa_allowed",np.nan)
    row["def_rush_epa_diff"]=ap.get("def_rush_epa_allowed",np.nan)-hp.get("def_rush_epa_allowed",np.nan)
    row["pass_yards_diff"]=hp.get("passing_yards",np.nan)-ap.get("passing_yards",np.nan)
    row["rush_yards_diff"]=hp.get("rushing_yards",np.nan)-ap.get("rushing_yards",np.nan)
    row["explosive_pass_diff"]=hp.get("passing_20",np.nan)-ap.get("passing_20",np.nan)
    row["explosive_rush_diff"]=hp.get("rushing_10",np.nan)-ap.get("rushing_10",np.nan)

    ctx=st.session_state.get("matchup_context") or {}
    hr=(ctx.get("home") or {}).get("rest",np.nan)
    ar=(ctx.get("away") or {}).get("rest",np.nan)
    row["rest_diff"]=0 if pd.isna(hr) or pd.isna(ar) else float(hr)-float(ar)

    # Current weather input when available; historical model learned temp/wind.
    temp=pack["medians"].get("temp",0)
    wind=pack["medians"].get("wind",0)
    if game.get("indoor") is True:
        wind=0
    else:
        weather,_=fetch_game_day_weather(
            game.get("venue_city",""),
            game.get("venue_state",""),
            game.get("date",""),
        )
        if weather:
            temp=(safe_float(weather.get("high_f"))+safe_float(weather.get("low_f")))/2
            wind=safe_float(weather.get("wind_mph"))
    row["temp"]=temp
    row["wind"]=wind

    frame=pd.DataFrame([row])
    for c in pack["features"]:
        frame[c]=pd.to_numeric(frame[c],errors="coerce").fillna(pack["medians"][c])

    raw=pack["model"].predict_proba(frame[pack["features"]])[:,1]
    p=float(np.clip(pack["calibrator"].predict(raw)[0],.001,.999))
    return p,None,pack

@st.cache_resource(show_spinner=False)
def build_espn_basketball_model(league,current_season,seasons_back=4):
    """
    Basketball model for leagues without the local NBA database (especially WNBA).
    Builds real historical game rows from ESPN team schedules, then trains on
    pregame rolling offense/defense/form/rest only.
    """
    teams,terr=fetch_all_teams(league)
    if terr or teams.empty:
        return None,None,terr or "Basketball teams unavailable."

    rows=[]
    errors=[]
    seasons=list(range(int(current_season)-seasons_back+1,int(current_season)+1))
    for _,team in teams.iterrows():
        for yr in seasons:
            df,err=fetch_team_recent(league,team["team_id"],yr)
            if err:
                errors.append(err)
            if df.empty:
                continue
            x=df.copy()
            x["team"]=str(team["abbr"])
            x["season"]=yr
            rows.append(x)

    if not rows:
        return None,None,"No historical basketball schedules were built."

    tg=pd.concat(rows,ignore_index=True)
    tg["date_dt"]=pd.to_datetime(tg["date"],errors="coerce",utc=True)
    tg["pf"]=pd.to_numeric(tg["points_for"],errors="coerce")
    tg["pa"]=pd.to_numeric(tg["points_against"],errors="coerce")
    tg["won"]=(tg["result"].astype(str)=="W").astype(int)
    tg=tg.dropna(subset=["date_dt","pf","pa"]).sort_values(["team","date_dt"])

    grp=tg.groupby("team",group_keys=False)
    tg["off"]=grp["pf"].transform(lambda s:s.shift(1).rolling(10,min_periods=5).mean())
    tg["def"]=grp["pa"].transform(lambda s:s.shift(1).rolling(10,min_periods=5).mean())
    tg["form"]=grp["won"].transform(lambda s:s.shift(1).rolling(10,min_periods=5).mean())
    tg["prev_date"]=grp["date_dt"].shift(1)
    tg["rest"]=(tg["date_dt"]-tg["prev_date"]).dt.days.clip(0,14)
    # Avoid pandas/PyArrow vectorized string concatenation on Streamlit Cloud.
    # Build plain Python strings so this works across pandas string backends.
    _date_keys=[
        x.strftime("%Y-%m-%d") if pd.notna(x) else ""
        for x in tg["date_dt"].tolist()
    ]
    _pair_keys=[
        "|".join(sorted([str(team),str(opp)]))
        for team,opp in zip(tg["team"].tolist(),tg["opponent"].tolist())
    ]
    tg["game_key"]=[
        f"{d}|{pair}"
        for d,pair in zip(_date_keys,_pair_keys)
    ]

    home=tg[tg["home_away"].astype(str).str.lower()=="home"].copy()
    away=tg[tg["home_away"].astype(str).str.lower()=="away"].copy()

    m=home.merge(
        away,
        on="game_key",
        suffixes=("_h","_a"),
        how="inner",
    )
    # Exact pairing protection.
    m=m[
        (m["team_h"].astype(str)==m["opponent_a"].astype(str))
        & (m["opponent_h"].astype(str)==m["team_a"].astype(str))
    ].copy()

    m["offense_diff"]=m["off_h"]-m["off_a"]
    m["defense_diff"]=m["def_a"]-m["def_h"]
    m["form_diff"]=m["form_h"]-m["form_a"]
    m["rest_diff"]=m["rest_h"]-m["rest_a"]
    m["home_won"]=m["won_h"]
    m["_date"]=m["date_dt_h"]

    feats=["offense_diff","defense_diff","form_diff","rest_diff"]
    m=m.dropna(subset=feats+["home_won","_date"]).sort_values("_date").reset_index(drop=True)
    if len(m)<120:
        return None,m,f"Only {len(m)} usable historical {league} games were built."

    n=len(m)
    cut1=int(n*.60)
    cut2=int(n*.80)
    train=m.iloc[:cut1]
    cal=m.iloc[cut1:cut2]
    test=m.iloc[cut2:].copy()

    model=Pipeline([
        ("scale",StandardScaler()),
        ("lr",LogisticRegression(max_iter=2500)),
    ])
    train_w=recency_training_weights(train["_date"],half_life_days=360)
    model.fit(train[feats],train["home_won"],lr__sample_weight=train_w)

    cal_raw=model.predict_proba(cal[feats])[:,1]
    iso=IsotonicRegression(out_of_bounds="clip")
    cal_w=recency_training_weights(cal["_date"],half_life_days=360)
    iso.fit(cal_raw,cal["home_won"],sample_weight=cal_w)

    raw=model.predict_proba(test[feats])[:,1]
    p=np.clip(iso.predict(raw),.001,.999)
    test["model_p"]=p

    metrics={
        "accuracy":accuracy_score(test["home_won"],p>=.5),
        "brier":brier_score_loss(test["home_won"],p),
        "logloss":log_loss(test["home_won"],p),
        "train_n":len(train),
        "cal_n":len(cal),
        "test_n":len(test),
        "first_date":str(m["_date"].min().date()),
        "last_date":str(m["_date"].max().date()),
        "seasons":f"{min(seasons)}-{max(seasons)}",
    }
    return {
        "model":model,
        "calibrator":iso,
        "features":feats,
        "metrics":metrics,
    },test,None

def current_wnba_team_probability(game):
    season=league_season_year("WNBA",st.session_state.get("selected_date",date.today()))
    pack,_,err=build_espn_basketball_model("WNBA",season,5)
    if err or not pack:
        return np.nan,err or "WNBA historical model unavailable.",pack

    hdf,herr=fetch_team_history_multi("WNBA",game["home_id"],season,3)
    adf,aerr=fetch_team_history_multi("WNBA",game["away_id"],season,3)
    if herr or aerr or hdf.empty or adf.empty:
        return np.nan,"Not enough current WNBA team form.",pack

    h=hdf.head(10); a=adf.head(10)
    if len(h)<5 or len(a)<5:
        return np.nan,"Not enough recent WNBA games.",pack

    game_dt=pd.to_datetime(game.get("date"),errors="coerce",utc=True)
    h_last=pd.to_datetime(h.iloc[0]["date"],errors="coerce",utc=True)
    a_last=pd.to_datetime(a.iloc[0]["date"],errors="coerce",utc=True)
    h_rest=(game_dt-h_last).days if not pd.isna(game_dt) and not pd.isna(h_last) else 2
    a_rest=(game_dt-a_last).days if not pd.isna(game_dt) and not pd.isna(a_last) else 2

    frame=pd.DataFrame([{
        "offense_diff":h["points_for"].mean()-a["points_for"].mean(),
        "defense_diff":a["points_against"].mean()-h["points_against"].mean(),
        "form_diff":(h["result"]=="W").mean()-(a["result"]=="W").mean(),
        "rest_diff":np.clip(h_rest,0,14)-np.clip(a_rest,0,14),
    }])
    raw=pack["model"].predict_proba(frame[pack["features"]])[:,1]
    p=float(np.clip(pack["calibrator"].predict(raw)[0],.001,.999))
    return p,None,pack

def current_league_model_probability(game,league):
    if league=="NBA":
        p,err=current_nba_team_probability(game)
        pack=None
        if NBA_HIST_FILE.exists():
            pack0,_,e=build_nba_model(NBA_HIST_FILE.stat().st_mtime)
            if pack0 and not e:
                pack={
                    "metrics":pack0[2],
                    "name":"NBA historical model",
                }
        return p,err,pack
    if league=="WNBA":
        return current_wnba_team_probability(game)
    if league=="NFL":
        return current_nfl_team_probability(game)
    return np.nan,"No model for this league.",None

def model_metrics_for_league(league):
    try:
        if league=="NBA":
            if not NBA_HIST_FILE.exists():
                return {}
            pack,_,err=build_nba_model(NBA_HIST_FILE.stat().st_mtime)
            return {} if err or not pack else pack[2]
        if league=="WNBA":
            season=league_season_year("WNBA",st.session_state.get("selected_date",date.today()))
            pack,_,err=build_espn_basketball_model("WNBA",season,5)
            return {} if err or not pack else pack["metrics"]
        if league=="NFL":
            season=league_season_year("NFL",st.session_state.get("selected_date",date.today()))
            pack,_,err=build_nfl_model(season)
            return {} if err or not pack else pack["metrics"]
    except Exception:
        return {}
    return {}

@st.cache_data(ttl=120, show_spinner=False)
def fetch_event_lineup_status(league,event_id):
    """
    Uses ESPN game summary. If starter flags are not published pregame,
    returns NOT CONFIRMED rather than inventing a lineup.
    """
    cfg=LEAGUES[league]
    url=(
        f"https://site.api.espn.com/apis/site/v2/sports/"
        f"{cfg['sport']}/{cfg['league']}/summary"
    )
    try:
        payload=request_json(url,{"event":event_id})
    except Exception as e:
        return {},f"Lineup feed failed: {e}"

    out={}
    players=((payload.get("boxscore") or {}).get("players") or [])
    for team_block in players:
        team=(team_block.get("team") or {})
        abbr=team.get("abbreviation") or team.get("displayName") or "Team"
        starters=[]
        active=[]
        for stat_group in team_block.get("statistics",[]) or []:
            for item in stat_group.get("athletes",[]) or []:
                athlete=item.get("athlete") or {}
                name=athlete.get("displayName") or athlete.get("fullName") or ""
                if not name:
                    continue
                if item.get("starter") is True:
                    starters.append(name)
                if item.get("didNotPlay") is not True:
                    active.append(name)
        out[abbr]={
            "starters":list(dict.fromkeys(starters)),
            "active":list(dict.fromkeys(active)),
        }
    return out,None

def market_model_dashboard_rows(game,league,fee_per_contract=0.01,bankroll=1000.0,kelly_fraction=.25,max_pct=.02):
    prices,kerr=selected_game_kalshi_prices_v22(game,league)
    p_home,model_err,pack=current_league_model_probability(game,league)
    metrics=model_metrics_for_league(league)

    if not prices or pd.isna(p_home):
        return [],kerr or model_err or "Model/market data unavailable."

    rows=[]
    for r in prices:
        model_p=p_home if r["Abbr"]==game["home_abbr"] else 1-p_home
        price=r["Kalshi probability"]
        if pd.isna(price):
            continue

        edge=model_p-price
        ev=model_p*(1-price)-(1-model_p)*price-fee_per_contract
        kelly=fractional_kelly(model_p,price*100,kelly_fraction)
        risk_frac=min(max(kelly,0),max_pct) if ev>0 else 0
        position=bankroll*risk_frac

        rows.append({
            "Event ID":game.get("event_id",""),
            "Date":str(st.session_state.get("selected_date",date.today())),
            "League":league,
            "Game":f"{game['away_abbr']} at {game['home_abbr']}",
            "Market":f"{r['Team']} to win",
            "Side":r["Team"],
            "Side Abbr":r["Abbr"],
            "Ticker":r["Ticker"],
            "Kalshi price":price,
            "Model probability":model_p,
            "Edge":edge,
            "EV per $1":ev,
            "Historical accuracy":metrics.get("accuracy",np.nan),
            "Brier":metrics.get("brier",np.nan),
            "Suggested position":position,
            "Fee allowance":fee_per_contract,
        })
    return rows,None

def record_market_predictions(rows):
    if not rows:
        return
    old=load_csv(MODEL_MARKET_HISTORY_FILE)
    hour=datetime.now().strftime("%Y-%m-%d %H")
    new_rows=[]
    for r in rows:
        key=f"{hour}|{r.get('Event ID')}|{r.get('Ticker')}|{r.get('Side Abbr')}"
        rec={
            "snapshot_hour":hour,
            "record_key":key,
            "timestamp":datetime.now().isoformat(timespec="seconds"),
            "event_id":r.get("Event ID",""),
            "date":r.get("Date",""),
            "league":r.get("League",""),
            "game":r.get("Game",""),
            "market":r.get("Market",""),
            "side":r.get("Side",""),
            "side_abbr":r.get("Side Abbr",""),
            "ticker":r.get("Ticker",""),
            "kalshi_price":r.get("Kalshi price",np.nan),
            "model_probability":r.get("Model probability",np.nan),
            "edge":r.get("Edge",np.nan),
            "ev_per_dollar":r.get("EV per $1",np.nan),
            "historical_accuracy":r.get("Historical accuracy",np.nan),
            "suggested_position":r.get("Suggested position",np.nan),
            "settled":"",
            "won":"",
            "profit_loss_100":"",
        }
        new_rows.append(rec)

    new=pd.DataFrame(new_rows)
    if not old.empty and "record_key" in old.columns:
        new=new[~new["record_key"].isin(old["record_key"].astype(str))]
    if new.empty:
        return
    out=pd.concat([old,new],ignore_index=True) if not old.empty else new
    out.to_csv(MODEL_MARKET_HISTORY_FILE,index=False)

def settle_market_prediction_history():
    hist=load_csv(MODEL_MARKET_HISTORY_FILE)
    if hist.empty:
        return hist,0

    if "settled" not in hist.columns:
        hist["settled"]=""
    if "won" not in hist.columns:
        hist["won"]=""
    if "profit_loss_100" not in hist.columns:
        hist["profit_loss_100"]=""

    updated=0
    schedule_cache={}
    for idx,row in hist.iterrows():
        if str(row.get("settled","")).lower() in {"true","1","yes"}:
            continue
        lg=str(row.get("league",""))
        d=pd.to_datetime(row.get("date"),errors="coerce")
        if lg not in LEAGUES or pd.isna(d):
            continue
        key=(lg,d.date().isoformat())
        if key not in schedule_cache:
            games,_=fetch_schedule(lg,d.date())
            schedule_cache[key]=games
        game=next(
            (g for g in schedule_cache[key] if str(g.get("event_id",""))==str(row.get("event_id",""))),
            None,
        )
        if not game or game.get("state")!="post":
            continue

        hs=safe_float(game.get("home_score"))
        a_s=safe_float(game.get("away_score"))
        if pd.isna(hs) or pd.isna(a_s) or hs==a_s:
            continue
        winner=game["home_abbr"] if hs>a_s else game["away_abbr"]
        won=str(row.get("side_abbr",""))==str(winner)
        price=safe_float(row.get("kalshi_price"))
        pnl=np.nan
        if not pd.isna(price) and 0<price<1:
            pnl=(100/price-100) if won else -100.0

        hist.at[idx,"settled"]="true"
        hist.at[idx,"won"]="true" if won else "false"
        hist.at[idx,"profit_loss_100"]=pnl
        updated+=1

    if updated:
        hist.to_csv(MODEL_MARKET_HISTORY_FILE,index=False)
    return hist,updated

def historical_edge_performance(hist):
    if hist is None or hist.empty:
        return pd.DataFrame()
    x=hist.copy()
    settled=x["settled"].astype(str).str.lower().isin(["true","1","yes"])
    x=x[settled].copy()
    if x.empty:
        return pd.DataFrame()
    x["edge"]=pd.to_numeric(x["edge"],errors="coerce")
    x["profit_loss_100"]=pd.to_numeric(x["profit_loss_100"],errors="coerce")
    x["won_bool"]=x["won"].astype(str).str.lower().isin(["true","1","yes"]).astype(int)
    x["edge_band"]=pd.cut(
        x["edge"],
        bins=[-1,0,.025,.05,.075,.10,1],
        labels=["≤0","0–2.5%","2.5–5%","5–7.5%","7.5–10%","10%+"],
        include_lowest=True,
    )
    grp=x.groupby("edge_band",observed=False).agg(
        markets=("record_key","count"),
        wins=("won_bool","sum"),
        avg_edge=("edge","mean"),
        profit_per_100=("profit_loss_100","mean"),
        total_profit=("profit_loss_100","sum"),
    ).reset_index()
    grp["win_rate"]=np.where(grp["markets"]>0,grp["wins"]/grp["markets"],np.nan)
    return grp

def model_coverage_rows():
    return [
        ("Real historical game data","✅","NBA local historical database; WNBA ESPN historical schedules; NFL nflverse schedules/team stats."),
        ("Feature engineering","✅","Basketball form/efficiency/pace/matchups/injuries/minutes; NFL EPA, pass/rush EPA, explosive plays, rest, weather, injuries."),
        ("Backtesting on unseen games","✅","Chronological held-out tests; NFL and WNBA use separate train/calibration/test periods."),
        ("Probability calibration","✅","Player props use calibration; NFL/WNBA team models use isotonic calibration; NBA calibration chart remains available."),
        ("Kalshi market-price input","✅","Current selected-game Kalshi winner prices are pulled automatically."),
        ("Edge / EV calculator","✅","Model probability minus market price; EV includes configurable fee/slippage allowance."),
        ("Bankroll / risk module","✅","Fractional Kelly plus hard maximum bankroll percentage cap."),
        ("Live/current updates","✅","Schedule, injuries, roster, lineup status when published, weather, odds, Kalshi."),
        ("Separate NFL and basketball models","✅","NFL model is EPA/football-feature based; basketball model uses basketball historical form/efficiency features."),
        ("Performance tracking","✅","Market predictions are logged; resolved game-winner contracts can be settled and grouped by edge band."),
        ("Data collection","✅","NBA local history + live ESPN/NBA/WNBA + nflverse historical/current NFL + Kalshi + weather."),
        ("Team/player statistics","✅","Team and player reports."),
        ("Recent form","✅","Team and player recent windows."),
        ("Home/away","✅","Team and player splits where returned."),
        ("Injuries/availability","✅","Verified/unverified distinction plus player status."),
        ("Opponent strength","✅","Team defensive/efficiency context and exact-opponent history."),
        ("Expected minutes","✅","Likely minutes/range and explicit restriction detection."),
        ("Statistical probability model","✅","Logistic/statistical models plus player-prop probability models."),
        ("Calibrated 60% meaning","✅","Calibration/backtesting panels show predicted-vs-actual behavior."),
        ("Kalshi implied probability","✅","Contract price displayed as probability."),
        ("Expected profit/loss","✅","EV per $1 and settled P/L per $100 tracked for priced markets."),
        ("Historical edge profitability","✅","Settled Kalshi model snapshots grouped by edge band."),
        ("Dashboard: game/market/price/model/edge/EV/accuracy/position size","✅","Model & Market table in Markets & Kalshi."),
    ]


# ============================================================
# V21 COMPLETE: SPREAD/TOTAL MODELS + ORDERBOOK/FEE + AUTO LOGGING
# ============================================================

KALSHI_GENERAL_TAKER_FEE_RATE = 0.07
KALSHI_FEE_SCHEDULE_EFFECTIVE = "2026-07-07"
KALSHI_FEE_SCHEDULE_URL = "https://kalshi.com/regulatory/fee-schedule"

def _ceil_cent(value):
    """Kalshi general fee schedule rounds the fee up to the next cent."""
    if value is None or pd.isna(value) or value <= 0:
        return 0.0
    return math.ceil(float(value)*100 - 1e-12)/100.0

def kalshi_general_taker_fee(contracts,price,multiplier=1.0):
    """
    General event-contract taker fee:
    ceil(M * 0.07 * C * P * (1-P)) to the next cent.
    The sports game/spread/total series used here are treated as the general
    schedule unless Kalshi later publishes a special series multiplier.
    """
    c=max(0.0,float(contracts or 0))
    p=float(price)
    if c<=0 or p<=0 or p>=1:
        return 0.0
    raw=float(multiplier)*KALSHI_GENERAL_TAKER_FEE_RATE*c*p*(1-p)
    return _ceil_cent(raw)

def kalshi_fee_adjusted_kelly(model_p,price,fee_per_contract,fraction=.25):
    """
    Fractional Kelly using the all-in entry cost (price + taker fee).
    """
    p=float(model_p)
    cost=float(price)+float(fee_per_contract)
    if not (0<p<1) or cost<=0 or cost>=1:
        return 0.0
    win_profit=1.0-cost
    b=win_profit/cost
    q=1-p
    full=(b*p-q)/b if b>0 else 0
    return max(0.0,float(full)*float(fraction))

@st.cache_data(ttl=20,show_spinner=False)
def fetch_kalshi_orderbook(ticker,depth=100):
    url=(
        "https://external-api.kalshi.com/trade-api/v2/markets/"
        f"{ticker}/orderbook"
    )
    try:
        payload=request_json(url,{"depth":int(depth)},timeout=20)
    except Exception as e:
        return {},str(e)

    yes=[]
    no=[]
    fp=payload.get("orderbook_fp") or {}
    if fp:
        for p,q in fp.get("yes_dollars") or []:
            pp=safe_float(p); qq=safe_float(q)
            if not pd.isna(pp) and not pd.isna(qq):
                yes.append((pp,qq))
        for p,q in fp.get("no_dollars") or []:
            pp=safe_float(p); qq=safe_float(q)
            if not pd.isna(pp) and not pd.isna(qq):
                no.append((pp,qq))

    if not yes and not no:
        ob=payload.get("orderbook") or {}
        def _old_levels(side):
            x=ob.get(side) or []
            if isinstance(x,dict):
                x=x.get("bids") or []
            out=[]
            for level in x:
                if not isinstance(level,(list,tuple)) or len(level)<2:
                    continue
                p=safe_float(level[0]); q=safe_float(level[1])
                if pd.isna(p) or pd.isna(q):
                    continue
                if p>1:
                    p=p/100.0
                out.append((p,q))
            return out
        yes=_old_levels("yes")
        no=_old_levels("no")

    yes=sorted(yes,key=lambda x:x[0],reverse=True)
    no=sorted(no,key=lambda x:x[0],reverse=True)

    # Kalshi orderbook consists of YES bids and NO bids.
    # A NO bid at x is a YES ask at 1-x.
    yes_asks=sorted(
        [(max(0.0,min(1.0,1-p)),q) for p,q in no],
        key=lambda x:x[0]
    )
    return {
        "yes_bids":yes,
        "no_bids":no,
        "yes_asks":yes_asks,
        "best_yes_bid":yes[0][0] if yes else np.nan,
        "best_yes_ask":yes_asks[0][0] if yes_asks else np.nan,
        "best_yes_ask_qty":yes_asks[0][1] if yes_asks else np.nan,
    },None

def simulate_yes_taker_fill(ticker,target_dollars,fallback_ask=np.nan):
    """
    Simulates an immediate YES purchase through the visible orderbook.
    Falls back to the quoted YES ask when orderbook depth is unavailable.
    """
    budget=max(0.0,float(target_dollars or 0))
    book,err=fetch_kalshi_orderbook(ticker,100)
    asks=(book or {}).get("yes_asks") or []

    if not asks and not pd.isna(fallback_ask):
        asks=[(float(fallback_ask),float("inf"))]

    if budget<=0 or not asks:
        return {
            "contracts":0,
            "avg_price":safe_float(fallback_ask),
            "gross_cost":0.0,
            "fee":0.0,
            "all_in_cost":0.0,
            "filled":False,
            "depth_status":"Orderbook unavailable" if err else "No ask liquidity",
            "orderbook_error":err or "",
        }

    fills=[]
    gross=0.0
    remaining_budget=budget

    # First pass targets gross contract spend. We then trim if fee pushes all-in
    # cost over the conservative bankroll cap.
    for price,qty in asks:
        price=float(price); qty=float(qty)
        if price<=0 or qty<=0:
            continue
        max_here=int(min(qty,math.floor(remaining_budget/price)))
        if max_here<=0:
            continue
        fills.append([price,max_here])
        cost=price*max_here
        gross+=cost
        remaining_budget-=cost
        if remaining_budget < min(p for p,_ in asks if p>0):
            break

    contracts=int(sum(q for _,q in fills))
    if contracts<=0:
        return {
            "contracts":0,
            "avg_price":asks[0][0],
            "gross_cost":0.0,
            "fee":0.0,
            "all_in_cost":0.0,
            "filled":False,
            "depth_status":"Budget below best ask",
            "orderbook_error":err or "",
        }

    avg=gross/contracts
    fee=kalshi_general_taker_fee(contracts,avg,1.0)

    # Trim highest-priced fills until gross + exact rounded fee fits budget.
    while contracts>0 and gross+fee>budget:
        price,qty=fills[-1]
        over=gross+fee-budget
        drop=max(1,int(math.ceil(over/max(price,0.01))))
        drop=min(drop,qty)
        fills[-1][1]-=drop
        gross-=price*drop
        contracts-=drop
        if fills[-1][1]<=0:
            fills.pop()
        if contracts<=0:
            break
        avg=gross/contracts
        fee=kalshi_general_taker_fee(contracts,avg,1.0)

    if contracts<=0:
        return {
            "contracts":0,
            "avg_price":asks[0][0],
            "gross_cost":0.0,
            "fee":0.0,
            "all_in_cost":0.0,
            "filled":False,
            "depth_status":"Budget consumed by minimum price/fee",
            "orderbook_error":err or "",
        }

    avg=gross/contracts
    fee=kalshi_general_taker_fee(contracts,avg,1.0)
    return {
        "contracts":contracts,
        "avg_price":avg,
        "gross_cost":gross,
        "fee":fee,
        "fee_per_contract":fee/contracts,
        "all_in_cost":gross+fee,
        "filled":True,
        "depth_status":(
            "Visible orderbook depth"
            if not err and (book or {}).get("yes_asks")
            else "Top quote fallback"
        ),
        "orderbook_error":err or "",
        "best_ask":asks[0][0],
    }

def _empirical_over_probability(pred,residuals,threshold):
    r=np.asarray(residuals,dtype=float)
    r=r[np.isfinite(r)]
    if len(r)<10 or pd.isna(pred) or pd.isna(threshold):
        return np.nan
    return float(np.mean(float(pred)+r>float(threshold)))

def _empirical_under_probability(pred,residuals,threshold):
    r=np.asarray(residuals,dtype=float)
    r=r[np.isfinite(r)]
    if len(r)<10 or pd.isna(pred) or pd.isna(threshold):
        return np.nan
    return float(np.mean(float(pred)+r<float(threshold)))

def _regression_validation(y_true,y_pred):
    y=np.asarray(y_true,dtype=float)
    p=np.asarray(y_pred,dtype=float)
    mask=np.isfinite(y)&np.isfinite(p)
    if mask.sum()==0:
        return {"mae":np.nan,"rmse":np.nan,"n":0}
    return {
        "mae":float(mean_absolute_error(y[mask],p[mask])),
        "rmse":float(mean_squared_error(y[mask],p[mask])**0.5),
        "n":int(mask.sum()),
    }

def _threshold_validation(test,pred_col,actual_col,residuals,threshold,direction="over"):
    if test is None or test.empty or pred_col not in test.columns or actual_col not in test.columns:
        return {"accuracy":np.nan,"brier":np.nan,"n":0}
    probs=[]
    ys=[]
    for _,r in test.iterrows():
        pred=safe_float(r[pred_col]); actual=safe_float(r[actual_col])
        if pd.isna(pred) or pd.isna(actual):
            continue
        if direction=="over":
            prob=_empirical_over_probability(pred,residuals,threshold)
            y=int(actual>threshold)
        else:
            prob=_empirical_under_probability(pred,residuals,threshold)
            y=int(actual<threshold)
        if pd.isna(prob):
            continue
        probs.append(prob); ys.append(y)
    if not probs:
        return {"accuracy":np.nan,"brier":np.nan,"n":0}
    arr=np.asarray(probs)
    y=np.asarray(ys)
    return {
        "accuracy":float(accuracy_score(y,arr>=.5)),
        "brier":float(brier_score_loss(y,np.clip(arr,.001,.999))),
        "n":len(y),
    }

@st.cache_resource(show_spinner=False)
def build_nba_score_models(mtime):
    games=pd.read_csv(NBA_HIST_FILE)
    d=guess_col(games.columns,["game_date","date"])
    hteam=guess_col(games.columns,["team_abbreviation_home","home_team"])
    ateam=guess_col(games.columns,["team_abbreviation_away","away_team"])
    hpts=guess_col(games.columns,["pts_home","home_pts","home_score"])
    apts=guess_col(games.columns,["pts_away","away_pts","away_score"])
    if any(c is None for c in [d,hteam,ateam,hpts,apts]):
        return None,None,"Historical NBA columns not recognized."

    g=games[[d,hteam,ateam,hpts,apts]].copy()
    g[d]=pd.to_datetime(g[d],errors="coerce")
    g[hpts]=pd.to_numeric(g[hpts],errors="coerce")
    g[apts]=pd.to_numeric(g[apts],errors="coerce")
    g=g.dropna().sort_values(d).reset_index(drop=True)

    home=pd.DataFrame({"idx":g.index,"date":g[d],"team":g[hteam].astype(str),"pf":g[hpts],"pa":g[apts],"home":1})
    away=pd.DataFrame({"idx":g.index,"date":g[d],"team":g[ateam].astype(str),"pf":g[apts],"pa":g[hpts],"home":0})
    tg=pd.concat([home,away],ignore_index=True).sort_values(["team","date","idx"])
    tg["won"]=(tg["pf"]>tg["pa"]).astype(int)
    grp=tg.groupby("team",group_keys=False)
    tg["off"]=grp["pf"].transform(lambda s:s.shift(1).rolling(10,min_periods=5).mean())
    tg["def"]=grp["pa"].transform(lambda s:s.shift(1).rolling(10,min_periods=5).mean())
    tg["form"]=grp["won"].transform(lambda s:s.shift(1).rolling(10,min_periods=5).mean())
    tg["prev"]=grp["date"].shift(1)
    tg["rest"]=(tg["date"]-tg["prev"]).dt.days.clip(0,14)

    hh=tg[tg["home"]==1].set_index("idx")
    aa=tg[tg["home"]==0].set_index("idx")
    m=pd.DataFrame({
        "date":g[d],
        "offense_diff":hh["off"]-aa["off"],
        "defense_diff":aa["def"]-hh["def"],
        "form_diff":hh["form"]-aa["form"],
        "rest_diff":hh["rest"]-aa["rest"],
        "home_margin":g[hpts]-g[apts],
        "game_total":g[hpts]+g[apts],
    }).dropna().sort_values("date").reset_index(drop=True)

    feats=["offense_diff","defense_diff","form_diff","rest_diff"]
    if len(m)<300:
        return None,m,"Not enough usable NBA score-model games."

    n=len(m); cut1=int(n*.70); cut2=int(n*.85)
    train=m.iloc[:cut1]; cal=m.iloc[cut1:cut2]; test=m.iloc[cut2:].copy()

    margin_model=Pipeline([("scale",StandardScaler()),("ridge",Ridge(alpha=8.0))])
    total_model=Pipeline([("scale",StandardScaler()),("ridge",Ridge(alpha=8.0))])
    train_w=recency_training_weights(train["date"],half_life_days=420)
    margin_model.fit(train[feats],train["home_margin"],ridge__sample_weight=train_w)
    total_model.fit(train[feats],train["game_total"],ridge__sample_weight=train_w)

    cal_margin_pred=margin_model.predict(cal[feats])
    cal_total_pred=total_model.predict(cal[feats])
    margin_resid=(cal["home_margin"].to_numpy()-cal_margin_pred)
    total_resid=(cal["game_total"].to_numpy()-cal_total_pred)

    test["margin_pred"]=margin_model.predict(test[feats])
    test["total_pred"]=total_model.predict(test[feats])

    return {
        "margin_model":margin_model,
        "total_model":total_model,
        "features":feats,
        "margin_residuals":margin_resid,
        "total_residuals":total_resid,
        "margin_metrics":_regression_validation(test["home_margin"],test["margin_pred"]),
        "total_metrics":_regression_validation(test["game_total"],test["total_pred"]),
        "test":test,
        "first_date":str(m["date"].min().date()),
        "last_date":str(m["date"].max().date()),
    },test,None

def _current_basketball_score_features(game,league):
    season=league_season_year(league,st.session_state.get("selected_date",date.today()))
    hdf,_=fetch_team_history_multi(league,game["home_id"],season,3)
    adf,_=fetch_team_history_multi(league,game["away_id"],season,3)
    if hdf.empty or adf.empty:
        return None,"Not enough current team history."
    h=hdf.head(10); a=adf.head(10)
    if len(h)<5 or len(a)<5:
        return None,"Not enough recent games."

    game_dt=pd.to_datetime(game.get("date"),errors="coerce",utc=True)
    h_last=pd.to_datetime(h.iloc[0]["date"],errors="coerce",utc=True)
    a_last=pd.to_datetime(a.iloc[0]["date"],errors="coerce",utc=True)
    h_rest=(game_dt-h_last).days if not pd.isna(game_dt) and not pd.isna(h_last) else 2
    a_rest=(game_dt-a_last).days if not pd.isna(game_dt) and not pd.isna(a_last) else 2

    frame=pd.DataFrame([{
        "offense_diff":h["points_for"].mean()-a["points_for"].mean(),
        "defense_diff":a["points_against"].mean()-h["points_against"].mean(),
        "form_diff":(h["result"]=="W").mean()-(a["result"]=="W").mean(),
        "rest_diff":np.clip(h_rest,0,14)-np.clip(a_rest,0,14),
    }])
    return frame,None

def current_nba_score_projection(game):
    if not NBA_HIST_FILE.exists():
        return None,"Historical NBA file is missing."
    pack,_,err=build_nba_score_models(NBA_HIST_FILE.stat().st_mtime)
    if err or not pack:
        return None,err
    frame,ferr=_current_basketball_score_features(game,"NBA")
    if ferr:
        return None,ferr
    return {
        "home_margin":float(pack["margin_model"].predict(frame[pack["features"]])[0]),
        "game_total":float(pack["total_model"].predict(frame[pack["features"]])[0]),
        "score_pack":pack,
    },None

@st.cache_resource(show_spinner=False)
def build_wnba_score_models(current_season,seasons_back=5):
    teams,terr=fetch_all_teams("WNBA")
    if terr or teams.empty:
        return None,None,terr or "WNBA teams unavailable."

    rows=[]
    seasons=list(range(int(current_season)-seasons_back+1,int(current_season)+1))
    for _,team in teams.iterrows():
        for yr in seasons:
            df,_=fetch_team_recent("WNBA",team["team_id"],yr)
            if df.empty:
                continue
            x=df.copy()
            x["team"]=str(team["abbr"])
            rows.append(x)

    if not rows:
        return None,None,"No WNBA historical schedules were built."

    tg=pd.concat(rows,ignore_index=True)
    tg["date_dt"]=pd.to_datetime(tg["date"],errors="coerce",utc=True)
    tg["pf"]=pd.to_numeric(tg["points_for"],errors="coerce")
    tg["pa"]=pd.to_numeric(tg["points_against"],errors="coerce")
    tg["won"]=(tg["result"].astype(str)=="W").astype(int)
    tg=tg.dropna(subset=["date_dt","pf","pa"]).sort_values(["team","date_dt"])

    grp=tg.groupby("team",group_keys=False)
    tg["off"]=grp["pf"].transform(lambda s:s.shift(1).rolling(10,min_periods=5).mean())
    tg["def"]=grp["pa"].transform(lambda s:s.shift(1).rolling(10,min_periods=5).mean())
    tg["form"]=grp["won"].transform(lambda s:s.shift(1).rolling(10,min_periods=5).mean())
    tg["prev_date"]=grp["date_dt"].shift(1)
    tg["rest"]=(tg["date_dt"]-tg["prev_date"]).dt.days.clip(0,14)
    # Same cloud-safe game-key construction used by the basketball model above.
    _date_keys=[
        x.strftime("%Y-%m-%d") if pd.notna(x) else ""
        for x in tg["date_dt"].tolist()
    ]
    _pair_keys=[
        "|".join(sorted([str(team),str(opp)]))
        for team,opp in zip(tg["team"].tolist(),tg["opponent"].tolist())
    ]
    tg["game_key"]=[
        f"{d}|{pair}"
        for d,pair in zip(_date_keys,_pair_keys)
    ]

    home=tg[tg["home_away"].astype(str).str.lower()=="home"].copy()
    away=tg[tg["home_away"].astype(str).str.lower()=="away"].copy()
    m=home.merge(away,on="game_key",suffixes=("_h","_a"),how="inner")
    m=m[
        (m["team_h"].astype(str)==m["opponent_a"].astype(str))
        &(m["opponent_h"].astype(str)==m["team_a"].astype(str))
    ].copy()

    m["offense_diff"]=m["off_h"]-m["off_a"]
    m["defense_diff"]=m["def_a"]-m["def_h"]
    m["form_diff"]=m["form_h"]-m["form_a"]
    m["rest_diff"]=m["rest_h"]-m["rest_a"]
    m["home_margin"]=m["pf_h"]-m["pa_h"]
    m["game_total"]=m["pf_h"]+m["pa_h"]
    m["date"]=m["date_dt_h"]

    feats=["offense_diff","defense_diff","form_diff","rest_diff"]
    m=m.dropna(subset=feats+["home_margin","game_total","date"]).sort_values("date").reset_index(drop=True)
    if len(m)<120:
        return None,m,f"Only {len(m)} usable WNBA score-model games were built."

    n=len(m); cut1=int(n*.70); cut2=int(n*.85)
    train=m.iloc[:cut1]; cal=m.iloc[cut1:cut2]; test=m.iloc[cut2:].copy()

    margin_model=Pipeline([("scale",StandardScaler()),("ridge",Ridge(alpha=6.0))])
    total_model=Pipeline([("scale",StandardScaler()),("ridge",Ridge(alpha=6.0))])
    train_w=recency_training_weights(train["date"],half_life_days=360)
    margin_model.fit(train[feats],train["home_margin"],ridge__sample_weight=train_w)
    total_model.fit(train[feats],train["game_total"],ridge__sample_weight=train_w)

    cal_margin_pred=margin_model.predict(cal[feats])
    cal_total_pred=total_model.predict(cal[feats])
    margin_resid=cal["home_margin"].to_numpy()-cal_margin_pred
    total_resid=cal["game_total"].to_numpy()-cal_total_pred

    test["margin_pred"]=margin_model.predict(test[feats])
    test["total_pred"]=total_model.predict(test[feats])

    return {
        "margin_model":margin_model,
        "total_model":total_model,
        "features":feats,
        "margin_residuals":margin_resid,
        "total_residuals":total_resid,
        "margin_metrics":_regression_validation(test["home_margin"],test["margin_pred"]),
        "total_metrics":_regression_validation(test["game_total"],test["total_pred"]),
        "test":test,
        "first_date":str(m["date"].min().date()),
        "last_date":str(m["date"].max().date()),
    },test,None

def current_wnba_score_projection(game):
    season=league_season_year("WNBA",st.session_state.get("selected_date",date.today()))
    pack,_,err=build_wnba_score_models(season,5)
    if err or not pack:
        return None,err
    frame,ferr=_current_basketball_score_features(game,"WNBA")
    if ferr:
        return None,ferr
    return {
        "home_margin":float(pack["margin_model"].predict(frame[pack["features"]])[0]),
        "game_total":float(pack["total_model"].predict(frame[pack["features"]])[0]),
        "score_pack":pack,
    },None

@st.cache_resource(show_spinner=False)
def build_nfl_score_models(current_season):
    """
    Separate NFL spread/total score models using real nflverse historical games
    and team EPA/efficiency inputs. Chronological calibration residuals are used
    to turn point projections into cover/total probabilities.
    """
    seasons=list(range(max(2019,int(current_season)-7),int(current_season)+1))
    frames=[]
    errors=[]
    for yr in seasons:
        df,err=fetch_nflverse_team_stats(yr)
        if err:
            errors.append(err)
        if not df.empty:
            frames.append(df)
    schedules,serr=fetch_nflverse_schedules()
    if serr:
        errors.append(serr)
    if not frames or schedules.empty:
        return None,None," | ".join(errors) or "NFL historical data unavailable."

    ts=_nfl_stats_with_defense(pd.concat(frames,ignore_index=True))
    if ts.empty:
        return None,None,"NFL team stats unavailable."
    ts["season"]=pd.to_numeric(ts["season"],errors="coerce")
    ts["week"]=pd.to_numeric(ts["week"],errors="coerce")
    ts=ts.sort_values(["team","season","week","game_id"]).reset_index(drop=True)
    grp=ts.groupby("team",group_keys=False)

    base_cols=[
        "off_epa","passing_epa","rushing_epa","def_epa_allowed",
        "def_pass_epa_allowed","def_rush_epa_allowed",
        "passing_yards","rushing_yards","pass_yards_allowed","rush_yards_allowed",
        "passing_20","rushing_10","explosive_passes_allowed","explosive_rushes_allowed",
        "sacks_suffered",
    ]
    for c in base_cols:
        if c not in ts.columns:
            ts[c]=np.nan
        ts[f"{c}_r5"]=_rolling_shift_mean(grp,c,5,3)

    schedules=schedules.copy()
    for c in ["season","week","home_score","away_score","home_rest","away_rest","temp","wind"]:
        if c in schedules.columns:
            schedules[c]=pd.to_numeric(schedules[c],errors="coerce")
    if "game_type" in schedules.columns:
        schedules=schedules[schedules["game_type"].astype(str)=="REG"].copy()

    h=ts.add_prefix("h_")
    a=ts.add_prefix("a_")
    m=schedules.merge(
        h,left_on=["game_id","home_team"],right_on=["h_game_id","h_team"],how="inner"
    ).merge(
        a,left_on=["game_id","away_team"],right_on=["a_game_id","a_team"],how="inner"
    )

    defs={
        "off_epa_diff":("h_off_epa_r5","a_off_epa_r5"),
        "pass_epa_diff":("h_passing_epa_r5","a_passing_epa_r5"),
        "rush_epa_diff":("h_rushing_epa_r5","a_rushing_epa_r5"),
        "def_epa_diff":("a_def_epa_allowed_r5","h_def_epa_allowed_r5"),
        "def_pass_epa_diff":("a_def_pass_epa_allowed_r5","h_def_pass_epa_allowed_r5"),
        "def_rush_epa_diff":("a_def_rush_epa_allowed_r5","h_def_rush_epa_allowed_r5"),
        "pass_yards_diff":("h_passing_yards_r5","a_passing_yards_r5"),
        "rush_yards_diff":("h_rushing_yards_r5","a_rushing_yards_r5"),
        "explosive_pass_diff":("h_passing_20_r5","a_passing_20_r5"),
        "explosive_rush_diff":("h_rushing_10_r5","a_rushing_10_r5"),
    }
    for out,(x1,x2) in defs.items():
        m[out]=pd.to_numeric(m.get(x1),errors="coerce")-pd.to_numeric(m.get(x2),errors="coerce")

    m["rest_diff"]=pd.to_numeric(m.get("home_rest"),errors="coerce")-pd.to_numeric(m.get("away_rest"),errors="coerce")
    m["temp"]=pd.to_numeric(m.get("temp"),errors="coerce")
    m["wind"]=pd.to_numeric(m.get("wind"),errors="coerce")
    m["home_margin"]=pd.to_numeric(m["home_score"],errors="coerce")-pd.to_numeric(m["away_score"],errors="coerce")
    m["game_total"]=pd.to_numeric(m["home_score"],errors="coerce")+pd.to_numeric(m["away_score"],errors="coerce")
    m["_date"]=pd.to_datetime(m.get("gameday"),errors="coerce")

    feats=[
        "off_epa_diff","pass_epa_diff","rush_epa_diff",
        "def_epa_diff","def_pass_epa_diff","def_rush_epa_diff",
        "pass_yards_diff","rush_yards_diff",
        "explosive_pass_diff","explosive_rush_diff",
        "rest_diff","temp","wind",
    ]
    medians={}
    for c in feats:
        med=pd.to_numeric(m[c],errors="coerce").median()
        medians[c]=0.0 if pd.isna(med) else float(med)
        m[c]=pd.to_numeric(m[c],errors="coerce").fillna(medians[c])

    m=m.dropna(subset=["_date","home_margin","game_total"]).sort_values("_date").reset_index(drop=True)
    if len(m)<400:
        return None,m,f"Only {len(m)} usable NFL score-model games were built."

    n=len(m); cut1=int(n*.70); cut2=int(n*.85)
    train=m.iloc[:cut1]; cal=m.iloc[cut1:cut2]; test=m.iloc[cut2:].copy()

    margin_model=Pipeline([("scale",StandardScaler()),("ridge",Ridge(alpha=10.0))])
    total_model=Pipeline([("scale",StandardScaler()),("ridge",Ridge(alpha=10.0))])
    train_w=recency_training_weights(train["_date"],half_life_days=420)
    margin_model.fit(train[feats],train["home_margin"],ridge__sample_weight=train_w)
    total_model.fit(train[feats],train["game_total"],ridge__sample_weight=train_w)

    cal_margin_pred=margin_model.predict(cal[feats])
    cal_total_pred=total_model.predict(cal[feats])
    margin_resid=cal["home_margin"].to_numpy()-cal_margin_pred
    total_resid=cal["game_total"].to_numpy()-cal_total_pred

    test["margin_pred"]=margin_model.predict(test[feats])
    test["total_pred"]=total_model.predict(test[feats])

    return {
        "margin_model":margin_model,
        "total_model":total_model,
        "features":feats,
        "medians":medians,
        "margin_residuals":margin_resid,
        "total_residuals":total_resid,
        "margin_metrics":_regression_validation(test["home_margin"],test["margin_pred"]),
        "total_metrics":_regression_validation(test["game_total"],test["total_pred"]),
        "test":test,
        "first_date":str(m["_date"].min().date()),
        "last_date":str(m["_date"].max().date()),
    },test,None

def _current_nfl_score_feature_frame(game,pack):
    season=league_season_year("NFL",st.session_state.get("selected_date",date.today()))
    frames=[]
    for yr in [season-1,season]:
        df,_=fetch_nflverse_team_stats(yr)
        if not df.empty:
            frames.append(df)
    if not frames:
        return None,"Current NFL team statistics unavailable."

    ts=_nfl_stats_with_defense(pd.concat(frames,ignore_index=True))
    hp=_latest_nfl_team_profile(ts,game["home_abbr"])
    ap=_latest_nfl_team_profile(ts,game["away_abbr"])
    if not hp or not ap:
        return None,"Not enough current NFL EPA/team history."

    row={
        "off_epa_diff":hp.get("off_epa",np.nan)-ap.get("off_epa",np.nan),
        "pass_epa_diff":hp.get("passing_epa",np.nan)-ap.get("passing_epa",np.nan),
        "rush_epa_diff":hp.get("rushing_epa",np.nan)-ap.get("rushing_epa",np.nan),
        "def_epa_diff":ap.get("def_epa_allowed",np.nan)-hp.get("def_epa_allowed",np.nan),
        "def_pass_epa_diff":ap.get("def_pass_epa_allowed",np.nan)-hp.get("def_pass_epa_allowed",np.nan),
        "def_rush_epa_diff":ap.get("def_rush_epa_allowed",np.nan)-hp.get("def_rush_epa_allowed",np.nan),
        "pass_yards_diff":hp.get("passing_yards",np.nan)-ap.get("passing_yards",np.nan),
        "rush_yards_diff":hp.get("rushing_yards",np.nan)-ap.get("rushing_yards",np.nan),
        "explosive_pass_diff":hp.get("passing_20",np.nan)-ap.get("passing_20",np.nan),
        "explosive_rush_diff":hp.get("rushing_10",np.nan)-ap.get("rushing_10",np.nan),
    }
    ctx=st.session_state.get("matchup_context") or {}
    hr=(ctx.get("home") or {}).get("rest",np.nan)
    ar=(ctx.get("away") or {}).get("rest",np.nan)
    row["rest_diff"]=0 if pd.isna(hr) or pd.isna(ar) else float(hr)-float(ar)

    row["temp"]=pack["medians"].get("temp",0)
    row["wind"]=pack["medians"].get("wind",0)
    if game.get("indoor") is True:
        row["wind"]=0
    else:
        weather,_=fetch_game_day_weather(
            game.get("venue_city",""),game.get("venue_state",""),game.get("date","")
        )
        if weather:
            hi=safe_float(weather.get("high_f")); lo=safe_float(weather.get("low_f"))
            if not pd.isna(hi) and not pd.isna(lo):
                row["temp"]=(hi+lo)/2
            w=safe_float(weather.get("wind_mph"))
            if not pd.isna(w):
                row["wind"]=w

    frame=pd.DataFrame([row])
    for c in pack["features"]:
        frame[c]=pd.to_numeric(frame[c],errors="coerce").fillna(pack["medians"][c])
    return frame,None

def current_nfl_score_projection(game):
    season=league_season_year("NFL",st.session_state.get("selected_date",date.today()))
    pack,_,err=build_nfl_score_models(season)
    if err or not pack:
        return None,err
    frame,ferr=_current_nfl_score_feature_frame(game,pack)
    if ferr:
        return None,ferr
    return {
        "home_margin":float(pack["margin_model"].predict(frame[pack["features"]])[0]),
        "game_total":float(pack["total_model"].predict(frame[pack["features"]])[0]),
        "score_pack":pack,
    },None

def current_score_projection(game,league):
    if league=="NBA":
        return current_nba_score_projection(game)
    if league=="WNBA":
        return current_wnba_score_projection(game)
    if league=="NFL":
        return current_nfl_score_projection(game)
    return None,"No spread/total model for this league."

def parse_kalshi_spread_total_contract(row,game):
    typ=str(row.get("Type",""))
    ticker=str(row.get("Ticker","")).upper()
    text=" ".join([
        str(row.get("Title","")),
        str(row.get("Subtitle","")),
        str(row.get("YES subtitle","")),
        str(row.get("Market","")),
    ])
    low=text.lower()

    if typ=="Total":
        threshold=np.nan
        m=re.search(r"(?:over|more than)\s+(\d+(?:\.\d+)?)\s*(?:points|pts)",low)
        if m:
            threshold=float(m.group(1))
        if pd.isna(threshold):
            m=re.search(r"-(\d+)$",ticker)
            if m:
                threshold=float(m.group(1))+.5
        if pd.isna(threshold):
            return None
        return {
            "market_type":"Total",
            "threshold":threshold,
            "relation":"over",
            "label":f"Over {threshold:g} total points",
            "side_abbr":"",
        }

    if typ=="Spread":
        threshold=np.nan
        m=re.search(r"wins?\s+by\s+(?:over|more than)\s+(\d+(?:\.\d+)?)",low)
        if m:
            threshold=float(m.group(1))

        team_abbr=""
        # Prefer exact team names/abbr in contract text.
        for side in ["home","away"]:
            name=str(game[f"{side}_name"]).lower()
            abbr=str(game[f"{side}_abbr"]).upper()
            if name and name in low:
                team_abbr=game[f"{side}_abbr"]
                break
            if re.search(rf"\b{re.escape(abbr)}\b",text.upper()):
                team_abbr=game[f"{side}_abbr"]
                break

        # Kalshi sports spread tickers conventionally end TEAM + integer strike,
        # where TEAM3 corresponds to >3.5.
        tm=re.search(r"-([A-Z]{2,4})(\d+)$",ticker)
        if tm:
            code=tm.group(1)
            if not team_abbr:
                for side in ["home","away"]:
                    if normalize_team_code(game[f"{side}_abbr"])==normalize_team_code(code):
                        team_abbr=game[f"{side}_abbr"]
                        break
            if pd.isna(threshold):
                threshold=float(tm.group(2))+.5

        if not team_abbr or pd.isna(threshold):
            return None
        team_name=(
            game["home_name"]
            if team_abbr==game["home_abbr"]
            else game["away_name"]
        )
        return {
            "market_type":"Spread",
            "threshold":threshold,
            "relation":"win_by_over",
            "label":f"{team_name} wins by more than {threshold:g}",
            "side_abbr":team_abbr,
        }
    return None

def spread_total_probability_and_validation(game,league,contract,projection):
    pack=projection["score_pack"]
    threshold=float(contract["threshold"])
    if contract["market_type"]=="Total":
        pred=projection["game_total"]
        p=_empirical_over_probability(pred,pack["total_residuals"],threshold)
        val=_threshold_validation(
            pack["test"],"total_pred","game_total",
            pack["total_residuals"],threshold,"over"
        )
        metric=pack["total_metrics"]
        return p,val,metric

    # Spread YES means selected team wins by MORE THAN threshold.
    if contract["side_abbr"]==game["home_abbr"]:
        pred=projection["home_margin"]
        p=_empirical_over_probability(pred,pack["margin_residuals"],threshold)
        val=_threshold_validation(
            pack["test"],"margin_pred","home_margin",
            pack["margin_residuals"],threshold,"over"
        )
    else:
        # Away margin > threshold <=> home margin < -threshold.
        pred=projection["home_margin"]
        p=_empirical_under_probability(pred,pack["margin_residuals"],-threshold)
        val=_threshold_validation(
            pack["test"],"margin_pred","home_margin",
            pack["margin_residuals"],-threshold,"under"
        )
    metric=pack["margin_metrics"]
    return p,val,metric

def exact_market_row_with_fill(
    base_row,
    bankroll=1000.0,
    kelly_fraction=.25,
    max_pct=.02,
):
    p=float(base_row["Model probability"])
    quoted=float(base_row["Kalshi price"])
    ticker=base_row["Ticker"]

    # One-contract fee first, used to make the Kelly estimate fee-aware.
    one_fee=kalshi_general_taker_fee(1,quoted,1.0)
    kelly=kalshi_fee_adjusted_kelly(p,quoted,one_fee,kelly_fraction)
    target=min(max(kelly,0),float(max_pct))*float(bankroll)

    fill=simulate_yes_taker_fill(ticker,target,quoted)
    actual_price=(
        fill["avg_price"]
        if fill.get("filled") and not pd.isna(fill.get("avg_price",np.nan))
        else quoted
    )
    contracts=int(fill.get("contracts",0) or 0)
    total_fee=float(fill.get("fee",0) or 0)
    fee_per_contract=(total_fee/contracts) if contracts>0 else kalshi_general_taker_fee(1,actual_price,1.0)

    edge=p-actual_price
    ev_per_contract=p-actual_price-fee_per_contract
    actual_position=float(fill.get("all_in_cost",0) or 0)

    out=dict(base_row)
    out.update({
        "Quoted Kalshi price":quoted,
        "Kalshi price":actual_price,
        "Edge":edge,
        "Exact taker fee":total_fee,
        "Fee per contract":fee_per_contract,
        "EV per contract":ev_per_contract,
        "EV per $1":(
            ev_per_contract/max(actual_price,1e-9)
            if actual_price>0 else np.nan
        ),
        "Contracts":contracts,
        "Suggested position":actual_position,
        "Fill source":fill.get("depth_status",""),
        "Best ask":fill.get("best_ask",quoted),
        "Orderbook note":fill.get("orderbook_error",""),
    })
    return out

def full_market_model_dashboard_rows(
    game,league,bankroll=1000.0,kelly_fraction=.25,max_pct=.02
):
    rows=[]
    errors=[]

    # ---------- Winner contracts ----------
    prices,kerr=selected_game_kalshi_prices_v22(game,league)
    p_home,model_err,_=current_league_model_probability(game,league)
    metrics=model_metrics_for_league(league)
    if kerr:
        errors.append(str(kerr))
    if model_err:
        errors.append(str(model_err))

    if prices and not pd.isna(p_home):
        for r in prices:
            model_p=p_home if r["Abbr"]==game["home_abbr"] else 1-p_home
            quote=r["Kalshi probability"]
            if pd.isna(quote):
                continue
            base={
                "Event ID":game.get("event_id",""),
                "Date":str(st.session_state.get("selected_date",date.today())),
                "League":league,
                "Game":f"{game['away_abbr']} at {game['home_abbr']}",
                "Market Type":"Winner",
                "Market":f"{r['Team']} to win",
                "Side":r["Team"],
                "Side Abbr":r["Abbr"],
                "Ticker":r["Ticker"],
                "Threshold":np.nan,
                "Kalshi price":quote,
                "Model probability":model_p,
                "Historical accuracy":metrics.get("accuracy",np.nan),
                "Historical Brier":metrics.get("brier",np.nan),
                "Validation N":metrics.get("test_n",np.nan),
                "Projection":np.nan,
                "Regression MAE":np.nan,
            }
            rows.append(
                exact_market_row_with_fill(
                    base,bankroll,kelly_fraction,max_pct
                )
            )

    # ---------- Spread / total contracts ----------
    projection,perr=current_score_projection(game,league)
    if perr:
        errors.append(str(perr))
    others=selected_game_other_kalshi_markets_v22(game,league)

    if projection:
        for r in others:
            contract=parse_kalshi_spread_total_contract(r,game)
            if not contract:
                continue
            quote=r.get("YES probability",np.nan)
            if pd.isna(quote):
                continue
            model_p,val,regmetric=spread_total_probability_and_validation(
                game,league,contract,projection
            )
            if pd.isna(model_p):
                continue
            proj_value=(
                projection["game_total"]
                if contract["market_type"]=="Total"
                else (
                    projection["home_margin"]
                    if contract["side_abbr"]==game["home_abbr"]
                    else -projection["home_margin"]
                )
            )
            base={
                "Event ID":game.get("event_id",""),
                "Date":str(st.session_state.get("selected_date",date.today())),
                "League":league,
                "Game":f"{game['away_abbr']} at {game['home_abbr']}",
                "Market Type":contract["market_type"],
                "Market":contract["label"],
                "Side":contract["label"],
                "Side Abbr":contract.get("side_abbr",""),
                "Ticker":r["Ticker"],
                "Threshold":contract["threshold"],
                "Kalshi price":quote,
                "Model probability":model_p,
                "Historical accuracy":val.get("accuracy",np.nan),
                "Historical Brier":val.get("brier",np.nan),
                "Validation N":val.get("n",0),
                "Projection":proj_value,
                "Regression MAE":regmetric.get("mae",np.nan),
            }
            rows.append(
                exact_market_row_with_fill(
                    base,bankroll,kelly_fraction,max_pct
                )
            )

    return rows," | ".join(dict.fromkeys(x for x in errors if x)) if errors else None

def record_all_market_evaluations(rows):
    """
    Logs EVERY matched, priced market row produced by the model evaluator:
    winner + spread + total. One row per ticker per hourly snapshot.
    """
    if not rows:
        return 0
    old=load_csv(MODEL_MARKET_HISTORY_FILE)
    hour=datetime.now().strftime("%Y-%m-%d %H")
    new_rows=[]
    for r in rows:
        key=f"{hour}|{r.get('Event ID')}|{r.get('Ticker')}"
        rec={
            "snapshot_hour":hour,
            "record_key":key,
            "timestamp":datetime.now().isoformat(timespec="seconds"),
            "event_id":r.get("Event ID",""),
            "date":r.get("Date",""),
            "league":r.get("League",""),
            "game":r.get("Game",""),
            "market_type":r.get("Market Type",""),
            "market":r.get("Market",""),
            "side":r.get("Side",""),
            "side_abbr":r.get("Side Abbr",""),
            "ticker":r.get("Ticker",""),
            "threshold":r.get("Threshold",np.nan),
            "kalshi_price":r.get("Kalshi price",np.nan),
            "quoted_kalshi_price":r.get("Quoted Kalshi price",np.nan),
            "model_probability":r.get("Model probability",np.nan),
            "edge":r.get("Edge",np.nan),
            "ev_per_contract":r.get("EV per contract",np.nan),
            "ev_per_dollar":r.get("EV per $1",np.nan),
            "historical_accuracy":r.get("Historical accuracy",np.nan),
            "historical_brier":r.get("Historical Brier",np.nan),
            "validation_n":r.get("Validation N",np.nan),
            "projection":r.get("Projection",np.nan),
            "regression_mae":r.get("Regression MAE",np.nan),
            "contracts":r.get("Contracts",0),
            "entry_fee":r.get("Exact taker fee",0),
            "fee_per_contract":r.get("Fee per contract",0),
            "suggested_position":r.get("Suggested position",0),
            "fill_source":r.get("Fill source",""),
            "settled":"",
            "won":"",
            "profit_loss_100":"",
        }
        new_rows.append(rec)

    new=pd.DataFrame(new_rows)
    if not old.empty and "record_key" in old.columns:
        new=new[~new["record_key"].isin(old["record_key"].astype(str))]
    if new.empty:
        return 0
    out=pd.concat([old,new],ignore_index=True) if not old.empty else new
    out.to_csv(MODEL_MARKET_HISTORY_FILE,index=False)
    return len(new)

def auto_evaluate_log_selected_game(game,league):
    settings=load_json(
        SETTINGS_FILE,
        {"bankroll":1000.0,"kelly_fraction":.25,"max_pct":.02}
    )
    settings.setdefault("bankroll",1000.0)
    settings.setdefault("kelly_fraction",.25)
    settings.setdefault("max_pct",.02)

    rows,err=full_market_model_dashboard_rows(
        game,league,
        bankroll=float(settings["bankroll"]),
        kelly_fraction=float(settings["kelly_fraction"]),
        max_pct=float(settings["max_pct"]),
    )
    added=record_all_market_evaluations(rows)
    return rows,err,added

def selected_game_edge_alerts(game,league,rows,ctx=None):
    """Promote only model edges that also have outside sportsbook support when comparable."""
    settings=load_json(SETTINGS_FILE,{})
    min_edge=float(settings.get("edge_alert_min",.05))
    min_books=int(settings.get("edge_alert_min_books",2))
    consensus,cons_err=fetch_espn_moneyline_consensus(league,game.get("event_id",""))
    injury_verified=False
    if ctx:
        injury_verified=all(bool((ctx.get(side) or {}).get("injury_verified",False)) for side in ["home","away"])
    alerts=[]
    for r in rows or []:
        edge=safe_float(r.get("Edge")); ev=safe_float(r.get("EV per contract")); price=safe_float(r.get("Kalshi price")); model=safe_float(r.get("Model probability"))
        if pd.isna(edge) or pd.isna(model) or pd.isna(price) or edge<min_edge or (not pd.isna(ev) and ev<=0):
            continue
        item=dict(r)
        item["Sportsbook probability"]=np.nan
        item["Sportsbook books"]=0
        item["Sportsbook support"]=False
        item["Sportsbook note"]=cons_err or "No comparable sportsbook consensus."
        if str(r.get("Market Type"))=="Winner" and consensus:
            side=str(r.get("Side Abbr",""))
            if side==game.get("home_abbr"):
                bp=consensus.get("home_probability",np.nan)
            elif side==game.get("away_abbr"):
                bp=consensus.get("away_probability",np.nan)
            else:
                bp=np.nan
            item["Sportsbook probability"]=bp
            item["Sportsbook books"]=int(consensus.get("books",0) or 0)
            item["Sportsbook support"]=(not pd.isna(bp) and bp>price+.01 and item["Sportsbook books"]>=min_books)
            item["Sportsbook note"]=(
                f"De-vigged consensus {bp*100:.1f}% from {item['Sportsbook books']} book(s)."
                if not pd.isna(bp) else "Sportsbook consensus unavailable."
            )
            item["Sportsbook quotes"]=consensus.get("quotes",[])
        # Strong alert requires outside confirmation for winner markets.
        # Spread/total rows stay in the market table until comparable multi-book prices exist.
        item["Alert strength"]=(
            "STRONG" if item["Sportsbook support"] and injury_verified
            else "SUPPORTED" if item["Sportsbook support"]
            else "MODEL-ONLY"
        )
        if item["Sportsbook support"]:
            alerts.append(item)
    alerts.sort(key=lambda x:(safe_float(x.get("Edge")) if not pd.isna(safe_float(x.get("Edge"))) else -1),reverse=True)
    return alerts,cons_err

def render_edge_alerts(game,league,rows,ctx=None,compact=False):
    alerts,_=selected_game_edge_alerts(game,league,rows,ctx)
    if not alerts:
        return False
    top=alerts[0]
    st.error(
        f"🚨 EDGE ALERT — {top.get('Market','Market')} | "
        f"Kalshi {fmt_pct(top.get('Kalshi price'))} vs model {fmt_pct(top.get('Model probability'))} "
        f"({safe_float(top.get('Edge'))*100:+.1f} pts)"
    )
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Kalshi",fmt_pct(top.get("Kalshi price")))
    c2.metric("Model",fmt_pct(top.get("Model probability")))
    c3.metric("Sportsbooks",fmt_pct(top.get("Sportsbook probability")))
    c4.metric("EV / contract",fmt_money(top.get("EV per contract")))
    st.caption(
        f"{top.get('Sportsbook note','')} • Historical model accuracy {fmt_pct(top.get('Historical accuracy'))} • "
        f"Risk-size calculation {fmt_money(top.get('Suggested position'))}. "
        "This is a model/price discrepancy, not a guarantee."
    )
    if not compact:
        with st.expander("Why this alert fired"):
            st.write(f"**Model edge:** {safe_float(top.get('Edge'))*100:+.1f} percentage points")
            st.write(f"**Outside support:** {top.get('Sportsbook books',0)} de-vigged sportsbook quote(s) also price this outcome above Kalshi.")
            st.write(f"**Data status:** injury feeds {'verified' if all(bool((ctx or {}).get(x,{}).get('injury_verified',False)) for x in ['home','away']) else 'not fully verified'}.")
            quotes=top.get("Sportsbook quotes",[]) or []
            if quotes:
                qdf=pd.DataFrame([{
                    "Book":q.get("provider","Book"),
                    game.get("away_abbr","Away"): ("—" if pd.isna(safe_float(q.get("away_ml"))) else f"{int(safe_float(q.get('away_ml'))):+d}"),
                    game.get("home_abbr","Home"): ("—" if pd.isna(safe_float(q.get("home_ml"))) else f"{int(safe_float(q.get('home_ml'))):+d}"),
                } for q in quotes])
                st.dataframe(qdf,use_container_width=True,hide_index=True)
            if get_tavily_key():
                if st.button("Check current outside-model/news corroboration",key=f"edge_corroborate_{top.get('Ticker','')}"):
                    with st.spinner("Checking current public models/news..."):
                        result=run_free_research(
                            [
                                f"{game.get('away_name')} {game.get('home_name')} prediction model {st.session_state.get('selected_date',date.today())}",
                                f"{game.get('away_name')} {game.get('home_name')} injury lineup prediction odds today",
                            ],
                            "Summarize only verifiable current outside model predictions, injury/lineup changes, or market context relevant to this game. Do not invent a probability. Cite the returned sources.",
                            context=build_ai_dashboard_context(),
                        )
                    if result.get("summary"):
                        st.markdown(result["summary"])
                    if result.get("sources"):
                        for src in result["sources"][:6]:
                            st.markdown(f"- [{src.get('title','Source')}]({src.get('url','')})")
    if len(alerts)>1 and not compact:
        st.caption(f"{len(alerts)-1} additional supported edge alert(s) are available in Markets & Edge Alerts.")
    return True

def settle_all_market_prediction_history():
    """
    Settles winner, spread and total model snapshots from final game scores.
    Profit/loss is standardized per $100 of gross contract spend using the
    actual modeled entry price plus the recorded taker fee per contract.
    """
    hist=load_csv(MODEL_MARKET_HISTORY_FILE)
    if hist.empty:
        return hist,0

    for c in ["settled","won","profit_loss_100"]:
        if c not in hist.columns:
            hist[c]=""

    updated=0
    schedule_cache={}
    for idx,row in hist.iterrows():
        if str(row.get("settled","")).lower() in {"true","1","yes"}:
            continue

        lg=str(row.get("league",""))
        d=pd.to_datetime(row.get("date"),errors="coerce")
        if lg not in LEAGUES or pd.isna(d):
            continue

        key=(lg,d.date().isoformat())
        if key not in schedule_cache:
            games,_=fetch_schedule(lg,d.date())
            schedule_cache[key]=games

        game=next(
            (g for g in schedule_cache[key]
             if str(g.get("event_id",""))==str(row.get("event_id",""))),
            None,
        )
        if not game or game.get("state")!="post":
            continue

        hs=safe_float(game.get("home_score"))
        a_s=safe_float(game.get("away_score"))
        if pd.isna(hs) or pd.isna(a_s):
            continue

        market_type=str(row.get("market_type","Winner"))
        threshold=safe_float(row.get("threshold"))
        yes=False

        if market_type=="Winner":
            if hs==a_s:
                continue
            winner=game["home_abbr"] if hs>a_s else game["away_abbr"]
            yes=str(row.get("side_abbr",""))==str(winner)

        elif market_type=="Spread":
            if pd.isna(threshold):
                continue
            side_abbr=str(row.get("side_abbr",""))
            margin=(hs-a_s) if side_abbr==game["home_abbr"] else (a_s-hs)
            yes=margin>threshold

        elif market_type=="Total":
            if pd.isna(threshold):
                continue
            yes=(hs+a_s)>threshold
        else:
            continue

        price=safe_float(row.get("kalshi_price"))
        fee_pc=safe_float(row.get("fee_per_contract"))
        if pd.isna(fee_pc):
            fee_pc=kalshi_general_taker_fee(1,price,1.0) if not pd.isna(price) else 0

        pnl=np.nan
        if not pd.isna(price) and 0<price<1:
            units=100.0/price
            per_contract=(1-price-fee_pc) if yes else -(price+fee_pc)
            pnl=units*per_contract

        hist.at[idx,"settled"]="true"
        hist.at[idx,"won"]="true" if yes else "false"
        hist.at[idx,"profit_loss_100"]=pnl
        updated+=1

    if updated:
        hist.to_csv(MODEL_MARKET_HISTORY_FILE,index=False)
    return hist,updated

def full_market_checklist_rows():
    return [
        ("Historical game data retained with recency weighting","✅","Older games remain available to the models, but newer seasons receive greater training weight."),
        ("Feature engineering — EPA, efficiency, injuries, pace, matchup, availability","✅","NFL EPA/pass/rush/explosive/rest/weather; basketball efficiency/form/pace when available; injuries, matchup and minutes layered into research."),
        ("Backtesting on unseen games","✅","Chronological held-out validation for team models; rolling out-of-sample player-prop backtests."),
        ("Probability calibration","✅","Isotonic calibration for NFL/WNBA winner models; empirical calibration residuals for spread/total probabilities; player prop calibration retained."),
        ("Kalshi market-price input","✅","Winner, spread and total contracts are matched to the selected game and priced automatically."),
        ("Edge / EV calculator with fees","✅","Uses model probability minus actual simulated fill price; EV subtracts Kalshi general taker fee."),
        ("Bankroll / risk module","✅","Fee-aware fractional Kelly plus hard max-bankroll cap."),
        ("Live/current updates","✅","Injuries, roster, published starters/lineups, weather, sportsbook context, Kalshi quotes/orderbook."),
        ("Separate NFL and basketball models","✅","NFL uses football-specific EPA features; NBA/WNBA use basketball historical features."),
        ("Performance tracking","✅","Every matched priced winner/spread/total market the evaluator produces is logged and can be settled."),
        ("Data collection","✅","NBA/WNBA/ESPN/nflverse/Kalshi/Open-Meteo plus a persistent local player-game database and manual CSV import."),
        ("Team/player statistics","✅","Team and player report pages."),
        ("Recent form","✅","Recent team/player windows used."),
        ("Home/away","✅","Team and player splits where available."),
        ("Injuries/availability","✅","Verified/unverified distinction and player status."),
        ("Opponent strength","✅","Opponent efficiency/profile plus exact-opponent history."),
        ("Expected minutes for player markets","✅","Likely minutes/range and explicit restriction parsing."),
        ("Probability model","✅","Statistical models; probabilities are not manually guessed."),
        ("Calibrate 60% to behave roughly like 60%","✅","Calibration data is separated chronologically from final held-out tests."),
        ("Kalshi implied probability","✅","YES price is treated as market-implied probability."),
        ("Edge = Model probability − Market probability","✅","Shown after orderbook fill simulation."),
        ("Expected value","✅","Expected net profit per contract and per $1 after modeled taker fee."),
        ("Potential profit/loss per contract","✅","Win/loss economics and standardized settled P/L are tracked."),
        ("Historical edge profitability","✅","Settled winner/spread/total rows are grouped by edge band with win rate/P&L."),
        ("Dashboard — Game","✅","Included."),
        ("Dashboard — Market","✅","Winner, spread and total."),
        ("Dashboard — Kalshi price","✅","Simulated immediate-fill average price where depth is available."),
        ("Dashboard — Model probability","✅","Included."),
        ("Dashboard — Edge","✅","Included."),
        ("Dashboard — Expected value","✅","Included."),
        ("Dashboard — Historical model accuracy","✅","Contract-threshold held-out accuracy/Brier for spread/total; held-out accuracy for winner models."),
        ("Dashboard — Suggested conservative position size","✅","Fee-aware Kelly fraction capped by bankroll percentage and visible liquidity."),
    ]


# ============================================================
# V22: CURRENT-GAME COVERAGE / POSSIBLE LINEUPS / ROBUST MARKET MATCHING
# ============================================================

def _team_search_tokens(game,side):
    name=str(game.get(f"{side}_name","")).upper()
    abbr=str(game.get(f"{side}_abbr","")).upper()
    norm=normalize_team_code(abbr)
    short_name=name.split()[-1] if name else ""
    tokens={abbr,norm,name,short_name}
    # Common WNBA/NBA display aliases.
    aliases={
        "LV":{"LV","LVA","LAS VEGAS","ACES"},
        "LA":{"LA","LAS","LOS ANGELES","SPARKS"},
        "CON":{"CON","CONN","CONNECTICUT","SUN"},
        "POR":{"POR","PDX","PORTLAND","FIRE"},
        "GSW":{"GS","GSW","GOLDEN STATE","VALKYRIES","WARRIORS"},
        "NYK":{"NY","NYK","NEW YORK","LIBERTY","KNICKS"},
    }
    tokens |= aliases.get(norm,set())
    return {t for t in tokens if t}

def market_matches_selected_game(market,game,selected_date=None):
    """
    Match a Kalshi market directly to the selected game instead of relying only
    on one outcome ticker -> schedule roundtrip.
    """
    raw=" ".join([
        str(market.get("event_ticker","")),
        str(market.get("ticker","")),
        str(market.get("title","")),
        str(market.get("subtitle","")),
        str(market.get("yes_sub_title","")),
        str(market.get("no_sub_title","")),
    ]).upper()

    kd=kalshi_event_date(market)
    sd=selected_date or st.session_state.get("selected_date")
    if isinstance(sd,datetime):
        sd=sd.date()
    if isinstance(sd,pd.Timestamp):
        sd=sd.date()
    if kd and sd and kd!=sd:
        return False

    away_tokens=_team_search_tokens(game,"away")
    home_tokens=_team_search_tokens(game,"home")

    def token_present(token):
        if len(token)<=4:
            return bool(re.search(rf"(?<![A-Z]){re.escape(token)}(?![A-Z])",raw))
        return token in raw

    away_hit=any(token_present(t) for t in away_tokens)
    home_hit=any(token_present(t) for t in home_tokens)
    if away_hit and home_hit:
        return True

    # Ticker event suffix often concatenates the two team codes, e.g. LALV.
    event=str(market.get("event_ticker") or "").upper()
    tail=re.sub(r"^.*-\d{2}[A-Z]{3}\d{2}","",event)
    a_codes={re.sub(r"[^A-Z]","",x) for x in away_tokens if 1<len(re.sub(r"[^A-Z]","",x))<=4}
    h_codes={re.sub(r"[^A-Z]","",x) for x in home_tokens if 1<len(re.sub(r"[^A-Z]","",x))<=4}
    for a in a_codes:
        for h in h_codes:
            if tail in {a+h,h+a} or (a+h) in event or (h+a) in event:
                return True
    return False

def selected_game_kalshi_prices_v22(game,league):
    markets,err=fetch_kalshi_series_markets(
        KALSHI_SPORT_SERIES[league]["Game winner"],
        max_pages=12,
    )
    if err and not markets:
        return [],err

    rows=[]
    home=normalize_team_code(game["home_abbr"])
    away=normalize_team_code(game["away_abbr"])
    for m in markets:
        if not market_matches_selected_game(m,game):
            continue
        outcome=kalshi_market_outcome_code(m)
        # If ticker outcome alias is odd, use yes subtitle/title to identify side.
        side_abbr=""
        if outcome==home:
            side_abbr=game["home_abbr"]
        elif outcome==away:
            side_abbr=game["away_abbr"]
        else:
            yes_text=" ".join([
                str(m.get("yes_sub_title","")),
                str(m.get("title","")),
                str(m.get("subtitle","")),
            ]).upper()
            if any(t in yes_text for t in _team_search_tokens(game,"home")):
                side_abbr=game["home_abbr"]
            elif any(t in yes_text for t in _team_search_tokens(game,"away")):
                side_abbr=game["away_abbr"]
        if not side_abbr:
            continue
        name=game["home_name"] if side_abbr==game["home_abbr"] else game["away_name"]
        rows.append({
            "Team":name,
            "Abbr":side_abbr,
            "Kalshi probability":kalshi_yes_ask(m),
            "Ticker":m.get("ticker",""),
            "Bid":kalshi_yes_bid(m),
            "Spread":kalshi_market_spread(m),
        })

    # Deduplicate per team, preferring a row with a real ask.
    out=[]
    for abbr in [game["away_abbr"],game["home_abbr"]]:
        candidates=[r for r in rows if r["Abbr"]==abbr]
        if candidates:
            candidates=sorted(candidates,key=lambda r:pd.isna(r["Kalshi probability"]))
            out.append(candidates[0])
    return out,err

def selected_game_other_kalshi_markets_v22(game,league):
    rows=[]
    for typ in ["Spread","Total"]:
        series=KALSHI_SPORT_SERIES[league].get(typ)
        if not series:
            continue
        markets,err=fetch_kalshi_series_markets(series,max_pages=12)
        for m in markets:
            if not market_matches_selected_game(m,game):
                continue
            rows.append({
                "Type":typ,
                "Market":m.get("title") or m.get("subtitle") or m.get("ticker"),
                "Title":m.get("title",""),
                "Subtitle":m.get("subtitle",""),
                "YES subtitle":m.get("yes_sub_title",""),
                "NO subtitle":m.get("no_sub_title",""),
                "YES probability":kalshi_yes_ask(m),
                "YES bid":kalshi_yes_bid(m),
                "Spread":kalshi_market_spread(m),
                "Ticker":m.get("ticker",""),
                "Event ticker":m.get("event_ticker",""),
                "Raw market":m,
            })
    return rows[:120]

@st.cache_data(ttl=300,show_spinner=False)
def fetch_recent_possible_starters(league,team_id,current_season,max_games=5):
    """
    Derive a POSSIBLE lineup from starter flags in the team's most recent
    completed games. This is deliberately not labeled confirmed.
    """
    histories=[]
    errors=[]
    for yr in [int(current_season),int(current_season)-1]:
        hist,err=fetch_team_recent(league,team_id,yr)
        if err:
            errors.append(err)
        if not hist.empty:
            histories.append(hist)
    if not histories:
        return [],0," | ".join(errors) if errors else "No recent completed games."

    hist=pd.concat(histories,ignore_index=True)
    hist["_d"]=pd.to_datetime(hist["date"],errors="coerce",utc=True)
    hist=hist.sort_values("_d",ascending=False).drop_duplicates("event_id")
    event_ids=[x for x in hist["event_id"].astype(str).tolist() if x][:max_games]

    counts={}
    last_seen={}
    games_checked=0
    for event_id in event_ids:
        info,err=fetch_event_lineup_status(league,event_id)
        if err:
            errors.append(err)
        if not info:
            continue

        # More reliable: get this team's roster abbreviations from the recent event
        # by matching starter names against current roster below. First collect all
        # blocks and choose the one with the largest current-roster name overlap.
        roster,_=fetch_roster(league,team_id)
        roster_names=set(_name_key(x) for x in roster.get("player",pd.Series(dtype=str)).astype(str))
        best=None
        best_overlap=-1
        for abbr,data in info.items():
            starters=data.get("starters") or []
            overlap=sum(1 for n in starters if _name_key(n) in roster_names)
            if overlap>best_overlap:
                best_overlap=overlap
                best=data
        if not best or not best.get("starters"):
            continue

        games_checked+=1
        for name in best.get("starters",[]):
            key=_name_key(name)
            if not key:
                continue
            counts[key]=counts.get(key,{"name":name,"starts":0})
            counts[key]["starts"]+=1
            last_seen.setdefault(key,games_checked)

    if not counts:
        return [],games_checked," | ".join(dict.fromkeys(errors)) if errors else "Recent box scores did not expose starter flags."

    limit=5 if league in {"NBA","WNBA"} else 11
    ranked=sorted(
        counts.values(),
        key=lambda r:(-r["starts"],r["name"])
    )[:limit]
    return ranked,games_checked," | ".join(dict.fromkeys(errors)) if errors else None

def possible_lineup_for_side(game,league,side,ctx=None):
    team_id=game[f"{side}_id"]
    season=league_season_year(
        league,
        st.session_state.get("selected_date",date.today())
    )
    possible,games_checked,err=fetch_recent_possible_starters(
        league,team_id,season,5
    )

    injuries=pd.DataFrame()
    verified=False
    if ctx:
        t=ctx.get(side,{})
        injuries=t.get("injury_df",pd.DataFrame())
        verified=t.get("injury_verified",False)

    rows=[]
    for item in possible:
        name=item["name"]
        status=""
        inj=_injury_row_for_player(injuries,name)
        if inj is not None:
            status=str(inj.get("Status","") or "").upper()
        rows.append({
            "Player":name,
            "Recent starts":f"{item['starts']}/{max(games_checked,1)}",
            "Current injury status":status or ("No designation on verified feed" if verified else "Not verified"),
        })

    return pd.DataFrame(rows),games_checked,err

def render_current_or_possible_lineups(game,league,ctx):
    """
    Show confirmed starters when the live feed exposes them.
    Otherwise show a clearly labeled POSSIBLE lineup from recent starts.
    """
    current,err=fetch_event_lineup_status(league,game["event_id"])
    confirmed_any=bool(current and any(v.get("starters") for v in current.values()))

    if confirmed_any:
        st.success("The live game feed currently exposes starter flags.")
        left,right=st.columns(2)
        for side,col in [("away",left),("home",right)]:
            with col:
                abbr=game[f"{side}_abbr"]
                block=next(
                    (v for k,v in current.items() if normalize_team_code(k)==normalize_team_code(abbr)),
                    None
                )
                st.markdown(f"### {game[f'{side}_name']}")
                starters=(block or {}).get("starters") or []
                if starters:
                    for n in starters:
                        st.write("✅ "+n)
                else:
                    st.caption("Starter flags were not returned for this team.")
        return

    st.warning(
        "Final starters are not available from this live feed yet. "
        "Below is a **POSSIBLE lineup based on recent starts**, not a confirmed starting lineup."
    )
    if err:
        st.caption("Live starter feed: "+str(err))

    left,right=st.columns(2)
    for side,col in [("away",left),("home",right)]:
        with col:
            st.markdown(f"### {game[f'{side}_name']} — POSSIBLE")
            poss,n,perr=possible_lineup_for_side(game,league,side,ctx)
            if not poss.empty:
                st.dataframe(poss,use_container_width=True,hide_index=True)
                st.caption(
                    f"Estimated from starter flags in up to {n} recent completed game(s). "
                    "Current injury status is shown separately so an injured player is not silently treated as confirmed."
                )
            else:
                st.info(
                    "A reliable possible lineup could not be built from recent starter flags. "
                    "The app will not invent five names."
                )
                if perr:
                    st.caption(perr)


# ============================================================
# V23: RESEARCH-FIRST SOURCE STACK / ROBUST FALLBACKS
# ============================================================

SOURCE_POLICY_V23 = {
    "NBA": ["NBA official stats", "ESPN current game/team feeds", "NBA official injury report", "Kalshi", "DraftKings via ESPN"],
    "WNBA": ["WNBA official stats", "ESPN current game/team feeds", "WNBA official injury report", "Kalshi", "DraftKings via ESPN"],
    "NFL": ["NFL.com official injury report", "ESPN current game/team feeds", "nflverse historical/EPA", "Open-Meteo", "Kalshi", "DraftKings via ESPN"],
}


def _completed_status_v23(event, comp):
    status=((event.get("status") or {}).get("type") or {})
    if status.get("state")=="post" or status.get("completed") is True:
        return True
    cstatus=((comp.get("status") or {}).get("type") or {})
    if cstatus.get("state")=="post" or cstatus.get("completed") is True:
        return True
    txt=" ".join([
        str(status.get("name") or ""), str(status.get("description") or ""),
        str(status.get("detail") or ""), str(cstatus.get("name") or ""),
        str(cstatus.get("description") or ""), str(cstatus.get("detail") or ""),
    ]).lower()
    return any(x in txt for x in ["final","completed","game over"])


@st.cache_data(ttl=600, show_spinner=False)
def fetch_team_recent(league,team_id,season):
    """Fast ESPN current-season team history fetch with short failover."""
    cfg=LEAGUES[league]
    url=f"https://site.api.espn.com/apis/site/v2/sports/{cfg['sport']}/{cfg['league']}/teams/{team_id}/schedule"
    attempts=[{"season":int(season),"seasontype":2},{"season":int(season)}]
    errors=[]; payloads=[]
    for params in attempts:
        try:
            payload=request_json(url,params,timeout=8)
            payloads.append(payload)
            if payload.get("events"): break
        except Exception as e: errors.append(str(e))
    rows=[]
    for payload in payloads:
        for event in payload.get("events",[]) or []:
            comps=event.get("competitions") or []
            if not comps: continue
            comp=comps[0]
            if not _completed_status_v23(event,comp): continue
            cs=comp.get("competitors") or []
            mine=next((c for c in cs if str((c.get("team") or {}).get("id",""))==str(team_id)),None)
            if mine is None: mine=next((c for c in cs if str(c.get("id",""))==str(team_id)),None)
            opp=next((c for c in cs if c is not mine),None) if mine is not None else None
            if not mine or not opp: continue
            pf,pa=safe_float(mine.get("score")),safe_float(opp.get("score"))
            won=mine.get("winner")
            if won is None and not pd.isna(pf) and not pd.isna(pa): won=pf>pa
            rows.append({"event_id":str(event.get("id","")),"date":event.get("date",""),"opponent":(opp.get("team") or {}).get("abbreviation","") or (opp.get("team") or {}).get("shortDisplayName",""),"opponent_id":str((opp.get("team") or {}).get("id","")),"home_away":mine.get("homeAway",""),"result":"W" if won else "L","points_for":pf,"points_against":pa,"source":"ESPN team schedule"})
    df=pd.DataFrame(rows)
    if not df.empty:
        df["_d"]=pd.to_datetime(df["date"],errors="coerce",utc=True)
        df=df.sort_values("_d",ascending=False).drop_duplicates("event_id").drop(columns="_d")
        return df,None
    return pd.DataFrame(),(" | ".join(dict.fromkeys(errors)) if errors else "ESPN team schedule returned no completed games for the requested season.")

@st.cache_data(ttl=180, show_spinner=False)
def fetch_event_summary_payload_v23(league,event_id):
    cfg=LEAGUES[league]
    url=f"https://site.api.espn.com/apis/site/v2/sports/{cfg['sport']}/{cfg['league']}/summary"
    try: return request_json(url,{"event":event_id},timeout=8),None
    except Exception as e: return {},f"ESPN game summary failed: {e}"

def _collect_injury_objects_v23(obj, out):
    if isinstance(obj,dict):
        athlete=obj.get("athlete") or obj.get("player")
        status=obj.get("status") or obj.get("gameStatus")
        # Only accept objects that look like an injury/availability record.
        injuryish=any(k in obj for k in ["injury","injuries","details","shortComment","longComment","returnDate"])
        if isinstance(athlete,dict) and (status is not None or injuryish):
            if isinstance(status,dict):
                status=status.get("name") or status.get("description") or status.get("type")
            details=obj.get("details") or obj.get("injury") or {}
            if not isinstance(details,dict): details={"type":str(details)}
            team=athlete.get("team") or obj.get("team") or {}
            out.append({
                "Player":athlete.get("displayName") or athlete.get("fullName") or athlete.get("name") or "",
                "Status":status or "",
                "Injury":details.get("type") or details.get("description") or "",
                "Detail":obj.get("shortComment") or obj.get("detail") or details.get("detail") or "",
                "LongComment":obj.get("longComment") or "",
                "ReturnDate":details.get("returnDate") or obj.get("returnDate") or "",
                "AthleteID":str(athlete.get("id","")),
                "TeamID":str(team.get("id","") if isinstance(team,dict) else ""),
                "Source":"ESPN game summary",
            })
        for v in obj.values(): _collect_injury_objects_v23(v,out)
    elif isinstance(obj,list):
        for v in obj: _collect_injury_objects_v23(v,out)


@st.cache_data(ttl=180, show_spinner=False)
def fetch_event_injuries_v23(league,event_id,team_id=""):
    payload,err=fetch_event_summary_payload_v23(league,event_id)
    if err: return pd.DataFrame(),err
    rows=[]; _collect_injury_objects_v23(payload,rows)
    df=pd.DataFrame(rows)
    if df.empty: return pd.DataFrame(),None
    if team_id and "TeamID" in df.columns and df["TeamID"].astype(str).str.len().gt(0).any():
        filt=df[df["TeamID"].astype(str)==str(team_id)]
        if not filt.empty: df=filt
    if "Player" in df.columns:
        df=df[df["Player"].astype(str).str.strip().ne("")]
        df=df.drop_duplicates(subset=[c for c in ["Player","Status","Injury","Detail"] if c in df.columns])
    return df,None


@st.cache_data(ttl=180, show_spinner=False)
def fetch_injuries(league,team_id,event_id=""):
    """Fast verified injury lookup with fallback calls only when needed."""
    cfg=LEAGUES[league]; team_id=str(team_id); messages=[]
    team_url=f"https://site.api.espn.com/apis/site/v2/sports/{cfg['sport']}/{cfg['league']}/teams/{team_id}/injuries"
    try:
        payload=request_json(team_url,timeout=7); frames=[]
        direct=payload.get("injuries") if isinstance(payload,dict) else None
        if isinstance(direct,list):
            d=_injury_rows_from_items(direct)
            if not d.empty: d["Source"]="ESPN team injury feed"; frames.append(d)
        found=[]; _walk_injuries(payload,found); d=pd.DataFrame(found)
        if not d.empty: d["Source"]="ESPN team injury feed"; frames.append(d)
        if frames:
            out=pd.concat(frames,ignore_index=True,sort=False)
            if "Player" in out.columns:
                out["_key"]=out["Player"].map(_name_key)
                out["_score"]=(out.get("Status",pd.Series("",index=out.index)).astype(str).str.len()>0).astype(int)*2 + (out.get("Detail",pd.Series("",index=out.index)).astype(str).str.len()>0).astype(int)
                out=out.sort_values("_score",ascending=False).drop_duplicates("_key").drop(columns=["_key","_score"],errors="ignore")
            return out,None
        return pd.DataFrame(),None
    except Exception as e: messages.append(f"team feed: {e}")
    if event_id:
        d,e=fetch_event_injuries_v23(league,event_id,team_id)
        if e: messages.append(e)
        else: return d,None
    league_url=f"https://site.api.espn.com/apis/site/v2/sports/{cfg['sport']}/{cfg['league']}/injuries"
    try:
        payload=request_json(league_url,timeout=7); groups=payload.get("injuries") or []
        matched=next((g for g in groups if isinstance(g,dict) and str(g.get("id",""))==team_id),None)
        if matched is not None:
            d=_injury_rows_from_items(matched.get("injuries") or [])
            if not d.empty: d["Source"]="ESPN league injury feed"
            return d,None
    except Exception as e: messages.append(f"league feed: {e}")
    return pd.DataFrame(),"Injury/availability could not be verified from the available ESPN feeds. " + " | ".join(messages)

def _active_season_month_v36(league,ref_date):
    d=pd.Timestamp(ref_date).date() if not isinstance(ref_date,date) else ref_date
    m=d.month
    if league=="NBA": return m in {10,11,12,1,2,3,4,5,6}
    if league=="WNBA": return m in {5,6,7,8,9,10}
    if league=="NFL": return m in {9,10,11,12,1,2}
    return True

def _latest_age_days_v36(df,ref_date=None,date_col="date"):
    if df is None or df.empty or date_col not in df.columns: return np.inf
    vals=pd.to_datetime(df[date_col],errors="coerce",utc=True).dropna()
    if vals.empty: return np.inf
    ref=pd.Timestamp(ref_date or date.today())
    ref=ref.tz_localize("UTC") if ref.tzinfo is None else ref.tz_convert("UTC")
    today_ref=pd.Timestamp(date.today()).tz_localize("UTC")
    if ref>today_ref: ref=today_ref
    return max(0.0,(ref-vals.max()).total_seconds()/86400.0)

@st.cache_data(show_spinner=False)
def local_team_history_v36(league,abbr,selected_date):
    abbr=str(abbr or "").upper(); cutoff=pd.Timestamp(selected_date or date.today())
    if league=="NFL":
        g=load_nfl_games_seed_v33(); rows=[]
        if g is None or g.empty: return pd.DataFrame()
        for _,r in g.iterrows():
            home=str(r.get("home_team","")).upper(); away=str(r.get("away_team","")).upper()
            if abbr not in {home,away}: continue
            gd=pd.to_datetime(r.get("gameday"),errors="coerce")
            if pd.isna(gd) or gd>cutoff: continue
            hs=safe_float(r.get("home_score")); ass=safe_float(r.get("away_score"))
            if pd.isna(hs) or pd.isna(ass): continue
            is_home=abbr==home; pf=hs if is_home else ass; pa=ass if is_home else hs
            rows.append({"event_id":str(r.get("game_id","")),"date":gd.isoformat(),"opponent":away if is_home else home,"home_away":"home" if is_home else "away","result":"W" if pf>pa else "L","points_for":pf,"points_against":pa,"source":"Bundled nflverse schedule"})
        out=pd.DataFrame(rows)
    else:
        seed=load_basketball_seed_v33()
        if seed is None or seed.empty: return pd.DataFrame()
        x=seed[(seed.get("LEAGUE","").astype(str).str.upper()==str(league).upper()) & (seed.get("TEAM","").astype(str).str.upper()==abbr)].copy()
        if x.empty: return pd.DataFrame()
        x["_d"]=pd.to_datetime(x.get("DATE"),errors="coerce")
        x=x[x["_d"].notna() & (x["_d"]<=cutoff)]
        if x.empty: return pd.DataFrame()
        if "GAME_ID" in x.columns and x["GAME_ID"].astype(str).str.strip().ne("").any(): x=x.sort_values("_d",ascending=False).drop_duplicates("GAME_ID")
        else: x=x.sort_values("_d",ascending=False).drop_duplicates([c for c in ["DATE","TEAM","OPP"] if c in x.columns])
        out=pd.DataFrame({"event_id":x.get("GAME_ID",pd.Series("",index=x.index)).astype(str).values,"date":x["DATE"].astype(str).values,"opponent":x.get("OPP",pd.Series("",index=x.index)).astype(str).values,"home_away":x.get("HOME_AWAY",pd.Series("",index=x.index)).astype(str).str.lower().values,"result":x.get("RESULT",pd.Series("",index=x.index)).astype(str).values,"points_for":np.nan,"points_against":np.nan,"source":"Bundled verified player-game logs"})
    if out is None or out.empty: return pd.DataFrame()
    out["_d"]=pd.to_datetime(out["date"],errors="coerce",utc=True)
    return out.sort_values("_d",ascending=False).drop(columns="_d",errors="ignore").reset_index(drop=True)

def fast_team_history_v36(league,team_id,abbr,season,selected_date):
    local=local_team_history_v36(league,abbr,selected_date)
    refresh=_active_season_month_v36(league,selected_date) and _latest_age_days_v36(local,selected_date)>10
    live=pd.DataFrame(); err=None
    if refresh: live,err=fetch_team_recent(league,team_id,season)
    if live is None or live.empty: return local,err
    if local is None or local.empty: return live,err
    out=pd.concat([live,local],ignore_index=True,sort=False)
    if "date" in out.columns: out["_d"]=pd.to_datetime(out["date"],errors="coerce",utc=True)
    if "event_id" in out.columns and out["event_id"].astype(str).str.strip().ne("").any():
        nonblank=out["event_id"].astype(str).str.strip().ne(""); out=pd.concat([out[nonblank].drop_duplicates("event_id",keep="first"),out[~nonblank]],ignore_index=True,sort=False)
    subset=[c for c in ["date","opponent"] if c in out.columns]
    if subset: out=out.drop_duplicates(subset=subset,keep="first")
    if "_d" in out.columns: out=out.sort_values("_d",ascending=False).drop(columns="_d")
    return out.reset_index(drop=True),err

def build_matchup_context(game,league):
    season=league_season_year(league,st.session_state.get("selected_date",date.today()))
    selected_date=st.session_state.get("selected_date",date.today())
    context={}
    for side in ["home","away"]:
        team_id=game[f"{side}_id"]; abbr=game[f"{side}_abbr"]; opp=game["away_abbr"] if side=="home" else game["home_abbr"]
        history,herr=fast_team_history_v36(league,team_id,abbr,season,selected_date)
        injuries,ierr=fetch_injuries(league,team_id,game.get("event_id",""))
        h2h,h2h_summary=head_to_head_summary(history,opp,10)
        context[side]={"abbr":abbr,"history":history,"history_error":herr,"recent5":split_summary(history,n=5),"recent10":split_summary(history,n=10),"season":split_summary(history),"home_split":split_summary(history,where="home"),"away_split":split_summary(history,where="away"),"h2h":h2h,"h2h_summary":h2h_summary,"rest":rest_days_before_game(history,game.get("date")),"injury_df":injuries,"injuries":injury_counts(injuries),"injury_error":ierr,"injury_verified":ierr is None}
    return context

def _games_from_record_v23(record):
    m=re.match(r"\s*(\d+)\s*[-–]\s*(\d+)",str(record or ""))
    return int(m.group(1))+int(m.group(2)) if m else 0


def _team_metric_display_v23(key,value,rank_text,games):
    if pd.isna(value): return "—"
    percent_keys={"fieldGoalPct","threePointPct","threePointFieldGoalPct","freeThrowPct"}
    volume_keys={"freeThrowsAttempted","threePointFieldGoalsAttempted","fieldGoalsAttempted"}
    if key in percent_keys:
        val=f"{float(value):.1f}%"
    elif key in volume_keys and games>0 and float(value)>max(80,games*5):
        val=f"{float(value)/games:.1f}/game"
    else:
        val=f"{float(value):.1f}"
    return val+(f" • {rank_text}" if rank_text else "")


def scoreboard_team_strengths_weaknesses(game,league,side):
    """Clear own-team statistical profile. Does not mislabel offensive weaknesses as opponent defensive weaknesses."""
    stats=game.get(f"{side}_stats") or {}; records=game.get(f"{side}_records") or {}; abbr=game.get(f"{side}_abbr","")
    opp_side="away" if side=="home" else "home"; opp_stats=game.get(f"{opp_side}_stats") or {}; opp_abbr=game.get(f"{opp_side}_abbr","")
    tiers=_league_rank_tiers(league); games=_games_from_record_v23(records.get("overall",""))
    labels={
        "avgPoints":"scoring","fieldGoalPct":"field-goal shooting","threePointPct":"three-point shooting",
        "threePointFieldGoalPct":"three-point shooting","avgRebounds":"rebounding","avgAssists":"playmaking/assists",
        "freeThrowPct":"free-throw shooting","freeThrowsAttempted":"free-throw attempts",
        "threePointFieldGoalsAttempted":"three-point attempts","fieldGoalsAttempted":"shot attempts",
    }
    keys=["avgPoints","fieldGoalPct","threePointPct","avgRebounds","avgAssists","freeThrowPct","freeThrowsAttempted","threePointFieldGoalsAttempted","fieldGoalsAttempted"]
    if "threePointPct" not in stats and "threePointFieldGoalPct" in stats: keys[2]="threePointFieldGoalPct"
    strengths=[]; weaknesses=[]; neutral=[]
    for key in keys:
        row=_team_stat_row(stats,key); value=row["value"]; rank=row["rank"]
        if pd.isna(value): continue
        label=labels[key]; display=_team_metric_display_v23(key,value,row["rank_text"],games)
        if pd.isna(rank):
            opp=_team_stat_row(opp_stats,key)
            if not pd.isna(opp["value"]):
                diff=value-opp["value"]; threshold={"avgPoints":2.0,"avgRebounds":1.5,"avgAssists":1.0}.get(key,1.0)
                if diff>=threshold: strengths.append(f"{label.title()}: {abbr} {display} vs {opp_abbr} {_team_metric_display_v23(key,opp['value'],opp['rank_text'],_games_from_record_v23(game.get(f'{opp_side}_records',{}).get('overall','')))}.")
                elif diff<=-threshold: weaknesses.append(f"{label.title()}: {abbr} {display} vs {opp_abbr} {_team_metric_display_v23(key,opp['value'],opp['rank_text'],_games_from_record_v23(game.get(f'{opp_side}_records',{}).get('overall','')))}.")
            continue
        if rank<=tiers["elite"]: strengths.append(f"{label.title()}: {display}.")
        elif rank>=tiers["weak"]: weaknesses.append(f"{label.title()}: {display}.")
        else: neutral.append(f"{label.title()}: {display}.")
    venue_key="home" if side=="home" else "road"; venue_label="at home" if side=="home" else "on the road"; rec=records.get(venue_key); pct=_record_pct(rec) if rec else np.nan
    if not pd.isna(pct):
        item=f"{abbr} is {rec} {venue_label} ({pct*100:.1f}% win rate)."
        (strengths if pct>=.60 else weaknesses if pct<=.40 else neutral).insert(0,item)
    return {"team":abbr,"strengths":list(dict.fromkeys(strengths)),"weaknesses":list(dict.fromkeys(weaknesses)),"neutral":list(dict.fromkeys(neutral))}


def opponent_style_exploit_report(game,league,target_side,player_position="",player_log=None):
    """Separate a team's OWN weaknesses from verified opponent-defense matchup evidence."""
    target_abbr=game[f"{target_side}_abbr"]; attacker_side="away" if target_side=="home" else "home"; attacker_abbr=game[f"{attacker_side}_abbr"]
    own=scoreboard_team_strengths_weaknesses(game,league,target_side)
    exact=(basketball_style_matchup_notes(game,league,target_abbr,player_position=player_position,player_log=player_log) if league in {"NBA","WNBA"} else nfl_style_matchup_notes(game,target_side,player_position=player_position))
    verified_help=[str(x) for x in exact.get("positives",[]) if "unavailable" not in str(x).lower()]
    verified_hurt=[str(x) for x in exact.get("negatives",[]) if "unavailable" not in str(x).lower()]
    prop=[]
    if verified_help:
        prop.append(f"Verified matchup evidence identifies a possible {attacker_abbr} prop-friendly area; use the player-specific report to see whether the player's role actually matches it.")
    else:
        prop.append("No defense-specific player-prop weakness was verified from the current source set. The app will not turn an offensive team weakness into a fake defensive matchup edge.")
    return {
        "target":target_abbr,"attacker":attacker_abbr,
        "weaknesses":own["weaknesses"],"strengths":own["strengths"],
        "exploit":verified_help or ["No verified defense-specific weakness was returned by the current source set."],
        "caution":verified_hurt + [f"Team strength: {x}" for x in own["strengths"][:4]],
        "prop_implications":prop,
        "role":exact.get("role","team"),
    }


def v20_team_profile_df(game):
    rows=[]
    for side in ["away","home"]:
        stats=game.get(f"{side}_stats") or {}; records=game.get(f"{side}_records") or {}; abbr=game[f"{side}_abbr"]
        rows.append({
            "Team":abbr,"Overall record":records.get("overall","—"),
            "Road/Home record":records.get("road","—") if side=="away" else records.get("home","—"),
            "Points/Game":_scoreboard_stat_text(stats,"avgPoints"),"Rebounds/Game":_scoreboard_stat_text(stats,"avgRebounds"),
            "Assists/Game":_scoreboard_stat_text(stats,"avgAssists"),"Shooting %":_scoreboard_stat_text(stats,"fieldGoalPct"),
            "3PT %":_scoreboard_stat_text(stats,"threePointPct" if "threePointPct" in stats else "threePointFieldGoalPct"),
        })
    return pd.DataFrame(rows)


# ---- Player multi-source / multi-season research ----
def _normalize_player_log_v23(df,season_label,source):
    if df is None or df.empty: return pd.DataFrame()
    out=df.copy(); out["SEASON"]=str(season_label); out["DATA_SOURCE"]=source
    if "DATE" in out.columns:
        out["_date_sort"]=pd.to_datetime(out["DATE"],errors="coerce",utc=True)
    elif "GAME_DATE" in out.columns:
        out["DATE"]=out["GAME_DATE"]; out["_date_sort"]=pd.to_datetime(out["DATE"],errors="coerce",utc=True)
    else: out["_date_sort"]=pd.NaT
    return out


def _exact_opponent_filter_v23(log,opponent):
    if log is None or log.empty or not opponent: return pd.DataFrame()
    target=normalize_team_code(opponent)
    if "OPP" in log.columns:
        mask=log["OPP"].map(normalize_team_code)==target
        return log[mask].copy()
    if "MATCHUP" in log.columns:
        def hit(x):
            toks=re.findall(r"[A-Z]{2,4}",str(x).upper())
            return target in {normalize_team_code(t) for t in toks}
        return log[log["MATCHUP"].map(hit)].copy()
    return pd.DataFrame()


def refresh_player_history_database_v32(row,league,selected_date=None):
    """Fetch current + previous two seasons for one roster player and persist them."""
    selected_date=selected_date or date.today()
    pname=str(row.get("player","")); pteam=str(row.get("team_abbr","")); athlete_id=str(row.get("athlete_id",""))
    season=league_season_year(league,selected_date)
    frames=[]; notes=[]
    if league in {"NBA","WNBA"}:
        if league=="NBA":
            base=int(nba_season_string(selected_date).split("-")[0])
            years=[f"{y}-{str(y+1)[-2:]}" for y in range(base,base-3,-1)]
        else:
            years=[str(y) for y in range(int(season),int(season)-3,-1)]
        for y in years:
            try:
                if league=="NBA": d,_,e=fetch_nba_player_by_name(pname,y)
                else: d,_,e=fetch_wnba_player_by_name(pname,int(y))
                if not d.empty: frames.append(_normalize_player_log_v23(d,y,f"{league} official stats"))
                elif e: notes.append(str(e))
            except Exception as e: notes.append(str(e))
        if athlete_id and not frames:
            d,e=fetch_espn_player_gamelog_multi(league,athlete_id,int(season),3)
            if not d.empty: frames.append(_normalize_player_log_v23(d,"multi-season","ESPN athlete game log"))
            elif e: notes.append(str(e))
    else:
        local=local_nfl_player_log_v33(pname,pteam,row.get("player_id",""))
        if not local.empty:
            frames.append(_normalize_player_log_v23(local,"2024-current","nflverse local seed"))
        if athlete_id:
            d,e=fetch_espn_player_gamelog_multi("NFL",athlete_id,int(season),3)
            if not d.empty: frames.append(_normalize_player_log_v23(d,"multi-season","ESPN athlete game log"))
            elif e: notes.append(str(e))
    log=pd.concat(frames,ignore_index=True,sort=False) if frames else pd.DataFrame()
    if not log.empty:
        log=_prepare_player_log_v35(log,league)
        if not log.empty:
            log=log.sort_values("DATE",ascending=False,kind="stable")
            added=upsert_player_log_database(log,league=league,player=pname,team=None,source="admin refresh verified rows")
            return len(log),added," | ".join(notes[-3:]) if notes else None
    return 0,0," | ".join(notes[-3:]) if notes else "No game log returned"

def player_history_needs_live_refresh_v36(log,league,selected_date):
    if log is None or log.empty: return True
    if not _active_season_month_v36(league,selected_date): return False
    if "DATE" not in log.columns: return True
    tmp=log.rename(columns={"DATE":"date"}).copy()
    return _latest_age_days_v36(tmp,selected_date)>10

def fetch_player_research_v23(row,game,league):
    """Local-first verified player research with one current-season refresh only when stale."""
    pname=str(row.get("player","")).strip(); pteam=str(row.get("team_abbr","")).strip().upper(); athlete_id=_canonical_id_v35(row.get("athlete_id",""))
    selected_date=st.session_state.get("selected_date",date.today()); season=league_season_year(league,selected_date)
    opponent=(game["home_abbr"] if pteam==game["away_abbr"] else game["away_abbr"] if pteam==game["home_abbr"] else "")
    opponent_team_id=(game["home_id"] if pteam==game["away_abbr"] else game["away_id"] if pteam==game["home_abbr"] else "")
    notes=[]; player={"full_name":pname}; player_id=""
    try:
        saved=load_player_log_database(league,pname)
        if not saved.empty and "PLAYER" in saved.columns: saved=saved[saved["PLAYER"].map(_name_key)==_name_key(pname)].copy()
        saved=_prepare_player_log_v35(saved,league)
    except Exception as e:
        notes.append(f"Local database read: {e}"); saved=pd.DataFrame()
    if not saved.empty:
        for c in ["PLAYER_ID","player_id","gsis_id"]:
            if c in saved.columns:
                ids=saved[c].dropna().map(_canonical_id_v35); ids=ids[ids.ne("")]
                if not ids.empty: player_id=str(ids.iloc[0]); break
    frames=[]; needs_live=player_history_needs_live_refresh_v36(saved,league,selected_date)
    if needs_live and league in {"NBA","WNBA"} and _active_season_month_v36(league,selected_date):
        current_label=nba_season_string(selected_date) if league=="NBA" else str(int(season))
        try:
            if league=="NBA": d,p,e=fetch_nba_player_by_name(pname,current_label)
            else: d,p,e=fetch_wnba_player_by_name(pname,int(current_label))
            if p: player=p; player_id=_canonical_id_v35(p.get("id","")) or player_id
            if d is not None and not d.empty:
                usable=_prepare_player_log_v35(_normalize_player_log_v23(d,current_label,f"{league} official current-season stats"),league)
                if not usable.empty: frames.append(usable)
            elif e: notes.append(f"{league} live refresh: {e}")
        except Exception as e: notes.append(f"{league} live refresh: {e}")
        if not frames and athlete_id:
            try:
                d,e=fetch_espn_player_gamelog(league,athlete_id,int(season))
                if d is not None and not d.empty:
                    nd=_normalize_player_log_v23(d,str(season),"ESPN current-season athlete game log")
                    nd,filled,attempted=enrich_schedule_only_rows_v35(nd,league,athlete_id=athlete_id,player_name=pname,max_events=6)
                    if attempted: notes.append(f"Verified {filled}/{attempted} newest ESPN event row(s) against exact boxscores.")
                    usable=_prepare_player_log_v35(nd,league)
                    if not usable.empty: frames.append(usable)
                elif e: notes.append(f"ESPN current-season fallback: {e}")
            except Exception as e: notes.append(f"ESPN current-season fallback: {e}")
    elif needs_live and league=="NFL" and athlete_id:
        try:
            d,e=fetch_espn_player_gamelog("NFL",athlete_id,int(season))
            if d is not None and not d.empty:
                nd=_normalize_player_log_v23(d,str(season),"ESPN current-season athlete game log")
                nd,filled,attempted=enrich_schedule_only_rows_v35(nd,league,athlete_id=athlete_id,player_name=pname,max_events=4)
                if attempted: notes.append(f"Verified {filled}/{attempted} newest ESPN NFL event row(s) against exact boxscores.")
                usable=_prepare_player_log_v35(nd,league)
                if not usable.empty: frames.append(usable)
            elif e: notes.append(str(e))
        except Exception as e: notes.append(str(e))
    if not needs_live: notes.append("Verified local history is fresh; skipped unnecessary live stat requests for faster loading.")
    live=pd.concat(frames,ignore_index=True,sort=False) if frames else pd.DataFrame(); live=_prepare_player_log_v35(live,league)
    log=merge_live_and_saved_player_logs(live,saved,league)
    if not live.empty:
        try:
            added=upsert_player_log_database(live,league=league,player=pname,team=None,source="automatic verified current-season refresh")
            if added: notes.append(f"Saved {added} newly verified current-season game row(s).")
        except Exception as e: notes.append(f"Local database save: {e}")
    if log.empty: notes.append("No verified stat-bearing player rows were available.")
    latest=str(log.iloc[0].get("DATE","") or "") if not log.empty and "DATE" in log.columns else ""
    matchup=_exact_opponent_filter_v23(log,opponent)
    return {"log":log,"matchup":matchup,"player":player,"player_id":player_id,"opponent":opponent,"opponent_team_id":opponent_team_id,"notes":notes,"verified_rows":len(log),"latest_verified_date":latest}

def load_player_v11(row,game,league):
    pname=str(row.get("player","")).strip(); pteam=str(row.get("team_abbr","")).strip(); athlete_id=str(row.get("athlete_id","")).strip()
    with st.spinner(f"Loading {pname} from verified local history first; checking live sources only if needed..."):
        result=fetch_player_research_v23(row,game,league)
    log=result["log"]; matchup=result["matchup"]; player=result["player"]
    if log.empty:
        st.session_state["research_error"]=("Automated stat feeds did not return a professional game log after official + ESPN fallbacks. "
            "The report will still provide roster/availability context and direct verification links; stats are never invented.")
    else: st.session_state["research_error"]=""
    opponent=result["opponent"]
    today_side="home" if pteam==game.get("home_abbr") else "away" if pteam==game.get("away_abbr") else ""
    st.session_state["research_log"]=log; st.session_state["research_matchup_log"]=matchup
    st.session_state["research_name"]=player.get("full_name",pname) if isinstance(player,dict) else pname
    st.session_state["research_league"]=league; st.session_state["research_team"]=pteam; st.session_state["research_opponent"]=opponent
    st.session_state["research_today_side"]=today_side; st.session_state["research_team_id"]=str(row.get("team_id",""))
    st.session_state["research_opponent_team_id"]=str(result["opponent_team_id"]); st.session_state["research_player_id"]=str(result["player_id"])
    st.session_state["research_athlete_id"]=athlete_id; st.session_state["research_source_notes_v23"]=result["notes"]
    st.session_state["research_verified_rows_v35"]=int(result.get("verified_rows",len(log)))
    st.session_state["research_latest_verified_v35"]=str(result.get("latest_verified_date","") or "")


@st.cache_data(ttl=1800,show_spinner=False)
def league_roster_index_v23(league):
    if league=="NFL":
        raw=load_nfl_rosters_seed_v33()
        if not raw.empty:
            season=league_season_year("NFL",st.session_state.get("selected_date",date.today()))
            x=raw[pd.to_numeric(raw.get("season"),errors="coerce")==int(season)].copy()
            if not x.empty:
                if "game_type" in x.columns:
                    reg=x[x["game_type"].astype(str).str.upper()=="REG"]
                    if not reg.empty: x=reg
                wk=pd.to_numeric(x.get("week"),errors="coerce")
                if wk.notna().any(): x=x[wk==wk.max()]
                out=pd.DataFrame({
                    "athlete_id":x.get("espn_id",pd.Series("",index=x.index)).fillna("").astype(str).replace("nan",""),
                    "player":x.get("full_name",pd.Series("",index=x.index)).astype(str),
                    "position":x.get("position",pd.Series("",index=x.index)).astype(str),
                    "jersey":x.get("jersey_number",pd.Series("",index=x.index)).fillna("").astype(str).replace("nan",""),
                    "roster_status":x.get("status",pd.Series("",index=x.index)).astype(str),
                    "team_id":x.get("team",pd.Series("",index=x.index)).astype(str),
                    "team_abbr":x.get("team",pd.Series("",index=x.index)).astype(str),
                    "team_name":x.get("team",pd.Series("",index=x.index)).astype(str),
                })
                return out[out["player"].str.strip().ne("")].drop_duplicates(["team_abbr","player"]),None
    teams,err=fetch_all_teams(league)
    if err or teams.empty: return pd.DataFrame(),err or "No league teams returned."
    frames=[]; errors=[]
    for _,t in teams.iterrows():
        r,e=fetch_roster(league,str(t.get("team_id","")))
        if e: errors.append(f"{t.get('abbr','')}: {e}")
        if not r.empty:
            x=r.copy(); x["team_id"]=str(t.get("team_id","")); x["team_abbr"]=str(t.get("abbr","")); x["team_name"]=str(t.get("team_name","")); frames.append(x)
    if not frames: return pd.DataFrame()," | ".join(errors[:5])
    return pd.concat(frames,ignore_index=True).drop_duplicates(["athlete_id","player"]),(" | ".join(errors[:5]) if errors else None)


def find_player_anywhere_v23(league,name):
    idx,err=league_roster_index_v23(league)
    target=_name_key(name)
    if not idx.empty:
        work=idx.copy(); work["_key"]=work["player"].map(_name_key)
        exact=work[work["_key"]==target]
        if not exact.empty: return exact.iloc[0],err
        close=get_close_matches(target,work["_key"].tolist(),n=1,cutoff=.58)
        if close:
            return work[work["_key"]==close[0]].iloc[0],err
    # Basketball official directory fallback for a player not currently on a roster.
    if league=="NBA":
        p,e=_resolve_nba_player(name,nba_season_string(st.session_state.get("selected_date",date.today())))
        if p:
            return pd.Series({"player":p.get("full_name",name),"athlete_id":"","team_id":"","team_abbr":"","position":""}),err or e
    if league=="WNBA":
        sy=league_season_year("WNBA",st.session_state.get("selected_date",date.today()))
        d,e=fetch_wnba_player_directory(sy); m=match_wnba_player(name,d)
        if m is not None:
            return pd.Series({"player":m.get("DISPLAY_FIRST_LAST",name),"athlete_id":"","team_id":"","team_abbr":"","position":""}),err or e
    return None,err or f"Could not resolve '{name}' in current league rosters/directories."


def player_research_facts_v23(log,league):
    if log is None or log.empty: return {"strengths":[],"weaknesses":[],"facts":[]}
    if league in {"NBA","WNBA"}: return basketball_player_tendency_profile(log.head(15))
    # NFL: concise role/workload facts from the local nflverse/ESPN log.
    strengths=[]; weaknesses=[]; facts=[]
    specs=[
        (["TARGETS","TGT"],"targets",7,3),(["RECEPTIONS","REC"],"receptions",5,2),
        (["RUSHING ATT","RUSH_ATT","CAR"],"carries",15,6),(["PASSING ATT","PASS_ATT"],"pass attempts",28,15),
        (["SNAP PCT"],"offensive snap share",.75,.45),
    ]
    for candidates,label,hi,lo in specs:
        c=next((z for z in candidates if z in log.columns),None)
        if not c: continue
        v=pd.to_numeric(log[c],errors="coerce").head(10).mean()
        if pd.isna(v): continue
        disp=f"{v*100:.0f}%" if "share" in label else f"{v:.1f}"
        facts.append(f"{disp} {label} recently")
        if v>=hi: strengths.append(f"High recent {label} ({disp})")
        elif v<=lo: weaknesses.append(f"Low recent {label} ({disp})")
    for c,label in [("PASSING EPA","passing EPA"),("RUSHING EPA","rushing EPA"),("RECEIVING EPA","receiving EPA")]:
        if c in log.columns:
            v=pd.to_numeric(log[c],errors="coerce").head(10).mean()
            if not pd.isna(v):
                facts.append(f"{v:+.2f} {label}/game recently")
                (strengths if v>0 else weaknesses).append(f"{label.title()} {'positive' if v>0 else 'negative'} recently")
    if not strengths:
        # Use factual workload/production rather than leaving the section blank.
        for c,label in [("SNAP PCT","snap share"),("TARGETS","targets"),("RUSHING ATT","carries"),("PASSING ATT","pass attempts")]:
            if c in log.columns:
                v=pd.to_numeric(log[c],errors="coerce").dropna().head(10).mean()
                if not pd.isna(v):
                    disp=f"{v*100:.0f}%" if c=="SNAP PCT" else f"{v:.1f}"
                    strengths.append(f"clearest recent role signal: {disp} {label}")
                    break
        if not strengths:
            strengths.append("no verified recent strength could be calculated from the available rows")
    if not weaknesses:
        snap=pd.to_numeric(log.get("SNAP PCT",pd.Series(dtype=float)),errors="coerce").dropna().head(10)
        if not snap.empty and snap.mean()<.50:
            weaknesses.append(f"limited offensive snap share ({snap.mean()*100:.0f}% recently)")
        else:
            weaknesses.append("no clear statistical weakness crossed the current evidence threshold")
    return {"strengths":strengths,"weaknesses":weaknesses,"facts":facts}


def v20_render_player_style(game,league,log,player_name,position,opponent):
    fit=player_vs_team_fit(game,league,log,player_name,position,opponent,prop_stat=None)
    own=player_research_facts_v23(log,league)
    left,right=st.columns(2)
    with left:
        st.markdown("### ✅ Research-backed strengths")
        items=own.get("strengths",[]) or fit.get("player_strengths",[])
        if items:
            for x in items[:4]: st.write("• "+str(x))
        else: st.caption("No strength label is shown unless the returned statistics support it.")
    with right:
        st.markdown("### ⚠️ Research-backed weak points")
        items=own.get("weaknesses",[]) or fit.get("player_weaknesses",[])
        if items:
            for x in items[:4]: st.write("• "+str(x))
        else: st.caption("No weakness label is shown unless the returned statistics support it.")
    st.markdown("### 🎯 Matchup-specific evidence")
    shown=0
    for x in fit.get("helps",[])[:2]:
        if "no clear" not in str(x).lower(): st.write("✅ "+str(x)); shown+=1
    for x in fit.get("hurts",[])[:2]:
        if "no clear" not in str(x).lower(): st.write("⚠️ "+str(x)); shown+=1
    if not shown: st.caption("No verified matchup-specific edge was found. The app does not manufacture one.")
    return fit


# ---- Cross-market price comparison ----
def american_profit_per_100_v23(odds):
    o=safe_float(odds)
    if pd.isna(o) or o==0: return np.nan
    return 10000.0/abs(o) if o<0 else float(o)


@st.cache_data(ttl=120,show_spinner=False)
def fetch_espn_book_prices_v23(league,event_id):
    cfg=LEAGUES[league]
    url=f"https://sports.core.api.espn.com/v2/sports/{cfg['sport']}/leagues/{cfg['league']}/events/{event_id}/competitions/{event_id}/odds"
    try: payload=request_json(url,{"limit":100},timeout=20)
    except Exception as e: return [],f"Sportsbook odds request failed: {e}"
    rows=[]
    for item in payload.get("items",[]) or []:
        provider=(item.get("provider") or {}).get("name") or "Sportsbook"
        home=item.get("homeTeamOdds") or {}; away=item.get("awayTeamOdds") or {}
        for side,obj in [("home",home),("away",away)]:
            ml=safe_float(obj.get("moneyLine")); raw=american_to_implied(ml)
            rows.append({"provider":provider,"side":side,"american":ml,"raw_implied":raw,"profit_100":american_profit_per_100_v23(ml)})
    return [r for r in rows if not pd.isna(r["american"])],None


def kalshi_profit_per_100_v23(price):
    p=safe_float(price)
    if pd.isna(p) or p<=0 or p>=1: return np.nan
    contracts=max(1,int(100//p)); cost=contracts*p; fee=kalshi_general_taker_fee(contracts,p,1.0)
    if cost+fee>100:
        while contracts>1 and contracts*p+kalshi_general_taker_fee(contracts,p,1.0)>100: contracts-=1
        cost=contracts*p; fee=kalshi_general_taker_fee(contracts,p,1.0)
    return contracts-(cost+fee)


def cross_market_winner_table_v23(game,league,model_rows=None):
    books,berr=fetch_espn_book_prices_v23(league,game.get("event_id","")); kal,kerr=selected_game_kalshi_prices_v22(game,league)
    model_map={}
    for r in model_rows or []:
        if r.get("Market Type")=="Winner": model_map[str(r.get("Side Abbr",""))]=safe_float(r.get("Model probability"))
    if not model_map:
        ph,e,_=current_league_model_probability(game,league)
        if not pd.isna(ph): model_map={game["home_abbr"]:ph,game["away_abbr"]:1-ph}
    out=[]
    for r in books:
        abbr=game[f"{r['side']}_abbr"]; raw=r["raw_implied"]
        out.append({"Side":abbr,"Market/App":r["provider"],"Price / odds":f"{int(r['american']):+d}","Market implied probability":raw,"Model probability":model_map.get(abbr,np.nan),"Model edge":model_map.get(abbr,np.nan)-raw if abbr in model_map and not pd.isna(raw) else np.nan,"Potential profit on $100 if correct":r["profit_100"],"Source":"ESPN odds feed"})
    for r in kal:
        p=safe_float(r.get("Kalshi probability")); abbr=r.get("Abbr","")
        out.append({"Side":abbr,"Market/App":"Kalshi","Price / odds":f"{p*100:.1f}¢ YES" if not pd.isna(p) else "—","Market implied probability":p,"Model probability":model_map.get(abbr,np.nan),"Model edge":model_map.get(abbr,np.nan)-p if abbr in model_map and not pd.isna(p) else np.nan,"Potential profit on $100 if correct":kalshi_profit_per_100_v23(p),"Source":"Kalshi public market/orderbook API"})
    return pd.DataFrame(out)," | ".join(x for x in [berr,kerr] if x)


def official_injury_report_url(league):
    if league=="WNBA": return "https://www.wnba.com/wnba-injury-report"
    if league=="NBA": return "https://official.nba.com/nba-injury-report-2025-26-season/"
    return "https://www.nfl.com/injuries/"


def source_table_v23():
    return pd.DataFrame([
        ["NBA/WNBA player stats & game logs","NBA/WNBA official stats endpoints via nba_api wrapper","Primary automated basketball player history; multi-season fallback"],
        ["Current game/team data","ESPN public sports feeds","Schedule, roster, score, records, team stats, leaders, current game summary"],
        ["Injuries / availability","ESPN team + league + game-summary feeds; official league injury-report link","Cross-check availability; missing data is never treated as healthy"],
        ["NFL historical efficiency","nflverse schedules + weekly team stats","EPA, pass/rush efficiency, explosive plays, historical modeling"],
        ["Weather","Open-Meteo","Outdoor NFL game-day weather"],
        ["Sportsbook prices","ESPN odds feed (DraftKings is ESPN's current official sportsbook/odds provider)","Moneylines/spread/total when exposed"],
        ["Prediction-market prices","Kalshi public API + orderbook","Winner/spread/total contract prices, orderbook depth, fees"],
        ["Persistent player-game database","Local CSV database built from verified feeds/imports","Keeps collected logs available when a live endpoint temporarily fails; duplicates are merged"],
        ["Optional current corroboration","Tavily/Groq only when configured","Used only for current outside-model/news corroboration on an edge alert; core stats and probabilities do not depend on it"],
    ],columns=["Information","Real source","How it is used"])

# ============================================================
# V23B: SOURCE VISIBILITY / ADVANCED-STATS FALLBACKS
# ============================================================

def render_basketball_advanced_or_fallback(game,league,ctx):
    selected_date=st.session_state.get("selected_date",date.today())
    tries=[("current season",selected_date)]
    # In the offseason / feed outage, use the latest prior-season official number,
    # but label it as prior-season rather than pretending it is current.
    prior_date=selected_date-timedelta(days=365)
    tries.append(("prior season",prior_date))
    messages=[]
    for label,d in tries:
        adv,err=fetch_basketball_advanced(league,d)
        if err: messages.append(f"{label}: {err}")
        rows=[]
        if not adv.empty and "TEAM_ABBREVIATION" in adv.columns:
            for abbr in [game["away_abbr"],game["home_abbr"]]:
                row=adv[adv["TEAM_ABBREVIATION"].astype(str).str.upper()==abbr.upper()]
                if not row.empty:
                    r=row.iloc[0]
                    rows.append({
                        "Team":abbr,
                        "Offensive rating":safe_float(r.get("OFF_RATING")),
                        "Defensive rating":safe_float(r.get("DEF_RATING")),
                        "Net rating":safe_float(r.get("NET_RATING")),
                        "Pace":safe_float(r.get("PACE")),
                        "Source season":(nba_season_string(d) if league=="NBA" else str(d.year)),
                    })
        if len(rows)==2:
            if label=="current season": st.success("Official current-season NBA/WNBA advanced team stats loaded.")
            else: st.warning("Current-season advanced stats were unavailable. Showing official prior-season pace/ratings and labeling the season explicitly.")
            st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
            return rows,True
    st.warning("Official pace/ratings did not load from either current or prior season. Using verified scoring-result fallbacks; these are not possession-based pace/ratings.")
    fallback=basketball_verified_fallback_profile(game,ctx).copy()
    show=[c for c in ["Team","Recent points scored","Recent points allowed","Recent margin","Recent combined scoring"] if c in fallback.columns]
    if show: st.dataframe(fallback[show],use_container_width=True,hide_index=True)
    for note in basketball_fallback_takeaways(game,ctx): st.write("• "+str(note))
    if messages:
        with st.expander("Advanced-stat source messages"):
            for m in messages: st.write("• "+m)
    return [],False


def v20_render_injuries(game,ctx):
    league=st.session_state.get("league")
    st.markdown(f"[Open official {league} injury report]({official_injury_report_url(league)})")
    left,right=st.columns(2)
    for side,col in [("away",left),("home",right)]:
        t=ctx.get(side,{})
        with col:
            st.markdown(f"### {t.get('abbr',game[f'{side}_abbr'])}")
            if not t.get("injury_verified",False):
                st.warning("Automated injury report was not verified. Missing rows do NOT mean nobody is injured. Use the official report link above as the final cross-check.")
            else:
                counts=t.get("injuries",{})
                st.write(f"**{counts.get('out',0)} out/doubtful • {counts.get('questionable',0)} questionable**")
            inj=t.get("injury_df",pd.DataFrame())
            if inj is not None and not inj.empty:
                show=[c for c in ["Player","Status","Injury","Detail","ReturnDate","Source"] if c in inj.columns]
                st.dataframe(inj[show].head(20),use_container_width=True,hide_index=True)
            elif t.get("injury_verified",False):
                st.success("The automated source stack returned no listed players. Still use the official league report for final pregame verification.")
# ============================================================
# SIMPLE UI — V11
# ============================================================

st.title("🏟️ Sports Research Lab")
st.caption(
    "v35 verified-data build • strict player identity • stat-bearing game rows only • visible loading/progress • fast local-first full-game scans • prop hit tables • sportsbook-confirmed Kalshi edge alerts."
)
st.caption(f"Build: {BUILD_ID}")

top1,top2=st.columns([1,5])
with top1:
    if st.button("🔄 Refresh live data"):
        st.cache_data.clear()
        st.rerun()
with top2:
    st.caption(
        "Refresh clears cached schedule, injury, roster, weather, player, and market data."
    )


with st.sidebar:
    st.header("📡 Data status")
    _db=load_csv(PLAYER_LOG_DB_FILE)
    _ball_seed=load_basketball_seed_v33()
    _nfl_seed=load_nfl_player_seed_v33()
    st.success(f"Seed data: {len(_ball_seed):,} basketball player-games • {len(_nfl_seed):,} NFL player-weeks")
    if _db.empty:
        st.caption("Persistent live/import database has no extra rows yet; bundled seed history is already available.")
    else:
        st.success(f"Additional saved/live rows: {len(_db):,}")
        if "INGESTED_AT" in _db.columns:
            _last=pd.to_datetime(_db["INGESTED_AT"],errors="coerce").max()
            if not pd.isna(_last): st.caption(f"Last player-data update: {_last}")
    st.caption("Live roster/injury/market feeds are refreshed separately. Public web research is optional corroboration only; there is no Ask-AI page.")

workspace=st.radio(
    "Sport workspace",
    ["🏀 Basketball","🏈 NFL","🗂️ Shared"],
    horizontal=True,
    key="v33_workspace",
)

if workspace=="🏀 Basketball":
    _labels=["🏀 Games","📊 Game Report","👤 Player Report","🔍 Find a Bet","📝 Type a Bet","🧾 Bet Slips","🚨 Edge Alerts"]
    _page_map={"🏀 Games":"🏠 This Week","📊 Game Report":"📊 Game Report","👤 Player Report":"👤 Player Report","🔍 Find a Bet":"🔍 Find a Bet","📝 Type a Bet":"📝 Type a Bet","🧾 Bet Slips":"🧾 Bet Slips","🚨 Edge Alerts":"🚨 Edge Alerts"}
    st.caption("NBA + WNBA share the basketball workflow: minutes, PTS/REB/AST/3PM, lineup/injury context and matchup style.")
    _choice=st.radio("Go to",_labels,horizontal=True,key="v33_basketball_page")
    page=_page_map[_choice]
elif workspace=="🏈 NFL":
    _labels=["🏈 Games","🏈 NFL Game Report","🏈 NFL Player Report","🔍 Find a Bet","📝 Type a Bet","🧾 Bet Slips","🚨 Edge Alerts"]
    _page_map={"🏈 Games":"🏠 This Week","🏈 NFL Game Report":"📊 Game Report","🏈 NFL Player Report":"👤 Player Report","🔍 Find a Bet":"🔍 Find a Bet","📝 Type a Bet":"📝 Type a Bet","🧾 Bet Slips":"🧾 Bet Slips","🚨 Edge Alerts":"🚨 Edge Alerts"}
    st.caption("NFL uses a separate football feature set: EPA, QB efficiency, snaps/workload, sacks/turnovers, rest, weather and opponent strength.")
    _choice=st.radio("Go to",_labels,horizontal=True,key="v33_nfl_page")
    page=_page_map[_choice]
else:
    _labels=["🎟️ My Bets","🔧 Data Admin","🧪 Advanced"]
    page=st.radio("Go to",_labels,horizontal=True,key="v33_shared_page")

_selected_game=st.session_state.get("matchup")
if _selected_game:
    st.success(
        "Selected game: "
        f"{_selected_game.get('away_name','Away')} at "
        f"{_selected_game.get('home_name','Home')}"
    )
    st.caption(
        "Session selection ID: "
        + str(_selected_game.get("event_id","unknown"))
    )
else:
    st.caption("No game is currently stored in this browser session.")

if _selected_game:
    _edge_rows=st.session_state.get("v21_complete_market_rows",[])
    if _edge_rows:
        render_edge_alerts(_selected_game,st.session_state.get("league"),_edge_rows,st.session_state.get("matchup_context"),compact=True)

# ============================================================
# THIS WEEK — SELECT A GAME ONLY
# ============================================================

if page=="🏠 This Week":
    st.subheader("🏠 This Week")
    st.write(
        "This page is only for choosing the game you want to research. "
        "It does not automatically recommend a bet."
    )

    if workspace=="🏈 NFL":
        league="NFL"
        st.caption("NFL workspace • schedule uses live ESPN first with the local nflverse games file as fallback.")
    else:
        league=st.selectbox(
            "League",
            ["NBA","WNBA"],
            key="v33_basketball_league",
        )

    window=st.radio(
        "Schedule",
        ["Today → next 6 days","Following 7 days"],
        horizontal=True,
    )
    start=(
        date.today()
        if window.startswith("Today")
        else date.today()+timedelta(days=7)
    )

    with st.spinner("Loading schedule..."):
        week_games,week_errors=fetch_week_schedule(
            league,start,7
        )

    if not week_games:
        st.info(f"No {league} games were returned in this window.")
    else:
        labels=[weekly_game_label(g) for g in week_games]
        gi=st.selectbox(
            "Choose a game",
            range(len(labels)),
            format_func=lambda i:labels[i],
            key="v20_game_select",
        )
        chosen=week_games[gi]

        st.markdown(
            f"### {chosen['away_name']} at {chosen['home_name']}"
        )
        st.caption(
            f"{chosen['schedule_date'].strftime('%A, %B %d')} • "
            f"{chosen.get('status','')} • "
            f"{chosen.get('venue') or 'venue not listed'}"
        )

        if st.button(
            "Use this game for research",
            type="primary",
            use_container_width=True,
        ):
            # Save the selection FIRST. If a roster/history/model source fails,
            # the rest of the app still knows which game the user selected.
            st.session_state["matchup"]=chosen
            st.session_state["league"]=league
            st.session_state["selected_date"]=chosen["schedule_date"]
            st.session_state["roster"]=pd.DataFrame()
            st.session_state["matchup_context"]=empty_matchup_context(chosen)

            for key in [
                "research_log","research_name",
                "research_matchup_log","research_error",
                "research_minutes_info",
                "latest_easy_analysis","latest_easy_selection",
                "full_game_ideas","full_game_scan_diagnostics",
                "full_game_scan_total","full_game_scan_event",
                "quick_ranked_ideas","v20_typed_result",
            ]:
                st.session_state.pop(key,None)

            roster=pd.DataFrame()
            source_messages=[]

            with st.spinner(
                "Loading rosters, team history, injuries, and matchup context..."
            ):
                try:
                    ar,aerr=fetch_roster(league,chosen["away_id"])
                    hr,herr=fetch_roster(league,chosen["home_id"])
                    if league=="NFL" and ar.empty:
                        ar=local_nfl_roster_v33(chosen["away_abbr"],chosen.get("schedule_date"))
                        if not ar.empty: source_messages.append("Away roster: live feed unavailable; using local weekly nflverse roster.")
                    if league=="NFL" and hr.empty:
                        hr=local_nfl_roster_v33(chosen["home_abbr"],chosen.get("schedule_date"))
                        if not hr.empty: source_messages.append("Home roster: live feed unavailable; using local weekly nflverse roster.")
                    if aerr and ar.empty:
                        source_messages.append("Away roster: "+str(aerr))
                    if herr and hr.empty:
                        source_messages.append("Home roster: "+str(herr))

                    if not ar.empty:
                        ar["team_abbr"]=chosen["away_abbr"]
                        ar["team_id"]=chosen["away_id"]
                    if not hr.empty:
                        hr["team_abbr"]=chosen["home_abbr"]
                        hr["team_id"]=chosen["home_id"]

                    roster=pd.concat(
                        [ar,hr],
                        ignore_index=True,
                    ) if (not ar.empty or not hr.empty) else pd.DataFrame()
                except Exception as e:
                    source_messages.append("Roster load: "+str(e))

                ctx,ctx_err=safe_build_matchup_context(chosen,league)
                if ctx_err:
                    source_messages.append("Matchup context: "+str(ctx_err))

            st.session_state["roster"]=roster
            st.session_state["matchup_context"]=ctx

            # Market evaluation is optional. A model failure must never cancel
            # the selected game.
            try:
                auto_rows,auto_err,auto_added=auto_evaluate_log_selected_game(
                    chosen,league
                )
                st.session_state["v21_complete_market_rows"]=auto_rows
                st.session_state["v21_complete_market_error"]=auto_err
                st.session_state["v21_complete_market_logged"]=auto_added
            except Exception as e:
                st.session_state["v21_complete_market_rows"]=[]
                st.session_state["v21_complete_market_error"]=str(e)
                st.session_state["v21_complete_market_logged"]=0
                source_messages.append("Automatic market model: "+str(e))

            st.success(
                f"Selected {chosen['away_abbr']} at {chosen['home_abbr']}. "
                f"Loaded {len(roster)} roster players. "
                "The game stays selected even if an optional source/model fails."
            )

            if source_messages:
                with st.expander("Some optional sources did not load"):
                    for msg in source_messages:
                        st.write("• "+msg)

    if week_errors:
        with st.expander("Schedule source messages"):
            for msg in week_errors[:20]:
                st.write("• "+msg)

# ============================================================
# GAME REPORT — RESEARCH, NOT BET RECOMMENDATIONS
# ============================================================

if page=="📊 Game Report":
    st.subheader("📊 Game Report")

    game=st.session_state.get("matchup")
    league=st.session_state.get("league")
    ctx=st.session_state.get("matchup_context")

    if not game:
        st.info("Choose a game in This Week first.")
    else:
        if not ctx:
            with st.spinner("Fetching team history, rosters, and injury data..."):
                ctx,ctx_err=safe_build_matchup_context(game,league)
            st.session_state["matchup_context"]=ctx
            if ctx_err:
                st.warning(
                    "Some matchup-context feeds failed, but the selected game is still loaded. "
                    "The report will use the sources that are available."
                )

        st.markdown(
            f"# {game['away_name']} at {game['home_name']}"
        )
        st.caption("Research is source-backed. If a feed fails, this page labels the missing source instead of filling the gap with a guess.")
        st.info("⏳ Detailed sections fetch and calculate data as you scroll. On the first load, you may briefly see a 'Fetching…' message; cached reloads should be much faster.")
        st.caption(
            f"{game.get('status','')} • "
            f"{game.get('venue') or 'venue not listed'}"
        )

        if league=="NFL":
            render_nfl_matchup_dashboard_v33(game)

        if league in {"NBA","WNBA"}:
            st.markdown("## At a glance")
            st.dataframe(
                v20_team_profile_df(game),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("## 🧠 What matters in this matchup")
        takeaways=matchup_scoreboard_takeaways(game)
        takeaways+=plain_matchup_factors(
            game,league,ctx,[],None
        )
        if takeaways:
            for note in list(dict.fromkeys(takeaways))[:12]:
                st.write("• "+note)
        else:
            st.caption(
                "No strong plain-English takeaway was returned from the current feeds."
            )

        st.markdown("## ⚔️ Team strengths, weaknesses & betting matchup")
        with st.expander(
            f"{game['away_abbr']} report",
            expanded=True,
        ):
            try:
                with st.spinner(f"Fetching {game['away_abbr']} strengths, defense, and matchup data..."):
                    render_team_attack_report(
                        game,league,"away"
                    )
            except Exception as e:
                st.warning(
                    "This team's advanced strength/weakness source returned an unexpected shape. "
                    "The rest of the Game Report will keep working."
                )
                st.caption("Team-report source error: "+str(e))
        with st.expander(
            f"{game['home_abbr']} report",
            expanded=True,
        ):
            try:
                with st.spinner(f"Fetching {game['home_abbr']} strengths, defense, and matchup data..."):
                    render_team_attack_report(
                        game,league,"home"
                    )
            except Exception as e:
                st.warning(
                    "This team's advanced strength/weakness source returned an unexpected shape. "
                    "The rest of the Game Report will keep working."
                )
                st.caption("Team-report source error: "+str(e))

        if league in {"NBA","WNBA"}:
            st.markdown("## ⚡ Pace / offense / defense")
            with st.spinner("Fetching pace, offense, and defense data (using a fast local fallback if the live advanced feed is slow)..."):
                render_basketball_advanced_or_fallback(
                    game,league,ctx
                )

        if league=="NFL":
            with st.spinner("Fetching NFL depth-chart and workload context..."):
                render_nfl_depth_context_v33(game)
        else:
            st.markdown("## 🧾 Starting lineup / possible lineup")
            with st.spinner("Fetching lineup and rotation context..."):
                render_current_or_possible_lineups(
                    game,league,ctx
                )

        st.markdown("## 🚑 Injuries / availability")
        # Re-filter cached injury frames against the selected teams before display.
        # This also fixes sessions created before the stricter fetch filter existed.
        for _side in ["away","home"]:
            _team_id=game.get(f"{_side}_id","")
            _team_ctx=ctx.get(_side,{})
            _inj=_team_ctx.get("injury_df",pd.DataFrame())
            if _inj is not None and not _inj.empty:
                _clean=_filter_injuries_to_team(_inj,league,_team_id)
                _team_ctx["injury_df"]=_clean
                _team_ctx["injuries"]=injury_counts(_clean)
                if _clean.empty and not _inj.empty:
                    _team_ctx["injury_verified"]=False
                    _team_ctx["injury_error"]="Returned injury rows could not be verified against this team's roster, so they were hidden to prevent cross-team mixing."
        with st.spinner("Verifying injury rows belong to the correct team..."):
            v20_render_injuries(
                game,ctx
            )

        st.markdown("## 🔥 Recent form / rest")
        left,right=st.columns(2)
        for side,col in [
            ("away",left),
            ("home",right),
        ]:
            t=ctx.get(side,{})
            with col:
                st.markdown(
                    f"### {t.get('abbr',game[f'{side}_abbr'])}"
                )
                recent=t.get("recent10",{})
                if recent:
                    bits=[
                        f"**Last 10:** {recent.get('wins',0)}-{recent.get('losses',0)}"
                    ]
                    if not pd.isna(recent.get("avg_for",np.nan)):
                        bits.append(f"{recent['avg_for']:.1f} scored")
                    if not pd.isna(recent.get("avg_against",np.nan)):
                        bits.append(f"{recent['avg_against']:.1f} allowed")
                    if not pd.isna(recent.get("margin",np.nan)):
                        bits.append(f"{recent['margin']:+.1f} average margin")
                    st.write(" • ".join(bits))
                else:
                    rec=game.get(f"{side}_records") or {}
                    st.write(
                        f"**Season:** {rec.get('overall','—')}"
                    )
                rest=t.get("rest",np.nan)
                st.write(
                    "**Rest:** "
                    + (
                        "not returned"
                        if pd.isna(rest)
                        else f"{int(rest)} day(s)"
                    )
                )

        st.markdown("## 🤝 Head-to-head")
        h2h=ctx.get("home",{}).get(
            "h2h",pd.DataFrame()
        )
        hh=ctx.get("home",{}).get(
            "h2h_summary",{}
        )
        if hh and hh.get("games",0):
            st.write(
                f"Found **{hh['games']}** recent matchup(s). "
                f"{game['home_abbr']} record: "
                f"**{hh['wins']}-{hh['losses']}** • "
                f"average margin **{hh.get('margin',0):+.1f}**."
            )
            show=[
                c for c in [
                    "date","home_away","result",
                    "points_for","points_against"
                ]
                if c in h2h.columns
            ]
            st.dataframe(
                h2h[show].head(10),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(
                "No usable recent team head-to-head sample was returned. "
                "This does not mean the teams have never played."
            )

        st.markdown("## 💹 Current market context")
        with st.spinner("Fetching current market and sportsbook context..."):
            v20_render_market_snapshot(
                game,league
            )

        if league=="NFL":
            st.markdown("## 🌦️ Weather")
            if game.get("indoor") is True:
                st.success(
                    "Indoor stadium — outdoor weather should have little direct impact."
                )
            else:
                weather,err=fetch_game_day_weather(
                    game.get("venue_city",""),
                    game.get("venue_state",""),
                    game.get("date",""),
                )
                if weather:
                    a,b,c,d=st.columns(4)
                    a.metric("High",f"{weather['high_f']:.0f}°F")
                    b.metric("Low",f"{weather['low_f']:.0f}°F")
                    c.metric("Rain",f"{weather['rain_pct']:.0f}%")
                    d.metric("Wind",f"{weather['wind_mph']:.0f} mph")
                else:
                    st.caption(
                        "Weather unavailable: "+str(err)
                    )

        st.caption(
            "This page is for researching the matchup. "
            "Automatic bet suggestions are kept in Find a Bet."
        )

# ============================================================
# PLAYER REPORT — CHOOSE A PLAYER AND RESEARCH THEM
# ============================================================

if page=="👤 Player Report":
    st.subheader("👤 Player Report")

    game=st.session_state.get("matchup")
    league=st.session_state.get("league")
    roster=st.session_state.get(
        "roster",pd.DataFrame()
    )
    ctx=st.session_state.get("matchup_context")

    if not game:
        st.info("Choose a game in This Week first.")
    else:
        if roster.empty:
            st.warning(
                "The structured roster feed did not load, but the game is selected. "
                "Use the manual player search below; saved/imported player logs can still be used if available."
            )
        st.write(
            f"Research any player from **{game['away_abbr']} at {game['home_abbr']}**."
        )

        away=roster[
            roster["team_abbr"]==game["away_abbr"]
        ]
        home=roster[
            roster["team_abbr"]==game["home_abbr"]
        ]

        left,right=st.columns(2)
        for frame,abbr,col in [
            (away,game["away_abbr"],left),
            (home,game["home_abbr"],right),
        ]:
            with col:
                st.markdown(f"### {abbr}")
                for i,row in frame.reset_index(drop=True).iterrows():
                    if st.button(
                        f"{row['player']} • {row.get('position','')}",
                        key=f"v20_player_{abbr}_{row.get('athlete_id','')}_{i}",
                        use_container_width=True,
                    ):
                        load_player_v11(
                            row,game,league
                        )

        st.markdown("### 🔎 Player not listed? Search the whole league")
        px,py=st.columns([4,1])
        with px:
            manual_name=st.text_input("Player name",key="v23_manual_player",placeholder="Type any NBA, WNBA, or NFL player name")
        with py:
            st.write("")
            st.write("")
            if st.button("Research player",key="v23_manual_player_go",use_container_width=True):
                found,ferr=find_player_anywhere_v23(league,manual_name)
                if found is not None:
                    load_player_v11(found,game,league)
                    if ferr: st.caption("Roster-source note: "+str(ferr))
                else:
                    clean_name=str(manual_name or "").strip()
                    if clean_name:
                        st.session_state["research_name"]=clean_name
                        st.session_state["research_log"]=pd.DataFrame()
                        st.session_state["research_matchup_log"]=pd.DataFrame()
                        st.session_state["research_opponent"]=""
                        st.session_state["research_team"]=""
                        st.session_state["research_team_id"]=""
                        st.session_state["research_today_side"]=""
                        st.warning(
                            "The structured league directory did not resolve this player. "
                            "Import a game-log file in Data Admin if this player should be in the database."
                        )
                        if ferr:
                            st.caption("Structured-source note: "+str(ferr))
                    else:
                        st.error("Type a player name first.")

        if st.session_state.get("research_error"):
            st.warning(st.session_state["research_error"])
            st.caption(
                "Structured statistics are unavailable for this attempt. Check Data Admin for saved/imported rows or refresh the live sources."
            )

        log=st.session_state.get(
            "research_log",pd.DataFrame()
        )

        if not log.empty:
            player_name=st.session_state.get(
                "research_name",""
            )
            opponent=st.session_state.get(
                "research_opponent",""
            )
            matchup_log=st.session_state.get(
                "research_matchup_log",pd.DataFrame()
            )
            today_side=st.session_state.get(
                "research_today_side",""
            )
            team_id=st.session_state.get(
                "research_team_id",""
            )

            st.divider()
            st.markdown(
                f"# {player_name} vs {opponent}"
            )
            _vr=int(st.session_state.get("research_verified_rows_v35",len(log)) or 0)
            _vd=str(st.session_state.get("research_latest_verified_v35","") or "")
            if _vr:
                st.success(
                    f"✅ Verified player log: {_vr} stat-bearing game rows"
                    + (f" • latest verified game {_vd}" if _vd else "")
                    + ". Schedule-only rows with no player stats are excluded."
                )
                try:
                    _latest=pd.to_datetime(_vd,errors="coerce",utc=True)
                    _gdate=pd.to_datetime(game.get("date") or st.session_state.get("selected_date"),errors="coerce",utc=True)
                    if not pd.isna(_latest) and not pd.isna(_gdate):
                        _age=int((_gdate.normalize()-_latest.normalize()).days)
                        if _age>10:
                            st.warning(f"⚠️ Latest verified player-stat row is {_age} days before this game. Current form may be incomplete until a live source verifies newer box scores.")
                except Exception:
                    pass
                if "DATA_SOURCE" in log.columns:
                    _src=log["DATA_SOURCE"].dropna().astype(str).value_counts().head(3)
                    if not _src.empty:
                        st.caption("Verified log sources: "+" • ".join(f"{k} ({v})" for k,v in _src.items()))
            else:
                st.warning("No stat-bearing game rows passed verification for this player.")

            if league=="NFL":
                render_nfl_player_snapshot_v33(log,player_name)

            injuries,inj_err=(
                fetch_injuries(league,team_id)
                if team_id
                else (
                    pd.DataFrame(),
                    "No team ID",
                )
            )

            player_team=st.session_state.get(
                "research_team",""
            )
            player_side=(
                "home"
                if player_team==game["home_abbr"]
                else "away"
            )
            team_ctx=(
                ctx.get(player_side,{})
                if ctx
                else {}
            )

            minutes_info=estimate_player_availability_minutes(
                log,
                injuries,
                player_name,
                injury_verified=team_ctx.get(
                    "injury_verified",
                    inj_err is None,
                ),
            )
            st.session_state[
                "research_minutes_info"
            ]=minutes_info

            if league=="NFL":
                st.markdown("## 🩺 Availability / workload")
                status="NOT VERIFIED"
                if injuries is not None and not injuries.empty:
                    nm=injuries[injuries.get("Player",pd.Series("",index=injuries.index)).astype(str).map(_name_key_v33)==_name_key_v33(player_name)] if "Player" in injuries.columns else pd.DataFrame()
                    if not nm.empty:
                        status=str(nm.iloc[0].get("Status") or nm.iloc[0].get("status") or "LISTED")
                snaps=pd.to_numeric(log.get("SNAP PCT",pd.Series(dtype=float)),errors="coerce").dropna().head(5)
                workload=snaps.mean() if not snaps.empty else np.nan
                a,b,c=st.columns(3)
                a.metric("Status",status)
                b.metric("Recent offensive snap %","—" if pd.isna(workload) else f"{workload*100:.0f}%")
                pos=str(log.get("POSITION",pd.Series([""])).dropna().iloc[0] if "POSITION" in log.columns and log["POSITION"].notna().any() else "")
                c.metric("Position",pos or "—")
                st.caption("NFL workload uses snap share rather than basketball-style minutes. Current injury/inactive status is checked separately from historical workload.")
            else:
                st.markdown("## ⏱️ Playing status / likely minutes")
                a,b,c=st.columns(3)
                a.metric(
                    "Status",
                    minutes_info["status"],
                )
                b.metric(
                    "Likely minutes",
                    "—"
                    if pd.isna(minutes_info["likely"])
                    else f"{minutes_info['likely']:.1f}",
                )
                c.metric(
                    "Likely range",
                    "—"
                    if (
                        pd.isna(minutes_info["low"])
                        or pd.isna(minutes_info["high"])
                    )
                    else (
                        f"{minutes_info['low']:.0f}–"
                        f"{minutes_info['high']:.0f}"
                    ),
                )

                if minutes_info.get("restriction"):
                    st.warning(
                        "Minutes restriction reported: "
                        + minutes_info["restriction"]["label"]
                    )
                if minutes_info["status"] in {
                    "QUESTIONABLE / DAY-TO-DAY",
                    "DOUBTFUL",
                    "INJURY STATUS NOT VERIFIED",
                }:
                    st.warning(
                        "Availability is uncertain. Treat the minute estimate cautiously."
                    )

                st.caption(
                    minutes_info["basis"]
                    + " • "
                    + minutes_info["source_note"]
                )

                st.markdown("## 📊 Recent production")
                overall=summarize_player_matchup(
                    log.head(10)
                )
                vs=summarize_player_matchup(
                    matchup_log
                )
                cols=st.columns(5)
                cols[0].metric(
                    "Recent PTS",
                    fmt_num(overall.get("PTS")),
                )
                cols[1].metric(
                    "Recent REB",
                    fmt_num(overall.get("REB")),
                )
                cols[2].metric(
                    "Recent AST",
                    fmt_num(overall.get("AST")),
                )
                cols[3].metric(
                    "Recent PRA",
                    fmt_num(overall.get("PRA")),
                )
                cols[4].metric(
                    "Games vs opponent",
                    vs.get("games",0),
                )

                st.caption(v20_readable_opponent_sample(vs.get("games",0)))

                st.markdown("### Last 20 games")
                recent_cols=[c for c in ["DATE","TEAM","MATCHUP","OPP","MIN","PTS","REB","AST","STL","BLK","FG3M","FG3A","TOV","RESULT","SEASON","DATA_SOURCE"] if c in log.columns]
                if recent_cols:
                    st.dataframe(log[recent_cols].head(20),use_container_width=True,hide_index=True)

                st.markdown(f"### Exact history vs {opponent}")
                if not matchup_log.empty:
                    show_cols=[c for c in ["DATE","SEASON","MATCHUP","OPP","MIN","PTS","REB","AST","FG3M","FG3A","TOV","RESULT","DATA_SOURCE"] if c in matchup_log.columns]
                    st.dataframe(matchup_log[show_cols].head(20),use_container_width=True,hide_index=True)
                else:
                    st.caption("No prior professional game against this exact opponent was found in the multi-season logs returned by the sources.")

                if not matchup_log.empty:
                    m1,m2,m3,m4=st.columns(4)
                    m1.metric(
                        "PTS vs opponent",
                        fmt_num(vs.get("PTS")),
                    )
                    m2.metric(
                        "REB vs opponent",
                        fmt_num(vs.get("REB")),
                    )
                    m3.metric(
                        "AST vs opponent",
                        fmt_num(vs.get("AST")),
                    )
                    m4.metric(
                        "PRA vs opponent",
                        fmt_num(vs.get("PRA")),
                    )

            render_targeted_player_status_search_v33(player_name,game,league)

            st.markdown("## ⚔️ Player strengths/weaknesses vs opponent")
            pos=""
            match_row=roster[
                roster["player"].astype(str)==str(player_name)
            ]
            if not match_row.empty:
                pos=str(
                    match_row.iloc[0].get(
                        "position",""
                    )
                )
            fit=v20_render_player_style(
                game,
                league,
                log,
                player_name,
                pos,
                opponent,
            )

            st.markdown("## 📈 Role / usage trend")
            usage=player_role_usage_change(
                log,league
            )
            for note in usage["notes"]:
                st.write("• "+note)
            if usage["rows"]:
                udf=pd.DataFrame(
                    usage["rows"]
                )
                if "Change %" in udf.columns:
                    udf["Change %"]=udf["Change %"].map(
                        lambda x:
                        "—"
                        if pd.isna(x)
                        else f"{x*100:+.0f}%"
                    )
                st.dataframe(
                    udf,
                    use_container_width=True,
                    hide_index=True,
                )

            st.markdown("## 🧮 Research a specific prop")
            props=available_props(
                log,league
            )
            if props:
                p1,p2,p3=st.columns(3)
                with p1:
                    prop_name=st.selectbox(
                        "Stat",
                        list(props.keys()),
                        key="v20_prop",
                    )
                with p2:
                    side=st.selectbox(
                        "Side",
                        [
                            "At least (X+)",
                            "Over",
                            "Under",
                        ],
                        key="v20_side",
                    )
                with p3:
                    line=st.number_input(
                        "Line",
                        min_value=0.0,
                        value=3.0,
                        step=.5,
                        key="v20_line",
                    )

                if st.button(
                    "Research this line",
                    type="primary",
                    key="v20_prop_research",
                ):
                    analysis=easy_prop_analysis(
                        log,
                        matchup_log,
                        league,
                        props[prop_name],
                        side,
                        line,
                        market_probability=None,
                        current_side=today_side,
                        opponent=opponent,
                    )
                    st.session_state[
                        "latest_easy_analysis"
                    ]=analysis
                    st.session_state[
                        "latest_easy_selection"
                    ]=(
                        f"{player_name} — "
                        f"{side} {line} {prop_name}"
                    )

                analysis=st.session_state.get(
                    "latest_easy_analysis"
                )
                if analysis:
                    a,b,c,d=st.columns(4)
                    a.metric(
                        "Estimated chance",
                        fmt_pct(
                            analysis["probability"]
                        ),
                    )
                    b.metric(
                        "Projection",
                        fmt_num(
                            analysis["projection"]
                        ),
                    )
                    c.metric(
                        "Last 10 hit rate",
                        fmt_pct(
                            analysis["last10"]
                        ),
                    )
                    d.metric(
                        "Vs opponent",
                        fmt_pct(
                            analysis["vs_opponent"]
                        ),
                    )

                    quality=research_data_quality(
                        log,
                        matchup_log,
                        team_ctx.get(
                            "injury_verified",False
                        ),
                        minutes_info,
                        analysis,
                    )
                    st.write(
                        f"**Data quality:** {quality['label']} "
                        f"({quality['score']}/100)"
                    )
                    st.write(
                        f"**Research verdict:** {analysis['verdict']}"
                    )
                    st.write(
                        analysis["verdict_text"]
                    )

                    flags=automatic_red_flags(
                        log,
                        matchup_log,
                        minutes_info,
                        analysis,
                    )
                    if flags:
                        st.markdown("### Main warnings")
                        for x in flags:
                            st.write(x)

                    render_last20_prop_chart(
                        log,
                        props[prop_name],
                        side,
                        line,
                        opponent,
                        title="Last 20 games vs this line",
                    )

                    with st.expander(
                        "Model/backtest details"
                    ):
                        st.write(
                            f"Out-of-sample test predictions: "
                            f"**{analysis['backtest_n']}**"
                        )
                        st.write(
                            f"Accuracy: **{fmt_pct(analysis['accuracy'])}**"
                        )
                        st.write(
                            f"Confidence: **{analysis['confidence']}**"
                        )
                        st.write(
                            f"Brier: **{fmt_num(analysis.get('brier'),3)}**"
                        )
            else:
                st.warning("No verified stat columns were available to build player props for this report.")

            st.markdown("## 🔗 Sources / verification")
            notes=st.session_state.get("research_source_notes_v23",[])
            if notes:
                with st.expander("Source fallback messages"):
                    for n in notes: st.write("• "+str(n))
            st.caption("These links are direct verification sources for the selected player/game. Saved/imported logs are also shown in Data Admin.")
            for label,url in research_source_links(
                game,
                league,
                player_id=st.session_state.get(
                    "research_player_id",""
                ),
                athlete_id=st.session_state.get(
                    "research_athlete_id",""
                ),
                team_id=team_id,
                opponent_team_id=st.session_state.get(
                    "research_opponent_team_id",""
                ),
            ):
                st.markdown(
                    f"- [{label}]({url})"
                )

# ============================================================
# FIND A BET — ONLY PLACE THAT AUTO-SURFACES IDEAS
# ============================================================

if page=="🔍 Find a Bet":
    st.subheader("🔍 Find a Bet")
    st.write(
        "This pulls the current sportsbook player-prop lines for the selected game, then checks those exact lines against the verified local player database. The broad scan stays local after the one market fetch; Final check refreshes the exact shortlisted player and market."
    )

    game=st.session_state.get("matchup")
    league=st.session_state.get("league")
    roster=st.session_state.get(
        "roster",pd.DataFrame()
    )
    ctx=st.session_state.get(
        "matchup_context"
    )

    if not game or roster.empty:
        st.info(
            "Choose a game in This Week first."
        )
    else:
        st.markdown(
            f"# {game['away_name']} at {game['home_name']}"
        )

        if not get_the_odds_api_key():
            st.warning("THE_ODDS_API_KEY is not configured in Streamlit Secrets, so live player-prop scanning is unavailable.")

        if st.button(
            "Fetch & scan current sportsbook props",
            type="primary",
            use_container_width=True,
            disabled=not bool(get_the_odds_api_key()),
        ):
            scan_status=st.status("⏳ Matching this game to current sportsbook markets...",expanded=True)
            scan_progress=st.progress(0.0)
            scan_line=st.empty()
            def _scan_progress_v37(i,total,name,stage):
                frac=0.0 if not total else min(1.0,max(0.0,float(i)/float(total)))
                scan_progress.progress(frac)
                if name:
                    scan_line.caption(f"{stage.title()}: {name}")
                else:
                    scan_line.caption(stage.title())
            ideas,diagnostics,odds_meta,scan_err=scan_live_sportsbook_props_v37(
                game,league,roster,ctx,progress_cb=_scan_progress_v37
            )
            ranked=v20_top_game_ideas(ideas)
            st.session_state["full_game_ideas"]=ideas
            st.session_state["full_game_scan_diagnostics"]=diagnostics
            st.session_state["full_game_scan_total"]=len(ideas)
            st.session_state["quick_ranked_ideas"]=ranked
            st.session_state["live_props_meta_v37"]=odds_meta
            st.session_state["live_props_error_v37"]=scan_err or ""
            if scan_err:
                scan_status.update(label="⚠️ Live prop scan could not be completed",state="error",expanded=True)
            else:
                scan_status.update(label="✅ Current sportsbook lines checked against verified player history",state="complete",expanded=False)
            scan_progress.progress(1.0)

        ranked=st.session_state.get("quick_ranked_ideas",[])
        odds_meta=st.session_state.get("live_props_meta_v37",{}) or {}
        scan_err=st.session_state.get("live_props_error_v37","")
        if scan_err:
            st.warning(scan_err)
        if odds_meta.get("event"):
            ev=odds_meta["event"]
            quota=odds_meta.get("remaining")
            qtxt=f" • API credits remaining: {quota}" if quota not in [None,""] else ""
            st.caption(
                f"Live market matched: {ev.get('away_team','')} at {ev.get('home_team','')} • "
                f"sportsbook lines cached for 5 minutes{qtxt}."
            )
        if "quick_ranked_ideas" in st.session_state:
            st.caption("Only exact player identities from the selected-game roster are allowed into the scan. Historical calculations use verified local logs for speed; live market price and line come from The Odds API.")

        if "quick_ranked_ideas" not in st.session_state:
            st.caption("Press the button once to fetch the current offered player props and evaluate those exact lines.")
        elif not ranked:
            st.warning(
                "### 🚫 No current sportsbook prop cleared the promotion filter\n\n"
                "The app checked the real posted lines but will not force a pick when the model-vs-market difference, price, data quality, or availability is not strong enough."
            )
            _all_checked=st.session_state.get("full_game_ideas",[]) or []
            if _all_checked:
                _rows=[]
                for _r in _all_checked[:20]:
                    _best=_r.get("Best odds",np.nan)
                    _edge=_r.get("Market edge",np.nan)
                    _rows.append({
                        "Player":_r.get("Player"),
                        "Prop":f"{_r.get('Line','')} {_r.get('Stat','')}",
                        "Best price":("—" if pd.isna(_best) else f"{float(_best):+.0f}"),
                        "Book":_r.get("Best book",""),
                        "Model":fmt_pct(_r.get("Model probability")),
                        "Sportsbook fair":fmt_pct(_r.get("Market probability")),
                        "Difference":("—" if pd.isna(_edge) else f"{float(_edge)*100:+.1f} pts"),
                        "Last 10":fmt_pct(_r.get("Last 10")),
                        "Quality":_r.get("Data quality","—"),
                    })
                st.markdown("### Closest current lines checked")
                st.dataframe(pd.DataFrame(_rows),use_container_width=True,hide_index=True)
        else:
            st.markdown("## ⭐ Strongest player-prop ideas")
            for rank,row in enumerate(
                ranked[:5],1
            ):
                agreement,checks=quick_idea_agreement(
                    row
                )
                low,high=quick_probability_band(
                    row.get("Model probability"),
                    row.get("Data quality"),
                )
                label,kind=quick_idea_label(
                    row
                )

                st.markdown(
                    f"### {rank}. {row['Player']} — "
                    f"{row['Line']} {row['Stat']}"
                )

                a,b,c=st.columns(3)
                a.metric(
                    "Estimated chance",
                    (
                        "—"
                        if pd.isna(low) or pd.isna(high)
                        else f"{low*100:.0f}–{high*100:.0f}%"
                    ),
                )
                b.metric(
                    "Research agreement",
                    f"{agreement}/8",
                )
                c.metric(
                    "Data quality",
                    row.get("Data quality","—"),
                )

                if row.get("Live market"):
                    m1,m2,m3,m4=st.columns(4)
                    _best=row.get("Best odds",np.nan)
                    m1.metric(
                        "Best current price",
                        "—" if pd.isna(_best) else f"{float(_best):+.0f}",
                        help=str(row.get("Best book","") or "Sportsbook"),
                    )
                    m2.metric("Sportsbook fair",fmt_pct(row.get("Market probability")))
                    _me=row.get("Market edge",np.nan)
                    m3.metric("Model vs market","—" if pd.isna(_me) else f"{float(_me)*100:+.1f} pts")
                    _pe=row.get("Price edge",np.nan)
                    m4.metric("Model vs break-even","—" if pd.isna(_pe) else f"{float(_pe)*100:+.1f} pts")
                    st.caption(
                        f"Best book: {row.get('Best book','—')} • books at this line: {row.get('Books',0)} • "
                        f"paired books used to de-vig consensus: {row.get('Consensus books',0)} • market update: {row.get('Odds last update','—')}"
                    )

                if kind=="success":
                    st.success(label)
                elif kind=="warning":
                    st.warning(label)
                else:
                    st.info(label)

                st.write(
                    ("**Workload / availability:** " if league=="NFL" else "**Minutes / availability:** ")
                    + str(row.get("Minutes","—"))
                )

                st.markdown("**Why it stands out**")
                for x in quick_idea_reasons(
                    row
                ):
                    st.write("✅ "+x)

                st.markdown("**Biggest concern**")
                st.write(
                    "⚠️ "
                    + quick_idea_concern(row)
                )

                with st.expander(
                    "Show full evidence"
                ):
                    left,right=st.columns(2)
                    with left:
                        st.markdown(
                            "**Player strengths / matchup help**"
                        )
                        for x in (
                            row.get(
                                "Player strengths",[]
                            )
                            + row.get("Helps",[])
                        )[:8]:
                            st.write("• "+str(x))
                    with right:
                        st.markdown(
                            "**Player weaknesses / matchup concerns**"
                        )
                        for x in (
                            row.get(
                                "Player weaknesses",[]
                            )
                            + row.get("Hurts",[])
                        )[:8]:
                            st.write("• "+str(x))

                    st.markdown(
                        "**8 research checks**"
                    )
                    for name,ok in checks:
                        st.write(
                            ("✅ " if ok else "❌ ")
                            + name
                        )

                key_base=re.sub(
                    r"[^A-Za-z0-9_]+",
                    "_",
                    (
                        f"{row['Player']}_"
                        f"{row['Stat']}_"
                        f"{row['Line']}_{rank}"
                    ),
                )

                b1,b2=st.columns(2)
                with b1:
                    if st.button(
                        "🔄 Final check",
                        key="v20_final_"+key_base,
                        use_container_width=True,
                    ):
                        with st.spinner(
                            "Refreshing this exact idea..."
                        ):
                            result=final_check_game_idea(
                                game,
                                league,
                                roster,
                                row,
                            )
                        st.session_state[
                            "v20_final_result_"+key_base
                        ]=result
                with b2:
                    if st.button(
                        "➕ Save to My Bets",
                        key="v20_save_"+key_base,
                        use_container_width=True,
                    ):
                        save_quick_idea_to_watchlist(
                            row,league
                        )
                        st.success("Saved.")

                result=st.session_state.get(
                    "v20_final_result_"+key_base
                )
                if result:
                    if result["kind"]=="success":
                        st.success(
                            "✅ "+result["status"]
                        )
                    elif result["kind"]=="error":
                        st.error(
                            "🚨 "+result["status"]
                        )
                    else:
                        st.warning(
                            "⚠️ "+result["status"]
                        )
                    for msg in result.get(
                        "messages",[]
                    ):
                        st.write("• "+msg)

                st.divider()

        _diag=st.session_state.get("full_game_scan_diagnostics",[]) or []
        if _diag:
            with st.expander("Verification / skipped market rows"):
                st.caption("These rows were excluded instead of guessed when player identity, stat history, or sportsbook data could not be verified.")
                st.dataframe(pd.DataFrame(_diag).drop_duplicates(),use_container_width=True,hide_index=True)

        st.markdown("## 🏆 Team winner research")
        team=team_win_research(
            game,league
        )
        cons=team.get("consensus")
        if cons:
            a,b=st.columns(2)
            a.metric(
                game["away_abbr"],
                fmt_pct(
                    cons["away_probability"]
                ),
            )
            b.metric(
                game["home_abbr"],
                fmt_pct(
                    cons["home_probability"]
                ),
            )
            st.caption(
                f"De-vigged outside estimate from {cons['books']} book(s). "
                "Higher probability does not automatically mean good betting value."
            )
        else:
            st.caption(
                "Outside team-win estimate could not be verified."
            )

        st.markdown("## 🔎 Selected-game Kalshi comparison")
        prices,kerr=selected_game_kalshi_prices_v22(
            game,league
        )
        if prices and cons:
            rows=[]
            for r in prices:
                fair=(
                    cons["home_probability"]
                    if r["Abbr"]==game["home_abbr"]
                    else cons["away_probability"]
                )
                rows.append({
                    "Team":r["Team"],
                    "Kalshi":fmt_pct(
                        r["Kalshi probability"]
                    ),
                    "Outside fair":fmt_pct(fair),
                    "Difference":(
                        "—"
                        if pd.isna(
                            r["Kalshi probability"]
                        )
                        else (
                            f"{(fair-r['Kalshi probability'])*100:+.1f} pts"
                        )
                    ),
                })
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )
        elif kerr:
            st.caption(
                "Kalshi comparison unavailable: "
                + str(kerr)
            )
        else:
            st.caption(
                "No matching selected-game Kalshi winner market was returned."
            )

# ============================================================
# TYPE A BET
# ============================================================

if page=="📝 Type a Bet":
    st.subheader("📝 Type a Bet")
    st.write(
        "Use this when he already has a specific bet in mind and wants the site to research it."
    )

    game=st.session_state.get("matchup")
    league=st.session_state.get("league")
    roster=st.session_state.get(
        "roster",pd.DataFrame()
    )
    ctx=st.session_state.get(
        "matchup_context"
    )

    if not game:
        st.info(
            "Choose a game in This Week first."
        )
    else:
        if roster.empty:
            st.warning(
                "The selected-game roster feed is empty, but Type a Bet can still "
                "resolve a typed player from the league roster index or saved player database."
            )
        typed=st.text_input(
            "Bet",
            placeholder=(
                "Example: Rachel Banham 3+ points "
                "or A'ja Wilson over 9.5 rebounds -110"
            ),
            key="v20_typed_bet",
        )

        if st.button(
            "Research this bet",
            type="primary",
            disabled=not typed.strip(),
        ):
            with st.spinner(
                "Researching that exact bet..."
            ):
                result=analyze_bet_line_for_slip(
                    typed,
                    game,
                    league,
                    roster,
                    ctx,
                )
            st.session_state[
                "v20_typed_result"
            ]=result


        result=st.session_state.get(
            "v20_typed_result"
        )
        if result:
            st.markdown(
                f"## {result.get('Bet',typed)}"
            )
            p=result.get(
                "Probability",np.nan
            )
            be=result.get(
                "Break-even",np.nan
            )
            edge=result.get(
                "Edge",np.nan
            )

            a,b,c=st.columns(3)
            a.metric(
                "Estimated chance",
                fmt_pct(p),
            )
            b.metric(
                "Break-even chance",
                fmt_pct(be),
            )
            c.metric(
                "Price difference",
                (
                    "—"
                    if pd.isna(edge)
                    else f"{edge*100:+.1f} pts"
                ),
            )

            st.write(
                "**Research verdict:** "
                + str(
                    result.get(
                        "Verdict","—"
                    )
                )
            )
            if result.get("Minutes"):
                st.write(
                    ("**Workload / availability:** " if league=="NFL" else "**Minutes / availability:** ")
                    + str(result["Minutes"])
                )
            if result.get(
                "Value explanation"
            ):
                st.info(
                    result[
                        "Value explanation"
                    ]
                )
            if result.get("Risk"):
                st.warning(
                    "**Main concern:** "
                    + str(result["Risk"])
                )

            render_typed_bet_detail(typed,game,league,roster,ctx)


# ============================================================
# BET SLIPS — FULL SLIP ANALYZER OUTSIDE ADVANCED
# ============================================================

if page=="🧾 Bet Slips":
    st.subheader("🧾 Bet Slips")
    st.write("Paste a full slip here. Every line is researched separately using the selected game's player/team data; unsupported facts are not invented.")
    game=st.session_state.get("matchup"); league=st.session_state.get("league"); roster=st.session_state.get("roster",pd.DataFrame()); ctx=st.session_state.get("matchup_context")
    if not game:
        st.info("Choose a game in This Week first.")
    else:
        if roster.empty:
            st.warning(
                "The roster feed is empty, but typed player names can still be resolved from the league index or saved player database."
            )
        slip=st.text_area("One bet per line",height=180,key="v23_slip_text",placeholder="Player over 19.5 points -110\nTeam to win +135\nPlayer 8+ rebounds")
        if st.button("Research full slip",key="v23_slip_go",type="primary",use_container_width=True):
            lines=[x.strip() for x in slip.splitlines() if x.strip()]
            results=[]
            with st.spinner("Researching every line on the slip..."):
                for raw in lines[:30]:
                    try: results.append(analyze_bet_line_for_slip(raw,game,league,roster,ctx))
                    except Exception as e: results.append({"Bet":raw,"Verdict":"Research error","Risk":str(e)})
            st.session_state["v23_slip_results"]=results
        results=st.session_state.get("v23_slip_results",[])
        if results:
            df=pd.DataFrame(results)
            for c in ["Probability","Break-even","Edge"]:
                if c in df.columns:
                    df[c]=df[c].map(lambda x:"—" if pd.isna(x) else (f"{x*100:+.1f} pts" if c=="Edge" else f"{x*100:.1f}%"))
            st.dataframe(df,use_container_width=True,hide_index=True)
            st.caption("A model probability is an estimate, not a guarantee. Market price, injury news, minutes and source quality still matter.")
            st.markdown("## Leg-by-leg research")
            for i,r in enumerate(results,1):
                raw=str(r.get("Bet",f"Leg {i}"))
                with st.expander(f"{i}. {raw}",expanded=(i==1)):
                    c1,c2,c3=st.columns(3)
                    c1.metric("Estimated chance",fmt_pct(r.get("Probability")))
                    c2.metric("Break-even",fmt_pct(r.get("Break-even")))
                    _e=safe_float(r.get("Edge"))
                    c3.metric("Price edge","—" if pd.isna(_e) else f"{_e*100:+.1f} pts")
                    st.write("**Verdict:** "+str(r.get("Verdict","—")))
                    if r.get("Minutes"): st.write(("**Workload / availability:** " if league=="NFL" else "**Minutes / availability:** ")+str(r.get("Minutes")))
                    if r.get("Risk"): st.warning(str(r.get("Risk")))
                    render_typed_bet_detail(raw,game,league,roster,ctx)

# ============================================================
# MARKETS & PRICES
# ============================================================

if page=="🚨 Edge Alerts":
    st.subheader("🚨 Edge Alerts")

    game=st.session_state.get("matchup")
    league=st.session_state.get("league")

    if not game:
        st.info("Choose a game in This Week first.")
    else:
        st.markdown(f"# {game['away_name']} at {game['home_name']}")
        v20_render_market_snapshot(game,league)

        st.markdown("## 🏷️ Same outcome, different price")
        st.write("This compares every sportsbook price returned by the ESPN odds feed with Kalshi for the same winner outcome. A higher implied probability does **not** mean a bigger payout — cheaper prices / longer odds generally pay more if the outcome wins.")
        _mr,_me=full_market_model_dashboard_rows(game,league)
        comp,cerr=cross_market_winner_table_v23(game,league,_mr)
        if not comp.empty:
            show=comp.copy()
            for c in ["Market implied probability","Model probability"]:
                if c in show.columns: show[c]=show[c].map(fmt_pct)
            if "Model edge" in show.columns: show["Model edge"]=show["Model edge"].map(lambda x:"—" if pd.isna(x) else f"{x*100:+.1f} pts")
            if "Potential profit on $100 if correct" in show.columns: show["Potential profit on $100 if correct"]=show["Potential profit on $100 if correct"].map(fmt_money)
            st.dataframe(show,use_container_width=True,hide_index=True)
            st.caption("Potential profit is a price comparison, not a recommendation. Kalshi estimate includes the current general taker-fee calculation; sportsbook profit uses the displayed American moneyline.")
        else:
            st.warning("No directly comparable winner prices were returned from the current automatic sources.")
        if cerr: st.caption("Price-source messages: "+str(cerr))

        with st.expander("➕ Compare odds from another sportsbook/app"):
            st.caption("Use this when another book is not in the automatic ESPN feed. Enter the exact moneyline you see; the app compares price, payout, and model edge without pretending it fetched that book automatically.")
            m1,m2,m3=st.columns(3)
            with m1: extra_book=st.text_input("Book/app name",key="v23_extra_book",placeholder="FanDuel / BetMGM / etc.")
            with m2: extra_side=st.selectbox("Team",[game["away_abbr"],game["home_abbr"]],key="v23_extra_side")
            with m3: extra_odds=st.number_input("American moneyline",value=100,step=5,key="v23_extra_odds")
            if st.button("Compare this price",key="v23_compare_extra"):
                raw=american_to_implied(extra_odds); ph,pe,_=current_league_model_probability(game,league); mp=(ph if extra_side==game["home_abbr"] else 1-ph) if not pd.isna(ph) else np.nan
                c1,c2,c3=st.columns(3)
                c1.metric("Market implied",fmt_pct(raw))
                c2.metric("Potential profit on $100",fmt_money(american_profit_per_100_v23(extra_odds)))
                c3.metric("Model edge",("—" if pd.isna(mp) or pd.isna(raw) else f"{(mp-raw)*100:+.1f} pts"))
                if pe: st.caption("Model message: "+str(pe))

        settings=load_json(
            SETTINGS_FILE,
            {"bankroll":1000.0,"kelly_fraction":.25,"max_pct":.02}
        )
        settings.setdefault("bankroll",1000.0)
        settings.setdefault("kelly_fraction",.25)
        settings.setdefault("max_pct",.02)

        r1,r2=st.columns([1,1])
        with r1:
            if st.button("🔄 Refresh + evaluate ALL matched markets",type="primary",use_container_width=True):
                st.cache_data.clear()
                with st.spinner("Refreshing winner, spread, total, orderbook depth, models and fees..."):
                    rows,err,added=auto_evaluate_log_selected_game(game,league)
                st.session_state["v21_complete_market_rows"]=rows
                st.session_state["v21_complete_market_error"]=err
                st.session_state["v21_complete_market_logged"]=added
        with r2:
            st.caption(
                "Every matched priced winner/spread/total contract produced by this evaluator is logged automatically."
            )

        rows=st.session_state.get("v21_complete_market_rows")
        if rows is None:
            with st.spinner("Evaluating current matched markets..."):
                rows,err,added=auto_evaluate_log_selected_game(game,league)
            st.session_state["v21_complete_market_rows"]=rows
            st.session_state["v21_complete_market_error"]=err
            st.session_state["v21_complete_market_logged"]=added

        err=st.session_state.get("v21_complete_market_error")

        st.markdown("## 🚨 Supported edge alerts")
        if not render_edge_alerts(game,league,rows,st.session_state.get("matchup_context"),compact=False):
            st.info("No supported Kalshi edge currently clears the alert threshold with sportsbook confirmation.")

        st.markdown("## 🧮 Full Model & Market dashboard")
        st.caption(
            "Winner + spread + total markets. Prices use visible orderbook depth when available, "
            "then the current Kalshi general taker-fee formula is applied."
        )

        if rows:
            display=pd.DataFrame(rows).copy()

            for c in [
                "Kalshi price","Quoted Kalshi price","Model probability","Edge",
                "Historical accuracy","Historical Brier","Fee per contract"
            ]:
                if c in display.columns:
                    display[c]=display[c].map(
                        lambda x:"—" if pd.isna(x) else f"{float(x)*100:.1f}%"
                    )

            if "EV per contract" in display.columns:
                display["EV per contract"]=display["EV per contract"].map(
                    lambda x:"—" if pd.isna(x) else f"${float(x):+.3f}"
                )
            if "EV per $1" in display.columns:
                display["EV per $1"]=display["EV per $1"].map(
                    lambda x:"—" if pd.isna(x) else f"${float(x):+.3f}"
                )
            if "Suggested position" in display.columns:
                display["Suggested position"]=display["Suggested position"].map(fmt_money)
            if "Exact taker fee" in display.columns:
                display["Exact taker fee"]=display["Exact taker fee"].map(fmt_money)
            if "Projection" in display.columns:
                display["Projection"]=display["Projection"].map(
                    lambda x:"—" if pd.isna(x) else f"{float(x):.1f}"
                )

            cols=[
                "Game","Market Type","Market","Kalshi price","Model probability",
                "Edge","EV per contract","Historical accuracy","Projection",
                "Suggested position","Contracts","Exact taker fee","Fill source"
            ]
            st.dataframe(
                display[[c for c in cols if c in display.columns]],
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                f"Risk settings: bankroll {fmt_money(settings['bankroll'])} • "
                f"{settings['kelly_fraction']:.0%} Kelly fraction • "
                f"max {settings['max_pct']:.1%} bankroll per market."
            )

            st.markdown("### Simple interpretation")
            for r in sorted(rows,key=lambda x:x.get("Edge",-99),reverse=True)[:8]:
                edge=r.get("Edge",np.nan)
                if pd.isna(edge):
                    continue
                icon="✅" if edge>0 and r.get("EV per contract",0)>0 else "⚠️"
                st.write(
                    f"{icon} **{r['Market']}** — model {fmt_pct(r['Model probability'])}, "
                    f"fill {fmt_pct(r['Kalshi price'])}, edge {edge*100:+.1f} pts, "
                    f"EV {fmt_money(r.get('EV per contract'))}/contract."
                )
        else:
            st.warning(
                "The app did not finish a model-vs-market row, but it will still show any current Kalshi contracts it found below."
            )

            winner_inventory,werr=selected_game_kalshi_prices_v22(game,league)
            other_inventory=selected_game_other_kalshi_markets_v22(game,league)

            inv=[]
            for r in winner_inventory:
                inv.append({
                    "Type":"Winner",
                    "Market":f"{r['Team']} to win",
                    "Kalshi price":fmt_pct(r.get("Kalshi probability")),
                    "Ticker":r.get("Ticker",""),
                })
            for r in other_inventory:
                inv.append({
                    "Type":r.get("Type",""),
                    "Market":r.get("Market",""),
                    "Kalshi price":fmt_pct(r.get("YES probability")),
                    "Ticker":r.get("Ticker",""),
                })

            if inv:
                st.dataframe(
                    pd.DataFrame(inv),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption(
                    "The market exists. A missing model probability is a model/data problem, not 'no market'. "
                    "Open Source/model messages below to see which modeling source failed."
                )
            else:
                st.error(
                    "No matching Kalshi contracts were returned after direct date + both-team matching."
                )
                if werr:
                    st.caption(str(werr))

        if err:
            with st.expander("Source/model messages"):
                st.write(err)

        st.markdown("## 💵 Fee + fill method")
        st.write(
            "For an immediate YES purchase, the app reads visible Kalshi orderbook depth when available, "
            "walks the asks to estimate the average fill, then applies the general taker-fee formula "
            "**ceil(0.07 × contracts × price × (1 − price))**. "
            "If depth is unavailable, it labels the result as a top-quote fallback."
        )
        st.caption(
            f"Fee schedule basis: general Kalshi event-contract taker formula, effective schedule tracked in-app as "
            f"{KALSHI_FEE_SCHEDULE_EFFECTIVE}. Market-specific future fee changes still need the app refreshed."
        )


# ============================================================
# MY BETS
# ============================================================

if page=="🎟️ My Bets":
    st.subheader("🎟️ My Bets / Research Tracker")

    watch=load_csv(
        WATCHLIST_FILE
    )

    if watch.empty:
        st.info(
            "Nothing saved yet."
        )
    else:
        summary=tracked_bet_history_summary(
            watch
        )

        a,b,c,d=st.columns(4)
        a.metric(
            "Tracked",
            summary.get("rows",0),
        )
        b.metric(
            "Resolved",
            summary.get("resolved",0),
        )
        c.metric(
            "Win rate",
            fmt_pct(
                summary.get("win_rate")
            ),
        )
        d.metric(
            "Realized P/L",
            fmt_money(
                summary.get("pnl")
            ),
        )

        edited=st.data_editor(
            watch,
            num_rows="dynamic",
            use_container_width=True,
        )
        if st.button(
            "Save tracker changes"
        ):
            edited.to_csv(
                WATCHLIST_FILE,
                index=False,
            )
            st.success("Saved.")


    st.divider()
    st.markdown("## 📈 Model performance / historical edge tracking")
    if st.button("Settle finished winner / spread / total predictions",key="v21_settle_market_history"):
        with st.spinner("Checking finished games..."):
            hist,updated=settle_all_market_prediction_history()
        st.success(f"Updated {updated} finished market prediction(s).")

    hist=load_csv(MODEL_MARKET_HISTORY_FILE)
    if hist.empty:
        st.caption(
            "No model-vs-market snapshots have been logged yet. "
            "Open Edge Alerts for selected games; those model/market rows are recorded automatically."
        )
    else:
        settled=hist[
            hist.get("settled",pd.Series(dtype=str)).astype(str).str.lower().isin(["true","1","yes"])
        ].copy()
        a,b,c=st.columns(3)
        a.metric("Market snapshots logged",len(hist))
        b.metric("Settled",len(settled))
        pnl=pd.to_numeric(
            settled.get("profit_loss_100",pd.Series(dtype=float)),
            errors="coerce"
        ).dropna()
        c.metric("Tracked P/L per $100 stakes",fmt_money(pnl.sum()) if len(pnl) else "—")

        edge_perf=historical_edge_performance(hist)
        if not edge_perf.empty:
            st.markdown("### Historical results by model edge")
            show=edge_perf.copy()
            show["avg_edge"]=show["avg_edge"].map(fmt_pct)
            show["win_rate"]=show["win_rate"].map(fmt_pct)
            show["profit_per_100"]=show["profit_per_100"].map(fmt_money)
            show["total_profit"]=show["total_profit"].map(fmt_money)
            st.dataframe(show,use_container_width=True,hide_index=True)
            st.caption(
                "This is the app's own tracked history. Small samples can be misleading; "
                "the table becomes more meaningful after many resolved markets."
            )

# ============================================================
# DATA ADMIN — PERSISTENT PLAYER DATABASE / MANUAL BACKUP
# ============================================================

if page=="🔧 Data Admin":
    st.subheader("🔧 Data Admin")
    st.write("Bundled seed files provide the historical foundation. Automatic feeds add newer rows; manual imports remain the backup when a source misses data.")
    _ball=load_basketball_seed_v33(); _np=load_nfl_player_seed_v33(); _ns=load_nfl_snaps_seed_v33(); _nr=load_nfl_rosters_seed_v33(); _ng=load_nfl_games_seed_v33()
    st.markdown("## Bundled data foundation")
    seed_status=pd.DataFrame([
        {"Dataset":"NBA/WNBA player games","Rows":len(_ball),"Status":"✅" if not _ball.empty else "❌"},
        {"Dataset":"NFL player weeks","Rows":len(_np),"Status":"✅" if not _np.empty else "❌"},
        {"Dataset":"NFL snap counts","Rows":len(_ns),"Status":"✅" if not _ns.empty else "❌"},
        {"Dataset":"NFL weekly rosters","Rows":len(_nr),"Status":"✅" if not _nr.empty else "❌"},
        {"Dataset":"NFL schedules/games","Rows":len(_ng),"Status":"✅" if not _ng.empty else "❌"},
    ])
    st.dataframe(seed_status,use_container_width=True,hide_index=True)
    st.caption("NFL play-by-play Parquet files are bundled separately for deeper EPA/success-rate processing when pyarrow is installed.")
    db=load_csv(PLAYER_LOG_DB_FILE)
    a,b,c=st.columns(3)
    a.metric("Saved game rows",len(db))
    b.metric("Players",0 if db.empty or "PLAYER_KEY" not in db.columns else db["PLAYER_KEY"].nunique())
    c.metric("Leagues",0 if db.empty or "LEAGUE" not in db.columns else db["LEAGUE"].nunique())
    if not db.empty:
        showcols=[c for c in ["LEAGUE","PLAYER","DATE","TEAM","OPP","PTS","REB","AST","MIN","SOURCE_DB","INGESTED_AT"] if c in db.columns]
        st.dataframe(db.sort_values("DATE",ascending=False)[showcols].head(250),use_container_width=True,hide_index=True)

    st.markdown("## Refresh automatic player logs")
    lg=st.selectbox("League to refresh",["NBA","WNBA","NFL"],key="v32_admin_league")
    if st.button("Refresh every current roster player in this league",key="v32_refresh_league",type="primary"):
        idxdf,err=league_roster_index_v23(lg)
        if idxdf.empty:
            st.error(err or "League roster could not be loaded.")
        else:
            prog=st.progress(0.0); added_total=0; players_ok=0; messages=[]
            for i,(_,r) in enumerate(idxdf.iterrows(),1):
                try:
                    n,added,msg=refresh_player_history_database_v32(r,lg,date.today())
                    added_total+=added
                    if n: players_ok+=1
                    if msg: messages.append(f"{r.get('player','')}: {msg}")
                except Exception as e:
                    messages.append(f"{r.get('player','')}: {e}")
                prog.progress(i/max(len(idxdf),1))
            st.success(f"Refreshed logs for {players_ok}/{len(idxdf)} roster players; {added_total} new game rows were added.")
            if messages:
                with st.expander("Refresh messages"):
                    for m in messages[:100]: st.write("• "+m)

    game=st.session_state.get("matchup"); league_now=st.session_state.get("league"); roster_now=st.session_state.get("roster",pd.DataFrame())
    if game and not roster_now.empty:
        if st.button("Refresh every player in the selected game",key="v32_refresh_game"):
            prog=st.progress(0.0); added_total=0
            for i,(_,r) in enumerate(roster_now.iterrows(),1):
                _,added,_=refresh_player_history_database_v32(r,league_now,st.session_state.get("selected_date",date.today()))
                added_total+=added; prog.progress(i/max(len(roster_now),1))
            st.success(f"Selected-game refresh complete. {added_total} new game rows were added.")

    st.markdown("## Import backup game logs")
    st.caption("Best format: one row per player-game with league, player, date, team/opponent and whatever stat columns you have. Common PTS/REB/AST/MIN aliases are normalized automatically.")
    _template=pd.DataFrame(columns=["LEAGUE","PLAYER","DATE","SEASON","TEAM","OPP","MATCHUP","MIN","PTS","REB","AST","FG3M","FG3A","STL","BLK","TOV"])
    st.download_button("Download player-log CSV template",_template.to_csv(index=False),file_name="player_game_log_template.csv",mime="text/csv",key="v32_template_download")
    if not db.empty:
        st.download_button("Download current player database backup",db.to_csv(index=False),file_name="player_game_logs_backup.csv",mime="text/csv",key="v32_db_download")
    default_lg=st.selectbox("Use this league if the file has no league column",["NBA","WNBA","NFL"],key="v32_import_league")
    up=st.file_uploader("CSV file",type=["csv"],key="v32_player_log_upload")
    if up is not None:
        try:
            raw=pd.read_csv(up)
            st.dataframe(raw.head(20),use_container_width=True,hide_index=True)
            has_league=_first_existing_col(raw,["LEAGUE","league","sport"]) is not None
            if st.button("Merge this file into the player database",key="v32_merge_upload"):
                added=upsert_player_log_database(raw,league=None if has_league else default_lg,source="manual CSV import")
                st.success(f"Import complete. {added} new player-game rows were added; duplicates were ignored/updated.")
        except Exception as e:
            st.error("Could not read/import the CSV: "+str(e))

    st.markdown("## Database freshness")
    updates=load_csv(DATA_UPDATE_LOG_FILE)
    if not updates.empty:
        st.dataframe(updates.tail(100).iloc[::-1],use_container_width=True,hide_index=True)
    else:
        st.caption("No update log yet.")

# ============================================================
# ADVANCED — KEEP THE DETAIL, HIDE THE CLUTTER
# ============================================================

if page=="🧪 Advanced":
    st.subheader("🧪 Advanced")
    st.write(
        "The detailed/raw tools from the older version are still here, but they are tucked into expanders so the main research pages stay readable."
    )


    st.markdown("## ✅ Full model/research checklist")
    coverage=pd.DataFrame(
        full_market_checklist_rows(),
        columns=["Requirement","Implemented","How the app covers it"],
    )
    st.dataframe(
        coverage,
        use_container_width=True,
        hide_index=True,
    )
    st.caption("Implementation status is not the same as live verification. The selected game's source status is shown below.")
    _g=st.session_state.get("matchup"); _c=st.session_state.get("matchup_context")
    if _g and _c:
        st.markdown("### 🔍 Selected-game reliability check")
        st.dataframe(
            v25_live_reliability_table(_g,_c),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Implemented means the feature exists. Verified means the selected game's structured/live data actually loaded."
        )

    st.info(
        "V22 current-game coverage: if final starters are not exposed, the app shows a clearly labeled POSSIBLE lineup from recent starts. "
        "If a model row fails, it still shows matched Kalshi inventory and the exact failure reason instead of hiding the market."
    )

    with st.expander("Bankroll / risk / fee settings"):
        settings=load_json(
            SETTINGS_FILE,
            {
                "bankroll":1000.0,
                "kelly_fraction":.25,
                "max_pct":.02,
                "kalshi_fee_per_contract":.01,
            },
        )
        bank=st.number_input(
            "Research bankroll",
            min_value=0.0,
            value=float(settings.get("bankroll",1000.0)),
            step=100.0,
        )
        kelly_frac=st.slider(
            "Kelly fraction",
            0.05,0.50,
            float(settings.get("kelly_fraction",.25)),
            .05,
        )
        max_pct=st.slider(
            "Maximum bankroll percentage for one idea",
            0.005,0.10,
            float(settings.get("max_pct",.02)),
            .005,
            format="%.1f%%",
        )
        fee=st.number_input(
            "Fee/slippage allowance per $1 contract",
            min_value=0.0,
            max_value=.20,
            value=float(settings.get("kalshi_fee_per_contract",.01)),
            step=.005,
            format="%.3f",
        )
        edge_alert_min=st.slider(
            "Minimum model edge for an alert",
            0.01,0.20,float(settings.get("edge_alert_min",.05)),0.01,format="%.0f%%"
        )
        edge_alert_min_books=st.slider(
            "Minimum sportsbook sources confirming a winner alert",
            1,5,int(settings.get("edge_alert_min_books",2)),1
        )
        if st.button("Save risk settings",key="v21_save_risk"):
            save_json(
                SETTINGS_FILE,
                {
                    "bankroll":bank,
                    "kelly_fraction":kelly_frac,
                    "max_pct":max_pct,
                    "kalshi_fee_per_contract":fee,
                    "edge_alert_min":edge_alert_min,
                    "edge_alert_min_books":edge_alert_min_books,
                },
            )
            st.success("Risk settings saved.")

    with st.expander("Spread / total historical models — held-out validation"):
        st.write(
            "Spread and total probabilities are built from separate historical point-margin / total-score models. "
            "A chronological calibration period supplies empirical residuals; the final test period stays unseen."
        )

        game_here=st.session_state.get("matchup")
        lg_here=st.session_state.get("league")
        if game_here and lg_here:
            if st.button("Run selected-game spread / total validation",key="v28_run_score_model"):
                try:
                    with st.spinner("Loading selected league score-model validation..."):
                        proj,score_err=current_score_projection(game_here,lg_here)
                    st.session_state["v28_score_model_result"]=(proj,score_err)
                except Exception as e:
                    st.session_state["v28_score_model_result"]=(None,str(e))

            proj,score_err=st.session_state.get(
                "v28_score_model_result",(None,None)
            )
            if score_err:
                st.warning(score_err)
            elif proj:
                sp=proj["score_pack"]
                a,b,c,d=st.columns(4)
                a.metric("Projected margin",f"{proj['home_margin']:+.1f} home")
                b.metric("Projected total",f"{proj['game_total']:.1f}")
                c.metric("Margin test MAE",f"{sp['margin_metrics']['mae']:.2f}")
                d.metric("Total test MAE",f"{sp['total_metrics']['mae']:.2f}")
                st.caption(
                    f"Historical span: {sp.get('first_date','—')} → {sp.get('last_date','—')} • "
                    f"margin test N {sp['margin_metrics']['n']} • total test N {sp['total_metrics']['n']}."
                )
            else:
                st.caption("Press the button to build this historical model.")
        else:
            st.caption("Choose a game first to see the selected league's spread/total model.")

    with st.expander("Separate NFL model — EPA / football features"):
        season=league_season_year("NFL",st.session_state.get("selected_date",date.today()))
        if st.button("Build / refresh NFL historical model",key="v21_build_nfl"):
            try:
                st.cache_resource.clear()
                with st.spinner("Loading NFL model status..."):
                    nfl_pack,nfl_test,nfl_err=build_nfl_model(season)
                st.session_state["v28_nfl_model_result"]=(nfl_pack,nfl_err)
            except Exception as e:
                st.session_state["v28_nfl_model_result"]=(None,str(e))

        nfl_pack,nfl_err=st.session_state.get(
            "v28_nfl_model_result",(None,None)
        )
        if nfl_err:
            st.warning(nfl_err)
        elif nfl_pack:
            met=nfl_pack["metrics"]
            a,b,c,d=st.columns(4)
            a.metric("Held-out accuracy",fmt_pct(met["accuracy"]))
            b.metric("Brier",f"{met['brier']:.3f}")
            c.metric("Log loss",f"{met['logloss']:.3f}")
            d.metric("Held-out games",met["test_n"])
            st.write(
                "**NFL features:** offensive EPA, passing EPA, rushing EPA, defensive EPA allowed, "
                "pass/rush EPA allowed, passing/rushing yards, explosive plays, rest, temperature, and wind."
            )
            st.caption(
                f"Historical span: {met['first_date']} → {met['last_date']} • "
                f"train {met['train_n']} • calibration {met['cal_n']} • test {met['test_n']}."
            )
        else:
            st.caption("This model is not built until you press the button.")

    with st.expander("WNBA historical model / calibration"):
        season=league_season_year("WNBA",st.session_state.get("selected_date",date.today()))
        if st.button("Build / refresh WNBA historical model",key="v21_build_wnba"):
            try:
                st.cache_resource.clear()
                with st.spinner("Loading WNBA model status..."):
                    wpack,wtest,werr=build_espn_basketball_model("WNBA",season,5)
                st.session_state["v28_wnba_model_result"]=(wpack,werr)
            except Exception as e:
                st.session_state["v28_wnba_model_result"]=(None,str(e))

        wpack,werr=st.session_state.get(
            "v28_wnba_model_result",(None,None)
        )
        if werr:
            st.warning(werr)
        elif wpack:
            met=wpack["metrics"]
            a,b,c,d=st.columns(4)
            a.metric("Held-out accuracy",fmt_pct(met["accuracy"]))
            b.metric("Brier",f"{met['brier']:.3f}")
            c.metric("Log loss",f"{met['logloss']:.3f}")
            d.metric("Held-out games",met["test_n"])
            st.caption(
                f"Historical span: {met['first_date']} → {met['last_date']} • seasons {met['seasons']}."
            )
        else:
            st.caption("This model is not built until you press the button.")


    game=st.session_state.get("matchup")
    league=st.session_state.get("league")
    roster=st.session_state.get(
        "roster",pd.DataFrame()
    )
    ctx=st.session_state.get(
        "matchup_context"
    )

    with st.expander(
        "Full raw matchup breakdown"
    ):
        if game and ctx:
            render_matchup_breakdown_v11(
                game,league,ctx
            )
        else:
            st.info(
                "Choose a game first."
            )

    with st.expander(
        "Full game-scan results / diagnostics"
    ):
        ideas=st.session_state.get(
            "full_game_ideas",[]
        )
        diagnostics=st.session_state.get(
            "full_game_scan_diagnostics",[]
        )
        if ideas:
            simple=[]
            for r in ideas:
                simple.append({
                    "Player":r.get("Player"),
                    "Team":r.get("Team"),
                    "Idea":(
                        f"{r.get('Line','')} "
                        f"{r.get('Stat','')}"
                    ),
                    "Type":r.get("Volatility"),
                    "Probability":fmt_pct(
                        r.get(
                            "Model probability"
                        )
                    ),
                    "Last 10":fmt_pct(
                        r.get("Last 10")
                    ),
                    "Vs opponent":fmt_pct(
                        r.get("Vs opponent")
                    ),
                    "Data quality":r.get(
                        "Data quality"
                    ),
                    "Backtest N":r.get(
                        "Backtest N"
                    ),
                })
            st.dataframe(
                pd.DataFrame(simple),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption(
                "Run Find a Bet first."
            )
        if diagnostics:
            st.markdown(
                "### Players checked with no promoted idea / source issue"
            )
            st.dataframe(
                pd.DataFrame(diagnostics),
                use_container_width=True,
                hide_index=True,
            )

    with st.expander(
        "Historical NBA model / backtesting / calibration"
    ):
        st.write(
            "Older games train the historical model; later unseen games are used to test it."
        )
        if NBA_HIST_FILE.exists():
            if st.button("Build / refresh NBA historical model",key="v28_build_nba"):
                try:
                    pack,test,err=build_nba_model(
                        NBA_HIST_FILE.stat().st_mtime
                    )
                    st.session_state["v28_nba_model_result"]=(pack,test,err)
                except Exception as e:
                    st.session_state["v28_nba_model_result"]=(None,None,str(e))

            pack,test,err=st.session_state.get(
                "v28_nba_model_result",(None,None,None)
            )
            if err:
                st.error(err)
            elif pack is not None and test is not None:
                _,_,metrics=pack
                a,b,c=st.columns(3)
                a.metric(
                    "Held-out accuracy",
                    fmt_pct(
                        metrics["accuracy"]
                    ),
                )
                b.metric(
                    "Brier score",
                    f"{metrics['brier']:.3f}",
                )
                c.metric(
                    "Log loss",
                    f"{metrics['logloss']:.3f}",
                )

                frac,mean=calibration_curve(
                    test["home_won"],
                    test["model_p"],
                    n_bins=8,
                    strategy="quantile",
                )
                cal=pd.DataFrame({
                    "Predicted probability":mean,
                    "Actual win rate":frac,
                })
                st.line_chart(
                    cal.set_index(
                        "Predicted probability"
                    )
                )
            else:
                st.caption("This model is not built until you press the button.")
        else:
            st.caption(
                "Historical NBA game.csv is not present in data/."
            )

    with st.expander("📚 Sources / research integrity"):
        st.dataframe(source_table_v23(),use_container_width=True,hide_index=True)
        st.markdown("**Source rule:** missing source data is labeled missing/unverified; zero, healthy, confirmed, or a matchup weakness is never inferred from an empty response.")
        game_src=st.session_state.get("matchup"); league_src=st.session_state.get("league")
        if game_src and league_src:
            st.markdown("**Selected-game verification links**")
            st.markdown(f"- [ESPN game page]({espn_game_page_url(league_src,game_src['event_id'])})")
            st.markdown(f"- [Official league injury report]({official_injury_report_url(league_src)})")
            st.markdown("- [Kalshi API documentation](https://docs.kalshi.com/)")
            st.markdown("- [Kalshi fee schedule](https://kalshi.com/regulatory/fee-schedule)")
        st.info("Core player/game research comes from structured league/ESPN/nflverse feeds plus the persistent local database. Optional public-web research is used only as corroboration for current market/news context.")

    with st.expander(
        "All-sports Kalshi discrepancy scanner"
    ):
        st.caption(
            "This is the technical scanner. The normal selected-game view is in Edge Alerts."
        )
        scan_leagues=st.multiselect(
            "Sports",
            ["NBA","WNBA","NFL"],
            default=["NBA","WNBA","NFL"],
            key="v20_scan_leagues",
        )
        min_books=st.slider(
            "Minimum outside sportsbooks",
            1,5,2,
            key="v20_min_books",
        )
        gap=st.slider(
            "Minimum difference to flag",
            2,15,5,
            key="v20_gap",
        )
        buffer=st.slider(
            "Safety margin",
            0,6,2,
            key="v20_buffer",
        )
        if st.button(
            "Run Kalshi scanner",
            key="v20_run_kalshi",
        ):
            with st.spinner(
                "Comparing Kalshi with outside prices..."
            ):
                results,inventory,internal,errors=scan_kalshi_sports_markets(
                    scan_leagues,
                    discrepancy_threshold=gap/100,
                    fee_buffer=buffer/100,
                    min_books=min_books,
                )
            st.session_state[
                "v20_scan_results"
            ]=results
            st.session_state[
                "v20_scan_errors"
            ]=errors

        results=st.session_state.get(
            "v20_scan_results",[]
        )
        if results:
            out=[]
            for r in results[:30]:
                out.append({
                    "League":r["League"],
                    "Game":r["Game"],
                    "Outcome":r["Outcome"],
                    "Kalshi":fmt_pct(
                        r["Kalshi ask"]
                    ),
                    "Outside":fmt_pct(
                        r["Consensus probability"]
                    ),
                    "Difference":(
                        f"{r['Adjusted gap']*100:+.1f} pts"
                    ),
                    "Flag":r["Flag"],
                })
            st.dataframe(
                pd.DataFrame(out),
                use_container_width=True,
                hide_index=True,
            )
        elif "v20_scan_results" in st.session_state:
            st.info(
                "No comparison passed the current scan settings."
            )

