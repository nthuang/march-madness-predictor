import numpy as np
import pandas as pd

#load data
data_dir = "mmlm2026"

M_regular_results = pd.read_csv(f"{data_dir}/MRegularSeasonDetailedResults.csv")
M_tourney_results = pd.read_csv(f"{data_dir}/MNCAATourneyCompactResults.csv")
M_seeds = pd.read_csv(f"{data_dir}/MNCAATourneySeeds.csv")
M_massey = pd.read_csv(f"{data_dir}/MMasseyOrdinals.csv")

W_regular_results = pd.read_csv(f"{data_dir}/WRegularSeasonDetailedResults.csv")
W_tourney_results = pd.read_csv(f"{data_dir}/WNCAATourneyCompactResults.csv")
W_seeds = pd.read_csv(f"{data_dir}/WNCAATourneySeeds.csv")

SAMPLE_PATH = f"{data_dir}/SampleSubmissionStage2.csv"

def tourney_labels(tourney_results):
    df = tourney_results[["Season", "WTeamID", "WScore", "LTeamID", "LScore"]].copy()
    df["Team1"] = df[["WTeamID", "LTeamID"]].min(axis=1)
    df["Team2"] = df[["WTeamID", "LTeamID"]].max(axis=1)
    df["w"] = (df["Team1"] == df["WTeamID"]).astype(int)
    df["margin"] = np.where(
        df["Team1"] == df["WTeamID"],
        df["WScore"] - df["LScore"],
        df["LScore"] - df["WScore"],
    ).astype(np.int16)
    return df[["Season", "Team1", "Team2", "w", "margin"]]

def seed_table(seeds):
    df = seeds[["Season", "TeamID", "Seed"]].copy()
    df["SeedN"] = df["Seed"].str[1:3].astype(int)
    return df[["Season", "TeamID", "SeedN"]]

def massey(massey_df, day_cap=133):
    m = massey_df[massey_df["RankingDayNum"] <= day_cap].copy()
    last_day = m.groupby("Season")["RankingDayNum"].max().reset_index(name="UseDay")
    m = m.merge(last_day, on="Season", how="inner")
    m = m[m["RankingDayNum"] == m["UseDay"]]

    out = (
        m.groupby(["Season", "TeamID"])["OrdinalRank"]
          .agg(
              MasseyMean="mean",
              MasseyMedian="median",
              MasseyMin="min",
              MasseyMax="max",
              MasseyStd="std",
              MasseyCount="count",
          )
          .reset_index()
    )
    return out

def team_game_table(regular_season):
    cols = ["Season", "DayNum", "WTeamID", "WScore", "LTeamID", "LScore", 
        "WLoc", "NumOT", "WFGM", "WFGA", "WFGM3", "WFGA3", "WFTM", "WFTA", 
        "WOR", "WDR", "WAst", "WTO", "WStl", "WBlk", "WPF", "LFGM", "LFGA", 
        "LFGM3", "LFGA3", "LFTM", "LFTA", "LOR", "LDR" ,"LAst", "LTO", 
        "LStl", "LBlk", "LPF"]
    df = regular_season[cols].copy()

    w = df.rename(columns={
        "Season": "Season",
        "DayNum": "DayNum",
        "WLoc": "Loc",
        "NumOT": "NumOT",

        "WTeamID": "TeamID",
        "LTeamID": "OppID",
        "WScore": "Score",
        "LScore": "OppScore",

        "WFGM": "FGM",
        "WFGA": "FGA",
        "WFGM3": "FGM3",
        "WFGA3": "FGA3",
        "WFTM": "FTM",
        "WFTA": "FTA",
        "WOR": "OR",
        "WDR": "DR",
        "WAst": "Ast",
        "WTO": "TO",
        "WStl": "Stl",
        "WBlk": "Blk",
        "WPF": "PF",

        "LFGM": "OppFGM",
        "LFGA": "OppFGA",
        "LFGM3": "OppFGM3",
        "LFGA3": "OppFGA3",
        "LFTM": "OppFTM",
        "LFTA": "OppFTA",
        "LOR": "OppOR",
        "LDR": "OppDR",
        "LAst": "OppAst",
        "LTO": "OppTO",
        "LStl": "OppStl",
        "LBlk": "OppBlk",
        "LPF": "OppPF",
    })
    w["Win"] = 1

    l = df.rename(columns={
        "Season": "Season",
        "DayNum": "DayNum",
        "WLoc": "Loc",     
        "NumOT": "NumOT",

        "LTeamID": "TeamID",
        "WTeamID": "OppID",
        "LScore": "Score",
        "WScore": "OppScore",

        "LFGM": "FGM",
        "LFGA": "FGA",
        "LFGM3": "FGM3",
        "LFGA3": "FGA3",
        "LFTM": "FTM",
        "LFTA": "FTA",
        "LOR": "OR",
        "LDR": "DR",
        "LAst": "Ast",
        "LTO": "TO",
        "LStl": "Stl",
        "LBlk": "Blk",
        "LPF": "PF",

        "WFGM": "OppFGM",
        "WFGA": "OppFGA",
        "WFGM3": "OppFGM3",
        "WFGA3": "OppFGA3",
        "WFTM": "OppFTM",
        "WFTA": "OppFTA",
        "WOR": "OppOR",
        "WDR": "OppDR",
        "WAst": "OppAst",
        "WTO": "OppTO",
        "WStl": "OppStl",
        "WBlk": "OppBlk",
        "WPF": "OppPF",
    })
    l["Win"] = 0

    flip = {"H": "A", "A": "H", "N": "N"}
    l["Loc"] = l["Loc"].map(flip)
    
    loc_map = {"H": 1, "N": 0, "A": -1}
    w["Loc"] = w["Loc"].map(loc_map)
    l["Loc"] = l["Loc"].map(loc_map)

    tg = pd.concat([w, l], ignore_index=True)

    tg["PtDif"] = tg["Score"] - tg["OppScore"]

    tg["Poss"] = tg["FGA"] - tg["OR"] + tg["TO"] + .44*tg["FTA"]
    tg["OppPoss"] = tg["OppFGA"] - tg["OppOR"] + tg["OppTO"] + 0.44*tg["OppFTA"]

    tg["ppp"] = tg["Score"] / tg["Poss"]
    tg["opp_ppp"] = tg["OppScore"] / tg["OppPoss"]
    tg["ORTG"] = 100*tg["ppp"]
    tg["DRTG"] = 100*tg["opp_ppp"] 
    tg["NetRtg"] = tg["ORTG"] - tg["DRTG"]

    tg["eFG"] = (tg["FGM"] + .5*tg["FGM3"])/tg["FGA"]
    tg["OppeFG"] = (tg["OppFGM"] + .5*tg["OppFGM3"])/tg["OppFGA"]
    tg["TS"] = tg["Score"] / (2 * (tg["FGA"] + 0.44 * tg["FTA"]))
    tg["OppTS"] = tg["OppScore"] / (2 * (tg["OppFGA"] + 0.44 * tg["OppFTA"]))
    
    tg["3PAr"] = tg["FGA3"] / tg["FGA"]
    tg["Opp3PAr"] = tg["OppFGA3"] / tg["OppFGA"]
    tg["FTr"] = tg["FTA"] / tg["FGA"]
    tg["OppFTr"] = tg["OppFTA"] / tg["OppFGA"]

    return tg

def team_season_table(team_game_table):
    ts = team_game_table.groupby(["Season", "TeamID"]).agg(
        NumGames = ("DayNum", "count"),
        WinPct = ("Win", "mean"),

        PtsSeason = ("Score", "sum"),
        OppPtsSeason = ("OppScore", "sum"),

        PossSeason = ("Poss", "sum"),
        OppPossSeason = ("OppPoss", "sum"),

        FGM=("FGM", "sum"),
        FGA=("FGA", "sum"),
        FGM3=("FGM3", "sum"),
        FGA3=("FGA3", "sum"),
        FTM=("FTM", "sum"),
        FTA=("FTA", "sum"),
        OR=("OR", "sum"),
        DR=("DR", "sum"),
        Ast=("Ast", "sum"),
        TO=("TO", "sum"),
        Stl=("Stl", "sum"),
        Blk=("Blk", "sum"),
        PF=("PF", "sum"),

        OppFGM=("OppFGM", "sum"),
        OppFGA=("OppFGA", "sum"),
        OppFGM3=("OppFGM3", "sum"),
        OppFGA3=("OppFGA3", "sum"),
        OppFTM=("OppFTM", "sum"),
        OppFTA=("OppFTA", "sum"),
        OppOR=("OppOR", "sum"),
        OppDR=("OppDR", "sum"),
        OppAst=("OppAst", "sum"),
        OppTO=("OppTO", "sum"),
        OppStl=("OppStl", "sum"),
        OppBlk=("OppBlk", "sum"),
        OppPF=("OppPF", "sum"),

        PtDiffMean=("PtDif", "mean"),
    ).reset_index()

    ts["PtsPerGame"] = ts["PtsSeason"] / ts["NumGames"]
    ts["OppPtsPerGame"] = ts["OppPtsSeason"] / ts["NumGames"]
    ts["AvgPtDiff"] = ts["PtsPerGame"] - ts["OppPtsPerGame"]
    ts["Pace"] = (ts["PossSeason"] + ts["OppPossSeason"]) / (2 * ts["NumGames"])

    ts["ORTG"] = 100 * ts["PtsSeason"] /ts["PossSeason"]
    ts["DRTG"] = 100 * ts["OppPtsSeason"] / ts["OppPossSeason"]
    ts["NetRtg"] = ts["ORTG"] - ts["DRTG"]

    ts["eFG_off"] = (ts["FGM"] + 0.5 * ts["FGM3"]) / ts["FGA"]
    ts["eFG_def"] = (ts["OppFGM"] + 0.5 * ts["OppFGM3"]) / ts["OppFGA"]

    ts["TS_off"] = ts["PtsSeason"] / (2 * (ts["FGA"] + 0.44 * ts["FTA"]))
    ts["TS_def"] = ts["OppPtsSeason"] / (2 * (ts["OppFGA"] + 0.44 * ts["OppFTA"]))

    ts["3PAr_off"] = ts["FGA3"] / ts["FGA"]
    ts["3PAr_def"] = ts["OppFGA3"] / ts["OppFGA"]

    ts["FTr_off"] = ts["FTA"] / ts["FGA"]
    ts["FTr_def"] = ts["OppFTA"] / ts["OppFGA"]

    drop_totals = [
    "PtsSeason","OppPtsSeason","PossSeason","OppPossSeason",
    "FGM","FGA","FGM3","FGA3","FTM","FTA","OR","DR","Ast","TO","Stl","Blk","PF",
    "OppFGM","OppFGA","OppFGM3","OppFGA3","OppFTM","OppFTA","OppOR","OppDR",
    "OppAst","OppTO","OppStl","OppBlk","OppPF",
]
    ts = ts.drop(columns=[c for c in drop_totals if c in ts.columns])

    return ts

def addprefix(df, prefix):
    exclude = ["Season", "TeamID"]
    rename = {c: f"{prefix}{c}" for c in df.columns if c not in exclude}
    return df.rename(columns=rename)



def swap_augment(df):
    swapped = df.copy()

    swapped[["Team1", "Team2"]] = swapped[["Team2", "Team1"]].values

    swapped["w"] = 1 - swapped["w"]

    if "margin" in swapped.columns:
        swapped["margin"] = -swapped["margin"]

    t1_cols = [c for c in df.columns if c.startswith("T1_")]
    for c1 in t1_cols:
        c2 = c1.replace("T1_", "T2_", 1)
        if c2 in swapped.columns:
            swapped[c1], swapped[c2] = df[c2].values, df[c1].values

    diff_cols = [c for c in df.columns if c.startswith("Diff_")]
    swapped[diff_cols] = -swapped[diff_cols]

    return pd.concat([df, swapped], ignore_index=True)

def parse_sample_submission(sample_path, season=2026):
    sub = pd.read_csv(sample_path)
    parts = sub["ID"].str.split("_", expand=True)
    sub["Season"] = parts[0].astype(int)
    sub["Team1"] = parts[1].astype(int) 
    sub["Team2"] = parts[2].astype(int)  
    return sub[sub["Season"] == season][["ID", "Season", "Team1", "Team2"]].reset_index(drop=True)

def build_matchup_features(sample_df, team_feats, feature_cols):
    t1 = addprefix(team_feats, "T1_")
    t2 = addprefix(team_feats, "T2_")

    df = (sample_df
          .merge(t1, left_on=["Season","Team1"], right_on=["Season","TeamID"], how="left")
          .drop(columns=["TeamID"])
          .merge(t2, left_on=["Season","Team2"], right_on=["Season","TeamID"], how="left")
          .drop(columns=["TeamID"])
    )

    for c1 in [c for c in df.columns if c.startswith("T1_")]:
        c2 = c1.replace("T1_", "T2_", 1)
        if c2 in df.columns and pd.api.types.is_numeric_dtype(df[c1]) and pd.api.types.is_numeric_dtype(df[c2]):
            df[c1.replace("T1_", "Diff_", 1)] = df[c1] - df[c2]

    X = df.reindex(columns=feature_cols)

    for c in feature_cols:
        if c not in X.columns:
            X[c] = 0.0
    X = X[feature_cols]

    out = pd.concat([df[["ID"]], X], axis=1)
    return out

sample_2026 = parse_sample_submission(SAMPLE_PATH, season=2026)

def build_train_aug(
    prefix,
    regular_results,
    tourney_results,
    seeds,
    min_season,
    massey_ordinals=None,   
):
    regular_results = regular_results[regular_results["Season"] >= min_season].copy()
    tourney_results = tourney_results[tourney_results["Season"] >= min_season].copy()
    seeds = seeds[seeds["Season"] >= min_season].copy()

    labels_df = tourney_labels(tourney_results)
    seeds_df_local = seed_table(seeds)

    tg = team_game_table(regular_results)
    tg.to_csv(f"data/{prefix}_2026_team_game_table.csv", index=False)

    ts = team_season_table(tg)
    ts.to_csv(f"data/{prefix}_2026_team_season_table.csv", index=False)

    team_feats = ts.merge(seeds_df_local, on=["Season", "TeamID"], how="left")
    team_feats["SeedN"] = team_feats["SeedN"].fillna(99).astype(int)
    team_feats["HasSeed"] = team_feats["SeedN"].ne(99).astype(int)

    if massey_ordinals is not None:
        massey_ordinals = massey_ordinals[massey_ordinals["Season"] >= min_season].copy()
        massey_df_local = massey(massey_ordinals)
        team_feats = team_feats.merge(massey_df_local, on=["Season", "TeamID"], how="left")

        massey_cols = [c for c in team_feats.columns if c.startswith("Massey")]
        for c in massey_cols:
            team_feats[c] = team_feats.groupby("Season")[c].transform(lambda s: s.fillna(s.median()))
            team_feats[c] = team_feats[c].fillna(team_feats[c].median())

    for c in team_feats.columns:
        if c in ["Season", "TeamID"]:
            continue
        if pd.api.types.is_numeric_dtype(team_feats[c]):
            team_feats[c] = team_feats.groupby("Season")[c].transform(lambda s: s.fillna(s.median()))
            team_feats[c] = team_feats[c].fillna(team_feats[c].median())

    t1 = addprefix(team_feats, "T1_")
    t2 = addprefix(team_feats, "T2_")

    train_df_local = (
        labels_df
        .merge(t1, left_on=["Season","Team1"], right_on=["Season","TeamID"], how="left")
        .drop(columns=["TeamID"])
        .merge(t2, left_on=["Season","Team2"], right_on=["Season","TeamID"], how="left")
        .drop(columns=["TeamID"])
    )

    for c1 in [c for c in train_df_local.columns if c.startswith("T1_")]:
        c2 = c1.replace("T1_", "T2_", 1)
        if c2 in train_df_local.columns and pd.api.types.is_numeric_dtype(train_df_local[c1]) and pd.api.types.is_numeric_dtype(train_df_local[c2]):
            train_df_local[c1.replace("T1_", "Diff_", 1)] = train_df_local[c1] - train_df_local[c2]

    train_df_local.to_csv(f"data/{prefix}_2026_train_base.csv", index=False)

    train_aug_local = swap_augment(train_df_local)
    train_aug_local.to_csv(f"data/{prefix}_2026_train_aug.csv", index=False)

    print(f"\n===== {prefix}_TRAIN_AUG REPORT =====")
    print("Rows, Cols:", train_aug_local.shape)
    print("Season range:", train_aug_local["Season"].min(), "-", train_aug_local["Season"].max())
    print("Unique seasons:", train_aug_local["Season"].nunique())

    feature_cols = [c for c in train_aug_local.columns if c not in ["Season","Team1","Team2","w","margin"]]

    if prefix == "M":
        sample_part = sample_2026[sample_2026["Team1"] < 2000].copy()
    else:
        sample_part = sample_2026[sample_2026["Team1"] >= 3000].copy()

    team_feats_2026 = team_feats[team_feats["Season"] == 2026].copy()

    matchups_2026 = build_matchup_features(sample_part, team_feats_2026, feature_cols)
    matchups_2026.to_csv(f"data/{prefix}_2026_matchups_features.csv", index=False)

    print(f"Saved: data/{prefix}_2026_matchups_features.csv | rows={len(matchups_2026)} cols={matchups_2026.shape[1]}")

    return train_aug_local

#Build Files
M_MIN_SEASON = 2003
W_MIN_SEASON = 2010  

M_train_aug = build_train_aug(
    prefix="M",
    regular_results=M_regular_results,
    tourney_results=M_tourney_results,
    seeds=M_seeds,
    min_season=M_MIN_SEASON,
    massey_ordinals=M_massey,    
)

W_train_aug = build_train_aug(
    prefix="W",
    regular_results=W_regular_results,
    tourney_results=W_tourney_results,
    seeds=W_seeds,
    min_season=W_MIN_SEASON,
    massey_ordinals=None,       
)

print("\nSaved:")
print("data/M_2026_train_aug.csv")
print("data/W_2026_train_aug.csv")
