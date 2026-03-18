import pandas as pd

SEASON = 2026

M_submission = pd.read_csv("data/2026_submission.csv")
M_seeds = pd.read_csv("mmlm2026/MNCAATourneySeeds.csv")
M_slots = pd.read_csv("mmlm2026/MNCAATourneySlots.csv")
M_teams = pd.read_csv("mmlm2026/MTeams.csv")

id_to_name = dict(zip(M_teams["TeamID"], M_teams["TeamName"]))

def parse_id(id):
    s, t1, t2 = id.split("_")
    return int(s), int(t1), int(t2)

def build_prob_lookup(sub_df, season):
    sub = sub_df[sub_df["ID"].str.startswith(f"{season}_")].copy()
    probs = {}
    for rid, p in zip(sub["ID"].values, sub["Pred"].values):
        s, a, b = parse_id(rid)
        if s != season:
            continue
        a0, b0 = (a, b) if a < b else (b, a)
        p = float(p)
        probs[(a0, b0)] = p if (a < b) else (1.0 - p)
    return probs

def win_prob(team_a, team_b, probs):
    if team_a == team_b:
        return 0.5
    x, y = (team_a, team_b) if team_a < team_b else (team_b, team_a)
    p_min = probs.get((x, y))
    return p_min if team_a == x else (1.0 - p_min)

def round_number(slot: str) -> int:
    if slot.startswith("R") and len(slot) >= 2 and slot[1].isdigit():
        return int(slot[1])
    return 0

def most_likely_bracket(season: int, seeds_df, slots_df, sub_df):
    probs = build_prob_lookup(sub_df, season)

    seeds_season = seeds_df[seeds_df["Season"] == season].copy()
    seed_to_team = dict(zip(seeds_season["Seed"], seeds_season["TeamID"]))
    team_to_seed = dict(zip(seeds_season["TeamID"], seeds_season["Seed"]))

    slots = slots_df[slots_df["Season"] == season].copy()
    slots["Round"] = slots["Slot"].apply(round_number)
    slots = slots.sort_values(["Round", "Slot"]).reset_index(drop=True)

    slot_winner = {}
    rows = []

    def resolve(ref: str) -> int:
        if ref in seed_to_team:
            return int(seed_to_team[ref])
        if ref in slot_winner:
            return int(slot_winner[ref])
        raise KeyError(f"Cannot resolve '{ref}' (not a seed and no prior slot winner).")

    for _, r in slots.iterrows():
        slot = r["Slot"]
        rnd = int(r["Round"])
        a = resolve(r["StrongSeed"])
        b = resolve(r["WeakSeed"])

        p_a = win_prob(a, b, probs)
        winner = a if p_a >= 0.5 else b
        slot_winner[slot] = winner

        rows.append({
            "Season": season,
            "Round": rnd,
            "Slot": slot,
            "TeamA": a,
            "TeamB": b,
            "TeamASeed": team_to_seed.get(a, ""),
            "TeamBSeed": team_to_seed.get(b, ""),
            "P(TeamA wins)": p_a,
            "Winner": winner,
            "WinnerSeed": team_to_seed.get(winner, ""),
            "WinnerProb": max(p_a, 1.0 - p_a),
        })

    games = pd.DataFrame(rows)

    final_round = games["Round"].max()
    champion = int(games[games["Round"] == final_round].iloc[-1]["Winner"])

    return games, champion

games, champ = most_likely_bracket(SEASON, M_seeds, M_slots, M_submission)

games["TeamA"] = games["TeamA"].map(id_to_name)
games["TeamB"] = games["TeamB"].map(id_to_name)
games["Winner"] = games["Winner"].map(id_to_name)

print("Most-likely champion TeamID:", champ)
games.to_csv(f"data/M_{SEASON}_most_likely_bracket.csv", index=False)
print("Saved:", f"data/M_{SEASON}_most_likely_bracket.csv")


for round in sorted(games["Round"].unique()):
    g = games[games["Round"] == round]
    print(f"\nRound {round} winners:")
    print(g[["Slot", "WinnerSeed", "Winner", "WinnerProb"]].to_string(index=False))