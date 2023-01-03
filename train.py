import pandas as pd

if __name__=="__main__":
    fileName = "cleaned_2.csv"
    df = pd.read_csv(fileName)
    all_players = df['player_id'].unique()
    player_stats = {}

    total_records = 0
    for id in all_players:
        id_stats = df[df['player_id'] == id]
        if id_stats.shape[0] >= 5:
            player_stats[id] = id_stats
            total_records += id_stats.shape[0]

    categorial_ones = ["surface", "draw_size", "tourney_level", "player_hand", "opponent_hand", "round"]
    numerical_ones = ["player_ht", "player_age", "opponent_ht", "opponent_age", "player_rank", "player_rank_points", "opponent_rank", "opponent_rank_points",
                      "set_score_diff", "set_score_sum", "game_score_diff", "game_score_sum"]
