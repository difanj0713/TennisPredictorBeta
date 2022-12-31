import pandas as pd
import logging
import numpy as np
import os
import re
from scipy.stats import zscore

def selectUsefulAttributes():
    directory = 'tennisData'
    df_list = []
    for file in os.listdir(directory):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(directory, file))
            selected_columns = ['surface', 'draw_size', 'tourney_level', 'tourney_date', 'winner_id', 'winner_name',
                                'winner_hand', 'winner_ht', 'winner_age', 'loser_id', 'loser_name', 'loser_hand',
                                'loser_ht',
                                'loser_age', 'score', 'round', 'winner_rank', 'winner_rank_points', 'loser_rank',
                                'loser_rank_points']
            df_selected = df.loc[:, selected_columns]
            df_list.append(df)
    fileName = 'atp_matches_2016-2022.csv'
    df_aggregated = pd.concat(df_list)
    df_aggregated.to_csv(fileName, index=False)

def collectPlayerStats():
    fileName = 'atp_matches_2016-2022.csv'
    df = pd.read_csv(fileName)
    player_ids = np.unique(df[['winner_id', 'loser_id']].values)
    player_ids = player_ids.tolist()
    player_dict = {}
    for player_id in player_ids:
        player_df = df[(df['winner_id'] == player_id) | (df['loser_id'] == player_id)]
        player_dict[player_id] = player_df

    return player_dict

def scoreRegexTransformer(score):
    # Split the score into different sets
    sets = score.split()

    # Initialize the set and game scores
    set_scores = [0, 0]
    game_scores = [0, 0]

    # Use a regular expression to extract the numerical values from each set
    try:
        for set_score in sets:
            match = re.search(r'(\d+)-(\d+)', set_score)
            if match:
                if int(match.group(1)) > int(match.group(2)):
                    set_scores[0] += 1
                else:
                    set_scores[1] += 1
                game_scores[0] += int(match.group(1))
                game_scores[1] += int(match.group(2))
    except Exception as e:
        logging.error(e)

    final_set_score = f'{set_scores[0]}:{set_scores[1]}'
    final_game_score = f'{game_scores[0]}:{game_scores[1]}'

    return final_set_score, final_game_score

def Normalization():
    fileName = 'atp_matches_2016-2022.csv'
    df = pd.read_csv(fileName)

    # Do some categorical replacement
    replacements_sur = {'Hard': 0, 'Clay': 1, 'Grass': 2}
    df['surface'] = df['surface'].replace(replacements_sur)
    df['surface'] = df['surface'].replace(r'^(?!Hard|Clay|Grass).*$', 3, regex=True)

    replacements_lev = {'G': 4, 'M': 3, 'A': 2, 'F': 3, 'D': 1}
    df['tourney_level'] = df['tourney_level'].replace(replacements_lev)
    df['tourney_level'] = df['tourney_level'].replace(r'^(?!F|G|A|M|D).*$', 1, regex=True)

    replacements_hands = {'L': 1, 'R': 2}
    df['winner_hand'] = df['winner_hand'].replace(replacements_hands)
    df['winner_hand'] = df['winner_hand'].replace(r'^(?!L|R).*$', 3, regex=True)
    df['loser_hand'] = df['loser_hand'].replace(replacements_hands)
    df['loser_hand'] = df['loser_hand'].replace(r'^(?!L|R).*$', 3, regex=True)

    # Change 6-4 6-3 format scores into set scores and game scores
    set_scores = []
    game_scores = []
    for score in df['score']:
        final_set_score, final_game_score = scoreRegexTransformer(score)
        set_scores.append(final_set_score)
        game_scores.append(final_game_score)
    df = df.assign(set_score=set_scores, game_score=game_scores)

    # Normalize draw_size, winner_ht, loser_ht, winner_age, loser_age, winner_rank_points, loser_rank_points by z-score; winner_rank, loser_rank like Zipf's law
    df[['draw_size', 'winner_ht', 'loser_ht', 'winner_age', 'loser_age', 'winner_rank_points', 'loser_rank_points']] = \
    df[['draw_size', 'winner_ht', 'loser_ht', 'winner_age', 'loser_age', 'winner_rank_points',
        'loser_rank_points']].apply(lambda x: (x - x.min() + 1) / (x.max() - x.min()))
    df['winner_rank'] = 1 / df['winner_rank']
    df['loser_rank'] = 1 / df['loser_rank']

    # fill nan with 0
    df.fillna(0, inplace=True)
    newfileName = 'cleaned.csv'
    df.to_csv(newfileName, index=False)

if __name__ == "__main__":
    Normalization()