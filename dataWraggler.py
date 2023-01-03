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
            selected_columns = ['surface', 'draw_size', 'round', 'tourney_level', 'tourney_date', 
                                'winner_id', 'winner_name', 'winner_hand', 'winner_ht', 'winner_age', 'winner_rank', 'winner_rank_points', 
                                'loser_id', 'loser_name', 'loser_hand', 'loser_ht', 'loser_age', 'loser_rank', 'loser_rank_points',
                                'score']
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
                set_scores[0] += int(int(match.group(1)) > int(match.group(2)))
                set_scores[1] += int(int(match.group(1)) < int(match.group(2)))
                game_scores[0] += int(match.group(1))
                game_scores[1] += int(match.group(2))
    except Exception as e:
        logging.error(e)

    final_set_score = f'{set_scores[0]}:{set_scores[1]}'
    final_set_score_diff = set_scores[0] - set_scores[1]
    final_set_score_sum = set_scores[0] + set_scores[1]
    final_game_score = f'{game_scores[0]}:{game_scores[1]}'
    final_game_score_diff = game_scores[0] - game_scores[1]
    final_game_score_sum = game_scores[0] + game_scores[1]

    return final_set_score, final_set_score_diff, final_set_score_sum, final_game_score, final_game_score_diff, final_game_score_sum

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

    replacements_round = {'R128': 7, 'R64': 6, 'R32': 5, 'R16': 4, 'QF': 3, 'SF': 2, 'F': 1, 'RR': 4, 'BR': 2}
    df['round'] = df['round'].replace(replacements_round)

    # Change the 'score' column with 6-4 6-3 format scores into set scores and game scores
    set_scores = []
    set_score_diffs = []
    set_score_sums = []
    game_scores = []
    game_score_diffs = []
    game_score_sums = []
    for score in df['score']:
        final_set_score, final_set_score_diff, final_set_score_sum, final_game_score, final_game_score_diff, final_game_score_sum = scoreRegexTransformer(score)
        set_scores.append(final_set_score)
        set_score_diffs.append(final_set_score_diff)
        set_score_sums.append(final_set_score_sum)
        game_scores.append(final_game_score)
        game_score_diffs.append(final_game_score_diff)
        game_score_sums.append(final_game_score_sum)
    df = df.assign(set_score=set_scores, 
                   set_score_diff=set_score_diffs,
                   set_score_sum=set_score_sums,
                   game_score=game_scores,
                   game_score_diff=game_score_diffs,
                   game_score_sum=game_score_sums)
    
    # Take the log2 of the values in the 'draw_size' and 'round' column 
    df[['draw_size']] = np.log2(df[['draw_size']])
    
    # Replace NaN values in multiple columns with the mean of each column
    df.fillna(df.mean(), inplace=True)

    # Normalize winner_ht, loser_ht, winner_age, loser_age, winner_rank_points, loser_rank_points by z-score; winner_rank, loser_rank like Zipf's law
    df[['winner_ht', 'loser_ht', 'winner_age', 'loser_age', 'winner_rank_points', 'loser_rank_points']] = \
    df[['winner_ht', 'loser_ht', 'winner_age', 'loser_age', 'winner_rank_points', 'loser_rank_points']].apply(zscore)
    #.apply(lambda x: (x - x.min() + 1) / (x.max() - x.min()))
    df['winner_rank'] = 1 / df['winner_rank']
    df['loser_rank'] = 1 / df['loser_rank']
    
    newFileName = 'cleaned.csv'
    df.to_csv(newFileName, index=False)

    # For each record in df, create two records from both the winner's and the loser's perspectives
    df_winner = df.copy()
    column_mapping_winner = {'winner_id': 'player_id', 'winner_name': 'player_name', 'winner_hand': 'player_hand', 'winner_ht': 'player_ht', 
                            'winner_age': 'player_age', 'winner_rank': 'player_rank', 'winner_rank_points': 'player_rank_points', 
                            'loser_id': 'opponent_id', 'loser_name': 'opponent_name', 'loser_hand': 'opponent_hand', 'loser_ht': 'opponent_ht',
                            'loser_age': 'opponent_age', 'loser_rank': 'opponent_rank', 'loser_rank_points': 'opponent_rank_points'}
    df_winner.rename(columns=column_mapping_winner, inplace=True)

    df_loser = df.copy()
    column_mapping_loser = {'winner_id': 'opponent_id', 'winner_name': 'opponent_name', 'winner_hand': 'opponent_hand', 'winner_ht': 'opponent_ht', 
                            'winner_age': 'opponent_age', 'winner_rank': 'opponent_rank', 'winner_rank_points': 'opponent_rank_points', 
                            'loser_id': 'player_id', 'loser_name': 'player_name', 'loser_hand': 'player_hand', 'loser_ht': 'player_ht',
                            'loser_age': 'player_age', 'loser_rank': 'player_rank', 'loser_rank_points': 'player_rank_points'}
    df_loser.rename(columns=column_mapping_loser, inplace=True)
    df_loser['set_score_diff'] = -df_loser['set_score_diff']
    df_loser['game_score_diff'] = -df_loser['game_score_diff']

    df_new = pd.concat([df_winner, df_loser])

    newFileName_2 = 'cleaned_2.csv'
    df_new.to_csv(newFileName_2, index=False)

if __name__ == "__main__":
    Normalization()