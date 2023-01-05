import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from dataWraggler import *

def DecisionTreeBenchmark():
    Normalization()

    fileName = 'cleaned_2.csv'
    df = pd.read_csv(fileName)

    # pick matches before 20210101 as training set
    df_train = df
    df_test = df.where(df['tourney_date'] >= 20220100)
    df_test = df_test.sample(frac=1)
    df_train = df_train.sample(frac=1)
    df_train = df_train.dropna()
    df_test = df_test.dropna()

    # Select some player stats
    X_train = df_train[
        ["surface", "draw_size", "tourney_level", "player_hand", "player_ht", "player_age", "opponent_hand",
         "opponent_ht", "opponent_age", "round"]]
    # X_train = df_train.drop(columns=['player_name', 'opponent_name', 'score', 'set_score_diff', 'set_score_sum', 'game_score_diff', 'game_score_sum'])
    y_train = df_train[['set_score_diff', 'set_score_sum', 'game_score_diff', 'game_score_sum']]

    X_test = df_test[
        ["surface", "draw_size", "tourney_level", "player_hand", "player_ht", "player_age", "opponent_hand",
         "opponent_ht", "opponent_age", "round"]]
    # X_test = df_test.drop(columns=['player_name', 'opponent_name', 'score', 'set_score_diff', 'set_score_sum', 'game_score_diff', 'game_score_sum'])
    y_test = df_test[['set_score_diff', 'set_score_sum', 'game_score_diff', 'game_score_sum']]

    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    set_score_diff_test = y_test['set_score_diff'].to_numpy()
    set_score_diff_pred = y_pred[:, 0]

    set_score_diff_mse = mean_squared_error(set_score_diff_test, set_score_diff_pred)
    print(f'set_score_diff\nMSE: {set_score_diff_mse:.2f}')

    l = len(set_score_diff_test)
    succ = 0
    no = 0
    for i in range(l):
        if set_score_diff_pred[i] == 0:
            no += 1
        if set_score_diff_test[i] * set_score_diff_pred[i] > 0:
            succ += 1
    print(l, succ, succ/l, no)

    return model


if __name__=="__main__":
    model = DecisionTreeBenchmark()
    X_test = pd.DataFrame({'player_ht': -1.705733536774848, 'opponent_ht': 0.0830841702762672}, index=[1])
    y_pred = model.predict(X_test)
    print(int((y_pred[:, 1] + y_pred[:, 0]) / 2), int((y_pred[:, 1] - y_pred[:, 0]) / 2), int(
        (y_pred[:, 3] + y_pred[:, 2]) / 2), int((y_pred[:, 3] - y_pred[:, 2]) / 2))

