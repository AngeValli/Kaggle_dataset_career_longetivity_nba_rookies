# File paths
TRAIN_DATA: str = "./Train_data.csv"
TEST_DATA: str = "./Test_data.csv"
LABELS_TEST: str = "./Sample_Submission.csv"

# Dataset column names
NAME: str = "Name" # Name
GP: str = "GP" # Games Played
MIN: str = "MIN" # Minutes Played
PTS: str = "PTS" # Points per Game
FGM: str = "FGM" # Field Goals Made
FGM_MIN: str = "FGM_MIN" # Field Goals Made per Minutes Played
FGA: str = "FGA" # Field Goal Attempts
FGA_MIN: str = "FGA_MIN" # Field Goal Attempts per Minutes Played
FG_PERC: str = "FG%" # Field Goal Percent
THREEP_MADE: str = "3P Made" # 3 Points Made
THREEP_MADE_MIN: str = "3P Made_MIN" # 3 Points Made per Minutes Played
THREEP_A: str = "3PA" # 3 Points Attempts
THREEP_A_MIN: str = "3PA_MIN" # 3 Points Attempts per Minutes Played
THREEP_PERC: str = "3P%" # 3 Points Attempts Percentage
FTM: str = "FTM" # Free Throw Made
FTM_MIN: str = "FTM_MIN" # Free Throw Made per Minutes Played
FTA: str = "FTA" # Free Throw Attempts
FTA_MIN: str = "FTA_MIN" # Free Throw Attempts per Minutes Played
FT_PERC: str = "FT%" # Free Throw Attempts Percentage
OREB: str = "OREB" # Offensive Rebounds
OREB_MIN: str = "OREB_MIN" # Rebounds per Minutes Played
DREB: str = "DREB" # Defensive Rebounds
DREB_MIN: str = "DREB_MIN" # Rebounds per Minutes Played
REB: str = "REB" # Rebounds
REB_MIN: str = "REB_MIN" # Rebounds per Minutes Played
AST: str = "AST" # Assists
AST_MIN: str = "AST_MIN" # Assists per Minutes Played
STL: str = "STL" # Steals
STL_MIN: str = "STL_MIN" # Steals per Minutes Played
BLK: str = "BLK" # Blocks
BLK_MIN: str = "BLK_MIN" # Blocks per Minutes Played
TOV: str = "TOV" # Turnovers
TOV_MIN: str = "TOV_MIN" # Turnovers per Minutes Played
TARGET: str = "Target" # Target column
K_MEANS_CLUSTERING: str = "K_MEANS_CLUSTERING" # K_Means clustering results

# Pickles names
MODEL_SVC: str = "model_svc.pickle"
MODEL_LGRG: str = "model_lgrg.pickle"
MODEL_RF: str = "model_rf.pickle"
IMPUTER: str = "imputer.pickle"
KMEANS_MODEL: str = "kmeans_model.pickle"
NORMALIZER: str = "normalizer.pickle"