# File paths
TRAIN_DATA: str = "./Train_data.csv"
TEST_DATA: str = "./Test_data.csv"
LABELS_TEST: str = "./Sample_Submission.csv"

# Dataset column names
TARGET: str = "Target"
FT_PERC: str = "FT%"
THREEP_PERC: str = "3P%"
FG_PERC: str = "FG%"

# Pickles names
MODEL_SVC: str = "model_svc.pickle"
MODEL_LGRG: str = "model_lgrg.pickle"
MODEL_RF: str = "model_rf.pickle"
IMPUTER: str = "imputer.pickle"
NORMALIZER: str = "normalizer.pickle"