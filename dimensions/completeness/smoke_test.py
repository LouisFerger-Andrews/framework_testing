import os, sys
import pandas as pd

# project root = .../framework_testing
FILE_DIR = os.path.dirname(__file__)
PROJ_ROOT = os.path.abspath(os.path.join(FILE_DIR, "..", ".."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

# import model trainers from the local models package
from models import train_logreg_en, train_random_forest, train_gbm

DATA_PATH = os.path.join(PROJ_ROOT, "data", "mimic_lung_day1.csv")
RESULTS_DIR = os.path.join(PROJ_ROOT, "results")
MODELS_DIR = os.path.join(PROJ_ROOT, "models")

FEATURES = [
    "age","vent_flag","pressors_flag",
    "hr_min","hr_max","map_min","map_max",
    "resp_rate_max","temp_max","spo2_min",
    "albumin_min","lactate_max","bun_max","creatinine_max","bilirubin_max",
    "hemoglobin_min","platelets_min","sodium_min","potassium_max","aniongap_max"
]
LABEL = "hospital_expire_flag"
CATS = []

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Missing data at {DATA_PATH}. Run data/make_fake_mimic.py first."
        )
    df = pd.read_csv(DATA_PATH)

    m1 = train_logreg_en(df, FEATURES, LABEL, cat_features=CATS,
                         results_dir=RESULTS_DIR, models_dir=MODELS_DIR)
    m2 = train_random_forest(df, FEATURES, LABEL, cat_features=CATS,
                             results_dir=RESULTS_DIR, models_dir=MODELS_DIR)
    m3 = train_gbm(df, FEATURES, LABEL, cat_features=CATS,
                   results_dir=RESULTS_DIR, models_dir=MODELS_DIR)

    out = pd.DataFrame([m1, m2, m3]).sort_values("auc", ascending=False)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "completeness_smoke_test.csv")
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"\nSaved metrics to {out_path}")

if __name__ == "__main__":
    main()
