import os, numpy as np, pandas as pd

np.random.seed(42)
N = 1200  # rows

def clip(a, lo, hi):
    return np.clip(a, lo, hi)

# base signals
age = clip(np.random.normal(65, 12, N).round(), 18, 90)
vent_flag = np.random.binomial(1, 0.30, N)
pressors_flag = np.random.binomial(1, 0.20, N)

hr_base = np.random.normal(90, 15, N)
hr_min = clip(hr_base - np.abs(np.random.normal(10, 5, N)), 40, 120)
hr_max = clip(hr_base + np.abs(np.random.normal(10, 5, N)), 60, 180)

map_base = np.random.normal(75, 12, N)
map_min = clip(map_base - np.abs(np.random.normal(8, 4, N)), 35, 100)
map_max = clip(map_base + np.abs(np.random.normal(8, 4, N)), 55, 130)

resp_rate_max = clip(np.random.normal(22, 6, N), 10, 45)
temp_max = clip(np.random.normal(37.6, 0.6, N), 35.5, 40.5)
spo2_min = clip(np.random.normal(94, 4, N), 70, 100)

albumin_min = clip(np.random.normal(3.1, 0.6, N), 1.2, 5.0)
lactate_max = clip(np.random.lognormal(mean=np.log(1.8), sigma=0.6, size=N), 0.5, 12)
bun_max = clip(np.random.normal(24, 12, N), 5, 100)
creatinine_max = clip(np.random.normal(1.4, 0.7, N), 0.3, 7)
bilirubin_max = clip(np.random.lognormal(mean=np.log(0.8), sigma=0.7, size=N), 0.2, 15)

hemoglobin_min = clip(np.random.normal(11, 2, N), 6, 17)
platelets_min = clip(np.random.normal(170, 60, N), 15, 500)
sodium_min = clip(np.random.normal(137, 4, N), 120, 155)
potassium_max = clip(np.random.normal(4.2, 0.6, N), 2.5, 7.0)
aniongap_max = clip(np.random.normal(14, 4, N), 6, 32)

# mortality risk with sensible directions
score = (
    0.03*(age-60) + 0.9*vent_flag + 1.1*pressors_flag
    + 0.45*(lactate_max-2) + 0.02*(bun_max-20) + 0.5*(creatinine_max-1)
    + 0.03*(bilirubin_max-1) + 0.02*(resp_rate_max-20)
    - 0.03*(spo2_min-92) - 0.28*(albumin_min-3.5)
    - 0.0008*(platelets_min-150)
)
p = 1 / (1 + np.exp(-(-2.0 + score)))   # target event rate around 20 percent
hospital_expire_flag = np.random.binomial(1, p)

df = pd.DataFrame({
    "age": age.astype(int),
    "vent_flag": vent_flag,
    "pressors_flag": pressors_flag,
    "hr_min": hr_min, "hr_max": hr_max,
    "map_min": map_min, "map_max": map_max,
    "resp_rate_max": resp_rate_max, "temp_max": temp_max, "spo2_min": spo2_min,
    "albumin_min": albumin_min, "lactate_max": lactate_max,
    "bun_max": bun_max, "creatinine_max": creatinine_max, "bilirubin_max": bilirubin_max,
    "hemoglobin_min": hemoglobin_min, "platelets_min": platelets_min,
    "sodium_min": sodium_min, "potassium_max": potassium_max, "aniongap_max": aniongap_max,
    "hospital_expire_flag": hospital_expire_flag.astype(int),
})

# add some missing values to mimic day one gaps
for col in [c for c in df.columns if c not in ["vent_flag","pressors_flag","hospital_expire_flag"]]:
    mask = np.random.rand(N) < 0.1
    df.loc[mask, col] = np.nan

out_dir = os.path.dirname(__file__)
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "mimic_lung_day1.csv")
df.to_csv(out_path, index=False)
print(f"Wrote {len(df)} rows to {out_path}")
