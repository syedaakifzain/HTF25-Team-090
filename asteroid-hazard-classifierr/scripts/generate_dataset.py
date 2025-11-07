import csv
import random
import os

random.seed(42)

p = os.path.join('asteroid-hazard-classifier', 'data', 'dataset.csv')
os.makedirs(os.path.dirname(p), exist_ok=True)

headers = [
    'Relative Velocity km per hr',
    'Miles per hour',
    'Miss Dist.(Astronomical)',
    'Miss Dist.(lunar)',
    'Miss Dist.(kilometers)',
    'Semi Major Axis',
    'Aphelion Dist',
    'Mean Motion',
    'Orbital Period',
    'Hazardous'
]

rows = 800
with open(p, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(headers)
    for i in range(rows):
        rv_kmh = random.uniform(50000, 120000)
        mph = rv_kmh * 0.621371
        miss_au = random.uniform(0.01, 3.0)
        miss_lunar = random.uniform(1, 1000)
        miss_km = miss_au * 149597870
        sma = random.uniform(0.5, 5.5)
        aphelion = sma + random.uniform(0.0, 2.0)
        mean_motion = random.uniform(0.1, 5.0)
        orbital_period = random.uniform(50, 10000)
        hazard = 1 if (miss_au < 0.05 and rv_kmh > 85000) else 0
        w.writerow([
            round(rv_kmh, 2),
            round(mph, 2),
            round(miss_au, 5),
            round(miss_lunar, 2),
            round(miss_km, 2),
            round(sma, 4),
            round(aphelion, 4),
            round(mean_motion, 4),
            round(orbital_period, 2),
            hazard,
        ])

print(f"Wrote {rows} rows to {p}")