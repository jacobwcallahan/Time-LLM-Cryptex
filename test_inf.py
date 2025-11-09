from utils.pipeline import inf_analysis
from pathlib import Path

run = None
inf = inf_analysis(Path("./inference(2).csv"))

for key, value in inf.items():
    print(key, value)