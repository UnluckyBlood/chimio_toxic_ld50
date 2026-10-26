import subprocess
import os

smiles = "CN(C)CCCN(CCCN(C)C)C(=O)C(Cc1ccccc1)NC(=O)C12CC3CC(CC(C3)C1)C2"

# ESKAPE ожидает CSV с колонкой "smiles"
with open("input.csv", "w") as f:
    f.write("smiles\n")
    f.write(smiles + "\n")

# Запуск (путь к папке db с моделями нужно указать свой)
cmd = [
    "eskape_model",
    "--input_file", "input.csv",
    "--output_directory", "output",
    "--models_directory", "path/to/db"  # <-- замените на путь к скачанным моделям
]
subprocess.run(cmd, check=True)

# Чтение результата (TSV)
import pandas as pd
df = pd.read_csv("output/predictions.tsv", sep="\t")
print(df.T)  # транспонируем для удобства