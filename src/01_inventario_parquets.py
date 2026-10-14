from pathlib import Path
import pandas as pd
from huggingface_hub import list_repo_files

REPO = "AKCIT-Deepfake/BRSpeech-DF"
saida = Path("data/metadados/inventario_parquets.csv")
saida.parent.mkdir(parents=True, exist_ok=True)
registros = []
for arquivo in list_repo_files(repo_id=REPO, repo_type="dataset"):
    if not arquivo.endswith(".parquet"):
        continue
    partes = arquivo.split("/")
    if len(partes) != 2 or partes[0] not in ("bonafide", "spoof"):
        continue
    divisao = partes[1].split("-")[0]
    if divisao not in ("train", "validation", "test"):
        continue
    registros.append(dict(arquivo_parquet=arquivo, classe=partes[0], particao=divisao))
df = pd.DataFrame(registros).sort_values(["particao", "classe", "arquivo_parquet"])
df.to_csv(saida, index=False)
print(pd.crosstab(df.particao, df.classe))
print("Total de Parquets:", len(df), "| Arquivo:", saida)
