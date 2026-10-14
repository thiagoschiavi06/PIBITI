from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

entrada = Path("data/metadados/inventario_parquets.csv")
saida = Path("data/metadados/inventario_quantidades.csv")
df = pd.read_csv(entrada)
fs = HfFileSystem()
registros = []
for i, linha in df.iterrows():
    nome = linha["arquivo_parquet"]
    try:
        with fs.open(f"datasets/AKCIT-Deepfake/BRSpeech-DF/{nome}", "rb") as f:
            quantidade = pq.ParquetFile(f).metadata.num_rows
        registros.append({**linha.to_dict(), "quantidade_audios": quantidade})
        print(f"[{i+1}/{len(df)}] {nome}: {quantidade}")
    except Exception as e:
        print(f"ERRO {nome}: {e}")
resultado = pd.DataFrame(registros)
resultado.to_csv(saida, index=False)
print("\nArquivos consultados:", len(resultado), "de", len(df))
if not resultado.empty:
    print(pd.pivot_table(resultado, index="particao", columns="classe", values="quantidade_audios", aggfunc="sum", fill_value=0))
print("Saída:", saida)
