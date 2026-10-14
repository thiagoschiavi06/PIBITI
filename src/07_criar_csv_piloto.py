from pathlib import Path
import pandas as pd
import soundfile as sf

pasta = Path("data/metadados")
fontes = sorted(pasta.glob("amostra_*.csv"))
if not fontes:
    raise SystemExit("Não foram encontrados arquivos amostra_*.csv")
df = pd.concat([pd.read_csv(arq) for arq in fontes], ignore_index=True)
df = df.drop_duplicates(subset=["arquivo", "parquet_origem", "indice_registro"])
def caminho_processado(original):
    p = Path(original)
    return str(Path("data/processados") / p.relative_to("data/audios").with_suffix(".wav"))
df["arquivo_processado"] = df["arquivo"].map(caminho_processado)
erros = []
for nome in df["arquivo_processado"]:
    arquivo = Path(nome)
    if not arquivo.is_file():
        erros.append(f"Ausente: {arquivo}")
        continue
    try:
        info = sf.info(arquivo)
        if info.samplerate != 16000 or info.channels != 1 or info.format != "WAV":
            erros.append(f"Formato incorreto: {arquivo}")
    except Exception as e:
        erros.append(f"Erro ao verificar {arquivo}: {e}")
if erros:
    print("Arquivos com problema:", len(erros))
    print("\n".join(erros[:20]))
    raise SystemExit("Corrija os arquivos antes de salvar o CSV")
saida = pasta / "subconjunto_piloto.csv"
df.to_csv(saida, index=False)
print(pd.crosstab(df.particao, df.classe))
print("Áudios validados:", len(df), "| Saída:", saida)
