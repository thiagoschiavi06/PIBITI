import argparse
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

parser = argparse.ArgumentParser()
parser.add_argument("--classe", choices=["bonafide", "spoof"], required=True)
parser.add_argument("--particao", choices=["train", "validation", "test"], required=True)
parser.add_argument("--shard", type=int, default=0)
parser.add_argument("--grupo", type=int, default=0)
args = parser.parse_args()
totais = {"bonafide": {"train": 37, "validation": 1, "test": 2}, "spoof": {"train": 274, "validation": 7, "test": 7}}
n = totais[args.classe][args.particao]
if not 0 <= args.shard < n:
    parser.error(f"shard deve estar entre 0 e {n-1}")
origem = f"{args.classe}/{args.particao}-{args.shard:05d}-of-{n:05d}.parquet"
colunas = ["audio", "label"] + (["model"] if args.classe == "spoof" else [])
with HfFileSystem().open(f"datasets/AKCIT-Deepfake/BRSpeech-DF/{origem}", "rb") as f:
    parquet = pq.ParquetFile(f)
    if not 0 <= args.grupo < parquet.num_row_groups:
        parser.error(f"grupo deve estar entre 0 e {parquet.num_row_groups-1}")
    offset = sum(parquet.metadata.row_group(i).num_rows for i in range(args.grupo))
    tabela = parquet.read_row_group(args.grupo, columns=colunas)

pasta = Path("data/audios") / args.classe / args.particao
pasta.mkdir(parents=True, exist_ok=True)
metadados = Path("data/metadados")
metadados.mkdir(parents=True, exist_ok=True)
registros = []
for i, registro in enumerate(tabela.to_pylist()):
    indice = offset+i
    blob = registro["audio"]["bytes"]
    if blob is None:
        raise ValueError(f"Registro {indice} sem bytes; revise acesso ao áudio")
    # Nomes legados preservados para shard=0, grupo=0
    sufixo = f"{indice:05d}" if args.shard == 0 else f"s{args.shard:03d}_{indice:05d}"
    destino = pasta / f"{args.classe}_{args.particao}_{sufixo}.audio"
    if destino.exists():
        if destino.read_bytes() != blob:
            raise RuntimeError(f"Arquivo já existe com conteúdo diferente: {destino}")
    else:
        destino.write_bytes(blob)
    registros.append({"arquivo": str(destino), "parquet_origem": origem,
                      "indice_registro": indice, "label_original": registro["label"],
                      "classe": args.classe, "particao": args.particao,
                      "modelo_sintese": registro.get("model")})
saida = metadados / f"amostra_{args.classe}_{args.particao}_s{args.shard:03d}_g{args.grupo:03d}.csv"
# Compatibilidade com os arquivos de metadados já criados nas primeiras etapas
if args.shard == 0 and args.grupo == 0:
    saida = metadados / f"amostra_{args.classe}_{args.particao}.csv"
pd.DataFrame(registros).to_csv(saida, index=False)
print("Áudios extraídos ou verificados:", len(registros), "\nDestino:", pasta, "\nCSV:", saida)
if args.classe == "spoof":
    print("Modelos:", pd.DataFrame(registros).modelo_sintese.value_counts().to_string())
