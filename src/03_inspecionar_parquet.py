import argparse
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

parser = argparse.ArgumentParser(description="Inspeciona metadados sem extrair áudio")
parser.add_argument("--classe", choices=["bonafide", "spoof"], default="bonafide")
parser.add_argument("--particao", choices=["train", "validation", "test"], default="train")
parser.add_argument("--indice", type=int, default=0)
args = parser.parse_args()
totais = {"bonafide": {"train": 37, "validation": 1, "test": 2}, "spoof": {"train": 274, "validation": 7, "test": 7}}
n = totais[args.classe][args.particao]
if not 0 <= args.indice < n:
    parser.error(f"indice deve estar entre 0 e {n-1}")
nome = f"{args.classe}/{args.particao}-{args.indice:05d}-of-{n:05d}.parquet"
with HfFileSystem().open(f"datasets/AKCIT-Deepfake/BRSpeech-DF/{nome}", "rb") as f:
    parquet = pq.ParquetFile(f)
    print("Arquivo:", nome, "\nRegistros:", parquet.metadata.num_rows, "\nColunas:", parquet.schema_arrow.names)
    print("Grupos:", parquet.num_row_groups, "\nRegistros no primeiro grupo:", parquet.metadata.row_group(0).num_rows)
    print("Tamanho descomprimido do primeiro grupo (MB):", round(parquet.metadata.row_group(0).total_byte_size / 1024**2, 2))
