from huggingface_hub import HfFileSystem
import pyarrow.parquet as pq

fs = HfFileSystem()
repo = "datasets/AKCIT-Deepfake/BRSpeech-DF"

arquivos = {
    "train": "bonafide/train-00000-of-00037.parquet",
    "validation": "bonafide/validation-00000-of-00001.parquet",
    "test": "bonafide/test-00000-of-00002.parquet"
}

locutores = {}

for particao, arquivo in arquivos.items():
    with fs.open(f"{repo}/{arquivo}", "rb") as f:
        parquet = pq.ParquetFile(f)
        tabela = parquet.read_row_group(0, columns=["audio"])

    nomes = [
        registro["audio"]["path"]
        for registro in tabela.to_pylist()
    ]

    if any(nome is None for nome in nomes):
        raise ValueError(f"Nome ausente em {particao}")

    ids = {
        nome.split("_")[0]
        for nome in nomes
    }

    locutores[particao] = ids

    print(f"\n=== {particao.upper()} ===")
    print("Áudios analisados:", len(nomes))
    print("Locutores identificados:", len(ids))
    print("IDs:", sorted(ids))

print("\n=== VERIFICAÇÃO DE SOBREPOSIÇÃO ===")

pares = [
    ("train", "validation"),
    ("train", "test"),
    ("validation", "test")
]

vazamento = False

for a, b in pares:
    repetidos = locutores[a] & locutores[b]

    print(f"{a} x {b}: {sorted(repetidos)}")

    if repetidos:
        vazamento = True

print("\n=== RESULTADO ===")

if vazamento:
    print("ATENÇÃO: identificadores repetidos!")
else:
    print("Nenhum identificador repetido na amostra real.")
    print("A separação oficial é descrita no artigo.")
    print("A amostra sintética não possui IDs explícitos.")
