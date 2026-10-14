from pathlib import Path
from itertools import combinations
from huggingface_hub import HfFileSystem
import pyarrow.parquet as pq
import csv

REPO = "datasets/AKCIT-Deepfake/BRSpeech-DF"
SAIDA = Path("results")
SAIDA.mkdir(exist_ok=True)

fs = HfFileSystem()

arquivos = {
    "train": 37,
    "validation": 1,
    "test": 2,
}

locutores = {particao: set() for particao in arquivos}
contagens = {particao: 0 for particao in arquivos}
ausentes = {particao: 0 for particao in arquivos}

for particao, total in arquivos.items():
    for indice in range(total):
        nome = f"{particao}-{indice:05d}-of-{total:05d}.parquet"
        caminho = f"{REPO}/bonafide/{nome}"

        print(f"[{particao}] {indice + 1}/{total}: {nome}", flush=True)

        with fs.open(caminho, "rb") as f:
            parquet = pq.ParquetFile(f)

            for grupo in range(parquet.metadata.num_row_groups):
                tabela = parquet.read_row_group(
                    grupo,
                    columns=["audio.path"]
                )

                for registro in tabela.to_pylist():
                    nome_audio = registro["audio"]["path"]
                    contagens[particao] += 1

                    if not nome_audio:
                        ausentes[particao] += 1
                        continue

                    locutor = nome_audio.split("_")[0]
                    locutores[particao].add(locutor)

linhas = [
    "VALIDACAO DE IDENTIFICADORES - BRSpeech-DF",
    "",
]

for particao in arquivos:
    linhas.extend([
        f"PARTICAO: {particao}",
        f"Audios analisados: {contagens[particao]}",
        f"Nomes ausentes: {ausentes[particao]}",
        f"Identificadores distintos: {len(locutores[particao])}",
        f"IDs: {', '.join(sorted(locutores[particao]))}",
        "",
    ])

linhas.append("SOBREPOSICAO ENTRE PARTICOES")
linhas.append("")

sobreposicao = False

for a, b in combinations(arquivos, 2):
    comuns = locutores[a] & locutores[b]
    linhas.append(f"{a} x {b}: {sorted(comuns)}")

    if comuns:
        sobreposicao = True

linhas.extend([
    "",
    "RESULTADO",
    (
        "Foram encontrados identificadores repetidos."
        if sobreposicao
        else "Nenhuma sobreposicao de identificadores encontrada."
    ),
    "",
    "OBSERVACOES:",
    "- Verificacao restrita aos registros bonafide.",
    "- O prefixo do nome foi utilizado como identificador presumido.",
    "- A correspondencia com locutores deve seguir a documentacao original.",
    "- Audios sinteticos nao foram verificados por locutor.",
])

relatorio = "\n".join(linhas)

print("\n" + relatorio)

(SAIDA / "validacao_locutores_completa.txt").write_text(
    relatorio + "\n",
    encoding="utf-8"
)

with (SAIDA / "locutores_por_particao.csv").open(
    "w", newline="", encoding="utf-8"
) as f:
    escritor = csv.writer(f)
    escritor.writerow(["particao", "identificador"])

    for particao, ids in locutores.items():
        for identificador in sorted(ids):
            escritor.writerow([particao, identificador])

print("\nRelatorios salvos em results/")
