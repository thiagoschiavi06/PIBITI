from collections import Counter
from pathlib import Path
import argparse
import csv
import time

from huggingface_hub import HfFileSystem
import pyarrow.parquet as pq

REPO = "datasets/AKCIT-Deepfake/BRSpeech-DF/spoof"
SAIDA = Path("data/metadados/inventario_modelos_sintese.csv")
CAMPOS = ["particao", "parquet", "modelo", "quantidade"]

PARTICOES = {
    "validation": 7,
    "test": 7,
    "train": 274,
}


def carregar():
    if not SAIDA.exists():
        return []

    with SAIDA.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def salvar(registros):
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    temporario = SAIDA.with_suffix(".csv.tmp")

    with temporario.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CAMPOS)
        writer.writeheader()
        writer.writerows(registros)

    temporario.replace(SAIDA)


def resumo(registros):
    concluidos = {
        (r["particao"], r["parquet"])
        for r in registros
    }

    total = sum(PARTICOES.values())

    print(f"\n=== PROGRESSO: {len(concluidos)}/{total} ===")

    for particao in PARTICOES:
        quantidade = sum(
            1 for p, _ in concluidos if p == particao
        )
        print(f"{particao}: {quantidade}/{PARTICOES[particao]}")

    contagem = Counter()

    for r in registros:
        chave = (r["particao"], r["modelo"])
        contagem[chave] += int(r["quantidade"])

    print("\n=== MODELOS IDENTIFICADOS ===")

    for (particao, modelo), qtd in sorted(contagem.items()):
        print(f"{particao:12} {modelo:20} {qtd:>8}")


def analisar(particao, nome):
    caminho = f"{REPO}/{nome}"
    modelos = Counter()

    with HfFileSystem().open(caminho, "rb") as f:
        parquet = pq.ParquetFile(f)

        for grupo in range(parquet.metadata.num_row_groups):
            tabela = parquet.read_row_group(
                grupo, columns=["model"]
            )

            modelos.update(
                str(m) if m is not None else "DESCONHECIDO"
                for m in tabela.column("model").to_pylist()
            )

    return [
        {
            "particao": particao,
            "parquet": nome,
            "modelo": modelo,
            "quantidade": quantidade,
        }
        for modelo, quantidade in modelos.items()
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resumo", action="store_true")
    parser.add_argument("--limite", type=int, default=None)
    args = parser.parse_args()

    registros = carregar()
    resumo(registros)

    if args.resumo:
        return

    concluidos = {
        (r["particao"], r["parquet"])
        for r in registros
    }

    novos = 0
    falhas = []

    for particao, total in PARTICOES.items():
        for indice in range(total):
            nome = (
                f"{particao}-{indice:05d}-of-"
                f"{total:05d}.parquet"
            )

            chave = (particao, nome)

            if chave in concluidos:
                continue

            if args.limite is not None and novos >= args.limite:
                resumo(registros)
                return

            print(f"\n[{particao}] {indice + 1}/{total}: {nome}",
                  flush=True)

            for tentativa in range(1, 4):
                try:
                    resultado = analisar(particao, nome)

                    if not resultado:
                        raise ValueError("Arquivo sem modelos")

                    registros.extend(resultado)
                    salvar(registros)

                    concluidos.add(chave)
                    novos += 1

                    print("OK — progresso salvo", flush=True)
                    break

                except Exception as erro:
                    print(
                        f"Tentativa {tentativa}/3 falhou: "
                        f"{type(erro).__name__}: {erro}",
                        flush=True
                    )

                    if tentativa < 3:
                        time.sleep(2 ** tentativa)
                    else:
                        falhas.append((particao, nome))
                        print("Pulado temporariamente.", flush=True)

    resumo(registros)

    if falhas:
        print("\nArquivos pendentes por erro de rede:")
        for particao, nome in falhas:
            print(f"- {particao}/{nome}")
        print("Execute novamente quando a conexão melhorar.")


if __name__ == "__main__":
    main()
