from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ENTRADA = Path("results/wpt_features_db4_l4_janela4s.csv")
PASTA = Path("reports/experimentos_wpt_mes3/figuras")
SAIDA_CSV = Path("results/resumo_features_wpt_por_particao.csv")

PASTA.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ENTRADA)

PARTICOES = ["train", "validation", "test"]

CONFIGURACOES = [
    ("entropia_", "Entropia de Shannon", "entropia"),
    ("energia_rel_", "Energia relativa", "energia_relativa"),
]

resumos = []

for particao in PARTICOES:
    dados = df[df["particao"] == particao]

    for prefixo, titulo, nome in CONFIGURACOES:
        colunas = sorted(
            c for c in df.columns if c.startswith(prefixo)
        )

        if len(colunas) != 16:
            raise ValueError(
                f"Esperadas 16 sub-bandas para {prefixo}"
            )

        plt.figure(figsize=(12, 5))

        for classe, legenda in [
            ("bonafide", "Real"),
            ("spoof", "Sintético")
        ]:
            grupo = dados[dados["classe"] == classe]

            medianas = grupo[colunas].median()

            if not np.isfinite(medianas.to_numpy()).all():
                raise ValueError("Medianas inválidas encontradas")

            bandas = [
                coluna.removeprefix(prefixo)
                for coluna in colunas
            ]

            plt.plot(
                bandas,
                medianas.to_numpy(),
                marker="o",
                label=f"{legenda} (n={len(grupo)})"
            )

            for banda, valor in zip(
                bandas, medianas.to_numpy()
            ):
                resumos.append({
                    "particao": particao,
                    "classe": classe,
                    "caracteristica": nome,
                    "subbanda": banda,
                    "mediana": float(valor),
                    "quantidade_audios": len(grupo),
                })

        plt.title(f"{titulo} por sub-banda — {particao}")
        plt.xlabel("Sub-banda WPT db4, nível 4")
        plt.ylabel(f"{titulo} — mediana")
        plt.xticks(rotation=45)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()

        destino = PASTA / f"{nome}_{particao}.png"
        plt.savefig(destino, dpi=180)
        plt.close()

        print(f"Gráfico criado: {destino}")

resumo = pd.DataFrame(resumos)
resumo.to_csv(SAIDA_CSV, index=False)

print("\n=== RESULTADO ===")
print("Gráficos gerados:", len(PARTICOES) * len(CONFIGURACOES))
print("Linhas do resumo:", len(resumo))
print("CSV:", SAIDA_CSV)
