from pathlib import Path
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

resultados = Path("results")
figuras = Path("reports/fichamento_wpt/figuras")
figuras.mkdir(parents=True, exist_ok=True)

# GRÁFICO 1 — Tempo de execução WPT

df = pd.read_csv(resultados / "comparacao_wavelets.csv")

tabela = df.pivot(
    index="wavelet",
    columns="nivel",
    values="tempo_ms"
)

ax = tabela.plot(kind="bar", figsize=(9, 5))
ax.set_title("Tempo de decomposição e reconstrução WPT")
ax.set_xlabel("Wavelet-mãe")
ax.set_ylabel("Tempo (ms)")
ax.legend(title="Nível")
ax.tick_params(axis="x", rotation=0)

plt.tight_layout()
plt.savefig(figuras / "comparacao_wavelets.png", dpi=300)
plt.close()

# GRÁFICO 2 — Entropia das 16 sub-bandas

df = pd.read_csv(
    resultados / "entropia_shannon_db4_l4.csv"
)

plt.figure(figsize=(10, 5))
plt.bar(df["banda"], df["entropia"])
plt.title("Entropia de Shannon por sub-banda — db4, L=4")
plt.xlabel("Sub-banda")
plt.ylabel("Entropia")
plt.xticks(range(1, 17))
plt.tight_layout()

plt.savefig(figuras / "entropia_subbandas.png", dpi=300)
plt.close()

# GRÁFICO 3 — Real x Deepfake

df = pd.read_csv(
    resultados / "comparacao_entropia_real_spoof.csv"
)

tabela = df.pivot(
    index="banda",
    columns="classe",
    values="entropia"
)

ax = tabela.plot(kind="bar", figsize=(11, 5))
ax.set_title("Entropia WPT — Áudio real x Deepfake")
ax.set_xlabel("Sub-banda")
ax.set_ylabel("Entropia de Shannon")
ax.legend(title="Classe")
ax.tick_params(axis="x", rotation=0)

plt.tight_layout()
plt.savefig(figuras / "comparacao_real_spoof.png", dpi=300)
plt.close()

print("=== GRÁFICOS GERADOS ===")

for arquivo in sorted(figuras.glob("*.png")):
    print(arquivo)
