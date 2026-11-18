from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ENTRADA = Path("results/features_wpt_96_janela4s.csv")
PASTA_FIGURAS = Path("reports/experimentos_wpt_mes4/figuras")
SAIDA_RESUMO_CSV = Path("results/resumo_descritores_96_por_particao.csv")
SAIDA_RELATORIO_TXT = Path("results/analise_consistencia_descritores_mes4.txt")

PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)

PARTICOES = ["train", "validation", "test"]
DESCRITORES = [
    ("energia", "Energia"),
    ("rms", "Valor Eficaz (RMS)"),
    ("zcr", "Taxa de Cruzamento por Zero (ZCR)"),
    ("env_media", "Envelope de Hilbert — Média"),
    ("env_desvio", "Envelope de Hilbert — Desvio Padrão"),
    ("env_curtose", "Envelope de Hilbert — Curtose"),
]


def extrair_sufixo_e_subbanda(coluna):
    partes = coluna.split("_")
    indice_banda = partes[0]
    caminho_no = partes[1]
    nome_descritor = "_".join(partes[2:])
    return indice_banda, caminho_no, nome_descritor


def main():
    if not ENTRADA.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {ENTRADA}")

    df = pd.read_csv(ENTRADA)

    todas_features = [c for c in df.columns if c not in ["arquivo_processado", "particao", "classe", "label"]]
    if len(todas_features) != 96:
        raise ValueError(f"Esperadas 96 colunas de caracteristicas, encontradas {len(todas_features)}")

    resumos = []

    for particao in PARTICOES:
        dados_part = df[df["particao"] == particao]

        for sufixo_desc, titulo_desc in DESCRITORES:
            colunas_desc = [c for c in todas_features if c.endswith(f"_{sufixo_desc}")]
            colunas_desc.sort()

            if len(colunas_desc) != 16:
                raise ValueError(f"Esperadas 16 bandas para {sufixo_desc}, encontradas {len(colunas_desc)}")

            plt.figure(figsize=(12, 5))

            for classe, legenda, cor, estilo in [
                ("bonafide", "Bona fide (Real)", "#1f77b4", "-"),
                ("spoof", "Spoof (F5-TTS)", "#d62728", "--")
            ]:
                grupo = dados_part[dados_part["classe"] == classe]
                medianas = grupo[colunas_desc].median()

                bandas_labels = [c.split("_")[0] for c in colunas_desc]

                plt.plot(
                    bandas_labels,
                    medianas.to_numpy(),
                    marker="o",
                    linestyle=estilo,
                    color=cor,
                    lw=1.8,
                    label=f"{legenda} (n={len(grupo)})"
                )

                for col, valor in zip(colunas_desc, medianas.to_numpy()):
                    idx_b, no_path, _ = extrair_sufixo_e_subbanda(col)
                    resumos.append({
                        "particao": particao,
                        "classe": classe,
                        "descritor": sufixo_desc,
                        "subbanda_indice": idx_b,
                        "caminho_no": no_path,
                        "coluna": col,
                        "mediana": float(valor),
                        "quantidade": len(grupo),
                    })

            plt.title(f"{titulo_desc} por Sub-banda WPT (db4, L=4) — {particao.upper()}", fontsize=12, fontweight="bold")
            plt.xlabel("Sub-banda (b00 = 0-500 Hz ... b15 = 7500-8000 Hz)")
            plt.ylabel("Mediana do Descritor")
            plt.xticks(rotation=45)
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()

            destino_img = PASTA_FIGURAS / f"{sufixo_desc}_{particao}.png"
            plt.savefig(destino_img, dpi=180)
            plt.close()

    df_resumo = pd.DataFrame(resumos)
    df_resumo.to_csv(SAIDA_RESUMO_CSV, index=False)

    df_train = df[df["particao"] == "train"]
    df_val = df[df["particao"] == "validation"]

    med_train_bf = df_train[df_train["classe"] == "bonafide"][todas_features].median()
    med_train_sp = df_train[df_train["classe"] == "spoof"][todas_features].median()
    diff_train = med_train_bf - med_train_sp

    med_val_bf = df_val[df_val["classe"] == "bonafide"][todas_features].median()
    med_val_sp = df_val[df_val["classe"] == "spoof"][todas_features].median()
    diff_val = med_val_bf - med_val_sp

    denominador_train = (med_train_bf.abs() + med_train_sp.abs()) / 2.0
    denominador_train = denominador_train.replace(0, 1e-12)
    contraste_train = diff_train.abs() / denominador_train

    consistentes = np.sign(diff_train) == np.sign(diff_val)
    taxa_consistencia = float(consistentes.sum() / len(todas_features) * 100.0)

    ranking_treino = contraste_train.sort_values(ascending=False)

    linhas_rel = [
        "ANALISE DE CONSISTENCIA DAS 96 CARACTERISTICAS WPT (MES 4)",
        f"Data: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total de caracteristicas analisadas: {len(todas_features)}",
        f"Total de audios: {len(df)} (Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df[df['particao'] == 'test'])})",
        f"Consistencia de sinal entre Treino e Validacao: {taxa_consistencia:.2f}% ({consistentes.sum()}/{len(todas_features)})",
        "=" * 80,
        "",
        "TOP 20 CARACTERISTICAS COM MAIOR CONTRASTE NO TREINO E STATUS NA VALIDACAO:",
        "-" * 80,
    ]

    for rank, (col, score) in enumerate(ranking_treino.head(20).items(), 1):
        sinal_igual = "SIM" if consistentes[col] else "NAO"
        sinal_tr = "+" if diff_train[col] > 0 else "-"
        sinal_vl = "+" if diff_val[col] > 0 else "-"
        linhas_rel.append(
            f"{rank:02d}. {col:30} | Contraste: {score:7.4f} | "
            f"Tr: {sinal_tr} ({diff_train[col]:10.3e}) | "
            f"Val: {sinal_vl} ({diff_val[col]:10.3e}) | Consistente: {sinal_igual}"
        )

    SAIDA_RELATORIO_TXT.write_text("\n".join(linhas_rel), encoding="utf-8")

    print(f"Graficos gerados em: {PASTA_FIGURAS}")
    print(f"Resumo salvo em:     {SAIDA_RESUMO_CSV}")
    print(f"Relatorio salvo em:  {SAIDA_RELATORIO_TXT}")


if __name__ == "__main__":
    main()
