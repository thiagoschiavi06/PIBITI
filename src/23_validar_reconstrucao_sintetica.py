from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils_wpt import (
    decompor_wpt,
    reconstruir_wpt,
    reconstruir_todas_subbandas,
    validar_reconstrucao,
    calcular_energia
)


FS = 16000  
DURACAO = 1.0  
N_AMOSTRAS = int(FS * DURACAO)
WAVELET = "db4"
NIVEL = 4
MODO = "symmetric"
ORDEM = "freq"
N_SUBBANDAS = 2 ** NIVEL  
LARGURA_BANDA_HZ = (FS / 2) / N_SUBBANDAS  

PASTA_RESULTS = Path("results")
PASTA_FIGURAS = Path("reports/experimentos_wpt_mes3/figuras")
PASTA_RESULTS.mkdir(parents=True, exist_ok=True)
PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)

SAIDA_CSV = PASTA_RESULTS / "validacao_reconstrucao_sintetica.csv"
SAIDA_TXT = PASTA_RESULTS / "validacao_reconstrucao_sintetica.txt"
SAIDA_PNG = PASTA_FIGURAS / "validacao_reconstrucao_sintetica.png"


def gerar_sinais_sinteticos() -> dict:

    t = np.arange(0, DURACAO, 1 / FS)

    sinal_1 = np.sin(2 * np.pi * 440 * t)

    sinal_2 = np.sin(2 * np.pi * 440 * t) + 0.8 * np.sin(2 * np.pi * 2200 * t)

    np.random.seed(42)
    sinal_3 = np.random.randn(N_AMOSTRAS)

    sinal_4 = sinal_1 + 0.3 * np.random.randn(N_AMOSTRAS)

    sinal_5 = np.zeros(N_AMOSTRAS, dtype=np.float64)
    sinal_5[5000] = 1.0

    sinal_6 = np.ones(N_AMOSTRAS, dtype=np.float64)

    sinal_7 = (
        np.sin(2 * np.pi * 200 * t) +
        np.sin(2 * np.pi * 1500 * t) +
        np.sin(2 * np.pi * 6000 * t)
    )

    return {
        "Senoide 440 Hz": (sinal_1, "Banda 0 (0-500 Hz)"),
        "Duas Senoides (440 + 2200 Hz)": (sinal_2, "Bandas 0 e 4"),
        "Ruído Gaussiano Branco": (sinal_3, "Todas as bandas"),
        "Senoide 440 Hz + Ruído": (sinal_4, "Banda 0 predominante"),
        "Impulso Unitário": (sinal_5, "Banda ampla"),
        "Sinal Constante (DC)": (sinal_6, "Banda 0 (0 Hz)"),
        "Três Senoides (200, 1500, 6000 Hz)": (sinal_7, "Bandas 0, 3 e 12"),
    }


def main():
    print("=" * 70)
    print("PIBITI — VALIDAÇÃO DA RECONSTRUÇÃO WPT COM SINAIS SINTÉTICOS")
    print("=" * 70)
    print(f"Taxa de amostragem: {FS} Hz | Duração: {DURACAO} s ({N_AMOSTRAS} amostras)")
    print(f"Wavelet: {WAVELET} | Nível L: {NIVEL} | Sub-bandas: {N_SUBBANDAS}")
    print(f"Modo de borda: {MODO} | Ordenação: {ORDEM} | Largura por sub-banda: {LARGURA_BANDA_HZ} Hz")
    print("-" * 70)

    sinais = gerar_sinais_sinteticos()
    resultados = []
    linhas_relatorio = [
        "RELATÓRIO DE VALIDAÇÃO DA RECONSTRUÇÃO WPT COM SINAIS SINTÉTICOS",
        f"Data de execução: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Configuração: Wavelet={WAVELET}, Nível={NIVEL}, Modo={MODO}, Ordem={ORDEM}",
        f"Sub-bandas: {N_SUBBANDAS} (largura teórica: {LARGURA_BANDA_HZ:.1f} Hz cada)",
        "=" * 70,
        ""
    ]

    dados_para_grafico = {}

    for nome, (sinal_orig, expectativa) in sinais.items():
        print(f"\n>>> Testando: {nome} (Expectativa: {expectativa})")
		
        t0 = time.perf_counter()
        wp = decompor_wpt(sinal_orig, wavelet=WAVELET, nivel=NIVEL, modo=MODO)
        t_decomp_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        sinal_rec_completo = reconstruir_wpt(wp, tamanho_original=len(sinal_orig))
        t_rec_completa_ms = (time.perf_counter() - t0) * 1000.0

        val_completa = validar_reconstrucao(sinal_orig, sinal_rec_completo)

        t0 = time.perf_counter()
        subbandas = reconstruir_todas_subbandas(
            wp, nivel=NIVEL, tamanho_original=len(sinal_orig), ordem=ORDEM
        )
        t_rec_subbandas_ms = (time.perf_counter() - t0) * 1000.0

        soma_subbandas = np.sum([s for _, s in subbandas], axis=0)
        val_soma = validar_reconstrucao(sinal_orig, soma_subbandas)

        energias_subbandas = [calcular_energia(s) for _, s in subbandas]
        energia_total_subbandas = sum(energias_subbandas)
        banda_max_energia = int(np.argmax(energias_subbandas))

        dados_para_grafico[nome] = {
            "sinal_orig": sinal_orig,
            "sinal_rec": sinal_rec_completo,
            "soma_subbandas": soma_subbandas,
            "energias": energias_subbandas,
            "t_decomp_ms": t_decomp_ms,
            "t_rec_subbandas_ms": t_rec_subbandas_ms,
        }

        print(f"  [Decomposição]      Tempo CPU: {t_decomp_ms:.2f} ms")
        print(f"  [Rec. Completa]     MSE: {val_completa['mse']:.2e} | Max Erro: {val_completa['erro_maximo']:.2e} | Equivalente: {val_completa['equivalente_atol_1e-10']}")
        print(f"  [Soma 16 Sub-bandas] MSE: {val_soma['mse']:.2e} | Max Erro: {val_soma['erro_maximo']:.2e} | Equivalente: {val_soma['equivalente_atol_1e-10']}")
        print(f"  [Energia]           Original: {val_completa['energia_original']:.2f} | Soma Sub-bandas: {energia_total_subbandas:.2f} | Dif: {val_soma['erro_energia_perc']:.4f}%")
        print(f"  [Banda dominante]   Banda {banda_max_energia} ({banda_max_energia * LARGURA_BANDA_HZ:.0f}-{(banda_max_energia + 1) * LARGURA_BANDA_HZ:.0f} Hz) com {energias_subbandas[banda_max_energia] / (energia_total_subbandas or 1) * 100:.2f}% da energia")

        registro = {
            "sinal": nome,
            "amostras": len(sinal_orig),
            "tempo_decomp_ms": round(t_decomp_ms, 3),
            "tempo_rec_completa_ms": round(t_rec_completa_ms, 3),
            "tempo_rec_subbandas_ms": round(t_rec_subbandas_ms, 3),
            "mse_rec_completa": val_completa["mse"],
            "max_erro_rec_completa": val_completa["erro_maximo"],
            "equiv_rec_completa": val_completa["equivalente_atol_1e-10"],
            "mse_soma_subbandas": val_soma["mse"],
            "max_erro_soma_subbandas": val_soma["erro_maximo"],
            "equiv_soma_subbandas": val_soma["equivalente_atol_1e-10"],
            "energia_original": val_completa["energia_original"],
            "energia_soma_subbandas": energia_total_subbandas,
            "erro_energia_perc": val_soma["erro_energia_perc"],
            "banda_max_energia": banda_max_energia,
            "perc_energia_banda_max": (
                energias_subbandas[banda_max_energia] / (energia_total_subbandas or 1) * 100.0
            ),
        }
        resultados.append(registro)

        linhas_relatorio.extend([
            f"SINAL: {nome}",
            f"- Amostras: {len(sinal_orig)} | Duração: {DURACAO} s",
            f"- Tempo CPU decomposição: {t_decomp_ms:.3f} ms",
            f"- Tempo CPU reconstrução das 16 sub-bandas: {t_rec_subbandas_ms:.3f} ms",
            f"- Reconstrução completa nativa: MSE = {val_completa['mse']:.2e}, Max Erro = {val_completa['erro_maximo']:.2e} (Equivalente: {val_completa['equivalente_atol_1e-10']})",
            f"- Soma temporal das 16 sub-bandas: MSE = {val_soma['mse']:.2e}, Max Erro = {val_soma['erro_maximo']:.2e} (Equivalente: {val_soma['equivalente_atol_1e-10']})",
            f"- Energia original: {val_completa['energia_original']:.4f} | Soma energias sub-bandas: {energia_total_subbandas:.4f}",
            f"- Diferença percentual de energia: {val_soma['erro_energia_perc']:.6f}%",
            f"- Banda com pico de energia: Sub-banda {banda_max_energia} ({banda_max_energia * LARGURA_BANDA_HZ:.0f}-{(banda_max_energia + 1) * LARGURA_BANDA_HZ:.0f} Hz) detém {registro['perc_energia_banda_max']:.2f}% da energia total",
            "-" * 70,
            ""
        ])

    df = pd.DataFrame(resultados)
    df.to_csv(SAIDA_CSV, index=False)
    SAIDA_TXT.write_text("\n".join(linhas_relatorio), encoding="utf-8")

    print("\n" + "=" * 70)
    print("GERANDO FIGURA COMPARATIVA DE VALIDAÇÃO CIENTÍFICA...")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    s1_orig = dados_para_grafico["Senoide 440 Hz"]["sinal_orig"][:200]
    s1_soma = dados_para_grafico["Senoide 440 Hz"]["soma_subbandas"][:200]
    res_s1 = s1_orig - s1_soma
    ax1 = axes[0, 0]
    ax1.plot(s1_orig, label="Original (440 Hz)", color="#1f77b4", lw=1.5)
    ax1.plot(s1_soma, label="Soma das 16 sub-bandas", color="#ff7f0e", linestyle="--", lw=1.2)
    ax1.plot(res_s1, label=f"Resíduo (Max={np.max(np.abs(res_s1)):.1e})", color="#2ca02c", alpha=0.7)
    ax1.set_title("1. Senoide Pura 440 Hz: Original vs Soma das Sub-bandas", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Amostras (zoom primeiras 200)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right", fontsize=9)

    s7_orig = dados_para_grafico["Três Senoides (200, 1500, 6000 Hz)"]["sinal_orig"][:300]
    s7_soma = dados_para_grafico["Três Senoides (200, 1500, 6000 Hz)"]["soma_subbandas"][:300]
    res_s7 = s7_orig - s7_soma
    ax2 = axes[0, 1]
    ax2.plot(s7_orig, label="Original (200+1500+6000 Hz)", color="#1f77b4", lw=1.5)
    ax2.plot(s7_soma, label="Soma das 16 sub-bandas", color="#ff7f0e", linestyle="--", lw=1.2)
    ax2.plot(res_s7, label=f"Resíduo (Max={np.max(np.abs(res_s7)):.1e})", color="#2ca02c", alpha=0.7)
    ax2.set_title("2. Desafio Final Mês 3: Três Senoides Reconstruídas", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Amostras (zoom primeiras 300)")
    ax2.set_ylabel("Amplitude")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper right", fontsize=9)

    e1 = np.array(dados_para_grafico["Senoide 440 Hz"]["energias"])
    e1_perc = e1 / sum(e1) * 100.0
    e7 = np.array(dados_para_grafico["Três Senoides (200, 1500, 6000 Hz)"]["energias"])
    e7_perc = e7 / sum(e7) * 100.0

    x_bandas = np.arange(N_SUBBANDAS)
    largura = 0.35
    ax3 = axes[1, 0]
    ax3.bar(x_bandas - largura / 2, e1_perc, width=largura, label="Senoide 440 Hz (esperado Banda 0)", color="#1f77b4")
    ax3.bar(x_bandas + largura / 2, e7_perc, width=largura, label="3 Senoides (esperado Bandas 0, 3, 12)", color="#d62728")
    ax3.set_title("3. Distribuição de Energia (%) nas 16 Sub-bandas WPT (order='freq')", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Índice da Sub-banda (0 = 0-500 Hz ... 15 = 7500-8000 Hz)")
    ax3.set_ylabel("Energia Relativa (%)")
    ax3.set_xticks(x_bandas)
    ax3.grid(alpha=0.3)
    ax3.legend(fontsize=9)

    nomes_curtos = [
        "Senoide 440", "Duas Sen.", "Ruído", "Sen+Ruído", "Impulso", "DC", "3 Senoides"
    ]
    t_decomp = [d["t_decomp_ms"] for d in dados_para_grafico.values()]
    t_rec = [d["t_rec_subbandas_ms"] for d in dados_para_grafico.values()]

    ax4 = axes[1, 1]
    ax4.bar(nomes_curtos, t_decomp, label="Decomposição WPT (ms)", color="#2ca02c")
    ax4.bar(nomes_curtos, t_rec, bottom=t_decomp, label="Rec. 16 Sub-bandas (ms)", color="#9467bd")
    ax4.set_title("4. Custo Computacional em CPU por Sinal (16.000 amostras)", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Tempo de CPU (ms)")
    ax4.set_xticks(range(len(nomes_curtos)))
    ax4.set_xticklabels(nomes_curtos, rotation=30, ha="right")
    ax4.grid(alpha=0.3)
    ax4.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(SAIDA_PNG, dpi=200)
    plt.close()

    print(f"Resultados tabulares salvos em: {SAIDA_CSV}")
    print(f"Relatório textual salvo em:     {SAIDA_TXT}")
    print(f"Figura gerada em:               {SAIDA_PNG}")
    print("=" * 70)
    print("VALIDAÇÃO CONCLUÍDA COM SUCESSO!")
    print("=" * 70)


if __name__ == "__main__":
    main()
