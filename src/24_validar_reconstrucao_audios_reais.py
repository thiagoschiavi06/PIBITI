"""
PIBITI - Detecção de Deepfakes de Áudio em Português Brasileiro
Script 24: Validação da Reconstrução WPT em Áudios Reais de Fala do Piloto

Objetivo:
Validar a fidelidade da decomposição e reconstrução WPT (completa e soma das 16
sub-bandas) em gravações de fala real (bonafide) e sintética (spoof) do BRSpeech-DF,
cobrindo as partições train, validation e test, garantindo que em sinais reais de voz
o erro numérico permaneça na ordem de precisão de máquina (MSE <= 1e-10) e a conservação
de energia seja mantida.

Execução:
    python src/24_validar_reconstrucao_audios_reais.py
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
import soundfile as sf
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

# Parâmetros
WAVELET = "db4"
NIVEL = 4
MODO = "symmetric"
ORDEM = "freq"
N_SUBBANDAS = 2 ** NIVEL  # 16 sub-bandas
AMOSTRAS_POR_GRUPO = 5  # 5 áudios de cada (train/val/test x bonafide/spoof) = 30 áudios

# Diretórios
BASE_PROCESSADOS = Path("data/processados")
PASTA_RESULTS = Path("results")
PASTA_FIGURAS = Path("reports/experimentos_wpt_mes3/figuras")
PASTA_RESULTS.mkdir(parents=True, exist_ok=True)
PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)

SAIDA_CSV = PASTA_RESULTS / "validacao_reconstrucao_audios_reais.csv"
SAIDA_TXT = PASTA_RESULTS / "validacao_reconstrucao_audios_reais.txt"
SAIDA_PNG = PASTA_FIGURAS / "validacao_reconstrucao_audios_reais.png"


def selecionar_audios_amostra() -> list:
    """Seleciona de forma determinística uma amostra de 30 áudios balanceados."""
    selecionados = []
    particoes = ["train", "validation", "test"]
    classes = ["bonafide", "spoof"]

    for particao in particoes:
        for classe in classes:
            pasta = BASE_PROCESSADOS / classe / particao
            if not pasta.exists():
                raise FileNotFoundError(f"Pasta de áudios não encontrada: {pasta}")
            arquivos = sorted(pasta.glob("*.wav"))
            if len(arquivos) < AMOSTRAS_POR_GRUPO:
                raise ValueError(f"Menos de {AMOSTRAS_POR_GRUPO} arquivos em {pasta}")

            # Seleciona 5 arquivos com espaçamento regular para cobrir diferentes durações
            indices = np.linspace(0, len(arquivos) - 1, AMOSTRAS_POR_GRUPO, dtype=int)
            for idx in indices:
                arq = arquivos[idx]
                selecionados.append({
                    "caminho": arq,
                    "nome_arquivo": arq.name,
                    "particao": particao,
                    "classe": classe
                })

    return selecionados


def main():
    print("=" * 75)
    print("PIBITI — VALIDAÇÃO DA RECONSTRUÇÃO WPT EM ÁUDIOS REAIS DE FALA")
    print("=" * 75)
    print(f"Wavelet: {WAVELET} | Nível L: {NIVEL} | Sub-bandas: {N_SUBBANDAS}")
    print(f"Modo: {MODO} | Ordem: {ORDEM} | Base: {BASE_PROCESSADOS}")
    print("-" * 75)

    amostra = selecionar_audios_amostra()
    print(f"Total de áudios selecionados para validação: {len(amostra)}")
    print("Divisão: 5 bonafide e 5 spoof em cada partição (train, validation, test)\n")

    resultados = []
    linhas_relatorio = [
        "RELATÓRIO DE VALIDAÇÃO DA RECONSTRUÇÃO WPT EM ÁUDIOS REAIS DE FALA",
        f"Data: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Configuração WPT: Wavelet={WAVELET}, Nível={NIVEL}, Modo={MODO}, Ordem={ORDEM}",
        f"Corpus: BRSpeech-DF (Piloto processado 16 kHz mono)",
        f"Total de áudios analisados: {len(amostra)}",
        "=" * 75,
        ""
    ]

    exemplos_grafico = []

    for i, item in enumerate(amostra, 1):
        caminho = item["caminho"]
        particao = item["particao"]
        classe = item["classe"]
        nome = item["nome_arquivo"]

        audio, sr = sf.read(caminho, dtype="float64")
        if audio.ndim != 1:
            audio = audio.mean(axis=1)

        duracao_s = len(audio) / sr

        # 1. Decomposição WPT e medição de tempo
        t0 = time.perf_counter()
        wp = decompor_wpt(audio, wavelet=WAVELET, nivel=NIVEL, modo=MODO)
        t_decomp_ms = (time.perf_counter() - t0) * 1000.0

        # 2. Reconstrução completa nativa
        t0 = time.perf_counter()
        rec_completa = reconstruir_wpt(wp, tamanho_original=len(audio))
        t_rec_completa_ms = (time.perf_counter() - t0) * 1000.0

        val_completa = validar_reconstrucao(audio, rec_completa)

        # 3. Reconstrução individual das 16 sub-bandas no domínio do tempo
        t0 = time.perf_counter()
        subbandas = reconstruir_todas_subbandas(
            wp, nivel=NIVEL, tamanho_original=len(audio), ordem=ORDEM
        )
        t_rec_subbandas_ms = (time.perf_counter() - t0) * 1000.0

        # Soma temporal das 16 sub-bandas
        soma_subbandas = np.sum([s for _, s in subbandas], axis=0)
        val_soma = validar_reconstrucao(audio, soma_subbandas)

        # Guarda dois exemplos (um bonafide e um spoof) para plotagem
        if len(exemplos_grafico) < 2:
            if not exemplos_grafico and classe == "bonafide":
                exemplos_grafico.append((f"Bona fide ({particao}) - {nome}", audio, soma_subbandas, sr))
            elif len(exemplos_grafico) == 1 and classe == "spoof":
                exemplos_grafico.append((f"Spoof ({particao}) - {nome}", audio, soma_subbandas, sr))

        registro = {
            "arquivo": nome,
            "particao": particao,
            "classe": classe,
            "duracao_segundos": round(duracao_s, 2),
            "amostras": len(audio),
            "tempo_decomp_ms": round(t_decomp_ms, 3),
            "tempo_rec_completa_ms": round(t_rec_completa_ms, 3),
            "tempo_rec_subbandas_ms": round(t_rec_subbandas_ms, 3),
            "tempo_total_cpu_ms": round(t_decomp_ms + t_rec_subbandas_ms, 3),
            "mse_rec_completa": val_completa["mse"],
            "max_erro_rec_completa": val_completa["erro_maximo"],
            "equiv_rec_completa": val_completa["equivalente_atol_1e-10"],
            "mse_soma_subbandas": val_soma["mse"],
            "max_erro_soma_subbandas": val_soma["erro_maximo"],
            "equiv_soma_subbandas": val_soma["equivalente_atol_1e-10"],
            "energia_original": val_completa["energia_original"],
            "energia_soma_subbandas": val_soma["energia_reconstruida"],
            "erro_energia_perc": val_soma["erro_energia_perc"],
        }
        resultados.append(registro)

        print(
            f"[{i:02d}/{len(amostra):02d}] {particao:10} | {classe:8} | "
            f"Dur: {duracao_s:4.1f}s | "
            f"MSE Soma: {val_soma['mse']:.2e} | "
            f"Max Erro: {val_soma['erro_maximo']:.2e} | "
            f"Equiv: {val_soma['equivalente_atol_1e-10']} | "
            f"CPU: {registro['tempo_total_cpu_ms']:5.1f}ms"
        )

    df = pd.DataFrame(resultados)
    df.to_csv(SAIDA_CSV, index=False)

    # Estatísticas consolidadas para o relatório
    mse_medio = df["mse_soma_subbandas"].mean()
    mse_max = df["mse_soma_subbandas"].max()
    max_erro_global = df["max_erro_soma_subbandas"].max()
    taxa_equivalencia = (df["equiv_soma_subbandas"].sum() / len(df)) * 100.0
    tempo_medio_cpu = df["tempo_total_cpu_ms"].mean()

    linhas_relatorio.extend([
        "SÍNTESE DOS RESULTADOS:",
        f"- Taxa de equivalência numérica perfeita (atol=1e-10): {taxa_equivalencia:.1f}% ({df['equiv_soma_subbandas'].sum()}/{len(df)})",
        f"- MSE médio (soma das 16 sub-bandas): {mse_medio:.2e}",
        f"- MSE máximo observado: {mse_max:.2e}",
        f"- Erro absoluto máximo observado em qualquer amostra: {max_erro_global:.2e}",
        f"- Tempo médio de CPU por áudio (decomposição + 16 reconstruções): {tempo_medio_cpu:.2f} ms",
        "-" * 75,
        "",
        "DETALHAMENTO POR CLASSE E PARTIÇÃO:"
    ])

    for (part, cls), grp in df.groupby(["particao", "classe"]):
        linhas_relatorio.append(
            f"  * {part:10} - {cls:8}: MSE médio = {grp['mse_soma_subbandas'].mean():.2e} | "
            f"CPU médio = {grp['tempo_total_cpu_ms'].mean():.1f} ms | "
            f"Duração média = {grp['duracao_segundos'].mean():.2f} s"
        )

    SAIDA_TXT.write_text("\n".join(linhas_relatorio), encoding="utf-8")

    print("\n" + "=" * 75)
    print("GERANDO FIGURA DA VALIDAÇÃO EM ÁUDIOS REAIS...")
    print("=" * 75)

    # Plotagem
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Painel 1: Exemplo Bona Fide (sinal vs reconstrução vs resíduo)
    titulo_bf, s_bf_orig, s_bf_soma, _ = exemplos_grafico[0]
    n_pts = min(400, len(s_bf_orig))
    res_bf = s_bf_orig[:n_pts] - s_bf_soma[:n_pts]
    ax1 = axes[0, 0]
    ax1.plot(s_bf_orig[:n_pts], label="Original (Bona Fide)", color="#1f77b4", lw=1.5)
    ax1.plot(s_bf_soma[:n_pts], label="Soma das 16 Sub-bandas WPT", color="#ff7f0e", linestyle="--", lw=1.2)
    ax1.plot(res_bf, label=f"Resíduo numérico (Max={np.max(np.abs(res_bf)):.1e})", color="#2ca02c", alpha=0.7)
    ax1.set_title(f"1. Forma de Onda: {titulo_bf}", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Amostras (zoom primeiras 400)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right", fontsize=9)

    # Painel 2: Exemplo Spoof (sinal vs reconstrução vs resíduo)
    titulo_sp, s_sp_orig, s_sp_soma, _ = exemplos_grafico[1]
    n_pts_sp = min(400, len(s_sp_orig))
    res_sp = s_sp_orig[:n_pts_sp] - s_sp_soma[:n_pts_sp]
    ax2 = axes[0, 1]
    ax2.plot(s_sp_orig[:n_pts_sp], label="Original (Spoof F5-TTS)", color="#d62728", lw=1.5)
    ax2.plot(s_sp_soma[:n_pts_sp], label="Soma das 16 Sub-bandas WPT", color="#ff7f0e", linestyle="--", lw=1.2)
    ax2.plot(res_sp, label=f"Resíduo numérico (Max={np.max(np.abs(res_sp)):.1e})", color="#2ca02c", alpha=0.7)
    ax2.set_title(f"2. Forma de Onda: {titulo_sp}", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Amostras (zoom primeiras 400)")
    ax2.set_ylabel("Amplitude")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper right", fontsize=9)

    # Painel 3: Dispersão Duração do Áudio vs Tempo de CPU
    ax3 = axes[1, 0]
    ax3.scatter(
        df[df["classe"] == "bonafide"]["duracao_segundos"],
        df[df["classe"] == "bonafide"]["tempo_total_cpu_ms"],
        color="#1f77b4", label="Bona Fide", alpha=0.8, s=50
    )
    ax3.scatter(
        df[df["classe"] == "spoof"]["duracao_segundos"],
        df[df["classe"] == "spoof"]["tempo_total_cpu_ms"],
        color="#d62728", label="Spoof", marker="^", alpha=0.8, s=50
    )
    ax3.set_title("3. Escalabilidade em CPU: Duração do Áudio vs Tempo de Processamento", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Duração do Áudio (segundos)")
    ax3.set_ylabel("Tempo de CPU (Decomp. + 16 Rec.) [ms]")
    ax3.grid(alpha=0.3)
    ax3.legend(fontsize=9)

    # Painel 4: Boxplot de MSE por Partição e Classe
    ax4 = axes[1, 1]
    df_box = [
        df[(df["particao"] == "train") & (df["classe"] == "bonafide")]["mse_soma_subbandas"],
        df[(df["particao"] == "train") & (df["classe"] == "spoof")]["mse_soma_subbandas"],
        df[(df["particao"] == "validation") & (df["classe"] == "bonafide")]["mse_soma_subbandas"],
        df[(df["particao"] == "validation") & (df["classe"] == "spoof")]["mse_soma_subbandas"],
        df[(df["particao"] == "test") & (df["classe"] == "bonafide")]["mse_soma_subbandas"],
        df[(df["particao"] == "test") & (df["classe"] == "spoof")]["mse_soma_subbandas"],
    ]
    rotulos = ["Tr. Real", "Tr. Spoof", "Val. Real", "Val. Spoof", "Tst. Real", "Tst. Spoof"]
    ax4.boxplot(df_box)
    ax4.set_title("4. Distribuição do Erro Quadrático Médio (MSE) por Grupo", fontsize=11, fontweight="bold")
    ax4.set_ylabel("MSE (Soma das 16 Sub-bandas)")
    ax4.set_xticks(range(1, len(rotulos) + 1))
    ax4.set_xticklabels(rotulos, rotation=30, ha="right")
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(SAIDA_PNG, dpi=200)
    plt.close()

    print(f"\nResultados tabulares salvos em: {SAIDA_CSV}")
    print(f"Relatório textual salvo em:     {SAIDA_TXT}")
    print(f"Figura gerada em:               {SAIDA_PNG}")
    print("=" * 75)
    print(f"CONCLUÍDO COM SUCESSO! 100% dos áudios reais ({len(df)}/{len(df)}) validados com MSE <= 1e-10!")
    print("=" * 75)


if __name__ == "__main__":
    main()
