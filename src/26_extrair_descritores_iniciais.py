import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import soundfile as sf
import pywt
from scipy.signal import hilbert
from scipy.stats import kurtosis
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils_wpt import decompor_wpt, reconstruir_todas_subbandas

FS = 16000
DURACAO_JANELA_S = 4.0
AMOSTRAS_JANELA = int(FS * DURACAO_JANELA_S)
WAVELET = "db4"
NIVEL = 4
MODO = "symmetric"
ORDEM = "freq"
N_SUBBANDAS = 2 ** NIVEL

NOMES_DESCRITORES = ["energia", "rms", "zcr", "env_media", "env_desvio", "env_curtose"]

PASTA_PROCESSADOS = Path("data/processados")
PASTA_RESULTS = Path("results")
PASTA_FIGURAS = Path("reports/experimentos_wpt_mes3/figuras")

PASTA_RESULTS.mkdir(parents=True, exist_ok=True)
PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)

SAIDA_CONFIG_JSON = PASTA_RESULTS / "feature_extraction_config_mes4.json"
SAIDA_CSV = PASTA_RESULTS / "features_iniciais_96_piloto.csv"
SAIDA_DESAFIO_TXT = PASTA_RESULTS / "desafio_mes4_contraste_features.txt"
SAIDA_PNG = PASTA_FIGURAS / "desafio_mes4_contraste_features.png"


def calcular_energia(x):
    return float(np.sum(np.square(x)))


def calcular_rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def calcular_zcr(x):
    if len(x) <= 1:
        return 0.0
    cruzamentos = np.where(np.diff(np.signbit(x)))[0]
    return float(len(cruzamentos) / len(x))


def calcular_descritores_envelope(x):
    sinal_analitico = hilbert(x)
    envelope = np.abs(sinal_analitico)
    media = float(np.mean(envelope))
    desvio = float(np.std(envelope))
    curt = float(kurtosis(envelope, fisher=True, bias=False)) if desvio > 0 else 0.0
    return media, desvio, curt


def extrair_descritores_subbanda(sinal_subbanda):
    arr = np.asarray(sinal_subbanda, dtype=np.float64)
    e = calcular_energia(arr)
    r = calcular_rms(arr)
    z = calcular_zcr(arr)
    m, s, k = calcular_descritores_envelope(arr)
    return [e, r, z, m, s, k]


def extrair_vetor_audio(sinal):
    wp = decompor_wpt(sinal, wavelet=WAVELET, nivel=NIVEL, modo=MODO)
    subbandas = reconstruir_todas_subbandas(
        wp, nivel=NIVEL, tamanho_original=len(sinal), ordem=ORDEM
    )

    vetor = []
    nomes_colunas = []

    for i, (caminho_no, sinal_b) in enumerate(subbandas):
        desc = extrair_descritores_subbanda(sinal_b)
        vetor.extend(desc)

        for nome_desc in NOMES_DESCRITORES:
            nomes_colunas.append(f"b{i:02d}_{caminho_no}_{nome_desc}")

    return np.array(vetor, dtype=np.float64), nomes_colunas


def carregar_janela_audio(caminho):
    audio, sr = sf.read(caminho, dtype="float64")
    if audio.ndim != 1:
        audio = audio.mean(axis=1)

    if len(audio) < AMOSTRAS_JANELA:
        inicio = 0
        sinal = np.pad(audio, (0, AMOSTRAS_JANELA - len(audio)), mode="constant")
    else:
        inicio = (len(audio) - AMOSTRAS_JANELA) // 2
        sinal = audio[inicio:inicio + AMOSTRAS_JANELA]

    return sinal


def salvar_configuracao():
    config = {
        "sampling_rate_hz": FS,
        "wavelet_mother": WAVELET,
        "decomposition_level": NIVEL,
        "padding_mode": MODO,
        "subband_order": ORDEM,
        "subbands_count": N_SUBBANDAS,
        "bandwidth_hz": (FS / 2) / N_SUBBANDAS,
        "window_duration_seconds": DURACAO_JANELA_S,
        "window_samples": AMOSTRAS_JANELA,
        "descriptors_per_subband": NOMES_DESCRITORES,
        "descriptors_count_per_subband": len(NOMES_DESCRITORES),
        "total_feature_dimensions": N_SUBBANDAS * len(NOMES_DESCRITORES),
        "implementation_reference": "mes04_descritores_iniciais.pdf",
    }
    with open(SAIDA_CONFIG_JSON, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    return config


def main():
    salvar_configuracao()

    arquivos_bf = sorted((PASTA_PROCESSADOS / "bonafide" / "train").glob("*.wav"))[:10]
    arquivos_sp = sorted((PASTA_PROCESSADOS / "spoof" / "train").glob("*.wav"))[:10]

    amostra = (
        [(f, "bonafide", 0) for f in arquivos_bf] +
        [(f, "spoof", 1) for f in arquivos_sp]
    )

    matriz_linhas = []
    nomes_colunas_features = None

    t0_geral = time.perf_counter()

    for caminho, classe, label in amostra:
        sinal = carregar_janela_audio(caminho)
        vetor_feats, colunas = extrair_vetor_audio(sinal)

        if nomes_colunas_features is None:
            nomes_colunas_features = colunas

        registro = {
            "arquivo": caminho.name,
            "classe": classe,
            "label": label,
        }
        for col, val in zip(nomes_colunas_features, vetor_feats):
            registro[col] = val

        matriz_linhas.append(registro)

    tempo_total_extracao = time.perf_counter() - t0_geral
    df_features = pd.DataFrame(matriz_linhas)
    df_features.to_csv(SAIDA_CSV, index=False)

    df_bf = df_features[df_features["classe"] == "bonafide"][nomes_colunas_features]
    df_sp = df_features[df_features["classe"] == "spoof"][nomes_colunas_features]

    media_bf = df_bf.mean()
    media_sp = df_sp.mean()

    denominador = (media_bf.abs() + media_sp.abs()) / 2.0
    denominador = denominador.replace(0, 1e-12)
    contraste_relativo = (media_bf - media_sp).abs() / denominador

    ranking = contraste_relativo.sort_values(ascending=False)

    linhas_desafio = [
        "DESAFIO DO MÊS 4 — CONTRASTE DE DESCRITORES: BONAFIDE VS SPOOF",
        f"Data: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Dimensões do vetor: {len(nomes_colunas_features)} características (16 sub-bandas x 6 descritores)",
        f"Amostra piloto analisada: {len(df_features)} áudios (10 bonafide, 10 spoof F5-TTS)",
        f"Tempo total de extração em CPU: {tempo_total_extracao * 1000:.2f} ms ({tempo_total_extracao / len(df_features) * 1000:.2f} ms por áudio)",
        "=" * 75,
        "",
        "TOP 15 CARACTERÍSTICAS COM MAIOR CONTRASTE RELATIVO (|BF - SP| / Média):",
        "-" * 75,
    ]

    for rank, (col, score) in enumerate(ranking.head(15).items(), 1):
        v_bf = float(media_bf[col])
        v_sp = float(media_sp[col])
        linhas_desafio.append(
            f"{rank:02d}. {col:32} | Contraste: {score:8.4f} | Bona fide: {v_bf:12.4e} | Spoof: {v_sp:12.4e}"
        )

    SAIDA_DESAFIO_TXT.write_text("\n".join(linhas_desafio), encoding="utf-8")

    top_10 = ranking.head(10)
    fig, ax = plt.subplots(figsize=(12, 6))
    y_pos = np.arange(len(top_10))
    ax.barh(y_pos, top_10.values[::-1], color="#1f77b4", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_10.index[::-1], fontsize=10)
    ax.set_xlabel("Contraste Relativo (|Bona Fide - Spoof| / Média)", fontsize=11)
    ax.set_title("Top 10 Características do Mês 4 com Maior Contraste Relativo Inicial (Desafio Mês 4)", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(SAIDA_PNG, dpi=200)
    plt.close()

    print(f"Configuração JSON salva em: {SAIDA_CONFIG_JSON}")
    print(f"Matriz de 96 features salva em: {SAIDA_CSV}")
    print(f"Relatório do Desafio salvo em: {SAIDA_DESAFIO_TXT}")
    print(f"Gráfico gerado em: {SAIDA_PNG}")


if __name__ == "__main__":
    main()
