from pathlib import Path
import time
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import hilbert
from scipy.stats import kurtosis

from utils_wpt import decompor_wpt, reconstruir_todas_subbandas

BASE = Path.cwd()
ENTRADA = BASE / "data/metadados/subconjunto_wpt_janelas4s.csv"
SAIDA = BASE / "results/features_wpt_96_janela4s.csv"

SR = 16000
SEGUNDOS = 4
AMOSTRAS = SR * SEGUNDOS
WAVELET = "db4"
NIVEL = 4
MODO = "symmetric"
ORDEM = "freq"
NOMES_DESCRITORES = ["energia", "rms", "zcr", "env_media", "env_desvio", "env_curtose"]


def calcular_energia(x):
    return float(np.sum(np.square(x)))


def calcular_rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def calcular_zcr(x):
    if len(x) <= 1:
        return 0.0
    cruzamentos = np.where(np.diff(np.signbit(x)))[0]
    return float(len(cruzamentos) / len(x))


def calcular_envelope(x):
    sinal_analitico = hilbert(x)
    envelope = np.abs(sinal_analitico)
    media = float(np.mean(envelope))
    desvio = float(np.std(envelope))
    curt = float(kurtosis(envelope, fisher=True, bias=False)) if desvio > 0 else 0.0
    return media, desvio, curt


def carregar_janela(caminho):
    with sf.SoundFile(caminho) as audio:
        if audio.samplerate != SR:
            raise ValueError(f"Frequencia incorreta: {caminho}")
        if audio.channels != 1:
            raise ValueError(f"Audio nao mono: {caminho}")
        if len(audio) < AMOSTRAS:
            raise ValueError(f"Audio menor que 4 segundos: {caminho}")

        inicio = (len(audio) - AMOSTRAS) // 2
        audio.seek(inicio)
        sinal = audio.read(AMOSTRAS, dtype="float64", always_2d=False)

    if len(sinal) != AMOSTRAS:
        raise ValueError(f"Janela incompleta: {caminho}")

    return sinal


def extrair_vetor(sinal):
    wp = decompor_wpt(sinal, wavelet=WAVELET, nivel=NIVEL, modo=MODO)
    subbandas = reconstruir_todas_subbandas(
        wp, nivel=NIVEL, tamanho_original=len(sinal), ordem=ORDEM
    )

    if len(subbandas) != 16:
        raise ValueError(f"Esperadas 16 sub-bandas, obtidas {len(subbandas)}")

    features = {}

    for i, (caminho_no, sinal_b) in enumerate(subbandas):
        arr = np.asarray(sinal_b, dtype=np.float64)
        e = calcular_energia(arr)
        r = calcular_rms(arr)
        z = calcular_zcr(arr)
        m, s, k = calcular_envelope(arr)

        prefixo = f"b{i:02d}_{caminho_no}"
        features[f"{prefixo}_energia"] = e
        features[f"{prefixo}_rms"] = r
        features[f"{prefixo}_zcr"] = z
        features[f"{prefixo}_env_media"] = m
        features[f"{prefixo}_env_desvio"] = s
        features[f"{prefixo}_env_curtose"] = k

    return features


def main():
    if not ENTRADA.exists():
        raise FileNotFoundError(f"Entrada nao encontrada: {ENTRADA}")

    dados = pd.read_csv(ENTRADA)

    print("=== EXTRACAO EM LOTE DOS 96 DESCRITORES WPT (MES 4) ===")
    print("Total de audios:", len(dados))
    print("Wavelet:", WAVELET, "| Nivel:", NIVEL, "| Sub-bandas: 16")
    print("Janela:", SEGUNDOS, "s (", AMOSTRAS, "amostras)")
    print("Descritores por sub-banda:", NOMES_DESCRITORES)
    print("Total de features por audio: 16 x 6 = 96")
    print()

    resultados = []
    t0_inicio = time.perf_counter()

    for i, registro in enumerate(dados.itertuples(index=False), start=1):
        caminho = BASE / registro.arquivo_processado
        sinal = carregar_janela(caminho)
        features = extrair_vetor(sinal)

        linha = {
            "arquivo_processado": registro.arquivo_processado,
            "particao": registro.particao,
            "classe": registro.classe,
            "label": 0 if registro.classe == "bonafide" else 1,
            **features,
        }
        resultados.append(linha)

        if i % 50 == 0 or i == len(dados):
            tempo_decorrido = time.perf_counter() - t0_inicio
            taxa = i / (tempo_decorrido or 1e-6)
            print(f"Processados: {i}/{len(dados)} ({taxa:.1f} audios/s)", flush=True)

    tempo_total = time.perf_counter() - t0_inicio
    resultado = pd.DataFrame(resultados)

    if resultado.isna().any().any():
        raise ValueError("Valores ausentes (NaN) encontrados na matriz extraida")

    if not np.isfinite(resultado.iloc[:, 4:].to_numpy()).all():
        raise ValueError("Valores infinitos encontrados na matriz extraida")

    saida_temporaria = SAIDA.with_suffix(".csv.tmp")
    resultado.to_csv(saida_temporaria, index=False)
    saida_temporaria.replace(SAIDA)

    print()
    print("=== RESULTADO DA EXTRACAO ===")
    print("Total de linhas:", len(resultado))
    print("Colunas de metadados: 4 (arquivo, particao, classe, label)")
    print("Colunas de caracteristicas WPT:", len(resultado.columns) - 4)
    print("Tempo total em CPU:", f"{tempo_total:.2f} s")
    print("Tempo medio por audio:", f"{tempo_total / len(dados) * 1000:.2f} ms")
    print()
    print(pd.crosstab(resultado["particao"], resultado["classe"]))
    print("\nArquivo salvo:", SAIDA)


if __name__ == "__main__":
    main()
