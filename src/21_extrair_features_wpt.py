from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import pywt

BASE = Path.cwd()

ENTRADA = BASE / "data/metadados/subconjunto_wpt_janelas4s.csv"
SAIDA = BASE / "results/wpt_features_db4_l4_janela4s.csv"

SR = 16000
SEGUNDOS = 4
AMOSTRAS = SR * SEGUNDOS
WAVELET = "db4"
NIVEL = 4


def entropia_shannon(coeficientes):
    energia = np.square(coeficientes)
    total = np.sum(energia)

    if total <= 0:
        return 0.0

    probabilidades = energia / total
    positivas = probabilidades[probabilidades > 0]

    return float(-np.sum(positivas * np.log(positivas)))


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

        sinal = audio.read(
            AMOSTRAS,
            dtype="float64",
            always_2d=False
        )

    if len(sinal) != AMOSTRAS:
        raise ValueError(f"Janela incompleta: {caminho}")

    return sinal


def extrair_features(sinal):
    pacote = pywt.WaveletPacket(
        data=sinal,
        wavelet=WAVELET,
        mode="symmetric",
        maxlevel=NIVEL
    )

    nos = pacote.get_level(NIVEL, order="natural")

    if len(nos) != 16:
        raise ValueError("Quantidade inesperada de sub-bandas")

    energias = [
        float(np.sum(np.square(no.data)))
        for no in nos
    ]

    energia_total = sum(energias)
    features = {}

    for no, energia in zip(nos, energias):
        caminho = no.path
        coef = np.asarray(no.data, dtype=np.float64)

        features[f"energia_rel_{caminho}"] = (
            energia / energia_total
            if energia_total > 0 else 0.0
        )

        features[f"entropia_{caminho}"] = (
            entropia_shannon(coef)
        )

        features[f"rms_{caminho}"] = float(
            np.sqrt(np.mean(np.square(coef)))
        )

        features[f"desvio_{caminho}"] = float(
            np.std(coef)
        )

    return features


def main():
    dados = pd.read_csv(ENTRADA)

    print("=== EXTRACAO WPT EM LOTE ===")
    print("Audios:", len(dados))
    print("Wavelet:", WAVELET)
    print("Nivel:", NIVEL)
    print("Janela:", SEGUNDOS, "segundos")
    print()

    resultados = []

    for i, registro in enumerate(
        dados.itertuples(index=False), start=1
    ):
        caminho = BASE / registro.arquivo_processado
        sinal = carregar_janela(caminho)

        features = extrair_features(sinal)

        linha = {
            "arquivo_processado": registro.arquivo_processado,
            "particao": registro.particao,
            "classe": registro.classe,
            "label": 0 if registro.classe == "bonafide" else 1,
            **features,
        }

        resultados.append(linha)

        if i % 50 == 0 or i == len(dados):
            print(f"Processados: {i}/{len(dados)}", flush=True)

    resultado = pd.DataFrame(resultados)

    if resultado.isna().any().any():
        raise ValueError("Foram encontrados valores ausentes")

    saida_temporaria = SAIDA.with_suffix(".csv.tmp")
    resultado.to_csv(saida_temporaria, index=False)
    saida_temporaria.replace(SAIDA)

    print("\n=== RESULTADO FINAL ===")
    print("Linhas:", len(resultado))
    print("Caracteristicas WPT:", len(resultado.columns) - 4)
    print()
    print(pd.crosstab(resultado["particao"], resultado["classe"]))
    print("\nArquivo salvo:", SAIDA)


if __name__ == "__main__":
    main()
