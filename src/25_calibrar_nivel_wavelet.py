from pathlib import Path
import time
import numpy as np
import pandas as pd
import soundfile as sf
import pywt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FS = 16000
DURACAO = 1.0
NIVEIS = [3, 4, 5]
WAVELETS = ["db4", "db8", "sym8"]
MODO = "symmetric"

AUDIO_REAL = Path("data/processados/bonafide/train/bonafide_train_00000.wav")
PASTA_RESULTS = Path("results")
PASTA_FIGURAS = Path("reports/experimentos_wpt_mes3/figuras")

PASTA_RESULTS.mkdir(parents=True, exist_ok=True)
PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)

SAIDA_CSV = PASTA_RESULTS / "calibracao_nivel_wavelet.csv"
SAIDA_TXT = PASTA_RESULTS / "calibracao_nivel_wavelet.txt"
SAIDA_PNG = PASTA_FIGURAS / "calibracao_nivel_wavelet.png"


def gerar_sinal_sintetico():
    t = np.linspace(0, DURACAO, int(FS * DURACAO), endpoint=False)
    np.random.seed(42)
    sinal = np.sin(2 * np.pi * 1000 * t) + 0.1 * np.random.randn(len(t))
    return sinal


def carregar_sinal_real():
    if not AUDIO_REAL.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {AUDIO_REAL}")
    audio, sr = sf.read(AUDIO_REAL, dtype="float64")
    if audio.ndim != 1:
        audio = audio.mean(axis=1)
    return audio[:int(FS * DURACAO)]


def avaliar_configuracao(sinal, wavelet, nivel):
    t0 = time.perf_counter()
    wp = pywt.WaveletPacket(
        data=sinal,
        wavelet=wavelet,
        mode=MODO,
        maxlevel=nivel
    )
    t_decomp_ms = (time.perf_counter() - t0) * 1000.0

    nos = wp.get_level(nivel, order="freq")
    qtd_subbandas = len(nos)
    tamanho_no = len(nos[0].data)
    largura_banda_hz = (FS / 2) / qtd_subbandas

    t0 = time.perf_counter()
    sinal_rec = wp.reconstruct(update=False)[:len(sinal)]
    t_rec_ms = (time.perf_counter() - t0) * 1000.0

    erro = sinal - sinal_rec
    mse = float(np.mean(erro ** 2))
    erro_max = float(np.max(np.abs(erro)))

    return {
        "wavelet": wavelet,
        "nivel": nivel,
        "subbandas": qtd_subbandas,
        "largura_banda_hz": largura_banda_hz,
        "tamanho_no_amostras": tamanho_no,
        "tempo_decomp_ms": round(t_decomp_ms, 3),
        "tempo_rec_ms": round(t_rec_ms, 3),
        "tempo_total_ms": round(t_decomp_ms + t_rec_ms, 3),
        "mse": mse,
        "erro_max": erro_max,
        "equivalente_atol_1e-10": bool(np.allclose(sinal, sinal_rec, atol=1e-10)),
    }


def main():
    sinal_sint = gerar_sinal_sintetico()
    sinal_real = carregar_sinal_real()

    testes = [
        ("Sintético (Senoide 1kHz + Ruído)", sinal_sint),
        ("Áudio Real (Fala Bonafide 1s)", sinal_real),
    ]

    registros = []
    linhas_relatorio = [
        "CALIBRAÇÃO DE NÍVEL DE DECOMPOSIÇÃO E WAVELET-MÃE (MÊS 4)",
        f"Data: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Frequência de amostragem: {FS} Hz",
        f"Modo de extensão: {MODO}",
        "=" * 75,
        ""
    ]

    for tipo_sinal, sinal in testes:
        linhas_relatorio.append(f"TIPO DE SINAL: {tipo_sinal}")
        linhas_relatorio.append(f"Amostras: {len(sinal)}")
        linhas_relatorio.append("-" * 75)

        for w in WAVELETS:
            for lvl in NIVEIS:
                res = avaliar_configuracao(sinal, w, lvl)
                res["tipo_sinal"] = tipo_sinal
                registros.append(res)

                linhas_relatorio.append(
                    f"Wavelet: {w:5} | L={lvl} | Sub-bandas: {res['subbandas']:2d} "
                    f"({res['largura_banda_hz']:5.0f} Hz cada) | Tam. Nó: {res['tamanho_no_amostras']:4d} | "
                    f"CPU: {res['tempo_total_ms']:5.2f} ms | MSE: {res['mse']:.2e} | "
                    f"Equiv: {res['equivalente_atol_1e-10']}"
                )

        linhas_relatorio.append("")

    df = pd.DataFrame(registros)
    df.to_csv(SAIDA_CSV, index=False)
    SAIDA_TXT.write_text("\n".join(linhas_relatorio), encoding="utf-8")

    df_real = df[df["tipo_sinal"] == "Áudio Real (Fala Bonafide 1s)"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    largura = 0.25
    x = np.arange(len(NIVEIS))

    for i, w in enumerate(WAVELETS):
        sub = df_real[df_real["wavelet"] == w]
        tempos = sub["tempo_total_ms"].tolist()
        ax1.bar(x + i * largura, tempos, width=largura, label=w)

    ax1.set_title("Custo Computacional em CPU por Nível e Wavelet", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Nível de Decomposição (L)")
    ax1.set_ylabel("Tempo Total CPU (ms)")
    ax1.set_xticks(x + largura)
    ax1.set_xticklabels([f"L={lvl} ({2**lvl} bandas)" for lvl in NIVEIS])
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2 = axes[1]
    for w in WAVELETS:
        sub = df_real[df_real["wavelet"] == w]
        mses = sub["mse"].tolist()
        ax2.plot(NIVEIS, mses, marker="o", lw=1.5, label=w)

    ax2.set_title("Erro Quadrático Médio (MSE) de Reconstrução", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Nível de Decomposição (L)")
    ax2.set_ylabel("MSE (Escala Log)")
    ax2.set_yscale("log")
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(SAIDA_PNG, dpi=200)
    plt.close()

    print(f"Resultados salvos em: {SAIDA_CSV}")
    print(f"Relatório salvo em:  {SAIDA_TXT}")
    print(f"Figura gerada em:    {SAIDA_PNG}")


if __name__ == "__main__":
    main()
