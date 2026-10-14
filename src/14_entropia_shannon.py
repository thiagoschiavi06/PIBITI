from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import pywt

arquivo = Path(
    "data/processados/bonafide/train/"
    "bonafide_train_00000.wav"
)

audio, sr = sf.read(arquivo)

def entropia_shannon(coeficientes):
    energia = np.abs(coeficientes) ** 2
    total = np.sum(energia)

    if total == 0:
        return 0.0

    p = energia / total
    p = p[p > 0]

    return float(-np.sum(p * np.log(p)))

wpt = pywt.WaveletPacket(
    data=audio,
    wavelet="db4",
    mode="symmetric",
    maxlevel=4
)

bandas = wpt.get_level(4, order="freq")

resultados = []

for i, banda in enumerate(bandas, 1):
    energia = np.sum(np.abs(banda.data) ** 2)
    entropia = entropia_shannon(banda.data)

    resultados.append({
        "banda": i,
        "no": banda.path,
        "energia": energia,
        "entropia": entropia
    })

df = pd.DataFrame(resultados)

saida = Path("results/entropia_shannon_db4_l4.csv")
saida.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(saida, index=False)

print("\n=== ENTROPIA DE SHANNON ===")
print("Arquivo:", arquivo.name)
print("Wavelet: db4")
print("Nivel: 4")
print("Sub-bandas:", len(bandas))

print("\n", df.to_string(index=False))

print("\nMaior entropia:")
print(df.loc[df["entropia"].idxmax()].to_string())

print("\nMenor entropia:")
print(df.loc[df["entropia"].idxmin()].to_string())

print("\nResultado salvo em:", saida)
