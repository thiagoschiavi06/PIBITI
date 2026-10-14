import time
import numpy as np
import soundfile as sf
import pywt
import pandas as pd
from pathlib import Path

arquivo = Path(
    "data/processados/bonafide/train/"
    "bonafide_train_00000.wav"
)

audio, sr = sf.read(arquivo)

resultados = []

for wavelet in ["db4", "db8", "sym8"]:
    for nivel in [4, 5]:
        inicio = time.perf_counter()

        wpt = pywt.WaveletPacket(
            data=audio,
            wavelet=wavelet,
            mode="symmetric",
            maxlevel=nivel
        )

        bandas = wpt.get_level(nivel, order="freq")
        reconstruido = wpt.reconstruct(update=False)
        reconstruido = reconstruido[:len(audio)]

        tempo = time.perf_counter() - inicio

        mse = np.mean((audio - reconstruido) ** 2)

        resultados.append({
            "wavelet": wavelet,
            "nivel": nivel,
            "subbandas": len(bandas),
            "mse": mse,
            "tempo_ms": tempo * 1000
        })

df = pd.DataFrame(resultados)

saida = Path("results/comparacao_wavelets.csv")
saida.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(saida, index=False)

print("\n=== COMPARAÇÃO WPT ===")
print(df.to_string(index=False))

print("\nResultados salvos em:", saida)
