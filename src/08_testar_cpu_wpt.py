import time
from pathlib import Path
import numpy as np
import soundfile as sf
import pywt

arquivos = sorted(Path("data/processados").rglob("*.wav"))[:100]
tempos = []
for arquivo in arquivos:
    audio, sr = sf.read(arquivo)
    if sr != 16000 or audio.ndim != 1:
        print("Ignorado (não 16k/mono):", arquivo)
        continue
    t0 = time.perf_counter()
    wp = pywt.WaveletPacket(data=audio, wavelet="db4", mode="symmetric", maxlevel=4)
    _ = wp.get_level(4, order="freq")
    tempos.append(time.perf_counter() - t0)
if tempos:
    print("Áudios:", len(tempos), "| WPT médio:", round(np.mean(tempos), 4), "s | WPT total:", round(sum(tempos), 4), "s")
else:
    print("Nenhum áudio adequado encontrado")
