from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import pywt

arquivos = {
    "bonafide": Path(
        "data/processados/bonafide/train/"
        "bonafide_train_00000.wav"
    ),
    "spoof": Path(
        "data/processados/spoof/train/"
        "spoof_train_00000.wav"
    )
}

def entropia_shannon(coeficientes):
    energia = np.abs(coeficientes) ** 2
    total = np.sum(energia)

    if total == 0:
        return 0.0

    p = energia / total
    p = p[p > 0]

    return float(-np.sum(p * np.log(p)))

resultados = []

for classe, arquivo in arquivos.items():
    audio, sr = sf.read(arquivo)

    assert sr == 16000, "Frequencia incorreta"

    wpt = pywt.WaveletPacket(
        data=audio,
        wavelet="db4",
        mode="symmetric",
        maxlevel=4
    )

    bandas = wpt.get_level(4, order="freq")

    for numero, banda in enumerate(bandas, 1):
        resultados.append({
            "classe": classe,
            "arquivo": arquivo.name,
            "banda": numero,
            "no": banda.path,
            "energia": float(
                np.sum(np.abs(banda.data) ** 2)
            ),
            "entropia": entropia_shannon(banda.data)
        })

df = pd.DataFrame(resultados)

tabela = df.pivot(
    index="banda",
    columns="classe",
    values="entropia"
)

tabela["diferenca"] = (
    tabela["spoof"] - tabela["bonafide"]
)

saida = Path(
    "results/comparacao_entropia_real_spoof.csv"
)

saida.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(saida, index=False)

print("\n=== ENTROPIA: REAL X DEEPFAKE ===")
print(tabela.round(4).to_string())

print("\nMaior diferenca absoluta:")
print(tabela["diferenca"].abs().idxmax())

print("\nCSV salvo em:", saida)
