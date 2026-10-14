from pathlib import Path
import numpy as np
import soundfile as sf
import pywt

arquivo = Path(
    "data/processados/bonafide/train/"
    "bonafide_train_00000.wav"
)

audio, sr = sf.read(arquivo)

wpt = pywt.WaveletPacket(
    data=audio,
    wavelet="db4",
    mode="periodization",
    maxlevel=4
)

def custo_entropia(coeficientes):
    energia = np.abs(coeficientes) ** 2
    energia = energia[energia > 0]

    if len(energia) == 0:
        return 0.0

    return float(-np.sum(energia * np.log(energia)))

pai = custo_entropia(wpt["a"].data)

filho_aa = custo_entropia(wpt["aa"].data)
filho_ad = custo_entropia(wpt["ad"].data)

filhos = filho_aa + filho_ad

print("=== SELEÇÃO DE BASE POR ENTROPIA ===")
print("Wavelet: db4")
print("Nó pai: a")
print(f"Custo do pai: {pai:.6f}")
print(f"Custo do filho aa: {filho_aa:.6f}")
print(f"Custo do filho ad: {filho_ad:.6f}")
print(f"Custo conjunto dos filhos: {filhos:.6f}")

if filhos < pai:
    print("\nDECISÃO: dividir o nó pai.")
else:
    print("\nDECISÃO: manter o nó pai.")

saida = Path("results/selecao_base_entropia.txt")
saida.parent.mkdir(parents=True, exist_ok=True)

with open(saida, "w", encoding="utf-8") as f:
    f.write(f"Custo do pai: {pai:.6f}\n")
    f.write(f"Custo dos filhos: {filhos:.6f}\n")
    f.write(
        "Decisão: dividir\n" if filhos < pai
        else "Decisão: manter\n"
    )

print("\nResultado salvo em:", saida)
