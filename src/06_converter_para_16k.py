import argparse
from math import gcd
from pathlib import Path
import soundfile as sf
from scipy.signal import resample_poly

parser = argparse.ArgumentParser()
parser.add_argument("--classe", choices=["bonafide", "spoof"], required=True)
parser.add_argument("--particao", choices=["train", "validation", "test"], required=True)
args = parser.parse_args()
entrada = Path("data/audios") / args.classe / args.particao
saida = Path("data/processados") / args.classe / args.particao
saida.mkdir(parents=True, exist_ok=True)
arquivos = sorted(entrada.glob("*.audio"))
if not arquivos:
    raise SystemExit(f"Nenhum .audio em {entrada}")
validos = 0
falhas = 0
for i, arquivo in enumerate(arquivos, 1):
    destino = saida / f"{arquivo.stem}.wav"
    try:
        if destino.exists():
            info = sf.info(destino)
            if info.samplerate == 16000 and info.channels == 1 and info.format == "WAV":
                validos += 1
                print(f"[{i}/{len(arquivos)}] Já convertido: {destino.name}")
                continue
        audio, sr = sf.read(arquivo)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != 16000:
            fator = gcd(sr,16000)
            audio = resample_poly(audio, up=16000//fator, down=sr//fator)
        sf.write(destino, audio, 16000, subtype="PCM_16")
        info = sf.info(destino)
        assert info.samplerate == 16000 and info.channels == 1
        validos += 1
        print(f"[{i}/{len(arquivos)}] Convertido: {destino.name}")
    except Exception as e:
        falhas += 1
        print(f"ERRO {arquivo}: {e}")
print("\nArquivos encontrados:", len(arquivos), "| Convertidos ou válidos:", validos, "| Erros:", falhas)
