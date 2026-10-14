import argparse
from pathlib import Path
import soundfile as sf

parser = argparse.ArgumentParser()
parser.add_argument("--classe", choices=["bonafide", "spoof"], required=True)
parser.add_argument("--particao", choices=["train", "validation", "test"], required=True)
args = parser.parse_args()
pasta = Path("data/audios") / args.classe / args.particao
arquivos = sorted(pasta.glob("*.audio"))
print("Arquivos encontrados:", len(arquivos))
for arquivo in arquivos[:5]:
    try:
        info = sf.info(arquivo)
        print(arquivo.name, "|", info.format, "|", info.samplerate, "Hz |", info.channels, "canal(is) |", round(info.duration,2), "s")
    except Exception as e:
        print("ERRO", arquivo, e)
