from pathlib import Path
import numpy as np
import soundfile as sf
import pywt

ARQUIVO = Path('data/processados/bonafide/train/bonafide_train_00000.wav')


def main():
    if not ARQUIVO.is_file():
        raise FileNotFoundError(f'Áudio não encontrado: {ARQUIVO}. Execute na raiz do PIBITI-main.')
    audio, sr = sf.read(ARQUIVO)
    if audio.ndim != 1 or sr != 16000:
        raise ValueError('Esperado áudio mono em 16 kHz.')
    wpt = pywt.WaveletPacket(data=audio, wavelet='db4', mode='symmetric', maxlevel=4)
    bandas = wpt.get_level(4, order='freq')
    print('=== ÁUDIO ===')
    print('Arquivo:', ARQUIVO)
    print('Amostras:', len(audio), 'Frequência:', sr, 'Duração:', len(audio)/sr)
    print('=== WPT db4 L=4 ===')
    for i, banda in enumerate(bandas, start=1):
        energia = float(np.sum(banda.data**2))
        print(f'Banda {i:02d} | Nó: {banda.path} | Coeficientes: {len(banda.data)} | Energia: {energia:.4f}')


if __name__ == '__main__':
    main()
