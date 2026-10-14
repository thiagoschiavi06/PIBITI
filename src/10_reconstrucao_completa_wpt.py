from pathlib import Path
import numpy as np
import soundfile as sf
import pywt

ARQUIVO = Path('data/processados/bonafide/train/bonafide_train_00000.wav')


def main():
    if not ARQUIVO.is_file():
        raise FileNotFoundError(f'Áudio não encontrado: {ARQUIVO}. Execute na raiz do PIBITI-main.')
    audio, sr = sf.read(ARQUIVO)
    wpt = pywt.WaveletPacket(data=audio, wavelet='db4', mode='symmetric', maxlevel=4)
    bandas = wpt.get_level(4, order='freq')
    reconstruido = wpt.reconstruct(update=False)
    if len(reconstruido) < len(audio):
        raise ValueError('Reconstrução mais curta que o áudio.')
    reconstruido = reconstruido[:len(audio)]
    erro = audio - reconstruido
    mse = np.mean(erro**2)
    max_erro = np.max(np.abs(erro))
    print('=== RECONSTRUÇÃO WPT ===')
    print('Wavelet: db4 | Nível: 4 | Sub-bandas:', len(bandas))
    print('Amostras originais:', len(audio))
    print('Amostras reconstruídas:', len(reconstruido))
    print('MSE:', f'{mse:.12e}')
    print('Erro máximo:', f'{max_erro:.12e}')
    print('Reconstrução numericamente equivalente:', np.allclose(audio, reconstruido, atol=1e-10))


if __name__ == '__main__':
    main()
