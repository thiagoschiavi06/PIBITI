from pathlib import Path
import numpy as np
import soundfile as sf
import pywt

ARQUIVO = Path('data/processados/bonafide/train/bonafide_train_00000.wav')
NO = 'aaaa'
SAIDA = Path('results/subbanda_aaaa.wav')


def main():
    if not ARQUIVO.is_file():
        raise FileNotFoundError(f'Áudio não encontrado: {ARQUIVO}. Execute na raiz do PIBITI-main.')
    audio, sr = sf.read(ARQUIVO)
    wpt = pywt.WaveletPacket(data=audio, wavelet='db4', mode='symmetric', maxlevel=4)
    bandas = wpt.get_level(4, order='freq')
    if NO not in {b.path for b in bandas}:
        raise ValueError('Sub-banda não encontrada.')
    isolada = pywt.WaveletPacket(data=None, wavelet='db4', mode='symmetric', maxlevel=4)
    for banda in bandas:
        isolada[banda.path] = banda.data.copy() if banda.path == NO else np.zeros_like(banda.data)
    reconstruido = isolada.reconstruct(update=False)[:len(audio)]
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    sf.write(SAIDA, reconstruido, sr, subtype='PCM_16')
    print('=== RECONSTRUÇÃO INDIVIDUAL ===')
    print('Sub-banda:', NO)
    print('Amostras originais:', len(audio))
    print('Amostras reconstruídas:', len(reconstruido))
    print('Duração:', round(len(reconstruido)/sr, 2), 's')
    print('Energia da sub-banda no domínio do tempo:', float(np.sum(reconstruido**2)))
    print('Arquivo salvo:', SAIDA)
    print('Obs.: WAV PCM16 introduz quantização ao salvar; análise numérica usa reconstruido em memória.')


if __name__ == '__main__':
    main()
