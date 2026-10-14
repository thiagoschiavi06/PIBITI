from pathlib import Path
import numpy as np
import soundfile as sf
import pywt

ARQUIVO = Path('data/processados/bonafide/train/bonafide_train_00000.wav')
PASTA_SAIDA = Path('results/subbandas_db4_l4')


def main():
    if not ARQUIVO.is_file():
        raise FileNotFoundError(f'Áudio não encontrado: {ARQUIVO}. Execute na raiz do PIBITI-main.')
    audio, sr = sf.read(ARQUIVO)
    wpt = pywt.WaveletPacket(data=audio, wavelet='db4', mode='symmetric', maxlevel=4)
    bandas = wpt.get_level(4, order='freq')
    soma = np.zeros_like(audio, dtype=float)
    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
    for i, banda_alvo in enumerate(bandas, 1):
        arvore = pywt.WaveletPacket(data=None, wavelet='db4', mode='symmetric', maxlevel=4)
        for banda in bandas:
            arvore[banda.path] = banda.data.copy() if banda.path == banda_alvo.path else np.zeros_like(banda.data)
        sinal = arvore.reconstruct(update=False)[:len(audio)]
        soma += sinal
        saida = PASTA_SAIDA / f'{i:02d}_{banda_alvo.path}.wav'
        sf.write(saida, sinal, sr, subtype='PCM_16')
        print(f'{i:02d}: {banda_alvo.path} -> {saida}')
    mse = float(np.mean((audio - soma)**2))
    max_erro = float(np.max(np.abs(audio - soma)))
    print('\n=== SOMA DAS SUB-BANDAS (ANTES DE SALVAR EM PCM16) ===')
    print('Número de sub-bandas:', len(bandas))
    print('MSE:', f'{mse:.12e}')
    print('Erro máximo:', f'{max_erro:.12e}')
    print('Equivalente ao original:', np.allclose(audio, soma, atol=1e-10))
    print('Nota: a soma dos WAVs salvos pode ter pequeno erro adicional de quantização PCM16.')


if __name__ == '__main__':
    main()
