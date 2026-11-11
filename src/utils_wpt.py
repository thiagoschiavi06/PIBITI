"""
PIBITI - Detecção de Deepfakes de Áudio em Português Brasileiro
Módulo: utils_wpt.py
Funções modulares e reutilizáveis para decomposição e reconstrução WPT.

Atividades de referência:
- Guia Prático Completo do PIBITI 2026-2027 (projeto_completo.pdf)
- Guia do Mês 3: WPT-Piloto, Decomposição e Reconstrução (mes03_wpt_piloto.pdf)
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pywt


def decompor_wpt(
    sinal: np.ndarray,
    wavelet: str = "db4",
    nivel: int = 4,
    modo: str = "symmetric",
) -> pywt.WaveletPacket:
    """
    Decompõe um sinal 1D através da Wavelet Packet Transform (WPT).

    Parâmetros:
        sinal: array 1D com as amostras do sinal (áudio).
        wavelet: nome da wavelet-mãe (padrão 'db4').
        nivel: nível de decomposição L (padrão 4, gerando 2^L = 16 sub-bandas).
        modo: estratégia de tratamento de bordas (padrão 'symmetric').

    Retorna:
        Objeto pywt.WaveletPacket com a árvore completa de decomposição.
    """
    sinal_array = np.asarray(sinal, dtype=np.float64)
    if sinal_array.ndim != 1:
        raise ValueError(f"O sinal deve ser 1D. Shape recebido: {sinal_array.shape}")

    wp = pywt.WaveletPacket(
        data=sinal_array,
        wavelet=wavelet,
        mode=modo,
        maxlevel=nivel
    )
    return wp


def reconstruir_wpt(
    wp: pywt.WaveletPacket,
    tamanho_original: Optional[int] = None
) -> np.ndarray:
    """
    Reconstrói o sinal original a partir da árvore completa WPT.

    Parâmetros:
        wp: objeto WaveletPacket já decomposto.
        tamanho_original: tamanho esperado em amostras para ajuste de borda (padding).

    Retorna:
        Array 1D com o sinal reconstruído no domínio do tempo.
    """
    sinal_rec = wp.reconstruct(update=False)
    if tamanho_original is not None:
        sinal_rec = sinal_rec[:tamanho_original]
    return sinal_rec


def reconstruir_subbanda(
    wp_original: pywt.WaveletPacket,
    caminho_no: str,
    nivel: int = 4,
    tamanho_original: Optional[int] = None,
    ordem: str = "freq"
) -> np.ndarray:
    """
    Reconstrói uma única sub-banda específica no domínio do tempo.

    Zera todos os coeficientes dos demais nós no nível folha, preservando
    exclusivamente os coeficientes do nó alvo.

    Parâmetros:
        wp_original: objeto WaveletPacket original com a decomposição.
        caminho_no: string identificadora do nó (ex: 'aaaa', 'aaad').
        nivel: nível da sub-banda (padrão 4).
        tamanho_original: se fornecido, ajusta o comprimento final cortando o padding.
        ordem: ordenação dos nós ('freq' para frequência crescente ou 'natural').

    Retorna:
        Array 1D com a sub-banda isolada reconstruída no domínio do tempo.
    """
    bandas = wp_original.get_level(nivel, order=ordem)
    caminhos_validos = {b.path for b in bandas}
    if caminho_no not in caminhos_validos:
        raise ValueError(
            f"Nó '{caminho_no}' não encontrado no nível {nivel}. "
            f"Nós disponíveis: {sorted(caminhos_validos)}"
        )

    # Cria nova árvore idêntica vazia com a mesma wavelet e modo
    wp_isolada = pywt.WaveletPacket(
        data=None,
        wavelet=wp_original.wavelet,
        mode=wp_original.mode,
        maxlevel=nivel
    )

    for banda in bandas:
        if banda.path == caminho_no:
            wp_isolada[banda.path] = banda.data.copy()
        else:
            wp_isolada[banda.path] = np.zeros_like(banda.data)

    sinal_subbanda = wp_isolada.reconstruct(update=False)
    if tamanho_original is not None:
        sinal_subbanda = sinal_subbanda[:tamanho_original]

    return sinal_subbanda


def reconstruir_todas_subbandas(
    wp_original: pywt.WaveletPacket,
    nivel: int = 4,
    tamanho_original: Optional[int] = None,
    ordem: str = "freq"
) -> List[Tuple[str, np.ndarray]]:
    """
    Reconstrói individualmente todas as sub-bandas de um nível no domínio temporal.

    Parâmetros:
        wp_original: objeto WaveletPacket já decomposto.
        nivel: nível das sub-bandas (padrão 4, gerando 16 sub-bandas).
        tamanho_original: corte de amostras para remover expansão de borda.
        ordem: 'freq' para ordenação por frequência crescente (recomendado).

    Retorna:
        Lista de tuplas (caminho_no, sinal_temporal_reconstruido).
    """
    bandas = wp_original.get_level(nivel, order=ordem)
    subbandas_reconstruidas = []

    for banda in bandas:
        sinal_b = reconstruir_subbanda(
            wp_original=wp_original,
            caminho_no=banda.path,
            nivel=nivel,
            tamanho_original=tamanho_original,
            ordem=ordem
        )
        subbandas_reconstruidas.append((banda.path, sinal_b))

    return subbandas_reconstruidas


def calcular_energia(dados: np.ndarray) -> float:
    """
    Calcula a energia total de uma sequência discreta: E = sum(x[n]^2).
    """
    arr = np.asarray(dados, dtype=np.float64)
    return float(np.sum(np.square(arr)))


def validar_reconstrucao(
    sinal_original: np.ndarray,
    sinal_reconstruido: np.ndarray,
    tolerancia_atol: float = 1e-10
) -> Dict[str, Any]:
    """
    Valida numericamente a reconstrução em relação ao sinal de referência.

    Calcula Erro Máximo Absoluto, Erro Quadrático Médio (MSE), conservação
    de energia percentual e equivalência numérica via np.allclose.

    Parâmetros:
        sinal_original: vetor com o sinal original.
        sinal_reconstruido: vetor com o sinal reconstruído.
        tolerancia_atol: tolerância absoluta para teste de igualdade numérica.

    Retorna:
        Dicionário com métricas de validação.
    """
    min_len = min(len(sinal_original), len(sinal_reconstruido))
    s_orig = np.asarray(sinal_original[:min_len], dtype=np.float64)
    s_recon = np.asarray(sinal_reconstruido[:min_len], dtype=np.float64)

    erro = s_orig - s_recon
    erro_max = float(np.max(np.abs(erro)))
    mse = float(np.mean(np.square(erro)))

    energia_orig = calcular_energia(s_orig)
    energia_recon = calcular_energia(s_recon)

    if energia_orig > 0:
        erro_energia_perc = float(abs(energia_orig - energia_recon) / energia_orig * 100.0)
    else:
        erro_energia_perc = 0.0

    equivalente = bool(np.allclose(s_orig, s_recon, atol=tolerancia_atol))

    return {
        "amostras_comparadas": int(min_len),
        "erro_maximo": erro_max,
        "mse": mse,
        "energia_original": energia_orig,
        "energia_reconstruida": energia_recon,
        "erro_energia_perc": erro_energia_perc,
        "equivalente_atol_1e-10": equivalente,
    }
