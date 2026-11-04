# PIBITI — Análise preliminar WPT — Mês 3

## Configuração experimental

- Corpus: BRSpeech-DF.
- Subconjunto: 522 áudios do piloto.
- Classes: bona fide e spoof.
- Modelo sintético do piloto: F5-TTS.
- Janela central: 4 segundos.
- Frequência: 16 kHz, mono.
- Wavelet: db4.
- Nível: 4.
- Sub-bandas: 16.
- Características: 64 por áudio.

## Validação dos dados

Foram extraídas 64 características WPT para cada
um dos 522 áudios.

Não foram encontrados valores ausentes, infinitos,
caminhos duplicados ou características constantes.

As energias relativas somaram aproximadamente 1
para cada gravação.

## Comparação treino e validação

Foram analisadas oito características selecionadas
exploratoriamente a partir das diferenças de medianas
observadas no treino.

Sete mantiveram o sinal da diferença entre as classes
na validação.

Na entropia de Shannon:

- adad: +0,388759 no treino e -0,010075 na validação.
- adaa: +0,383178 no treino e +0,285790 na validação.
- ddaa: +0,300057 no treino e +0,308037 na validação.
- adda: +0,248735 no treino e +0,028686 na validação.
- ddda: +0,242344 no treino e +0,438621 na validação.

As três características de energia relativa analisadas
mantiveram o sinal da diferença, mas apresentaram
mudanças importantes de magnitude.

## Limitações

Os resultados são exploratórios e não demonstram
capacidade de detecção de deepfakes.

A análise não incluiu treinamento de classificadores,
testes estatísticos de significância nem estimativas
de desempenho preditivo.

O piloto possui baixa diversidade de locutores e
utiliza somente o modelo sintético F5-TTS.

Diferenças entre partições podem refletir variações
de locutores, gravações e outros fatores.

O conjunto de teste não foi utilizado na seleção
das características.

## Próximos passos

- Concluir o inventário dos modelos sintéticos.
- Preparar um subconjunto com maior diversidade.
- Repetir a extração WPT no novo subconjunto.
- Consolidar os resultados no relatório do mês 3.
