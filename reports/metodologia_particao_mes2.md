# PIBITI — Metodologia da Partição dos Dados

## 1. Corpus utilizado

BRSpeech-DF, disponível no Hugging Face:
https://huggingface.co/datasets/AKCIT-Deepfake/BRSpeech-DF

## 2. Subconjunto piloto

Foram selecionados 600 áudios:

| Partição | Bona fide | Spoof | Total |
|---|---:|---:|---:|
| Train | 100 | 100 | 200 |
| Validation | 100 | 100 | 200 |
| Test | 100 | 100 | 200 |
| Total | 300 | 300 | 600 |

## 3. Critério de seleção

Foram preservadas as divisões oficiais do corpus.

Em cada classe e partição, foram selecionados os
100 registros do primeiro grupo de linhas do
primeiro arquivo Parquet correspondente.

A seleção é determinística, mas não aleatória.
Por isso, o subconjunto pode não representar toda
a diversidade do corpus.

## 4. Padronização

Os áudios foram convertidos para:

- Formato WAV
- Frequência de amostragem de 16.000 Hz
- Canal mono
- Codificação PCM de 16 bits

Os arquivos originais foram preservados.

## 5. Balanceamento

O subconjunto contém 50% de áudios bona fide
e 50% de áudios spoof em cada partição.

## 6. Integridade

Os 600 registros foram verificados quanto à
existência dos arquivos, formato, frequência
de amostragem e quantidade de canais.

Também foi verificada a ausência de caminhos
duplicados no CSV consolidado.

Foi gerado um hash SHA-256 para o arquivo
subconjunto_piloto_600.csv.

## 7. Limitações

Os arquivos Parquet consultados não apresentam
identificadores explícitos de locutores.

Portanto, ainda não foi possível comprovar a
ausência de speaker leakage entre treino,
validação e teste.

O subconjunto deve ser considerado piloto,
e não uma partição definitivamente validada
por locutor.

Também não foi demonstrada a ausência de
conteúdo de áudio duplicado entre as partições.

## 8. Próximas atividades

- Investigar a separação dos locutores.
- Confirmar com o orientador o protocolo adotado.
- Consolidar o fichamento teórico da WPT.
- Validar a decomposição e reconstrução WPT.
