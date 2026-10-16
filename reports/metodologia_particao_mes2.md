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

## 7. Limitações e validação dos locutores

A análise dos 40 arquivos Parquet da classe bona fide
abrangeu 76.644 áudios reais do BRSpeech-DF.

Foram identificados 62 prefixos distintos:
- Treino: 42 identificadores.
- Validação: 10 identificadores.
- Teste: 10 identificadores.

Não foi encontrada sobreposição de identificadores
entre as três partições.

Os identificadores foram extraídos dos nomes dos arquivos
disponíveis em `audio.path`. A interpretação desses prefixos
como identificadores de locutores depende da convenção
de nomenclatura do corpus original.

A identidade dos locutores dos áudios sintéticos ainda
não foi verificada diretamente.

O subconjunto piloto de 600 áudios permanece exploratório,
pois foi selecionado de forma determinística, sem amostragem
aleatória e sem garantia de diversidade suficiente.

Também não foi verificada a ausência de conteúdo de áudio
duplicado entre as partições.

## 8. Próximas atividades

- Investigar a separação dos locutores.
- Confirmar com o orientador o protocolo adotado.
- Consolidar o fichamento teórico da WPT.
- Validar a decomposição e reconstrução WPT.

## Validacao completa dos identificadores de locutores

Foi realizada a leitura da coluna `audio.path` dos 40 arquivos
Parquet da classe bonafide do BRSpeech-DF.

Resultados:

| Particao | Audios analisados | Identificadores distintos |
|---|---:|---:|
| Treino | 74.163 | 42 |
| Validacao | 1.161 | 10 |
| Teste | 1.320 | 10 |
| Total | 76.644 | 62 |

Nao foram encontrados nomes ausentes nem identificadores
repetidos entre treino, validacao e teste.

O identificador foi extraido do prefixo anterior ao primeiro
caractere `_` presente em `audio.path`.

Essa verificacao fornece evidencia de separacao dos identificadores
dos arquivos reais entre as particoes oficiais.

Limitacoes:
- A interpretacao do prefixo como identificador de locutor depende
  da convencao de nomenclatura do corpus original.
- Os registros sinteticos nao foram verificados individualmente
  quanto a identidade do locutor.
- O subconjunto piloto de 600 audios permanece exploratorio,
  sem garantia de diversidade suficiente para avaliar generalizacao.

Arquivos de reproducao:
- `src/19_validar_locutores_completo.py`
- `results/validacao_locutores_completa.txt`
- `results/locutores_por_particao.csv`
