# Projeto: Diagnóstico Precoce da Doença de Lyme via Visão Computacional

**Instituição:** FESA - Faculdade Engenheiro Salvador Arena  
**Curso:** Engenharia de Computação  
**Disciplina:** Análise de Dados  
**Avaliação:** N1 - 2º Bimestre • **Entrega 01: Relatório de Análise de Dados e Plano do Problema**  
**Autor:** Murilo Trevejo  
**Data:** Outubro de 2026  
**Relatório Técnico Oficial:** [`reports/Relatorio_Entrega_01_Doenca_de_Lyme.pdf`](reports/Relatorio_Entrega_01_Doenca_de_Lyme.pdf)

---

## 📌 Visão Geral do Projeto

Este repositório contém os artefatos de pesquisa, dados, código analítico e relatório técnico da **Entrega 01** da disciplina de **Análise de Dados** no curso de **Engenharia de Computação**. O trabalho foca no diagnóstico precoce da **Doença de Lyme** (*Borrelia burgdorferi*) através de técnicas de Visão Computacional e Aprendizado de Máquina aplicadas ao reconhecimento do **Eritema Migratório (Erythema Migrans - EM)** e sua diferenciação contra 12 diagnósticos diferenciais dermatológicos.

---

## 🏥 1. Contexto Clínico e Mapeamento do Domínio

- **Patologia:** A Doença de Lyme é uma zoonose infecciosa multissistêmica transmitida por carrapatos do gênero *Ixodes*.
- **Biomarcador Primário:** A manifestação inicial mais característica é o **Eritema Migratório (EM)**, uma lesão cutânea em expansão comumente em formato de alvo (*bullseye rash*) ou mancha eritematosa homogênea.
- **Gargalo Diagnóstico Atual:** A sorologia padrão (ELISA + Western Blot) possui sensibilidade inferior a **30% a 50%** nas primeiras semanas de infecção (janela imunológica). Por conseguinte, as diretrizes médicas mundiais (CDC/IDSA) preconizam o diagnóstico clínico-dermatológico imediato. Contudo, médicos generalistas e emergencistas erram ou hesitam no diagnóstico em mais de 50% dos casos atípicos devido ao mimetismo com outras patologias (tínea corporal, celulite bacteriana, picadas de artrópodes).
- **Impacto de IA:** Modelos de visão computacional proporcionam triagem automatizada no Ponto de Atendimento (*Point-of-Care*) e Telemedicina, acelerando a prescrição de antibioticoterapia oral de primeira linha (doxiciclina) e prevenindo sequelas crônicas severas (neuroborreliose, artrite de Lyme e cardite).

---

## 📊 2. Curadoria e Estrutura da Base de Dados

- **Repositório de Origem:** Kaggle (*Lyme Disease Rashes - Full Stratified*, curadoria por *sshikamaru*).
- **Volume Total:** **5.059 fotografias dermatológicas em formato RGB**.
- **Classes Binárias:**
  - **Negativo (Controles Dermatológicos):** 4.118 imagens (81,4%)
  - **Positivo (Doença de Lyme - EM):** 941 imagens (18,6%)
  - **Razão de Desbalanceamento:** 4,38 : 1.
- **Partições Estratificadas Prontas:**
  - Treinamento: 4.046 imagens (3.294 Neg / 752 Pos)
  - Validação: 506 imagens (412 Neg / 94 Pos)
  - Teste: 507 imagens (412 Neg / 95 Pos)
- **Subconjunto Balanceado:** O dataset inclui partição balanceada (`image_folder_balanced`) com 1.896 imagens (Treino: 1.506, Val: 200, Teste: 190) para mitigar viés a priori.
- **Subcategorias Catalogadas (15 condições):**
  - *Positivo:* `Erythema Migrans` (325), `bullseye rash` (310), `EM rash` (306).
  - *Negativo (Diagnósticos Diferenciais):* `superficial erythema annulare centrifugum` (447), `tinea corporis` (430), `spider bite rash` (411), `pityriasis rosea` (385), `Fixed drug eruption` (378), `contact dermatitis` (375), `Urticartia` (372), `Bacterial cellulitis` (323), `shingles rash` (314), `Local reaction to arthropod bite` (312), `serum sickness-like reactions` (189), `Erythema marginatum` (182).

---

## 📖 3. Dicionário de Dados

A tabela completa de atributos estruturados extraídos das 5.059 imagens encontra-se em `data/metadata_lyme_rashes.csv`:

| Atributo | Tipo de Dado | Unidade | Faixa / Domínio | Descrição Clínica / Computacional |
| :--- | :--- | :--- | :--- | :--- |
| **filename** | String | Adimensional | Texto único | Nome do arquivo com rótulo diagnóstico específico. |
| **binary_class** | Categórico | Classe | `{Positive, Negative}` | Rótulo binário: Positivo (Lyme EM) ou Negativo (Controle). |
| **condition** | Categórico | Patologia | 15 condições | Diagnóstico específico (ex.: *Erythema Migrans*, *tinea corporis*, *cellulitis*). |
| **split** | Categórico | Partição | `{train, val, test}` | Partição oficial para machine learning (80% treino, 10% val, 10% teste). |
| **width** | Numérico (Int) | Pixels (px) | [80, 457] (Média: 251.5) | Largura horizontal da fotografia/recorte. |
| **height** | Numérico (Int) | Pixels (px) | [66, 394] (Média: 202.9) | Altura vertical da fotografia/recorte. |
| **aspect_ratio** | Numérico (Float) | Razão L/A | [0.43, 3.03] (Média: 1.25) | Proporção de aspecto da imagem. |
| **color_mode** | Categórico | Espaço de cor | `{RGB}` | Modo de cores (100% RGB tridimensional). |
| **mean_r / g / b** | Numérico (Float) | Intensidade [0-255] | R: [42, 251] \| G: [20, 248] \| B: [6, 249] | Média espectral dos canais Vermelho, Verde e Azul. |
| **brightness** | Numérico (Float) | Luminância [0-255] | [15.6, 249.1] (Média: 139.5) | Luminância perceptiva ITU-R 601 (0.299R + 0.587G + 0.114B). |
| **contrast** | Numérico (Float) | Desvio Padrão | [3.2, 106.0] (Média: 45.9) | Desvio padrão da luminância global (medida de nitidez/foco). |
| **erythema_index**| Numérico (Float) | Diferença espectral | [-33.2, 112.4] (Média: 34.3) | Proxy de hiperemia cutânea (R - G) validado em dermatologia. |
| **file_size_bytes**| Numérico (Int) | Bytes | [575, 15.369] (Média: 4.860) | Tamanho binário em disco (indicador de compressão JPEG). |
| **corrupted** | Booleano | Flag binária | `{False}` (0% corrompidos) | Validação de integridade do arquivo. |

---

## 🎯 4. As Quatro Perguntas-Chave do Projeto

1. **Pergunta de Agrupamento (Clustering Não Supervisionado):**  
   > *"Existem subgrupos, padrões morfológicos latentes ou fenótipos visuais não rotulados entre as lesões de pacientes com Lyme que permitam categorizar diferentes perfis de gravidade ou tempo de evolução clínica?"*  
   - **Abordagem:** Extração de *embeddings* profundos via CNNs pré-treinadas, redução dimensional (UMAP/t-SNE) e algoritmos de densidade (HDBSCAN / K-Means com Silhueta).

2. **Pergunta de Classificação (Aprendizado Supervisionado):**  
   > *"É possível construir um modelo preditivo baseado em redes neurais convolucionais capaz de diagnosticar precocemente o eritema da Doença de Lyme e diferenciá-lo de outras 12 patologias dermatológicas similares em meio a imagens ruidosas?"*  
   - **Abordagem:** *Transfer Learning* (EfficientNet-B0/B2, ConvNeXt e Vision Transformers), funções de perda ponderadas (Focal Loss) para contornar o desbalanceamento de 4.38:1, avaliando Recall, Especificidade e AUC-ROC.

3. **Pergunta Preditiva (Regressão / Modelagem Temporal):**  
   > *"Como modelar e prever a dinâmica geométrica de expansão temporal da lesão (taxa de crescimento diametral em cm/dia) e estimar a probabilidade contínua de disseminação sistêmica ou transição para fases crônicas?"*  
   - **Abordagem:** Modelagem de crescimento radial não linear e regressão sobre parâmetros geométricos/espectrais para estimar a janela temporal de risco de invasão hematogênica bacteriana.

4. **Pergunta de Suporte à Decisão (Aplicação Prescritiva / CDSS):**  
   > *"Como as predições probabilísticas calibradas e mapas de explicabilidade visual (Grad-CAM) podem ser integrados em um Sistema de Suporte à Decisão Clínica (CDSS) para auxiliar na recomendação de condutas terapêuticas precoces e otimização de tratamento?"*  
   - **Abordagem:** Calibração de probabilidade associada a mapas de saliência Grad-CAM e motor de regras clínicas (estratificação de risco alto >80% com recomendação imediata de doxiciclina vs acompanhamento em 48-72h).

---

## 🔬 5. Principais Conclusões da Análise Exploratória (EDA)

1. **Completude Total:** 0,0% de dados faltantes e 0 arquivos corrompidos em 5.059 imagens analisadas.
2. **Desbalanceamento Significativo:** Razão de 4,38 controles para cada caso de Lyme. O repositório oferece partição balanceada correspondente para benchmarking.
3. **Insuficiência da Cor Pura:** O Índice Clínico de Eritema ($R - G$) apresenta médias quase idênticas (33,80 para Lyme vs 34,38 para controles), demonstrando que a hiperemia cutânea isolada não diferencia as doenças. Padrões morfológicos e bordas concêntricas aprendidos por CNNs são indispensáveis.
4. **Outliers e Ruído:** 2% das imagens apresentam iluminação extrema (subexposição severa < 66.8 ou reflexo de flash fotográfico > 205.6), além de artefatos de compressão JPEG em arquivos de baixa resolução, exigindo pré-processamento com normalização de contraste e aumento de dados (*data augmentation*).

---

## 📁 Estrutura do Repositório

```text
analise-de-dados-fesa-doenca-de-lyme/
├── data/
│   ├── DataSplit/                  # Dataset particionado (train/val/test, Positive/Negative)
│   ├── image_folder_balanced/      # Subconjunto balanceado para mitigar viés a priori
│   └── metadata_lyme_rashes.csv    # Tabela estruturada com 20 atributos de 5.059 imagens
├── notebooks/
│   └── 01_analise_exploratoria_lyme.ipynb  # Notebook interativo com a EDA completa
├── scripts/
│   ├── extract_metadata.py         # Script extrator de atributos computacionais e espectrais
│   ├── run_eda.py                  # Script gerador dos gráficos estatísticos em alta resolução
│   ├── generate_report_pdf.py      # Gerador ReportLab do documento técnico em PDF
│   └── create_notebook.py          # Gerador automatizado do Jupyter Notebook
├── reports/
│   ├── Relatorio_Entrega_01_Doenca_de_Lyme.pdf  # DOCUMENTO TÉCNICO OFICIAL DE ENTREGA
│   └── figures/                    # Gráficos em 300 DPI gerados na EDA
│       ├── 01_class_distribution.png
│       ├── 02_conditions_breakdown.png
│       ├── 03_dimensions_and_aspect_ratio.png
│       ├── 04_color_and_erythema.png
│       ├── 05_brightness_contrast_outliers.png
│       └── 06_sample_montage.png
├── requirements.txt                # Dependências do projeto
└── README.md                       # Documentação completa
```

---

## 🚀 Como Executar

### 1. Clonar o repositório e instalar dependências
```bash
git clone git@github.com:Trevejo/analise-de-dados-fesa-doenca-de-lyme.git
cd analise-de-dados-fesa-doenca-de-lyme
pip install -r requirements.txt
```

### 2. Executar extração de metadados
```bash
python3 scripts/extract_metadata.py
```

### 3. Gerar figuras da Análise Exploratória (EDA)
```bash
python3 scripts/run_eda.py
```

### 4. Compilar o Relatório Técnico em PDF
```bash
python3 scripts/generate_report_pdf.py
```
O documento final formatado será compilado em `reports/Relatorio_Entrega_01_Doenca_de_Lyme.pdf`.

---

## 📚 Referências Clínicas

1. **CDC** - *Centers for Disease Control and Prevention*. Lyme Disease: Clinical Guidance for Healthcare Providers and Erythema Migrans Diagnosis. Atlanta: CDC, 2024.
2. **Steere, A. C., et al.** *Lyme borreliosis*. Nature Reviews Disease Primers, 2(1), 16090, 2016.
3. **Lantos, P. M., et al.** *Clinical Practice Guidelines by the Infectious Diseases Society of America (IDSA), AAN, and ACR*. Clin Infect Dis, 72(1), e1-e48, 2021.
4. **Burlina, P., et al.** *AI-based detection of erythema migrans: Toward automated decision support for early Lyme disease*. Computers in Biology and Medicine, 125, 103986, 2020.
