"""Celulas Markdown (contextualizacao e interpretacao) do notebook da Entrega 02."""
from collections import defaultdict

TEXTOS = defaultdict(lambda: '> _[interpretação pendente]_')

TEXTOS['cabecalho'] = """
# Entrega 02 — Implementação Analítica: Análise de Dados, Suporte à Decisão Clínica e Ética
## Diagnóstico precoce da Doença de Lyme por visão computacional do Eritema Migratório

| | |
|---|---|
| **Instituição** | FESA — Faculdade Engenheiro Salvador Arena |
| **Curso / Disciplina** | Engenharia de Computação — Análise de Dados |
| **Avaliação** | N1 — 2º Bimestre — Entrega 02 (continuação da Entrega 01) |
| **Dupla** | Fábio Gomes Celestino — RA 081220016<br>Murilo Trevejo Santos — RA 081220025 |
| **Patologia** | Doença de Lyme (borreliose de Lyme) — manifestação cutânea precoce: eritema migratório (EM) |
| **Dataset** | *Lyme Disease Rashes* (Kaggle): https://www.kaggle.com/datasets/sshikamaru/lyme-disease-rashes |
| **Repositório do projeto** | https://github.com/Trevejo/analise-de-dados-fesa-doenca-de-lyme |

<a href="https://colab.research.google.com/github/Trevejo/analise-de-dados-fesa-doenca-de-lyme/blob/main/notebooks/02_pipeline_analitica_cdss_etica_lyme.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Abrir no Colab"/></a>

**Roteiro do notebook:** (1) identificação e carregamento → (2) qualidade dos dados e pré-processamento clínico →
(3) métricas estatísticas, agrupamento e classificação → (4) interpretação para o CDSS → (5) análise crítica e ética →
(6) conclusões e referências.
"""

TEXTOS['sintese_clinica'] = """
## Seção 01 — Identificação, resgate clínico e carregamento da base

### 1.1 Cabeçalho do projeto
Integrantes da dupla, título, patologia e link da fonte original do dataset estão no quadro de identificação acima.

### 1.2 Síntese do problema clínico (resgate da Entrega 01)
A Doença de Lyme é uma zoonose multissistêmica causada por espiroquetas do complexo *Borrelia burgdorferi* *sensu lato*,
transmitidas pela picada de carrapatos do gênero *Ixodes*. Após a inoculação, a bactéria se multiplica na derme e migra
centrifugamente, produzindo o **eritema migratório (EM)**: uma mácula/placa eritematosa em expansão, com ≥ 5 cm, que surge
entre 3 e 30 dias após a picada. O EM é o **biomarcador clínico primário** da fase localizada precoce — as diretrizes
IDSA/AAN/ACR (2020) recomendam diagnosticá-lo e tratá-lo **clinicamente**, sem esperar sorologia, porque ELISA e Western blot
são negativos na maioria dos pacientes nas primeiras semanas (janela imunológica). O gargalo é visual: a minoria dos EM tem
o aspecto clássico "em alvo"; a maioria é homogênea (Smith *et al.*, 2002) e se confunde com celulite, picada de artrópode, tinea, urticária,
herpes-zóster e outras dermatoses. Sem tratamento (doxiciclina por 10 dias), a infecção pode disseminar-se e causar
paralisia facial, meningite, cardite com bloqueio atrioventricular e artrite de Lyme.

Nesta etapa, os **biomarcadores analisados** são digitais e extraídos das fotografias: intensidade e contraste do eritema
(espaço CIELAB), padrão radial centro→borda (proxy do aspecto em alvo), textura (descamação, vesículas), densidade de bordas
e tom de pele (ITA), além de *embeddings* de uma CNN pré-treinada. As **4 perguntas norteadoras** da Entrega 01 são:

1. **Agrupamento:** existem fenótipos visuais latentes entre as lesões de Lyme? → *Seção 3.B.2 (PCA + K-Means + t-SNE).*
2. **Classificação:** é possível diferenciar o EM de 12 diagnósticos diferenciais em imagens ruidosas? → *Seção 3.B.1.*
3. **Predição temporal:** como modelar a expansão da lesão no tempo? → *Seção 3.B.3 (discussão de viabilidade: a base não tem dimensão temporal).*
4. **Suporte à decisão (CDSS):** como integrar probabilidades calibradas e explicabilidade em condutas? → *Seção 4.*
"""

TEXTOS['secao2_intro'] = """
## Seção 02 — Tratamento da qualidade dos dados e pré-processamento clínico
Cada decisão de limpeza abaixo é justificada sob a ótica médica: em um problema diagnóstico, remover ou imputar um dado
muda **quem** o modelo aprende a reconhecer como doente. Toda transformação que "aprende" parâmetros dos dados (medianas,
escalas) é ajustada **somente no conjunto de desenvolvimento** (treino + validação), nunca no teste.

### 2.1 Dados faltantes (missing values)
"""

TEXTOS['secao3_intro'] = """
## Seção 03 — Métricas estatísticas, análise exploratória e modelagem

### 3.A Métricas descritivas e inferenciais (biomarcadores × desfecho)
Para cada biomarcador comparamos Lyme × controles com **média ± desvio-padrão** e **mediana [IIQ]** (a mediana é a medida
robusta, dada a assimetria de vários atributos). A significância é avaliada pelo **teste de Mann-Whitney** (não paramétrico,
adequado a distribuições assimétricas) e, como referência, pelo **teste t de Welch**. Como são 14 testes simultâneos,
os p-valores são corrigidos por **Benjamini-Hochberg** (q-valor) e o tamanho de efeito é dado pelo **δ de Cliff**
(|δ| < 0,147 desprezível; < 0,33 pequeno; < 0,474 médio; ≥ 0,474 grande). Com ~4.400 fotos, p-valores minúsculos
aparecem mesmo para diferenças clinicamente irrelevantes — por isso o tamanho de efeito é o critério principal.
"""

TEXTOS['classif_intro'] = """
### 3.B Métricas de desempenho analítico e modelagem

#### 3.B.1 Diagnóstico diferencial (classificação supervisionada) — Pergunta 02
**Desenho experimental.** O conjunto de **teste** oficial (após deduplicação) fica intocado até a avaliação final. No
conjunto de **desenvolvimento** (treino + validação) usamos **validação cruzada estratificada de 5 *folds***: ela escolhe
hiperparâmetros e gera probabilidades *fora-do-fold* (nunca vistas pelo modelo) que servem para definir os limiares do CDSS
sem tocar no teste.

Três abordagens são comparadas com uma **linha de base** que sempre responde "controle":
- **A) Biomarcadores interpretáveis + Gradient Boosting** — modelo explicável, que permite SHAP por biomarcador;
- **B) *Embeddings* da CNN + Regressão Logística** — *transfer learning* (sonda linear sobre ConvNeXt-Tiny/ImageNet);
- **C) Fusão A + B** — média das probabilidades (combina forma/padrão espacial da CNN com cor/textura dos biomarcadores).

Métricas obrigatórias: matriz de confusão, **sensibilidade** (recall), **especificidade**, **VPP** (precisão), **VPN**,
**F1**, **AUC-ROC** e **AUC-PR**. Em base desbalanceada a AUC-PR é a métrica de seleção: sua linha de base é a
prevalência (~18%), enquanto a da AUC-ROC é sempre 0,5.
"""

TEXTOS['cluster_intro'] = """
#### 3.B.2 Descoberta de fenótipos (agrupamento) — Pergunta 01
Restringimos o agrupamento às **lesões de Lyme** (positivas) para responder à pergunta da Entrega 01: existem subtipos
visuais de EM? Os *embeddings* (768 dimensões) são padronizados e reduzidos por **PCA** (50 componentes); o **K-Means** é
avaliado para k = 2 … 8 com **inércia** (cotovelo), **coeficiente de silhueta** (maior = clusters mais coesos e separados;
varia de −1 a 1) e **índice de Davies-Bouldin** (menor = melhor). O perfil estatístico de cada cluster é descrito pelos
biomarcadores interpretáveis.
"""

TEXTOS['pergunta3'] = """
#### 3.B.3 Pergunta 03 (predição temporal / regressão) — por que não foi modelada nesta base
A Pergunta 03 da Entrega 01 pedia modelar a **expansão da lesão no tempo** (cm/dia) e o risco de disseminação. A análise
de qualidade da Seção 2.1 mostrou que a fonte **não registra** nenhuma variável temporal ou métrica física: não há data da
picada, dias de evolução, diâmetro em centímetros (as fotos não têm régua/escala) nem fotos seriadas do mesmo paciente.
Ajustar uma regressão sobre um alvo inventado produziria MAE/RMSE/R² sem significado clínico — o que seria pior do que não
responder. Por isso a dupla implementou as duas outras abordagens analíticas exigidas (classificação e agrupamento) e
documenta aqui o **protocolo** para responder à Pergunta 03 numa próxima coleta:

| Elemento | Especificação |
|---|---|
| Dados necessários | Fotos seriadas (D0, D2, D4…) com régua milimetrada, data da picada/início, local anatômico, desfecho (disseminação: EM múltiplo, paralisia facial, cardite) |
| Alvo contínuo | Diâmetro do EM em cm (segmentação + calibração pela régua) e taxa de expansão (cm/dia) |
| Modelo | Regressão de efeitos mistos (intercepto/inclinação por paciente) ou curva de crescimento radial não linear |
| Métricas | **MAE e RMSE em cm** (ex.: MAE de 0,5 cm não muda conduta; erro de 2–3 cm pode cruzar o critério de 5 cm do EM) e **R²** |
| Uso clínico | Lesão que cresce > 2–3 cm/dia após picada de carrapato reforça EM (picadas e reações de hipersensibilidade regridem em 24–48 h) |
"""

TEXTOS['cdss_intro'] = """
## Seção 04 — Interpretação dos dados para a tomada de decisão clínica (CDSS) — Pergunta 04
Esta seção traduz as métricas em **regras de uso clínico**. Os limiares foram escolhidos **apenas** com as probabilidades
fora-do-fold do desenvolvimento e depois verificados uma única vez no teste.
"""

TEXTOS['referencias'] = """
## Referências
1. CENTERS FOR DISEASE CONTROL AND PREVENTION (CDC). *Lyme Disease: Clinical Signs and Symptoms; Erythema Migrans*. Atlanta: CDC, 2024.
2. LANTOS, P. M. et al. Clinical Practice Guidelines by the IDSA, AAN and ACR: 2020 Guidelines for the Prevention, Diagnosis and Treatment of Lyme Disease. *Clinical Infectious Diseases*, v. 72, n. 1, p. e1–e48, 2021.
3. STEERE, A. C. et al. Lyme borreliosis. *Nature Reviews Disease Primers*, v. 2, 16090, 2016.
4. SMITH, R. P. et al. Clinical characteristics and treatment outcome of early Lyme disease in patients with microbiologically confirmed erythema migrans. *Annals of Internal Medicine*, v. 136, n. 6, p. 421–428, 2002.
5. BURLINA, P. M. et al. Automated detection of erythema migrans and other confounding skin lesions via deep learning. *Computers in Biology and Medicine*, v. 105, p. 151–156, 2019.
6. YOSHINARI, N. H. et al. Doença de Lyme-símile brasileira ou Síndrome Infecto-Reacional Baggio-Yoshinari: zoonose exótica e emergente transmitida por carrapatos. *Revista da Associação Médica Brasileira*, v. 56, n. 3, p. 363–369, 2010.
7. FIX, A. D.; PEÑA, C. A.; STRICKLAND, G. T. Racial differences in reported Lyme disease incidence. *American Journal of Epidemiology*, v. 152, n. 8, p. 756–759, 2000.
8. DANESHJOU, R. et al. Disparities in dermatology AI performance on a diverse, curated clinical image set. *Science Advances*, v. 8, n. 32, eabq6147, 2022.
9. ADELEKUN, A.; ONYEKABA, G.; LIPOFF, J. B. Skin color in dermatology textbooks: an updated evaluation and analysis. *Journal of the American Academy of Dermatology*, v. 84, n. 1, p. 194–196, 2021.
10. CHARDON, A.; CRETOIS, I.; HOURSEAU, C. Skin colour typology and suntanning pathways. *International Journal of Cosmetic Science*, v. 13, n. 4, p. 191–208, 1991.
11. ZHOU, B. et al. Learning Deep Features for Discriminative Localization (CAM). *CVPR*, 2016.
12. LUNDBERG, S. M.; LEE, S.-I. A Unified Approach to Interpreting Model Predictions (SHAP). *NeurIPS*, 2017.
13. LIU, Z. et al. A ConvNet for the 2020s (ConvNeXt). *CVPR*, 2022.
14. BRASIL. Lei nº 13.709, de 14 de agosto de 2018 — Lei Geral de Proteção de Dados Pessoais (LGPD), arts. 5º, 7º, 11 e 13.
15. ANVISA. Resolução RDC nº 657, de 24 de março de 2022 — regularização de *Software como Dispositivo Médico* (SaMD).
16. CONSELHO FEDERAL DE MEDICINA. Resolução CFM nº 2.314/2022 — define e regulamenta a telemedicina.
"""

TEXTOS['etica_intro'] = """
## Seção 05 — Análise crítica e ética contextualizada na Doença de Lyme
A discussão abaixo é ancorada nos números deste notebook (duplicatas, tom de pele, erros do modelo) e nas particularidades
clínicas e epidemiológicas da borreliose — não em considerações genéricas sobre IA.

### 5.1 Vieses de dados e representatividade epidemiológica — auditoria quantitativa
O dataset não traz sexo, idade nem etnia (Seção 2.1). A única dimensão demográfica que conseguimos **estimar** é o tom de
pele, pelo Ângulo Tipológico Individual (ITA° = arctan[(L* − 50)/b*]) medido na pele perilesional. É um *proxy* grosseiro
(depende de iluminação, balanço de branco e compressão das fotos da web), mas suficiente para auditar se o desempenho
muda entre grupos.
"""

TEXTOS['missing_decisao'] = """
**Leitura e decisão — dados faltantes.**
- **Atributos observados:** 37 de 38 colunas estão 100% completas. A única com ausência é o `ita_deg` (15 fotos = 0,3%).
  O ITA fica ausente quando a máscara de pele encontra menos de 200 pixels: as 15 fotos têm fração de pele mediana 0, o
  brilho não difere das demais (Mann-Whitney p = 0,21) e **todas pertencem ao grupo controle** (pitiríase rósea, erupção
  fixa por droga, eritema anular…). A ausência, portanto, **não é aleatória (MNAR)**: ela sinaliza fotos sem pele visível
  ou com cor muito alterada (*close-up* extremo, ilustração, filtro) e carrega informação sobre o desfecho.
- **Estratégia adotada:** (i) nas estatísticas descritivas, imputação pela **mediana estratificada por grupo clínico**
  (Lyme 37,4°; controle 33,6°); (ii) nos modelos preditivos, imputação pela **mediana global ajustada só no treino +
  indicador binário de ausência** (`add_indicator=True`). Não usamos a mediana por grupo no modelo porque ela exigiria
  saber o diagnóstico do paciente no momento da predição — seria vazamento do rótulo. Descartar as 15 fotos também foi
  rejeitado, pois apagaria justamente o sinal "foto sem pele analisável", útil para o CDSS pedir nova imagem.
- **Ausência estrutural (o achado mais importante):** as 10 variáveis clínicas que um prontuário de Lyme teria — idade,
  sexo, fototipo, local da lesão, diâmetro em cm, dias desde a picada, exposição a carrapato/área endêmica, sintomas
  sistêmicos, sorologia e região — estão **100% ausentes**, porque o dataset é formado por fotos coletadas da web. Não
  existe imputação possível para isso: é uma limitação de desenho que restringe a Pergunta 03 (temporal) e a auditoria
  demográfica da Seção 5.

### 2.2 Outliers e ruídos de coleta
"""

TEXTOS['outliers_decisao'] = """
**Leitura e decisão — outliers e ruídos.**

*Ruído de rótulo e duplicatas (problema mais grave).* A busca por duplicatas com critério duplo (similaridade semântica
dos *embeddings* ≥ 0,95 **e** correlação de pixels ≥ 0,95) encontrou **393 grupos de cópias envolvendo 908 fotos (17,9% da
base)**; **154 grupos cruzam treino/validação/teste** (vazamento: o modelo seria avaliado em fotos que já viu) e **93
grupos (248 fotos) têm rótulos contraditórios** — a mesma imagem aparece como EM e como diagnóstico diferencial. Os pares
mais frequentes (EM em alvo × picada de aranha; EM × eritema marginado; EM × doença do soro-símile) são exatamente os
mimetizadores clínicos do EM, o que indica que as fotos foram coletadas por busca na internet com o termo de cada
doença. **Decisão:** excluir os 248 casos contraditórios (não há verdade-terreno para eles; treinar com eles ensinaria o
modelo a errar) e manter **uma única cópia** de cada grupo (prioridade teste > validação > treino).

*Valores implausíveis/técnicos (A).* Não há arquivos corrompidos; 4 fotos têm resolução < 100 px; 40 estão subexpostas e
84 superexpostas; 26 quase não têm pele visível. São ruídos de aquisição, não de biologia, e **foram mantidos** e
sinalizados (`baixa_qualidade`): um CDSS real receberá fotos de celular mal iluminadas, e excluí-las daria uma estimativa
otimista do desempenho. O Qui-quadrado da Seção 3.A confirma que essas fotos não se associam ao desfecho.

*Extremos patológicos (B).* Valores extremos de vermelhidão (`a_p90_minus_p50`, até 64,8 unidades a*), de amplitude radial
e de textura são **manifestações reais** — eritema intenso de celulite, centro hemorrágico/necrótico de picada de aranha,
anéis muito marcados de EM — e são proporcionalmente **mais frequentes entre os casos de Lyme** em `a_p90_minus_p50`
(4,25% × 3,59%) e `radial_a_range` (3,61% × 2,43%). Excluí-los pela regra de Tukey removeria justamente os EM mais
exuberantes. **Decisão:** manter todos; usar escalonamento robusto (mediana/IIQ) e modelos pouco sensíveis a caudas.
"""

TEXTOS['graf1'] = """
**Leitura técnica e clínica do Gráfico 1.**
(A) Os atributos de imagem são praticamente completos (apenas `ita_deg` com 0,3% ausente), mas as barras cinza mostram o
verdadeiro problema de qualidade: **nenhuma variável clínica do paciente foi registrada**. Na prática, o modelo terá de
diagnosticar só pela foto, sem a anamnese que mais pesa no diagnóstico de EM (picada de carrapato, viagem a área endêmica,
evolução em dias). (B) Após a limpeza, a prevalência de Lyme caiu de 18,6% para **15,8% (razão 5,31 controles : 1 caso)**
porque todo grupo com rótulo contraditório continha ao menos uma foto registrada como EM. A proporção é semelhante nas três
partições (15,5%, 17,5% e 16,8%; Qui-quadrado p = 0,43), ou seja, a estratificação original foi preservada. Com esse
desbalanceamento, um classificador que responda sempre "controle" acerta 84% — ponto retomado na Seção 3.B.

#### Estratégias de mitigação do desbalanceamento
O desbalanceamento de 5,3 : 1 é tratado em três frentes, comparadas empiricamente na Seção 3.B.1: **ponderação de classes**
(`class_weight='balanced'`, que multiplica o custo de errar um caso de Lyme por ≈ 5,3), **subamostragem dos controles**
(*undersampling*, aplicada apenas dentro dos *folds* de treino) e **ajuste do limiar de decisão** (Seção 4). O SMOTE não foi
usado: interpolar *embeddings* de duas fotos não gera uma lesão clinicamente plausível e, em imagens, a forma correta de
sobreamostrar seria a aumentação de dados (rotação, recorte, brilho), que exige re-treinar a CNN.

### 2.4 Padronização e codificação
"""

TEXTOS['escala'] = """
**Leitura e decisão — padronização e codificação.** Oito biomarcadores têm |assimetria| > 1 (ex.: contraste GLCM 3,05;
nitidez 2,51; contraste eritematoso 1,72). Para eles foi escolhido o **RobustScaler**, que centraliza pela mediana e divide
pelo IIQ: os valores típicos ficam em torno de 0 (mediana ≈ 0 após a transformação) sem que os extremos patológicos mantidos
na Seção 2.2 dominem a escala — com o StandardScaler, média e desvio seriam puxados pelas caudas. Os *embeddings* da CNN,
aproximadamente simétricos, recebem **StandardScaler** (exigido pela regressão logística regularizada). Todos os
escalonadores são ajustados **somente no desenvolvimento** e embutidos em *pipelines*, de modo que, dentro da validação
cruzada, cada *fold* é escalonado apenas com os próprios dados de treino. Codificação: desfecho binário (`y` = 1 Lyme),
condição clínica em *one-hot* (15 colunas, usada apenas na descrição dos clusters — nunca como preditor, pois é o próprio
diagnóstico) e tom de pele em 3 categorias ordinais de ITA.
"""

TEXTOS['tab_desc'] = """
**Interpretação clínica das métricas descritivas e inferenciais.**
- **Sete biomarcadores diferem significativamente** entre Lyme e controles após a correção de Benjamini-Hochberg, mas
  **todos com efeito desprezível a pequeno** (|δ| ≤ 0,17). Os mais fortes são os ligados ao **padrão radial**: a amplitude
  radial da vermelhidão (mediana 10,0 × 8,2; δ = +0,17) e a diferença centro − anel intermediário (2,9 × 1,4; δ = +0,14),
  coerentes com a lesão centrada e anelar do EM. Seguem a **homogeneidade da textura** (EM é uma mácula lisa, sem
  descamação ou vesículas; δ = +0,12) e o **contraste eritematoso** (δ = +0,11).
- A **vermelhidão média** (a*, índice R − G) e a saturação **não diferem** (q > 0,15): confirma, com teste formal, a
  conclusão da Entrega 01 de que "estar vermelho" não distingue EM de celulite, urticária ou picada.
- Mediana e IIQ foram reportados ao lado de média ± DP porque vários marcadores são assimétricos (ex.: nitidez, assimetria
  2,51); Mann-Whitney e t de Welch concordam na significância, exceto `a_std` e brilho, que ficam no limiar.
- **Conclusão para a pergunta de classificação:** nenhum biomarcador isolado serve como teste diagnóstico — a separação
  precisa vir da **combinação multivariada** de forma, padrão espacial e textura, o que justifica o uso de modelos.
"""

TEXTOS['graf2'] = """
**Leitura técnica e clínica do Gráfico 2.** As distribuições de Lyme (laranja) e controles (azul) **se sobrepõem quase
totalmente**; o deslocamento das medianas é pequeno e consistente com os δ de Cliff da tabela. O EM tende a maior
amplitude radial e maior vermelhidão no centro em relação ao anel intermediário (painéis 1 e 2), o que corresponde ao
"alvo" e à lesão centrada na picada — porém muitos controles (picadas de aranha, reação a artrópode, eritema anular) também
são anelares, por isso a sobreposição. O EM é mais **homogêneo e menos "nítido"** em textura (painéis 3 e 5): mácula lisa,
sem as escamas da tinea ou as vesículas do herpes-zóster. As caudas longas (pontos) foram mantidas por serem extremos
patológicos reais. **Implicação clínica:** um limiar em qualquer um desses biomarcadores isolados geraria muitos falsos
positivos; eles são úteis como **sinais de apoio** combinados, não como critério único.
"""

TEXTOS['graf3'] = """
**Leitura técnica e clínica do Gráfico 3.**
- **Redundâncias:** `erythema_index` (R − G) e `lab_a_mean` (a*) medem a mesma coisa (ρ = 0,95) — basta um deles; nitidez,
  contraste GLCM e densidade de bordas formam um bloco de "textura/irregularidade" (|ρ| entre 0,80 e 0,88); o contraste
  eritematoso e o desvio de a* (ρ = 0,73) medem heterogeneidade da vermelhidão.
- **Confundidor de aquisição:** o ITA correlaciona-se fortemente com o **brilho da foto** (ρ = 0,68) — sinal de que o
  "tom de pele" estimado mistura fototipo com iluminação, o que é investigado na Seção 5.
- **Sinergias diagnósticas:** a linha DESFECHO mostra que **nenhum** biomarcador tem |ρ| > 0,11 com o diagnóstico; os de
  maior associação (amplitude radial 0,11; centro − anel 0,09; homogeneidade 0,08) são quase independentes da vermelhidão
  média (ρ ≈ 0,1), ou seja, **padrão espacial e cor carregam informações complementares** — combinação explorada pelos modelos.
- A largura da imagem (atributo de aquisição) não se associa ao desfecho (ρ = −0,06), o que reduz a preocupação de que o
  modelo aprenda o "site de origem" pela resolução.
"""

TEXTOS['quiquadrado'] = """
**Interpretação dos testes Qui-quadrado.** (1) A proporção de Lyme não difere entre treino, validação e teste após a
limpeza (p = 0,43; V = 0,02): a avaliação no teste é representativa. (2) Há associação estatisticamente significativa,
mas muito fraca, entre tom de pele aparente e desfecho (χ² = 10,3; p = 0,006; V de Cramér = 0,048): fotos "escuras ou
sombreadas" têm menor proporção de Lyme (13,1% × 17,3% e 16,7%) — o que reflete a sub-representação de EM em pele
pigmentada na literatura e nos atlas de onde as fotos foram coletadas. (3) A baixa qualidade técnica não se associa ao
desfecho (p = 0,37), justificando mantê-las sem viés de classe.
"""

TEXTOS['cv'] = """
**Leitura das métricas em validação cruzada (5 *folds*, 3.981 fotos).**
- A **linha de base** atinge **84,3% de acurácia com sensibilidade 0%**: não detecta nenhum caso de Lyme. É a prova
  numérica de que **a acurácia global é enganosa** em base desbalanceada — um "modelo" inútil parece excelente.
- O **modelo A (biomarcadores interpretáveis)** supera o acaso, mas de forma limitada (AUC-ROC 0,625; AUC-PR 0,252 contra
  0,157 da prevalência), coerente com os efeitos pequenos da Seção 3.A.
- O **modelo B (*transfer learning*)** é claramente superior (AUC-ROC 0,761; AUC-PR 0,449 — quase 3× a prevalência): a CNN
  captura forma, borda e contexto que os biomarcadores manuais não descrevem.
- A **fusão C** não melhora o B (AUC-PR 0,425): os biomarcadores acrescentam pouca informação nova e a média dilui o sinal.
  Pelo critério definido *a priori* (maior AUC-PR), **o modelo B é o selecionado para o CDSS**; o modelo A permanece como
  ferramenta de explicação por biomarcador (SHAP).
"""

TEXTOS['desbalanceamento'] = """
**Efeito das estratégias de desbalanceamento (mesmo modelo B, limiar 0,5).** Sem tratamento, o modelo aprende a "apostar
no controle": acurácia de 85,9% e especificidade de 98,9%, mas **sensibilidade de apenas 16,5%** — deixaria 5 de cada 6
pacientes com Lyme sem antibiótico. A **ponderação de classes** eleva a sensibilidade para 62,9% preservando a AUC (0,761) e
a AUC-PR (0,449); o *undersampling* chega a 69,3% de sensibilidade, mas descarta ~80% dos controles e perde AUC-PR
(0,405). As AUCs quase não mudam entre as estratégias — elas **deslocam o ponto de operação**, não a capacidade de
separação. Por isso adotamos a **ponderação de classes + ajuste explícito do limiar** (Seção 4), que mantém toda a
informação dos controles e deixa a escolha sensibilidade × especificidade para a decisão clínica.
"""

TEXTOS['teste'] = """
**Interpretação clínica das métricas no teste independente (470 fotos, 79 com Lyme).**
- O modelo B obtém **AUC-ROC 0,785 (IC95% 0,730–0,839)** e **AUC-PR 0,487 (0,396–0,594)**, valores próximos aos da
  validação cruzada (sem sinal de sobreajuste). No limiar padrão de 0,5: **sensibilidade 62,0%**, **especificidade 76,0%**,
  **VPP 34,3%**, **VPN 90,8%**, F1 0,441.
- **Custo clínico dos erros:** os **30 falsos negativos** são pacientes com EM que voltariam para casa sem doxiciclina,
  com risco de disseminação (paralisia facial, cardite, artrite); os **94 falsos positivos** receberiam antibiótico
  desnecessário e — mais grave — poderiam ter o **diagnóstico real atrasado** (celulite que precisa de cobertura
  antiestreptocócica, herpes-zóster que exige antiviral em até 72 h, farmacodermia que exige suspender o medicamento).
- A linha de base tem **acurácia de 83,2%**, superior à do modelo B (73,6%), apesar de não detectar nenhum caso: mais uma vez,
  a acurácia isolada é enganosa. As métricas relevantes são sensibilidade, VPN (segurança para descartar) e AUC-PR.
- A largura dos intervalos de confiança (sensibilidade de 52% a 72%) reflete o pequeno número de casos positivos no teste
  e deve constar em qualquer comunicação do desempenho.
"""

TEXTOS['vazamento'] = """
**O impacto da limpeza.** No teste original, **92 das 507 fotos (18%) tinham cópia idêntica no treino/validação**. O mesmo
modelo, avaliado na base original, mostra AUC-PR 0,512 e sensibilidade 65,3%, contra 0,487 e 62,0% na base limpa — a
diferença é modesta porque dois ruídos agem em sentidos opostos: as cópias **inflam** o desempenho (o modelo "lembra" a
foto), enquanto os rótulos contraditórios o **deprimem** (a mesma foto conta como acerto e erro). O número da base limpa
é o único que estima honestamente o desempenho em pacientes novos; sem a deduplicação, o relatório superestimaria a
sensibilidade em ≈ 3 pontos percentuais.
"""

TEXTOS['graf5'] = """
**Leitura técnica e clínica do Gráfico 5.**
- **(A) Matriz de confusão no limiar de triagem (0,21):** o modelo detecta **74 dos 79 casos de Lyme (sensibilidade
  93,7%)**, deixando apenas 5 falsos negativos, ao custo de 255 falsos positivos (especificidade 34,8%). O **VPN de 96,5%**
  significa que um resultado "baixo risco" descarta EM com segurança razoável nesta população — é o uso adequado para a
  triagem.
- **(B) ROC:** as curvas B (azul) e C (laranja) praticamente se sobrepõem e dominam A (verde). Os marcadores indicam os
  dois pontos de operação do CDSS: triagem (alta sensibilidade) e confirmação (alta especificidade).
- **(C) Precision-Recall:** com prevalência de 16,8% (linha tracejada), o VPP do modelo B só se mantém acima de 50% enquanto a
  sensibilidade é de no máximo ~50%. Ou seja, **um alerta positivo isolado não confirma Lyme**: no máximo dobra/triplica a probabilidade
  pré-teste.
- **(D) SHAP do modelo interpretável:** os biomarcadores que mais movem a decisão são a **amplitude radial de vermelhidão**
  (valores altos ↑ suspeita), a **luminosidade L\\***, o **contraste eritematoso P90 − mediana** (↑), a **densidade de bordas**
  (valores altos ↓ suspeita — lesões múltiplas, descamativas ou vesiculares sugerem diferenciais) e a **homogeneidade da
  textura** (↑). A tabela abaixo confirma a direção de cada efeito.
"""

TEXTOS['graf4'] = """
**Leitura técnica e clínica do Gráfico 4.**
(A) A variância dos *embeddings* é muito distribuída: PC1 explica 8,7%, os 10 primeiros componentes 35,6% e os 50 usados
no agrupamento 66,3% — não há um eixo dominante de variação entre as lesões. (B, C) O K-Means com **k = 2** teve a maior
silhueta (0,099) e o Davies-Bouldin fica em torno de 3,0 para todos os k: valores baixos que indicam **clusters
sobrepostos**, sem fronteira natural. A separação em B ocorre quase só ao longo de PC1 e, no t-SNE, os dois grupos são
regiões contíguas de uma mesma nuvem. **Resposta à Pergunta 01:** não há evidência de fenótipos visuais bem separados de
EM nesta base; os dois agrupamentos são tendências, e o perfil abaixo mostra o que os diferencia.
"""

TEXTOS['perfil_cluster'] = """
**Perfil dos fenótipos e correspondência clínica.**
- **Cluster 0 (n = 285) — "lesão em close-up":** fotos enquadradas na lesão (91% de pele visível), textura lisa
  (contraste GLCM 1,3; bordas 0,009), mais vermelhas (a* 13,2), mais claras e iluminadas (ITA 43,8°; brilho 148). Os
  exemplos mostram EM únicos, grandes, vários com aspecto em alvo — próximos da apresentação de livro-texto.
- **Cluster 1 (n = 420) — "lesão em contexto":** enquadramento aberto (71% de pele), mais bordas e textura (roupas,
  móveis, membros inteiros), vermelhidão média menor (a* 8,7), mais fotos escuras/sombreadas (27,9% × 16,5%). Inclui lesões
  pequenas, múltiplas e sutis, fotografadas como num atendimento real ou por telemedicina.
- As variáveis do **padrão radial não diferem** entre os clusters (Kruskal-Wallis p = 0,91 e 0,82) e a composição pelos
  rótulos "bullseye rash", "EM rash" e "Erythema Migrans" é quase a mesma (Rand ajustado = 0,009). Logo, **os clusters
  refletem sobretudo o modo de fotografar, não estágios clínicos** (precoce × tardio, localizado × disseminado), que
  exigiriam data de início e sintomas — inexistentes na base.
- **Utilidade para o CDSS:** o modelo atribui probabilidade mediana menor às lesões do Cluster 1 (0,567 × 0,668), embora a
  sensibilidade no limiar de triagem seja semelhante (91,4% × 89,1%). Na prática, **fotos em contexto, sombreadas ou com a
  lesão pequena no quadro devem gerar um pedido de nova foto padronizada** (close-up, luz difusa, régua) antes da decisão.
"""

TEXTOS['cdss_texto'] = """
### 4.1 Hierarquia de biomarcadores e triagem
Combinando os tamanhos de efeito (Seção 3.A), o SHAP (Gráfico 5D) e a literatura clínica, os indicadores mais
determinantes para suspeitar precocemente de EM são, em ordem:

1. **Padrão radial / anelar da vermelhidão** (`radial_a_range`, `radial_a_center_minus_mid`) — lesão centrada com anéis
   concêntricos; maior efeito univariado e maior importância SHAP.
2. **Contraste eritematoso relativo** (`a_p90_minus_p50`) — o quanto a lesão se destaca da pele vizinha, mais informativo
   que a vermelhidão absoluta (que não difere entre os grupos).
3. **Textura homogênea e poucas bordas** (`glcm_homogeneity` ↑, `edge_density` ↓) — mácula lisa favorece EM; descamação
   (tinea), vesículas em faixa (herpes-zóster) ou lesões múltiplas pequenas favorecem diferenciais.
4. **Padrão espacial aprendido pela CNN** (*embeddings*) — o preditor mais forte em conjunto (AUC 0,785), que integra
   forma, borda e tamanho relativo.
5. **Qualidade/contexto da foto** (fração de pele, brilho, ITA) — não é biomarcador da doença, mas **modula a confiança**
   do resultado e deve disparar o pedido de nova foto.

**Otimização de exames:** como as diretrizes não recomendam sorologia para EM típico, o ganho do CDSS não é "pedir menos
ELISA", e sim **evitar sorologias inúteis na fase aguda** (negativas na janela imunológica e geradoras de falsa
tranquilidade), **evitar biópsia/PCR de pele** em lesões de alta probabilidade e **direcionar exames apenas a quem
realmente precisa**: casos indeterminados com sintomas sistêmicos (sorologia pareada após 2–4 semanas) e casos
disseminados (ECG para bloqueio AV; avaliação neurológica/punção lombar apenas se houver sinais neurológicos).

### 4.2 Definição de limiares de risco (*threshold tuning*)
O corte padrão de 0,50 é inadequado: ele deixa 30 dos 79 casos de Lyme sem diagnóstico (sensibilidade 62%). A dupla
recomenda **dois limiares**, escolhidos com as probabilidades fora-do-fold e verificados no teste:

| Zona | Escore do modelo B | Prob. calibrada | Desempenho no teste | Uso |
|---|---|---|---|---|
| **Baixo risco** | < 0,21 | < 6,5% | Sens. 93,7%, **VPN 96,5%** (5 FN em 79) | Triagem primária: EM improvável → investigar o diagnóstico diferencial |
| **Indeterminado** | 0,21 – 0,74 | 6,5% – 36,8% | Mistura de casos | Avaliação clínica dirigida (anamnese de picada, evolução, sintomas) |
| **Alto risco** | ≥ 0,74 | ≥ 36,8% | **Espec. 93,1%**, VPP 54,2% | Confirmação: EM provável → tratar se a anamnese for compatível |

Dois cuidados quantitativos sustentam essa recomendação:
- **Calibração:** o escore bruto superestima o risco (efeito da ponderação de classes; Brier 0,173, pior que o modelo
  ingênuo 0,140). Após a **calibração de Platt** ajustada só no desenvolvimento, o Brier cai para **0,113** — a
  probabilidade exibida ao médico deve ser a calibrada.
- **Prevalência:** VPP e VPN dependem da população. Com os mesmos limiares, o VPP de um alerta de triagem é 38% numa clínica
  de área endêmica (prevalência 30%), 7% na atenção primária (5%) e **0,7% num cenário brasileiro** (0,5%), onde o VPN chega
  a 99,9%. **No Brasil, o modelo só faz sentido como ferramenta de descarte, nunca como confirmação.**
"""

TEXTOS['graf6'] = """
**Leitura técnica e clínica do Gráfico 6.**
(A) As densidades são normalizadas por classe: os controles (azul) concentram-se abaixo de 0,4 e os casos de Lyme
(laranja) têm o pico entre 0,75 e 0,9, mas há ampla sobreposição entre 0,2 e 0,7 — é por isso que existe a **zona
indeterminada**: o CDSS reconhece explicitamente que, nessa faixa, a foto não basta e a decisão volta para a anamnese.
Na zona de baixo risco quase só há controles (VPN 96,5%); na de alto risco, pouco mais da metade das fotos é de Lyme
(VPP 54,2%), porque há 5 controles para cada caso e mesmo uma pequena fração deles ultrapassa o limiar. (B) O modelo bruto (cinza) fica abaixo da diagonal: quando ele diz
"80%", a frequência observada de Lyme é ~55%. A versão calibrada (laranja) acompanha a diagonal — condição necessária para
que a probabilidade seja comunicada ao médico e ao paciente sem induzir excesso de confiança.

### 4.3 Explicabilidade visual (XAI)
"""

TEXTOS['cam'] = """
**Leitura dos mapas de ativação (CAM).**
- Nos **verdadeiros positivos**, parte da ativação recai sobre a lesão e seus anéis, mas também sobre **bordas da imagem
  e sobre a régua milimetrada** (3ª foto). Réguas são comuns em fotos de EM porque o critério diagnóstico exige ≥ 5 cm — o
  modelo pode estar aprendendo um **atalho (*shortcut learning*)**: "foto com régua = Lyme".
- **Falsos negativos:** o 1º é um **EM em pele negra** (P = 0,04) — o eritema fica violáceo/acastanhado e de baixo contraste,
  padrão raro no treino; o 2º é uma mão sem lesão visível rotulada como "EM rash", provável ruído de rótulo remanescente.
- **Falsos positivos:** reação a picada de artrópode e "picadas de aranha" com halo eritematoso concêntrico — mimetizadores
  legítimos do EM que também confundem clínicos. Aqui a anamnese (picada percebida, dor, evolução em 24–48 h) é o que
  desempata.
- **Conclusão:** o CAM confirma que o modelo **não deve ser usado sem revisão humana** da imagem e mostra onde ele falha —
  informação que deve ser apresentada junto com o escore no CDSS.
"""

TEXTOS['ita_validacao'] = """
**Validação do proxy de tom de pele.** Na amostra aleatória acima, das 16 fotos classificadas automaticamente como pele
"escura" (ITA ≤ 10°), **no máximo 3 (#2, #7 e #13) mostram de fato pele mais pigmentada**; as demais são peles claras em
sombra, com luz quente/avermelhada, *flash* ou fundo escuro, e a #12 sequer é uma foto de pele (é uma ilustração de
espiroquetas e hemácias rotulada como "celulite bacteriana" — mais um ruído de coleta). Junto com a correlação ITA × brilho
de 0,68 (Gráfico 3), isso mostra que, em fotos não padronizadas da internet, **o ITA mede "tom aparente" (fototipo +
iluminação), e não fototipo**. A auditoria abaixo deve ser lida assim: o grupo "escura ou sombreada" contém poucas peles
realmente pigmentadas e muitas fotos mal iluminadas — e **a própria escassez de pele pigmentada já é um achado de viés**.
"""

TEXTOS['graf7'] = """
**Leitura técnica e clínica do Gráfico 7.** (A) O grupo de pele clara é o mais representado (42,3%). Como mostrado acima, o
grupo "escura ou sombreada" (28,1%) é composto majoritariamente por iluminação desfavorável, e não por fototipos IV–VI.
(B) A sensibilidade no limiar de triagem é de **92–93% nos grupos claro e intermediário**, mas cai para **82,9% (IC95%
76,4–87,9%) no grupo escuro/sombreado**, abaixo da meta de 90%, com intervalos que não se sobrepõem aos do grupo
intermediário; a AUC também cai (0,705 × 0,790). Na prática: **a cada 100 pacientes com EM fotografados nessas condições,
cerca de 17 seriam liberados como "baixo risco", contra 7 a 8 nos demais grupos.** Como o falso negativo da Seção 4.3 em pele
negra ilustra, é exatamente o perfil de paciente que a literatura descreve como diagnosticado mais tarde e com doença mais
disseminada — o algoritmo, treinado com este dataset, **ampliaria a desigualdade existente**.
"""

TEXTOS['etica_texto'] = """
### 5.1 (cont.) Representatividade epidemiológica e aplicação no Brasil
- **Origem dos dados:** as fotos foram coletadas da internet (atlas dermatológicos, bancos de imagem com marca d'água,
  fóruns e sites como *BadSpiderBites.com*), sem protocolo, consentimento rastreável, país de origem ou dados demográficos.
  Não representam nenhuma população clínica definida: são **o que estava disponível on-line**, um viés de seleção que
  favorece casos clássicos e pele clara — os livros-texto de dermatologia mostram pele escura em minoria das imagens
  (Adelekun *et al.*, 2021), e modelos dermatológicos treinados assim perdem desempenho em fototipos altos (Daneshjou *et al.*, 2022).
- **Demografia da Lyme:** a doença tem distribuição bimodal por idade (crianças de 5–9 anos e adultos de 55–69 anos nos EUA)
  e predomina em populações brancas de áreas endêmicas do nordeste dos EUA e da Europa. Pacientes negros são minoria nas
  notificações, mas apresentam com mais frequência doença **já disseminada** (artrite, neuroborreliose) no diagnóstico,
  em parte porque o EM é menos reconhecido em pele pigmentada (Fix; Peña; Strickland, 2000). Sem idade e sexo na base,
  não é possível verificar se o modelo funciona em crianças (lesões em face/couro cabeludo, mais frequentes) ou em idosos.
- **Distorções no SUS / saúde suplementar:** (i) **Epidemiologia diferente:** no Brasil, o quadro é a Síndrome de
  Baggio-Yoshinari (Lyme-símile), transmitida por carrapatos *Amblyomma* e *Rhipicephalus*, com espiroquetas atípicas, EM
  presente em parte dos casos e sorologias de referência norte-americanas pouco sensíveis. Um modelo treinado com fotos de
  EM norte-americano/europeu não foi validado para essa apresentação. (ii) **Prevalência muito menor:** como calculado na
  Seção 4.2, o VPP de um alerta cai para < 1% — o CDSS geraria dezenas de alarmes falsos para cada caso real, sobrecarregando
  a atenção básica e induzindo antibioticoterapia em massa. (iii) **População miscigenada:** mais da metade da população
  brasileira (55,5%) se autodeclara parda ou preta (IBGE, Censo 2022), exatamente o grupo em que a sensibilidade cai e onde o
  dataset é mais pobre. (iv) **Diferenciais locais ausentes:** hanseníase (placas anulares), leishmaniose tegumentar,
  esporotricose e picadas de insetos regionais não estão entre os 12 diferenciais do dataset.

### 5.2 Impacto ético dos erros diagnósticos na Doença de Lyme
- **Falso negativo:** o paciente com EM não recebe doxiciclina na janela em que o tratamento oral de 10 dias é curativo em
  > 90% dos casos. Semanas depois, a borrélia pode disseminar-se por via hematogênica: **EM múltiplo, paralisia facial
  periférica (às vezes bilateral), meningite linfocitária, radiculoneurite, cardite de Lyme com bloqueio atrioventricular
  (que pode exigir marca-passo temporário e já causou mortes súbitas)** e, meses depois, **artrite de Lyme** em grandes
  articulações. O tratamento tardio passa a exigir ceftriaxona endovenosa e cursos mais longos, e uma parcela dos pacientes
  evolui com sintomas persistentes pós-tratamento. Em nosso modelo, 5 de 79 casos (6,3%) seriam perdidos na triagem —
  proporção que sobe para ~17% nas fotos de pele escura/sombreada.
- **Falso positivo:** (i) **atraso do diagnóstico correto** — é o dano mais sério: o herpes-zóster rotulado como Lyme
  perde a janela de 72 h do antiviral (risco de neuralgia pós-herpética); a celulite não recebe cobertura adequada; a
  farmacodermia continua sendo provocada pelo medicamento suspeito; a tinea tratada com corticoide piora. (ii) **Exposição
  desnecessária à doxiciclina** — fotossensibilidade, esofagite, distúrbios gastrointestinais; uso evitado na gestação;
  pressão de seleção para resistência bacteriana. (iii) **Rótulo de "doença de Lyme"** — pode levar a sorologias
  repetidas, investigação de "Lyme crônica" e a tratamentos antibióticos prolongados sem benefício comprovado e com
  danos documentados, além de ansiedade e custos. Com o limiar de triagem, 255 dos 391 controles do teste (65%) seriam
  sinalizados — por isso o alerta de triagem **não pode** acionar prescrição automática.

### 5.3 Explicabilidade (XAI), caixa-preta e autonomia médica
- Médicos e pacientes não devem confiar cegamente no alerta porque **este notebook demonstrou concretamente** como o modelo
  pode errar por motivos não clínicos: rótulos contraditórios na fonte (248 fotos), cópias que inflam métricas,
  atenção a réguas e bordas da imagem (CAM), queda de sensibilidade em pele escura/fotos sombreadas e probabilidades mal
  calibradas antes do ajuste de Platt.
- **Salvaguardas para que a ferramenta seja suporte, e não substituta:** (1) exibir sempre o **mapa CAM** e os
  **biomarcadores interpretáveis** junto com a probabilidade calibrada e seu intervalo; (2) **zona indeterminada** explícita,
  que devolve a decisão à anamnese (picada de carrapato, viagem, tempo de evolução, crescimento da lesão); (3) **controle de
  qualidade da foto** antes da inferência (pele visível, iluminação), com recusa e pedido de nova foto; (4) **nenhuma
  prescrição automática** — o médico registra a decisão final e pode discordar do sistema, e as discordâncias alimentam a
  auditoria do modelo; (5) monitoramento contínuo do desempenho por subgrupo após a implantação.
- **Regulação:** um software que sugere diagnóstico é **dispositivo médico (SaMD)** e exige regularização na ANVISA
  (RDC 657/2022); em teleconsulta, a responsabilidade pela decisão é do médico (Resolução CFM 2.314/2022).

### 5.4 Privacidade, dados sensíveis e conformidade com a LGPD
- Fotografias de lesões são **dados pessoais sensíveis referentes à saúde** (LGPD, art. 5º, II) e, quando mostram rosto,
  tatuagens ou marcas, também são **dados biométricos** potencialmente identificáveis. Neste dataset há rostos de crianças
  sem tarja, tatuagens, ambientes domésticos e marcas d'água com nomes de pessoas e sites — informação suficiente para
  reidentificar o paciente por busca reversa de imagem, mesmo sem nome no arquivo.
- **Reidentificação em doença rara:** no Brasil, a Lyme-símile tem poucos casos por município; a combinação "EM + cidade +
  data + idade" pode apontar um único paciente mesmo após a pseudonimização. A remoção do nome **não** basta.
- **Cuidados indispensáveis:** (1) **base legal** — consentimento específico e destacado (art. 11, I) ou tutela da saúde por
  profissional/serviço de saúde (art. 11, II, *f*); para pesquisa, aprovação em Comitê de Ética (CEP/CONEP) e anonimização
  sempre que possível (art. 13); (2) **minimização** — recortar a imagem na lesão, remover rosto, tatuagens e metadados EXIF
  (GPS, data, modelo do celular); (3) **generalização** de idade e localização (faixas etárias, região em vez de município)
  e supressão de células com poucos casos; (4) **segurança** — criptografia, controle de acesso e registro de quem acessa;
  (5) **governança do modelo** — embeddings e modelos treinados também podem vazar dados de treino, então não devem ser
  publicados sem avaliação; (6) **transparência** — informar o paciente de que uma IA participou da triagem (art. 20 garante
  revisão de decisões automatizadas). Por fim, o próprio dataset do Kaggle, coletado da web sem consentimento dos
  retratados, **não seria aceitável como base de um produto clínico**: serve apenas para ensino e prova de conceito.
"""

TEXTOS['conclusao'] = """
## Seção 06 — Conclusões
1. **Qualidade dos dados:** 17,9% da base estava em grupos de duplicatas e 248 fotos tinham rótulos contraditórios; após a
   limpeza restaram 4.451 fotos (705 de Lyme, razão 5,3 : 1). Não há dados faltantes relevantes nos atributos de imagem, mas
   todas as variáveis clínicas do paciente estão ausentes na fonte.
2. **Pergunta 02 (classificação):** biomarcadores visuais isolados diferem pouco entre EM e diferenciais (|δ de Cliff| ≤
   0,17); o *transfer learning* com ConvNeXt-Tiny atinge **AUC-ROC 0,785 e AUC-PR 0,487** no teste independente. Com o limiar
   de triagem, a sensibilidade é 93,7% e o VPN 96,5%, mas a especificidade é baixa (34,8%).
3. **Pergunta 01 (agrupamento):** os *embeddings* das lesões de Lyme formam dois agrupamentos fracos (silhueta 0,099), que
   correspondem ao enquadramento da foto (close-up × contexto), e não a estágios clínicos.
4. **Pergunta 03 (temporal):** não respondível com esta base, por ausência total de variáveis temporais e de escala física;
   foi proposto um protocolo de coleta seriada com métricas de erro em centímetros.
5. **Pergunta 04 (CDSS):** recomendamos um fluxo em três zonas (escore < 0,21 / 0,21–0,74 / ≥ 0,74) com probabilidades
   calibradas, CAM e controle de qualidade da foto, sempre subordinado à anamnese e à decisão médica.
6. **Ética:** o modelo perde sensibilidade em fotos de pele escura/sombreada (82,9% × 92–93%), não foi validado para a
   Lyme-símile brasileira e teria VPP < 1% na prevalência nacional. **Antes de qualquer uso clínico** seriam necessários
   dados prospectivos, com consentimento, fototipo registrado e diversidade populacional, validação externa no Brasil e
   regularização como SaMD.

**Próximos passos técnicos:** *fine-tuning* da CNN com aumentação de dados, segmentação da lesão para eliminar atalhos (réguas,
bordas), rotulagem dermatológica independente das fotos ambíguas e coleta de metadados clínicos para modelos multimodais.
"""
