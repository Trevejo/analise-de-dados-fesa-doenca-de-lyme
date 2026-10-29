"""Gera o notebook da Entrega 02 (pipeline analitica, CDSS e etica) a partir de celulas declaradas aqui.

Uso:
    python3 scripts/create_notebook_entrega02.py
O notebook gerado deve ser executado (Colab ou Jupyter) para que todas as saidas fiquem visiveis.
"""
import os
import nbformat as nbf

from notebook_entrega02_textos import TEXTOS

NB_PATH = os.path.join(os.path.dirname(__file__), '..', 'notebooks', '02_pipeline_analitica_cdss_etica_lyme.ipynb')

cells = []


def md(texto):
    cells.append(nbf.v4.new_markdown_cell(texto.strip('\n')))


def code(texto):
    cells.append(nbf.v4.new_code_cell(texto.strip('\n')))


def interp(chave):
    md(TEXTOS[chave])


# =====================================================================================
# SECAO 01 - Identificacao, resgate clinico e carregamento
# =====================================================================================
md(TEXTOS['cabecalho'])
md(TEXTOS['sintese_clinica'])

md("""
### 1.3 Configuração do ambiente (reprodutibilidade)
A célula abaixo funciona tanto no **Google Colab** quanto em um Jupyter local: se a base não for encontrada,
o repositório público do projeto é clonado (imagens + metadados + *cache* de atributos). Todas as etapas
aleatórias usam a semente fixa `RANDOM_STATE = 42`.
""")
code(r'''
# ---------------------------------------------------------------
# 1.3 Ambiente: bibliotecas, semente e localizacao dos dados
# ---------------------------------------------------------------
import os, sys, subprocess, warnings, importlib

# shap e scikit-image ja vem no Colab; instala apenas se faltar
for pacote, modulo in [('shap', 'shap'), ('scikit-image', 'skimage')]:
    if importlib.util.find_spec(modulo) is None:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pacote], check=True)

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from PIL import Image
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from skimage.color import rgb2lab, rgb2hsv
from skimage.feature import graycomatrix, graycoprops, canny
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.calibration import calibration_curve
from sklearn.metrics import (confusion_matrix, roc_auc_score, average_precision_score, roc_curve,
                             precision_recall_curve, f1_score, brier_score_loss, silhouette_score,
                             davies_bouldin_score, adjusted_rand_score)

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', module='tqdm')
import shap
pd.set_option('display.max_columns', 40)
pd.set_option('display.width', 160)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

REPO_URL = 'https://github.com/Trevejo/analise-de-dados-fesa-doenca-de-lyme.git'
CANDIDATOS = ['..', '.', '/content/analise-de-dados-fesa-doenca-de-lyme']
ROOT = next((c for c in CANDIDATOS if os.path.exists(os.path.join(c, 'data', 'metadata_lyme_rashes.csv'))), None)
if ROOT is None:  # execucao no Colab sem a base: clona o repositorio publico
    subprocess.run(['git', 'clone', '--depth', '1', REPO_URL, CANDIDATOS[-1]], check=True)
    ROOT = CANDIDATOS[-1]

IMG_DIR = os.path.join(ROOT, 'data', 'DataSplit', 'image_folder 2')
CACHE_DIR = os.path.join(ROOT, 'data', 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# Paleta acessivel (validada para daltonismo): azul = controles, laranja = Lyme
COR_NEG, COR_POS, COR_3, COR_4 = '#2a78d6', '#eb6834', '#1baf7a', '#6250d6'
COR_NEUTRA, COR_TEXTO = '#898781', '#52514e'
ROTULO = {0: 'Controle (diagnósticos diferenciais)', 1: 'Lyme (eritema migratório)'}
PALETA_CLASSE = {ROTULO[0]: COR_NEG, ROTULO[1]: COR_POS}
CMAP_DIV = LinearSegmentedColormap.from_list('azul_cinza_vermelho', ['#184f95', '#6da7ec', '#f0efec', '#ef8f8e', '#b3302f'])
mpl.rcParams.update({
    'figure.dpi': 110, 'savefig.dpi': 150, 'font.size': 9.5, 'axes.titlesize': 11, 'axes.titleweight': 'bold',
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#c3c2b7', 'axes.grid': True,
    'grid.color': '#e1e0d9', 'grid.linewidth': 0.6, 'axes.axisbelow': True, 'xtick.color': COR_TEXTO,
    'ytick.color': COR_TEXTO, 'legend.frameon': False,
})

print(f'Python {sys.version.split()[0]} | pandas {pd.__version__} | numpy {np.__version__}')
print('Executando no Google Colab' if 'google.colab' in sys.modules else 'Executando em Jupyter local')
print('Base localizada:', os.path.exists(IMG_DIR))
''')

md("""
### 1.4 Ingestão e inspeção inicial da base
A base tabular `metadata_lyme_rashes.csv` (construída na Entrega 01) descreve cada uma das 5.059 fotografias
dermatológicas do dataset *Lyme Disease Rashes* (Kaggle). Abaixo: dimensões (`shape`), tipos (`info()`) e amostra (`head()`).
""")
code(r'''
# ---------------------------------------------------------------
# 1.4 Ingestao da base de metadados (1 linha = 1 fotografia)
# ---------------------------------------------------------------
df_meta = pd.read_csv(os.path.join(ROOT, 'data', 'metadata_lyme_rashes.csv'))
assert df_meta['filename'].is_unique, 'nomes de arquivo devem ser unicos'

print('Dimensões da base (linhas, colunas):', df_meta.shape)
print()
df_meta.info()
display(df_meta.head())
''')

md("""
### 1.5 Dicionário resumido e extração dos biomarcadores visuais digitais
Na Entrega 01 mostramos que a cor média isolada **não** separa Lyme dos controles. Por isso, nesta etapa cada
fotografia é convertida em dois conjuntos de atributos:

1. **Biomarcadores visuais interpretáveis** (calculados com `scikit-image`), que traduzem os critérios que o
   dermatologista usa no exame físico: intensidade do eritema, contraste da lesão em relação à pele vizinha,
   padrão radial centro→borda (proxy do aspecto "em alvo"), textura/descamação e tom de pele;
2. ***Embeddings* profundos** de uma rede convolucional pré-treinada no ImageNet (*transfer learning*), que
   capturam forma, borda e padrão espacial da lesão em um vetor numérico.

Como a extração percorre todas as imagens (≈ 2 min para biomarcadores e ≈ 5–10 min para os *embeddings* em CPU),
os resultados são gravados em `data/cache/` e reaproveitados nas execuções seguintes.
""")
code(r'''
# ---------------------------------------------------------------
# 1.5 Dicionario resumido dos atributos usados na Entrega 02
# ---------------------------------------------------------------
dicionario = pd.DataFrame([
    ('binary_class',              'Categórico', '—',            'Desfecho: Positive = eritema migratório (Lyme); Negative = diagnóstico diferencial'),
    ('condition',                 'Categórico', '15 classes',   'Diagnóstico específico (3 rótulos de EM + 12 diferenciais)'),
    ('split',                     'Categórico', 'train/val/test','Partição oficial estratificada do dataset'),
    ('width, height',             'Inteiro',    'pixels',       'Resolução da fotografia (atributo de aquisição)'),
    ('brightness, contrast',      'Contínuo',   '0–255',        'Luminância média e desvio-padrão (qualidade da foto)'),
    ('erythema_index',            'Contínuo',   'R−G (0–255)',  'Índice de eritema clássico (hiperemia média)'),
    ('lab_a_mean',                'Contínuo',   'a* CIELAB',    'Vermelhidão média perceptual (eixo verde→vermelho)'),
    ('a_p90_minus_p50',           'Contínuo',   'Δa* CIELAB',   'Contraste eritematoso: quanto a área mais vermelha excede a pele típica da foto'),
    ('a_std',                     'Contínuo',   'a* CIELAB',    'Heterogeneidade da vermelhidão na imagem'),
    ('saturation_mean',           'Contínuo',   '0–1 (HSV)',    'Saturação média da cor (vivacidade da lesão)'),
    ('radial_a_center_minus_mid', 'Contínuo',   'Δa* CIELAB',   'Centro − anel intermediário: >0 lesão centralizada; <0 clareamento central (alvo)'),
    ('radial_a_range',            'Contínuo',   'Δa* CIELAB',   'Amplitude do perfil radial de vermelhidão (anéis concêntricos)'),
    ('radial_a_sign_changes',     'Discreto',   'contagem',     'Alternâncias claro/escuro no perfil radial (padrão anelar)'),
    ('glcm_contrast / homogeneity / energy / correlation', 'Contínuo', 'adimensional', 'Textura de Haralick (descamação, vesículas, pápulas)'),
    ('gray_entropy',              'Contínuo',   'bits',         'Complexidade/desordem da textura'),
    ('edge_density',              'Contínuo',   'fração 0–1',   'Densidade de bordas (Canny): lesões múltiplas/irregulares'),
    ('sharpness_lap',             'Contínuo',   'nível de cinza','Nitidez (energia do Laplaciano) — controle de foco'),
    ('skin_fraction',             'Contínuo',   'fração 0–1',   'Fração de pixels classificados como pele (YCbCr)'),
    ('ita_deg',                   'Contínuo',   'graus (°)',    'Ângulo Tipológico Individual da pele perilesional — proxy de fototipo'),
    ('emb_0 … emb_n',             'Contínuo',   'adimensional', 'Embedding da CNN pré-treinada (média global dos mapas de ativação)'),
], columns=['Atributo', 'Tipo', 'Unidade', 'Significado clínico / computacional'])
display(dicionario.style.hide(axis='index'))
''')

code(r'''
# ---------------------------------------------------------------
# 1.5 Extracao dos biomarcadores visuais interpretaveis (com cache)
# ---------------------------------------------------------------
BIO_CACHE = os.path.join(CACHE_DIR, 'biomarcadores_imagem.csv')

def caminho_imagem(linha):
    return os.path.join(IMG_DIR, linha['binary_class'], linha['filename'])

def mascara_pele(rgb):
    """Segmentacao classica de pele no espaco YCbCr (Chai & Ngan, 1999)."""
    ycc = np.asarray(Image.fromarray(rgb).convert('YCbCr'), dtype=np.float32)
    cb, cr = ycc[..., 1], ycc[..., 2]
    return (cr >= 133) & (cr <= 180) & (cb >= 77) & (cb <= 135)

def extrair_biomarcadores(caminho):
    rgb = np.asarray(Image.open(caminho).convert('RGB'))
    lab = rgb2lab(rgb)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    f = {'lab_L_mean': L.mean(), 'lab_a_mean': a.mean(), 'lab_b_mean': b.mean(),
         'a_p90_minus_p50': np.percentile(a, 90) - np.median(a), 'a_std': a.std(),
         'saturation_mean': rgb2hsv(rgb)[..., 1].mean()}

    # Tom de pele (ITA) medido apenas na pele perilesional: pixels de pele com a* abaixo da mediana
    pele = mascara_pele(rgb)
    f['skin_fraction'] = pele.mean()
    if pele.sum() >= 200:
        perilesional = pele & (a <= np.median(a[pele]))
        Lp, bp = L[perilesional].mean(), max(b[perilesional].mean(), 1e-3)
        f['ita_deg'] = np.degrees(np.arctan((Lp - 50) / bp))
    else:
        f['ita_deg'] = np.nan  # mascara de pele insuficiente -> valor ausente (investigado na Secao 2)

    # Perfil radial de vermelhidao em 5 aneis concentricos a partir do centro da foto
    h, w = a.shape
    yy, xx = np.mgrid[0:h, 0:w]
    raio = np.sqrt(((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2)
    aneis = np.array([a[(raio >= r0) & (raio < r0 + 0.2)].mean() for r0 in np.arange(0, 1.0, 0.2)])
    d = np.diff(aneis)
    f['radial_a_center_minus_mid'] = aneis[0] - aneis[2]
    f['radial_a_range'] = aneis.max() - aneis.min()
    f['radial_a_sign_changes'] = int(np.sum(np.sign(d[1:]) != np.sign(d[:-1])))

    # Textura (matriz de co-ocorrencia de Haralick), entropia, bordas e nitidez
    cinza = np.asarray(Image.fromarray(rgb).convert('L'))
    glcm = graycomatrix((cinza // 8).astype(np.uint8), [1, 3], [0, np.pi / 2], levels=32, symmetric=True, normed=True)
    for prop in ['contrast', 'homogeneity', 'energy', 'correlation']:
        f['glcm_' + prop] = graycoprops(glcm, prop).mean()
    hist = np.bincount(cinza.ravel(), minlength=256) / cinza.size
    f['gray_entropy'] = -(hist[hist > 0] * np.log2(hist[hist > 0])).sum()
    f['edge_density'] = canny(cinza / 255.0, sigma=2).mean()
    g = cinza.astype(np.float32)
    f['sharpness_lap'] = np.abs(np.diff(g, 2, axis=0)).mean() + np.abs(np.diff(g, 2, axis=1)).mean()
    return f

if os.path.exists(BIO_CACHE):
    df_bio = pd.read_csv(BIO_CACHE)
    print(f'Biomarcadores carregados do cache: {df_bio.shape}')
else:
    linhas = []
    for i, linha in df_meta.iterrows():
        linhas.append({'filename': linha['filename'], **extrair_biomarcadores(caminho_imagem(linha))})
    df_bio = pd.DataFrame(linhas)
    df_bio.to_csv(BIO_CACHE, index=False)
    print(f'Biomarcadores extraidos e salvos em cache: {df_bio.shape}')

df = df_meta.merge(df_bio, on='filename', how='left', validate='one_to_one')
df['y'] = (df['binary_class'] == 'Positive').astype(int)
df['classe'] = df['y'].map(ROTULO)
display(df[['filename', 'classe', 'lab_a_mean', 'a_p90_minus_p50', 'radial_a_center_minus_mid', 'glcm_contrast', 'ita_deg']].head())
''')

code(r'''
# ---------------------------------------------------------------
# 1.5 Embeddings profundos via transfer learning (com cache)
# ---------------------------------------------------------------
import torch
from torchvision import models

BACKBONE = 'convnext_tiny'   # ConvNeXt-Tiny pre-treinada no ImageNet-1k (torchvision)
EMB_CACHE = os.path.join(CACHE_DIR, f'embeddings_{BACKBONE}.npz')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def carregar_backbone():
    pesos = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    return models.convnext_tiny(weights=pesos).eval().to(DEVICE), pesos.transforms()

if os.path.exists(EMB_CACHE):
    pacote = np.load(EMB_CACHE, allow_pickle=True)
    assert (pacote['filename'] == df['filename'].values).all()
    EMB, THUMBS = pacote['emb'].astype(np.float32), pacote['thumb16'].astype(np.float32)
    print(f'Embeddings carregados do cache: {EMB.shape} (dispositivo disponivel: {DEVICE})')
else:
    rede, transformar = carregar_backbone()
    caminhos = [caminho_imagem(l) for _, l in df.iterrows()]
    lotes, miniaturas = [], []
    with torch.no_grad():
        for i in range(0, len(caminhos), 64):
            imgs = [Image.open(c).convert('RGB') for c in caminhos[i:i + 64]]
            x = torch.stack([transformar(im) for im in imgs]).to(DEVICE)
            lotes.append(rede.features(x).mean(dim=(2, 3)).cpu().numpy())   # media global (GAP) por canal
            miniaturas += [np.asarray(im.convert('L').resize((16, 16)), dtype=np.uint8).ravel() for im in imgs]
    EMB, THUMBS = np.concatenate(lotes).astype(np.float32), np.stack(miniaturas).astype(np.float32)
    np.savez_compressed(EMB_CACHE, emb=EMB.astype(np.float16), thumb16=THUMBS.astype(np.uint8), filename=df['filename'].values)
    print(f'Embeddings extraidos e salvos em cache: {EMB.shape}')

print(f'Cada fotografia -> vetor de {EMB.shape[1]} atributos profundos + {df_bio.shape[1] - 1} biomarcadores interpretaveis')
''')

# =====================================================================================
# SECAO 02 - Qualidade dos dados e pre-processamento clinico
# =====================================================================================
md(TEXTOS['secao2_intro'])

code(r'''
# ---------------------------------------------------------------
# 2.1 Quantificacao exata de valores ausentes (% por atributo)
# ---------------------------------------------------------------
BIOMARCADORES = [c for c in df_bio.columns if c != 'filename']
ausentes_obs = (df[list(df_meta.columns) + BIOMARCADORES].isna().mean() * 100).rename('% ausente')

# Variaveis clinicas que um prontuario de Lyme teria, mas que a fonte (fotos coletadas da web) nao registra
CLINICAS_AUSENTES = ['Idade', 'Sexo biológico', 'Fototipo (Fitzpatrick)', 'Local anatômico da lesão',
                     'Diâmetro da lesão (cm)', 'Dias desde a picada / início', 'Exposição a carrapato / área endêmica',
                     'Sintomas sistêmicos (febre, artralgia)', 'Sorologia (ELISA / Western blot)', 'País / região']
ausentes_clin = pd.Series(100.0, index=CLINICAS_AUSENTES, name='% ausente')

tabela_ausentes = pd.concat([
    ausentes_obs.to_frame().assign(Origem='Atributo observado (imagem/metadado)'),
    ausentes_clin.to_frame().assign(Origem='Variável clínica não registrada na fonte'),
])
print('Atributos observados com algum valor ausente:')
display(tabela_ausentes[(tabela_ausentes['% ausente'] > 0) & (tabela_ausentes['Origem'].str.startswith('Atributo'))].round(2))
print(f"Atributos observados 100% completos: {(ausentes_obs == 0).sum()} de {len(ausentes_obs)}")
print(f"Variáveis clínicas estruturalmente ausentes (100%): {len(ausentes_clin)}")
''')

code(r'''
# ---------------------------------------------------------------
# 2.1 O ITA ausente e aleatorio (MCAR) ou informativo (MNAR)?
# ---------------------------------------------------------------
df['ita_ausente'] = df['ita_deg'].isna()
comparacao = df.groupby('ita_ausente')[['skin_fraction', 'brightness', 'lab_L_mean', 'y']].median()
comparacao.index = ['ITA calculado', 'ITA ausente']
comparacao['n'] = df.groupby('ita_ausente').size().values
display(comparacao.round(3))

u = stats.mannwhitneyu(df.loc[df.ita_ausente, 'brightness'], df.loc[~df.ita_ausente, 'brightness'])
print(f'Mann-Whitney (brilho, ausente vs presente): p = {u.pvalue:.2e}')
print('Condições das fotos com ITA ausente:', df.loc[df.ita_ausente, 'condition'].value_counts().to_dict())

# Imputacao descritiva: mediana estratificada por grupo clinico (usada SOMENTE nas estatisticas descritivas)
mediana_por_grupo = df.groupby('y')['ita_deg'].median()
df['ita_deg_imput'] = df['ita_deg'].fillna(df['y'].map(mediana_por_grupo))
print('Mediana do ITA por grupo usada na imputação descritiva:', mediana_por_grupo.round(2).rename(ROTULO).to_dict())
''')

md(TEXTOS['missing_decisao'])

code(r'''
# ---------------------------------------------------------------
# 2.2 Ruido de coleta: duplicatas e conflitos de rotulo
# ---------------------------------------------------------------
# Criterio duplo e conservador: mesma "semantica" (cosseno dos embeddings >= 0,95)
# E mesma composicao de pixels (correlacao das miniaturas 16x16 >= 0,95)
emb_norm = EMB / np.linalg.norm(EMB, axis=1, keepdims=True)
sim_semantica = emb_norm @ emb_norm.T
mini = (THUMBS - THUMBS.mean(1, keepdims=True)) / (THUMBS.std(1, keepdims=True) + 1e-6)
sim_pixels = mini @ mini.T / mini.shape[1]
duplicata = (sim_semantica >= 0.95) & (sim_pixels >= 0.95)
np.fill_diagonal(duplicata, False)
del sim_semantica, sim_pixels

_, grupo = connected_components(csr_matrix(duplicata), directed=False)
df['grupo_dup'] = grupo
resumo_grupos = df.groupby('grupo_dup').agg(n=('y', 'size'), n_pos=('y', 'sum'), n_splits=('split', 'nunique'))
grupos_dup = resumo_grupos[resumo_grupos['n'] > 1]
grupos_conflito = grupos_dup[(grupos_dup['n_pos'] > 0) & (grupos_dup['n_pos'] < grupos_dup['n'])].index
grupos_vazamento = grupos_dup[grupos_dup['n_splits'] > 1].index

df['em_grupo_dup'] = df['grupo_dup'].isin(grupos_dup.index)
df['conflito_rotulo'] = df['grupo_dup'].isin(grupos_conflito)
print(f'Grupos de imagens duplicadas: {len(grupos_dup)} (envolvendo {df.em_grupo_dup.sum()} fotos = {df.em_grupo_dup.mean():.1%} da base)')
print(f'Grupos que cruzam treino/validação/teste (vazamento): {len(grupos_vazamento)}')
print(f'Grupos com rótulos contraditórios (mesma foto = Lyme e não-Lyme): {len(grupos_conflito)} '
      f'({df.conflito_rotulo.sum()} fotos)')

pares = (df[df.conflito_rotulo].groupby('grupo_dup')['condition']
         .apply(lambda s: ' × '.join(sorted(set(s)))).value_counts().head(8))
print('\nPares de rótulos contraditórios mais frequentes:')
print(pares.to_string())
''')

code(r'''
# ---------------------------------------------------------------
# 2.2 Evidencia visual: a mesma fotografia com rotulos clinicos diferentes
# ---------------------------------------------------------------
exemplos = df[df.conflito_rotulo].groupby('grupo_dup').head(2).groupby('grupo_dup').filter(lambda g: len(g) == 2)
exemplos = exemplos.groupby('grupo_dup').head(2).iloc[:8]
fig, eixos = plt.subplots(2, 4, figsize=(12, 6.2))
for ax, (_, linha) in zip(eixos.ravel(), exemplos.iterrows()):
    ax.imshow(Image.open(caminho_imagem(linha)).convert('RGB').resize((200, 200)))
    cor = COR_POS if linha['y'] == 1 else COR_NEG
    ax.set_title(f"{linha['condition']}\n({linha['split']})", fontsize=8.5, color=cor)
    ax.axis('off')
fig.suptitle('Ruído de rótulo: cada par lado a lado é a MESMA foto registrada com diagnósticos diferentes\n'
             '(laranja = rotulada como Lyme; azul = rotulada como diagnóstico diferencial)', fontsize=10.5, fontweight='bold')
plt.tight_layout()
plt.show()
''')

code(r'''
# ---------------------------------------------------------------
# 2.2 Outliers: implausiveis/tecnicos vs extremos patologicos reais
# ---------------------------------------------------------------
def outliers_iqr(serie):
    q1, q3 = serie.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (serie < q1 - 1.5 * iqr) | (serie > q3 + 1.5 * iqr)

lado_menor = df[['width', 'height']].min(axis=1)
tecnicos = pd.DataFrame({
    'Arquivo corrompido': df['corrupted'],
    'Resolução < 100 px no menor lado': lado_menor < 100,
    'Subexposição (brilho < 66,8)': df['brightness'] < 66.8,
    'Superexposição (brilho > 205,6)': df['brightness'] > 205.6,
    'Pouca pele visível (< 5% dos pixels)': df['skin_fraction'] < 0.05,
})
tab_tec = pd.DataFrame({'n fotos': tecnicos.sum(), '% base': (tecnicos.mean() * 100).round(2),
                        '% entre Lyme': (tecnicos[df.y == 1].mean() * 100).round(2),
                        '% entre controles': (tecnicos[df.y == 0].mean() * 100).round(2)})
print('A) Ruídos técnicos / valores implausíveis de aquisição')
display(tab_tec)

patologicos = ['lab_a_mean', 'a_p90_minus_p50', 'erythema_index', 'saturation_mean', 'radial_a_range', 'glcm_contrast', 'edge_density']
tab_pat = pd.DataFrame({
    'outliers IQR (n)': [outliers_iqr(df[c]).sum() for c in patologicos],
    '% entre Lyme': [outliers_iqr(df[c])[df.y == 1].mean() * 100 for c in patologicos],
    '% entre controles': [outliers_iqr(df[c])[df.y == 0].mean() * 100 for c in patologicos],
    'Mín.': [df[c].min() for c in patologicos], 'Máx.': [df[c].max() for c in patologicos],
}, index=patologicos).round(2)
print('B) Extremos de biomarcadores (regra de Tukey: fora de Q1 − 1,5·IIQ ou Q3 + 1,5·IIQ)')
display(tab_pat)
df['baixa_qualidade'] = tecnicos.drop(columns='Arquivo corrompido').any(axis=1)
''')

md(TEXTOS['outliers_decisao'])

code(r'''
# ---------------------------------------------------------------
# 2.2 Aplicacao das decisoes de limpeza
# ---------------------------------------------------------------
# 1) Remove grupos com rotulo contraditorio (verdade-terreno indefinida)
# 2) Mantem uma unica copia de cada grupo de duplicatas (prioridade teste > validacao > treino),
#    eliminando o vazamento de imagens identicas entre particoes
prioridade = {'test': 0, 'val': 1, 'train': 2}
df_limpo = (df[~df['conflito_rotulo']]
            .assign(_prio=df['split'].map(prioridade))
            .sort_values(['_prio', 'filename'])
            .drop_duplicates('grupo_dup')
            .drop(columns='_prio')
            .sort_index())
IDX_LIMPO = df_limpo.index.values           # posicoes no vetor EMB
EMB_LIMPO = EMB[IDX_LIMPO]
df_limpo = df_limpo.reset_index(drop=True)

auditoria = pd.DataFrame({
    'Etapa': ['Base original', 'Removidas: rótulo contraditório', 'Removidas: cópias redundantes', 'Base analítica final'],
    'Fotos': [len(df), df.conflito_rotulo.sum(), len(df) - df.conflito_rotulo.sum() - len(df_limpo), len(df_limpo)],
})
display(auditoria.style.hide(axis='index'))
display(pd.crosstab(df_limpo['split'], df_limpo['classe'], margins=True, margins_name='Total'))
''')

md("### 2.3 Análise de desbalanceamento de classes")
code(r'''
# ---------------------------------------------------------------
# 2.3 Desbalanceamento de classes (antes e depois da limpeza)
# ---------------------------------------------------------------
def resumo_classes(base, nome):
    n_pos, n_neg = base['y'].sum(), (1 - base['y']).sum()
    return {'Base': nome, 'Lyme (n)': n_pos, 'Controles (n)': n_neg, 'Prevalência Lyme': f'{n_pos / len(base):.1%}',
            'Razão Controle:Lyme': f'{n_neg / n_pos:.2f} : 1'}

display(pd.DataFrame([resumo_classes(df, 'Original (5.059)'), resumo_classes(df_limpo, 'Após limpeza')]).style.hide(axis='index'))
''')

md("""
#### Gráfico 1 — Qualidade e proporção da base
Painel A: percentual de valores ausentes por atributo **antes** do tratamento (atributos observados e variáveis
clínicas que a fonte não registra). Painel B: proporção de classes por partição após a limpeza.
""")
code(r'''
# ---------------------------------------------------------------
# GRAFICO 1 - Valores ausentes por atributo + proporcao de classes
# ---------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.3), gridspec_kw={'width_ratios': [1.25, 1]})

graf = tabela_ausentes.copy()
graf = pd.concat([graf[graf.Origem.str.startswith('Variável')],
                  graf[graf.Origem.str.startswith('Atributo') & (graf['% ausente'] > 0)],
                  graf[graf.Origem.str.startswith('Atributo') & (graf['% ausente'] == 0)].iloc[:12]])
cores = [COR_NEUTRA if o.startswith('Variável') else COR_NEG for o in graf['Origem']]
ax1.barh(graf.index[::-1], graf['% ausente'][::-1].clip(lower=0.4), color=cores[::-1], height=0.7)
for i, v in enumerate(graf['% ausente'][::-1]):
    ax1.text(max(v, 0.4) + 1, i, f'{v:.1f}%', va='center', fontsize=8, color=COR_TEXTO)
ax1.set_xlim(0, 112)
ax1.set_xlabel('Valores ausentes (% das 5.059 fotografias)\n'
               f'(+ {(ausentes_obs == 0).sum() - 12} outros atributos observados com 0% ausente omitidos)')
ax1.set_title('A. Valores ausentes por atributo (antes do tratamento)')
ax1.legend(handles=[mpl.patches.Patch(color=COR_NEUTRA, label='Variável clínica não registrada na fonte'),
                    mpl.patches.Patch(color=COR_NEG, label='Atributo observado (imagem/metadado)')], loc='lower right', fontsize=8)

prop = pd.crosstab(df_limpo['split'], df_limpo['y'], normalize='index').loc[['train', 'val', 'test']] * 100
cont = pd.crosstab(df_limpo['split'], df_limpo['y']).loc[['train', 'val', 'test']]
nomes = ['Treino', 'Validação', 'Teste']
ax2.bar(nomes, prop[0], color=COR_NEG, label=ROTULO[0], width=0.6, edgecolor='white', linewidth=2)
ax2.bar(nomes, prop[1], bottom=prop[0], color=COR_POS, label=ROTULO[1], width=0.6, edgecolor='white', linewidth=2)
for i, s in enumerate(['train', 'val', 'test']):
    ax2.text(i, prop.loc[s, 0] / 2, f'{prop.loc[s, 0]:.1f}%\n(n={cont.loc[s, 0]})', ha='center', va='center', color='white', fontsize=9, fontweight='bold')
    ax2.text(i, prop.loc[s, 0] + prop.loc[s, 1] / 2, f'{prop.loc[s, 1]:.1f}%\n(n={cont.loc[s, 1]})', ha='center', va='center', color='white', fontsize=9, fontweight='bold')
ax2.set_ylabel('Proporção de fotografias (%)')
ax2.set_ylim(0, 100)
ax2.set_title(f'B. Proporção de classes por partição (razão {(1 - df_limpo.y).sum() / df_limpo.y.sum():.2f} : 1)')
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=8.5)
ax2.grid(axis='x', visible=False)
plt.tight_layout()
plt.show()
''')
interp('graf1')

code(r'''
# ---------------------------------------------------------------
# 2.4 Padronizacao e codificacao
# ---------------------------------------------------------------
# Assimetria (skewness) justifica o RobustScaler (mediana e IIQ) nos biomarcadores
assimetria = df_limpo[BIOMARCADORES].skew().sort_values(key=np.abs, ascending=False).round(2)
print('Biomarcadores mais assimétricos (|skew| > 1 indica cauda longa):')
print(assimetria.head(8).to_string())

dev_mask = df_limpo['split'].isin(['train', 'val']).values
demo = RobustScaler().fit(df_limpo.loc[dev_mask, BIOMARCADORES])   # ajuste APENAS no conjunto de desenvolvimento
escalado = pd.DataFrame(demo.transform(df_limpo[BIOMARCADORES]), columns=BIOMARCADORES)
cols_demo = ['lab_a_mean', 'a_p90_minus_p50', 'glcm_contrast', 'edge_density']
antes_depois = pd.concat({'Antes (unidade original)': df_limpo[cols_demo].describe().loc[['50%', 'mean', 'std', 'min', 'max']],
                          'Depois (RobustScaler)': escalado[cols_demo].describe().loc[['50%', 'mean', 'std', 'min', 'max']]})
display(antes_depois.round(3))

# Codificacao das variaveis categoricas clinicas
df_limpo['grupo_tom_pele'] = pd.cut(df_limpo['ita_deg'], [-90, 10, 41, 90],
                                    labels=['Escura ou sombreada (ITA ≤ 10°)', 'Intermediária (10°–41°)', 'Clara (ITA > 41°)'])
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(df_limpo[['condition']])
condicao_ohe = pd.DataFrame(ohe.transform(df_limpo[['condition']]), columns=ohe.get_feature_names_out())
print(f"Desfecho codificado: y = 1 (Lyme) / 0 (controle). Condição clínica -> {condicao_ohe.shape[1]} colunas one-hot. "
      f"Tom de pele -> 3 categorias ordinais (ITA).")
display(condicao_ohe.iloc[:3, :6])
''')
interp('escala')

# =====================================================================================
# SECAO 03 - Metricas estatisticas, EDA e modelagem
# =====================================================================================
md(TEXTOS['secao3_intro'])

code(r'''
# ---------------------------------------------------------------
# 3.A Tendencia central, dispersao e testes de hipotese por grupo
# ---------------------------------------------------------------
CHAVE = ['lab_a_mean', 'a_p90_minus_p50', 'erythema_index', 'a_std', 'saturation_mean', 'radial_a_center_minus_mid',
         'radial_a_range', 'glcm_contrast', 'glcm_homogeneity', 'gray_entropy', 'edge_density', 'sharpness_lap',
         'ita_deg_imput', 'brightness']

def cliff_delta(x, y):
    """Tamanho de efeito nao parametrico: P(X>Y) - P(X<Y), a partir da estatistica U."""
    u = stats.mannwhitneyu(x, y).statistic
    return 2 * u / (len(x) * len(y)) - 1

linhas = []
for c in CHAVE:
    pos, neg = df_limpo.loc[df_limpo.y == 1, c], df_limpo.loc[df_limpo.y == 0, c]
    linhas.append({'Biomarcador': c,
                   'Lyme média ± DP': f'{pos.mean():.2f} ± {pos.std():.2f}',
                   'Lyme mediana [IIQ]': f'{pos.median():.2f} [{pos.quantile(.25):.2f}; {pos.quantile(.75):.2f}]',
                   'Controle média ± DP': f'{neg.mean():.2f} ± {neg.std():.2f}',
                   'Controle mediana [IIQ]': f'{neg.median():.2f} [{neg.quantile(.25):.2f}; {neg.quantile(.75):.2f}]',
                   'Assimetria': round(df_limpo[c].skew(), 2),
                   'p (Mann-Whitney)': stats.mannwhitneyu(pos, neg).pvalue,
                   'p (t de Welch)': stats.ttest_ind(pos, neg, equal_var=False).pvalue,
                   "δ de Cliff": cliff_delta(pos, neg)})
tab_desc = pd.DataFrame(linhas)
# Correcao de Benjamini-Hochberg para comparacoes multiplas
ordem = tab_desc['p (Mann-Whitney)'].rank(method='first').values
m = len(tab_desc)
q = (tab_desc['p (Mann-Whitney)'] * m / ordem).values
q_ord = np.minimum.accumulate(q[np.argsort(ordem)[::-1]])[::-1]
tab_desc['q (BH)'] = np.clip(q_ord[np.argsort(np.argsort(ordem))], 0, 1)
tab_desc['Significativo (q < 0,05)'] = np.where(tab_desc['q (BH)'] < 0.05, 'Sim', 'Não')
tab_desc = tab_desc.sort_values("δ de Cliff", key=np.abs, ascending=False).reset_index(drop=True)
display(tab_desc.style.format({'Assimetria': '{:.2f}', 'p (Mann-Whitney)': '{:.1e}', 'p (t de Welch)': '{:.1e}', 'q (BH)': '{:.1e}', "δ de Cliff": '{:+.3f}'}).hide(axis='index'))
''')
interp('tab_desc')

md("""
#### Gráfico 2 — Distribuição comparativa dos biomarcadores-chave (Lyme × controles)
Os seis biomarcadores com maior tamanho de efeito (|δ de Cliff|) são exibidos em *violin plots* com *boxplot* interno
(mediana = linha branca; caixa = IIQ). Os pontos além dos bigodes são os extremos patológicos mantidos na base.
""")
code(r'''
# ---------------------------------------------------------------
# GRAFICO 2 - Violin + boxplot dos 6 biomarcadores mais discriminantes
# ---------------------------------------------------------------
UNIDADES = {'lab_a_mean': 'a* médio (CIELAB)', 'a_p90_minus_p50': 'Δa* P90 − mediana (CIELAB)', 'erythema_index': 'R − G (0–255)',
            'a_std': 'DP de a* (CIELAB)', 'saturation_mean': 'Saturação HSV (0–1)', 'radial_a_center_minus_mid': 'Δa* centro − anel médio',
            'radial_a_range': 'Amplitude radial de a*', 'glcm_contrast': 'Contraste GLCM (adim.)', 'glcm_homogeneity': 'Homogeneidade GLCM (adim.)',
            'gray_entropy': 'Entropia (bits)', 'edge_density': 'Densidade de bordas (fração)', 'sharpness_lap': 'Nitidez (nível de cinza)',
            'ita_deg_imput': 'ITA (graus)', 'brightness': 'Luminância (0–255)'}
top6 = tab_desc['Biomarcador'].iloc[:6].tolist()
fig, eixos = plt.subplots(2, 3, figsize=(14, 8))
for ax, c in zip(eixos.ravel(), top6):
    lim = df_limpo[c].quantile([0.005, 0.995]).values
    dados = df_limpo[df_limpo[c].between(*lim)]
    sns.violinplot(data=dados, x='classe', y=c, hue='classe', palette=PALETA_CLASSE, order=[ROTULO[0], ROTULO[1]],
                   inner=None, cut=0, linewidth=0, ax=ax, legend=False)
    for col in ax.collections:
        col.set_alpha(0.35)
    sns.boxplot(data=dados, x='classe', y=c, hue='classe', palette=PALETA_CLASSE, order=[ROTULO[0], ROTULO[1]], width=0.18,
                fliersize=1.5, linewidth=1, ax=ax, legend=False, medianprops={'color': 'white', 'linewidth': 2},
                flierprops={'marker': 'o', 'markerfacecolor': COR_NEUTRA, 'markeredgecolor': 'none', 'alpha': 0.5})
    linha = tab_desc.set_index('Biomarcador').loc[c]
    ax.set_title(f"{c}\nδ de Cliff = {linha['δ de Cliff']:+.2f} | q = {linha['q (BH)']:.1e}", fontsize=9.5)
    ax.set_xlabel('')
    ax.set_ylabel(UNIDADES[c])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Controles\n(diferenciais)', 'Lyme\n(EM)'])
fig.suptitle('Distribuição dos biomarcadores visuais mais discriminantes: Lyme × diagnósticos diferenciais', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()
''')
interp('graf2')

md("""
#### Gráfico 3 — Mapa de calor de correlação de Spearman entre biomarcadores
Spearman foi escolhido (em vez de Pearson) porque vários biomarcadores são assimétricos e as relações esperadas são
monotônicas, não necessariamente lineares. A última linha mostra a correlação ponto-bisserial de cada biomarcador com o desfecho.
""")
code(r'''
# ---------------------------------------------------------------
# GRAFICO 3 - Heatmap de correlacao de Spearman
# ---------------------------------------------------------------
cols_corr = CHAVE + ['skin_fraction', 'width']
corr = df_limpo[cols_corr + ['y']].corr(method='spearman')
corr = corr.rename(index={'y': 'DESFECHO (Lyme=1)'}, columns={'y': 'DESFECHO (Lyme=1)'})
mascara = np.triu(np.ones_like(corr, dtype=bool), k=1)
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr, mask=mascara, cmap=CMAP_DIV, vmin=-1, vmax=1, center=0, annot=True, fmt='.2f', annot_kws={'size': 7},
            linewidths=1.5, linecolor='white', cbar_kws={'label': 'ρ de Spearman', 'shrink': 0.7}, ax=ax)
ax.set_title('Correlação de Spearman entre biomarcadores visuais e o desfecho diagnóstico', fontsize=12)
ax.grid(False)
plt.tight_layout()
plt.show()

pares_red = (corr.where(~mascara).stack().reset_index().rename(columns={0: 'rho'}))
pares_red = pares_red[(pares_red.level_0 != pares_red.level_1)].sort_values('rho', key=np.abs, ascending=False)
print('Pares mais redundantes (|ρ| mais alto):')
print(pares_red.head(6).to_string(index=False))
''')
interp('graf3')

code(r'''
# ---------------------------------------------------------------
# 3.A Tabelas de contingencia + Qui-quadrado (associacao categorica)
# ---------------------------------------------------------------
def qui_quadrado(var, titulo):
    tab = pd.crosstab(df_limpo[var], df_limpo['classe'])
    chi2, p, gl, _ = stats.chi2_contingency(tab)
    v_cramer = np.sqrt(chi2 / (tab.values.sum() * (min(tab.shape) - 1)))
    tab_pct = pd.crosstab(df_limpo[var], df_limpo['classe'], normalize='index').mul(100).round(1).add_suffix(' (% linha)')
    print(f'{titulo}: χ² = {chi2:.2f}, gl = {gl}, p = {p:.2e}, V de Cramér = {v_cramer:.3f}')
    display(pd.concat([tab, tab_pct], axis=1))

qui_quadrado('split', '1) Partição × desfecho (a estratificação foi preservada após a limpeza?)')
qui_quadrado('grupo_tom_pele', '2) Tom de pele (ITA) × desfecho')
qui_quadrado('baixa_qualidade', '3) Foto de baixa qualidade técnica × desfecho')
''')
interp('quiquadrado')

# ---------------------------- 3.B Classificacao ----------------------------
md(TEXTOS['classif_intro'])

code(r'''
# ---------------------------------------------------------------
# 3.B.1 Conjuntos de desenvolvimento (treino+validacao) e teste
# ---------------------------------------------------------------
dev = df_limpo[dev_mask].reset_index(drop=True)
teste = df_limpo[~dev_mask].reset_index(drop=True)
X_bio_dev, X_bio_te = dev[BIOMARCADORES].values, teste[BIOMARCADORES].values
X_emb_dev, X_emb_te = EMB_LIMPO[dev_mask], EMB_LIMPO[~dev_mask]
y_dev, y_te = dev['y'].values, teste['y'].values
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
print(f'Desenvolvimento: {len(dev)} fotos ({y_dev.sum()} Lyme) | Teste (nunca usado no ajuste): {len(teste)} fotos ({y_te.sum()} Lyme)')

def metricas(y, p, limiar=0.5):
    pred = (p >= limiar).astype(int)
    vn, fp, fn, vp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'Sensibilidade': vp / (vp + fn), 'Especificidade': vn / (vn + fp),
            'VPP (precisão)': vp / (vp + fp) if vp + fp else np.nan, 'VPN': vn / (vn + fn) if vn + fn else np.nan,
            'F1-Score': f1_score(y, pred, zero_division=0), 'Acurácia': (vp + vn) / len(y),
            'Acurácia balanceada': (vp / (vp + fn) + vn / (vn + fp)) / 2,
            'AUC-ROC': roc_auc_score(y, p), 'AUC-PR': average_precision_score(y, p),
            'VP': vp, 'FN': fn, 'FP': fp, 'VN': vn}

def ic_bootstrap(y, p, limiar, n=1000, seed=RANDOM_STATE):
    """IC 95% por bootstrap estratificado (reamostra Lyme e controles separadamente)."""
    rng = np.random.default_rng(seed)
    ip, ineg = np.where(y == 1)[0], np.where(y == 0)[0]
    amostras = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(ip, len(ip)), rng.choice(ineg, len(ineg))])
        amostras.append(metricas(y[idx], p[idx], limiar))
    a = pd.DataFrame(amostras)
    return a.quantile(0.025), a.quantile(0.975)
''')

code(r'''
# ---------------------------------------------------------------
# 3.B.1 Modelos candidatos avaliados por validacao cruzada (5 folds)
# ---------------------------------------------------------------
def modelo_biomarcadores():
    # Imputacao pela mediana + indicador de ausencia (ITA), escala robusta e gradient boosting balanceado
    return make_pipeline(SimpleImputer(strategy='median', add_indicator=True), RobustScaler(),
                         HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
                                                        l2_regularization=1.0, class_weight='balanced',
                                                        random_state=RANDOM_STATE))

def modelo_embeddings(C=0.003, class_weight='balanced'):
    # Transfer learning: sonda linear (regressao logistica regularizada) sobre os embeddings da CNN
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight=class_weight))

# Ajuste da regularizacao C da sonda linear por validacao cruzada (criterio: AUC-PR)
busca_C = {C: average_precision_score(y_dev, cross_val_predict(modelo_embeddings(C), X_emb_dev, y_dev, cv=CV,
                                                               method='predict_proba')[:, 1])
           for C in [0.0005, 0.001, 0.003, 0.01, 0.03]}
C_OTIMO = max(busca_C, key=busca_C.get)
print('AUC-PR em validação cruzada por valor de C:', {k: round(v, 3) for k, v in busca_C.items()}, '-> C escolhido =', C_OTIMO)

oof = {
    'Linha de base (classe majoritária)': cross_val_predict(DummyClassifier(strategy='prior'), X_bio_dev, y_dev, cv=CV, method='predict_proba')[:, 1],
    'A) Biomarcadores + GB': cross_val_predict(modelo_biomarcadores(), X_bio_dev, y_dev, cv=CV, method='predict_proba')[:, 1],
    'B) Embeddings CNN + RL': cross_val_predict(modelo_embeddings(C_OTIMO), X_emb_dev, y_dev, cv=CV, method='predict_proba')[:, 1],
}
oof['C) Fusão A + B'] = (oof['A) Biomarcadores + GB'] + oof['B) Embeddings CNN + RL']) / 2

tab_cv = pd.DataFrame({nome: metricas(y_dev, p) for nome, p in oof.items()}).T
# Criterio de selecao do modelo final definido a priori: maior AUC-PR na validacao cruzada
NOME_FINAL = tab_cv.drop(index='Linha de base (classe majoritária)')['AUC-PR'].idxmax()
print(f'Modelo selecionado para o CDSS (maior AUC-PR em validação cruzada): {NOME_FINAL}')
display(tab_cv[['AUC-ROC', 'AUC-PR', 'Sensibilidade', 'Especificidade', 'VPP (precisão)', 'F1-Score', 'Acurácia', 'Acurácia balanceada']]
        .style.format('{:.3f}').highlight_max(subset=['AUC-ROC', 'AUC-PR'], color='#cde2fb'))
''')
interp('cv')

code(r'''
# ---------------------------------------------------------------
# 3.B.1 Estrategias de mitigacao do desbalanceamento (CV, limiar 0,5)
# ---------------------------------------------------------------
def undersampling_cv(X, y):
    """Subamostragem aleatoria dos controles dentro de cada fold de treino (sem tocar no fold de avaliacao)."""
    p = np.zeros(len(y))
    rng = np.random.default_rng(RANDOM_STATE)
    for tr, va in CV.split(X, y):
        pos, neg = tr[y[tr] == 1], tr[y[tr] == 0]
        sel = np.concatenate([pos, rng.choice(neg, len(pos), replace=False)])
        p[va] = modelo_embeddings(C_OTIMO, class_weight=None).fit(X[sel], y[sel]).predict_proba(X[va])[:, 1]
    return p

estrategias = {
    'Sem tratamento': cross_val_predict(modelo_embeddings(C_OTIMO, class_weight=None), X_emb_dev, y_dev, cv=CV, method='predict_proba')[:, 1],
    'Ponderação de classes (class_weight)': oof['B) Embeddings CNN + RL'],
    'Undersampling dos controles': undersampling_cv(X_emb_dev, y_dev),
}
tab_desb = pd.DataFrame({k: metricas(y_dev, p) for k, p in estrategias.items()}).T
display(tab_desb[['Sensibilidade', 'Especificidade', 'VPP (precisão)', 'F1-Score', 'Acurácia', 'AUC-ROC', 'AUC-PR']].style.format('{:.3f}'))
''')
interp('desbalanceamento')

code(r'''
# ---------------------------------------------------------------
# 3.B.1 Modelo final: ajuste no desenvolvimento e avaliacao UNICA no teste
# ---------------------------------------------------------------
mod_A = modelo_biomarcadores().fit(X_bio_dev, y_dev)
mod_B = modelo_embeddings(C_OTIMO).fit(X_emb_dev, y_dev)
p_teste = {'A) Biomarcadores + GB': mod_A.predict_proba(X_bio_te)[:, 1],
           'B) Embeddings CNN + RL': mod_B.predict_proba(X_emb_te)[:, 1]}
p_teste['C) Fusão A + B'] = (p_teste['A) Biomarcadores + GB'] + p_teste['B) Embeddings CNN + RL']) / 2
p_teste['Linha de base (classe majoritária)'] = np.full(len(y_te), y_dev.mean())

P_OOF_FINAL, P_TE_FINAL = oof[NOME_FINAL], p_teste[NOME_FINAL]

tab_teste = pd.DataFrame({k: metricas(y_te, p, 0.5) for k, p in p_teste.items()}).T
display(tab_teste[['AUC-ROC', 'AUC-PR', 'Sensibilidade', 'Especificidade', 'VPP (precisão)', 'VPN', 'F1-Score', 'Acurácia', 'VP', 'FN', 'FP', 'VN']]
        .style.format('{:.3f}', subset=['AUC-ROC', 'AUC-PR', 'Sensibilidade', 'Especificidade', 'VPP (precisão)', 'VPN', 'F1-Score', 'Acurácia'])
        .format('{:.0f}', subset=['VP', 'FN', 'FP', 'VN']))

inf, sup = ic_bootstrap(y_te, P_TE_FINAL, 0.5)
pontual = pd.Series(metricas(y_te, P_TE_FINAL, 0.5))
ic = pd.DataFrame({'Estimativa': pontual, 'IC95% inf.': inf, 'IC95% sup.': sup}).loc[
    ['Sensibilidade', 'Especificidade', 'VPP (precisão)', 'VPN', 'F1-Score', 'AUC-ROC', 'AUC-PR']]
print(f'Modelo final [{NOME_FINAL}] no teste, limiar padrão 0,5, com IC 95% (bootstrap, 1.000 reamostragens):')
display(ic.style.format('{:.3f}'))
''')
interp('teste')

code(r'''
# ---------------------------------------------------------------
# 3.B.1 Prova do vazamento: mesmo modelo B treinado/avaliado na base ORIGINAL (com duplicatas)
# ---------------------------------------------------------------
orig_tr = df['split'].isin(['train', 'val']).values
orig_te = (df['split'] == 'test').values
p_orig = modelo_embeddings(C_OTIMO).fit(EMB[orig_tr], df.y.values[orig_tr]).predict_proba(EMB[orig_te])[:, 1]
copias_teste = df[orig_te & df.grupo_dup.isin(df.loc[orig_tr, 'grupo_dup']).values]
print(f'Fotos do teste original que possuem cópia idêntica no treino/validação: {len(copias_teste)} de {orig_te.sum()}')
display(pd.DataFrame({
    'Base original (com duplicatas e rótulos conflitantes)': metricas(df.y.values[orig_te], p_orig),
    'Base limpa (deduplicada)': metricas(y_te, p_teste['B) Embeddings CNN + RL']),
}).T[['AUC-ROC', 'AUC-PR', 'Sensibilidade', 'Especificidade']].style.format('{:.3f}'))
''')
interp('vazamento')

md("""
#### Gráfico 5 — Painel de avaliação de desempenho e decisão
Matriz de confusão (no limiar de triagem recomendado na Seção 4), curvas ROC e Precision-Recall dos três modelos no
conjunto de teste e importância dos biomarcadores (valores SHAP do modelo A, interpretável).
""")
code(r'''
# ---------------------------------------------------------------
# Limiares operacionais (definidos SOMENTE com as probabilidades fora-do-fold do desenvolvimento)
# ---------------------------------------------------------------
fpr_o, tpr_o, thr_o = roc_curve(y_dev, P_OOF_FINAL)
LIMIAR_TRIAGEM = thr_o[np.argmax(tpr_o >= 0.90)]               # maior limiar com sensibilidade >= 90%
LIMIAR_CONFIRMACAO = thr_o[np.where(1 - fpr_o >= 0.95)[0][-1]]  # menor limiar com especificidade >= 95%
LIMIAR_YOUDEN = thr_o[np.argmax(tpr_o - fpr_o)]
print(f'Limiar de TRIAGEM (sens. ≥ 90% no desenvolvimento): {LIMIAR_TRIAGEM:.3f}')
print(f'Limiar de CONFIRMAÇÃO (espec. ≥ 95% no desenvolvimento): {LIMIAR_CONFIRMACAO:.3f}')
print(f'Limiar de Youden (equilíbrio): {LIMIAR_YOUDEN:.3f}')

# SHAP do modelo A (biomarcadores interpretaveis)
pre_A = make_pipeline(*[passo for _, passo in mod_A.steps[:-1]])
X_te_A = pre_A.transform(X_bio_te)
nomes_A = [n.replace('missingindicator_', 'ausente_') for n in mod_A[:-1].get_feature_names_out(BIOMARCADORES)]
explicador = shap.TreeExplainer(mod_A[-1])
valores_shap = explicador.shap_values(X_te_A)
valores_shap = valores_shap[1] if isinstance(valores_shap, list) else valores_shap
importancia = pd.Series(np.abs(valores_shap).mean(0), index=nomes_A).sort_values()
''')
code(r'''
# ---------------------------------------------------------------
# GRAFICO 5 - Matriz de confusao + ROC + PR + importancia (SHAP)
# ---------------------------------------------------------------
fig, eixos = plt.subplots(2, 2, figsize=(14, 11))
ax_cm, ax_roc, ax_pr, ax_imp = eixos.ravel()

# (a) Matriz de confusao no limiar de triagem
m_tri = metricas(y_te, P_TE_FINAL, LIMIAR_TRIAGEM)
cm = np.array([[m_tri['VN'], m_tri['FP']], [m_tri['FN'], m_tri['VP']]])
cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
sns.heatmap(cm_pct, annot=False, cmap=LinearSegmentedColormap.from_list('azul', ['#f0efec', '#86b6ef', '#1c5cab']), vmin=0, vmax=100,
            cbar_kws={'label': '% da classe real (linha)'}, linewidths=2, linecolor='white', ax=ax_cm)
rotulos_cm = [['VN', 'FP'], ['FN', 'VP']]
for i in range(2):
    for j in range(2):
        ax_cm.text(j + 0.5, i + 0.5, f'{rotulos_cm[i][j]}\n{cm[i, j]}\n({cm_pct[i, j]:.1f}%)', ha='center', va='center',
                   fontsize=11, fontweight='bold', color='white' if cm_pct[i, j] > 55 else '#0b0b0b')
ax_cm.set_xticklabels(['Previsto: controle', 'Previsto: Lyme'])
ax_cm.set_yticklabels(['Real: controle', 'Real: Lyme'], rotation=0)
ax_cm.set_title(f'A. Matriz de confusão do modelo final {NOME_FINAL[:2]} — teste, limiar de triagem = {LIMIAR_TRIAGEM:.2f}\n'
                f"Sens. {m_tri['Sensibilidade']:.1%} | Espec. {m_tri['Especificidade']:.1%} | VPN {m_tri['VPN']:.1%}")

# (b) Curvas ROC
cores_mod = {'A) Biomarcadores + GB': COR_3, 'B) Embeddings CNN + RL': COR_NEG, 'C) Fusão A + B': COR_POS}
for nome, cor in cores_mod.items():
    f, t, _ = roc_curve(y_te, p_teste[nome])
    ax_roc.plot(f, t, color=cor, lw=2, label=f'{nome} (AUC = {roc_auc_score(y_te, p_teste[nome]):.3f})')
ax_roc.plot([0, 1], [0, 1], ls='--', color=COR_NEUTRA, lw=1, label='Acaso (AUC = 0,5)')
for lim, nome_l, mk in [(LIMIAR_TRIAGEM, 'Triagem', 'o'), (LIMIAR_CONFIRMACAO, 'Confirmação', 's')]:
    mm = metricas(y_te, P_TE_FINAL, lim)
    ax_roc.scatter(1 - mm['Especificidade'], mm['Sensibilidade'], s=70, marker=mk, color='#0b0b0b', zorder=5,
                   edgecolor='white', linewidth=1.5)
    ax_roc.annotate(f'{nome_l} (limiar {lim:.2f})', (1 - mm['Especificidade'], mm['Sensibilidade']), xytext=(8, -14),
                    textcoords='offset points', fontsize=8.5)
ax_roc.set_xlabel('1 − Especificidade (taxa de falsos positivos)')
ax_roc.set_ylabel('Sensibilidade (taxa de verdadeiros positivos)')
ax_roc.set_title('B. Curva ROC — conjunto de teste')
ax_roc.legend(loc='lower right', fontsize=8)

# (c) Curvas Precision-Recall
for nome, cor in cores_mod.items():
    pr, rc, _ = precision_recall_curve(y_te, p_teste[nome])
    ax_pr.plot(rc, pr, color=cor, lw=2, label=f'{nome} (AUC-PR = {average_precision_score(y_te, p_teste[nome]):.3f})')
ax_pr.axhline(y_te.mean(), ls='--', color=COR_NEUTRA, lw=1, label=f'Prevalência no teste = {y_te.mean():.1%}')
ax_pr.set_xlabel('Sensibilidade (recall)')
ax_pr.set_ylabel('Valor preditivo positivo (precisão)')
ax_pr.set_ylim(0, 1.02)
ax_pr.set_title('C. Curva Precision-Recall — conjunto de teste')
ax_pr.legend(loc='upper right', fontsize=8)

# (d) Importancia SHAP
top_imp = importancia.tail(12)
ax_imp.barh(top_imp.index, top_imp.values, color=COR_3, height=0.65)
for i, v in enumerate(top_imp.values):
    ax_imp.text(v, i, f' {v:.3f}', va='center', fontsize=8, color=COR_TEXTO)
ax_imp.set_xlabel('Importância média |SHAP| (impacto no log-odds de Lyme)')
ax_imp.set_title('D. Biomarcadores que mais pesam na decisão (modelo A, SHAP)')
ax_imp.grid(axis='y', visible=False)
fig.suptitle('Painel de desempenho diagnóstico no conjunto de teste independente', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
''')
interp('graf5')

code(r'''
# ---------------------------------------------------------------
# Direcao do efeito SHAP: valores altos do biomarcador aumentam ou reduzem a suspeita de Lyme?
# ---------------------------------------------------------------
rho_shap = pd.Series([stats.spearmanr(X_te_A[:, i], valores_shap[:, i]).statistic
                      if np.std(X_te_A[:, i]) > 0 and np.std(valores_shap[:, i]) > 0 else np.nan
                      for i in range(len(nomes_A))], index=nomes_A)
direcao = pd.DataFrame({'Importância |SHAP|': importancia, 'ρ(valor, SHAP)': rho_shap}).sort_values('Importância |SHAP|', ascending=False).head(8)
direcao['Leitura'] = np.where(direcao['ρ(valor, SHAP)'] > 0, 'valor ALTO → ↑ suspeita de Lyme', 'valor ALTO → ↓ suspeita de Lyme')
display(direcao.round(3))
''')

# ---------------------------- 3.B Agrupamento ----------------------------
md(TEXTOS['cluster_intro'])
code(r'''
# ---------------------------------------------------------------
# 3.B.2 Agrupamento das lesoes de Lyme: PCA + K-Means
# ---------------------------------------------------------------
pos_mask = df_limpo['y'].values == 1
df_pos = df_limpo[pos_mask].reset_index(drop=True)
Z = StandardScaler().fit_transform(EMB_LIMPO[pos_mask])
pca = PCA(n_components=50, random_state=RANDOM_STATE).fit(Z)
Z_pca = pca.transform(Z)
var_acum = np.cumsum(pca.explained_variance_ratio_)
print(f'Lesões de Lyme analisadas: {len(df_pos)}')
print(f'Variância explicada: PC1 = {pca.explained_variance_ratio_[0]:.1%} | PC1+PC2 = {var_acum[1]:.1%} | '
      f'10 PCs = {var_acum[9]:.1%} | 50 PCs = {var_acum[49]:.1%}')

resultados_k = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit(Z_pca)
    resultados_k.append({'k': k, 'Inércia': km.inertia_, 'Silhueta': silhouette_score(Z_pca, km.labels_),
                         'Davies-Bouldin': davies_bouldin_score(Z_pca, km.labels_)})
tab_k = pd.DataFrame(resultados_k).set_index('k')
display(tab_k.style.format({'Inércia': '{:,.0f}', 'Silhueta': '{:.3f}', 'Davies-Bouldin': '{:.3f}'})
        .highlight_max(subset=['Silhueta'], color='#cde2fb').highlight_min(subset=['Davies-Bouldin'], color='#cde2fb'))
K_OTIMO = int(tab_k['Silhueta'].idxmax())
km_final = KMeans(n_clusters=K_OTIMO, n_init=20, random_state=RANDOM_STATE).fit(Z_pca)
df_pos['cluster'] = km_final.labels_
print(f'k escolhido pela maior silhueta: {K_OTIMO}')
''')

md("""
#### Gráfico 4 — Padrões latentes: projeção 2D das lesões de Lyme (PCA e t-SNE) colorida pelos fenótipos encontrados
""")
code(r'''
# ---------------------------------------------------------------
# GRAFICO 4 - Variancia explicada + projecoes 2D (PCA e t-SNE) por cluster
# ---------------------------------------------------------------
tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=RANDOM_STATE).fit_transform(Z_pca)
CORES_CLUSTER = [COR_NEG, COR_POS, COR_3, COR_4, '#eda100', '#e87ba4', '#008300', '#e34948'][:K_OTIMO]
fig = plt.figure(figsize=(15, 5.6))
gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 1, 1])
ax0, ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])

ax0.bar(range(1, 21), pca.explained_variance_ratio_[:20] * 100, color='#86b6ef', label='Variância individual')
ax0.plot(range(1, 21), var_acum[:20] * 100, color=COR_NEG, marker='o', ms=4, lw=2, label='Variância acumulada')
ax0.set_xlabel('Componente principal')
ax0.set_ylabel('Variância explicada (%)')
ax0.set_title('A. Variância explicada (PCA dos embeddings)')
ax0.legend(fontsize=8)

for c in range(K_OTIMO):
    sel = df_pos['cluster'] == c
    rot = f'Cluster {c} (n={sel.sum()})'
    ax1.scatter(Z_pca[sel, 0], Z_pca[sel, 1], s=14, color=CORES_CLUSTER[c], alpha=0.75, edgecolor='white', linewidth=0.3, label=rot)
    ax2.scatter(tsne[sel, 0], tsne[sel, 1], s=14, color=CORES_CLUSTER[c], alpha=0.75, edgecolor='white', linewidth=0.3, label=rot)
ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} da variância)')
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} da variância)')
ax1.set_title('B. Projeção PCA (2 primeiros componentes)')
ax2.set_xlabel('t-SNE dimensão 1 (adimensional)')
ax2.set_ylabel('t-SNE dimensão 2 (adimensional)')
ax2.set_title(f'C. Projeção t-SNE — silhueta (k={K_OTIMO}) = {tab_k.loc[K_OTIMO, "Silhueta"]:.3f}')
for ax in (ax1, ax2):
    ax.legend(fontsize=8, markerscale=1.5)
fig.suptitle('Fenótipos visuais latentes entre as lesões de eritema migratório (K-Means sobre embeddings da CNN)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()
''')
interp('graf4')

code(r'''
# ---------------------------------------------------------------
# 3.B.2 Perfil estatistico medio de cada cluster (fenotipo)
# ---------------------------------------------------------------
perfil_cols = ['lab_a_mean', 'a_p90_minus_p50', 'radial_a_center_minus_mid', 'radial_a_range', 'glcm_contrast',
               'edge_density', 'gray_entropy', 'skin_fraction', 'ita_deg_imput', 'brightness']
perfil = df_pos.groupby('cluster')[perfil_cols].median().T
perfil.columns = [f'Cluster {c} (mediana)' for c in perfil.columns]
kw = {c: stats.kruskal(*[g[c].values for _, g in df_pos.groupby('cluster')]).pvalue for c in perfil_cols}
perfil['p (Kruskal-Wallis)'] = pd.Series(kw)
display(perfil.style.format({**{c: '{:.3f}' for c in perfil.columns}, 'p (Kruskal-Wallis)': '{:.1e}'}))

print('Composição de cada cluster pelos rótulos originais de EM (% coluna):')
display((pd.crosstab(df_pos['condition'], df_pos['cluster'], normalize='columns') * 100).round(1))
print('Distribuição de tom de pele (ITA) por cluster (% coluna):')
display((pd.crosstab(df_pos['grupo_tom_pele'], df_pos['cluster'], normalize='columns') * 100).round(1))
ari = adjusted_rand_score(df_pos['condition'], df_pos['cluster'])
print(f'Índice de Rand ajustado entre clusters e rótulos de EM do dataset: {ari:.3f} (0 = concordância ao acaso)')

# Sensibilidade do classificador (OOF + teste) por cluster: algum fenotipo e mais dificil de detectar?
prob_pos = pd.concat([pd.Series(P_OOF_FINAL[y_dev == 1], index=dev.loc[y_dev == 1, 'filename']),
                      pd.Series(P_TE_FINAL[y_te == 1], index=teste.loc[y_te == 1, 'filename'])])
df_pos['prob_lyme'] = df_pos['filename'].map(prob_pos)
sens_cluster = df_pos.groupby('cluster').apply(lambda g: pd.Series({'n': len(g), 'Sensibilidade (limiar triagem)': (g.prob_lyme >= LIMIAR_TRIAGEM).mean(),
                                                                    'Prob. mediana de Lyme': g.prob_lyme.median()}), include_groups=False)
display(sens_cluster.style.format({'n': '{:.0f}', 'Sensibilidade (limiar triagem)': '{:.1%}', 'Prob. mediana de Lyme': '{:.3f}'}))
''')

code(r'''
# ---------------------------------------------------------------
# 3.B.2 Exemplos representativos (mais proximos do centroide) de cada fenotipo
# ---------------------------------------------------------------
n_ex = 6
fig, eixos = plt.subplots(K_OTIMO, n_ex, figsize=(13, 2.3 * K_OTIMO))
eixos = np.atleast_2d(eixos)
for c in range(K_OTIMO):
    dist = np.linalg.norm(Z_pca - km_final.cluster_centers_[c], axis=1)
    dist[df_pos['cluster'].values != c] = np.inf
    for j, i in enumerate(np.argsort(dist)[:n_ex]):
        ax = eixos[c, j]
        ax.imshow(Image.open(caminho_imagem(df_pos.iloc[i])).convert('RGB').resize((180, 180)))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True); s.set_color(CORES_CLUSTER[c]); s.set_linewidth(3)
        if j == 0:
            ax.set_ylabel(f'Cluster {c}', fontsize=11, fontweight='bold', color=CORES_CLUSTER[c])
fig.suptitle('Lesões de Lyme mais típicas de cada fenótipo (mais próximas do centroide)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()
''')
interp('perfil_cluster')

md(TEXTOS['pergunta3'])

# =====================================================================================
# SECAO 04 - CDSS
# =====================================================================================
md(TEXTOS['cdss_intro'])

code(r'''
# ---------------------------------------------------------------
# 4.2 Pontos de operacao do CDSS avaliados no conjunto de teste
# ---------------------------------------------------------------
pontos = {'Padrão (0,50)': 0.5, f'Triagem (sens. ≥ 90%) = {LIMIAR_TRIAGEM:.2f}': LIMIAR_TRIAGEM,
          f'Youden = {LIMIAR_YOUDEN:.2f}': LIMIAR_YOUDEN, f'Confirmação (espec. ≥ 95%) = {LIMIAR_CONFIRMACAO:.2f}': LIMIAR_CONFIRMACAO}
linhas = []
for nome, lim in pontos.items():
    m = metricas(y_te, P_TE_FINAL, lim)
    inf, sup = ic_bootstrap(y_te, P_TE_FINAL, lim, n=500)
    linhas.append({'Ponto de operação': nome,
                   'Sensibilidade [IC95%]': f"{m['Sensibilidade']:.1%} [{inf['Sensibilidade']:.1%}–{sup['Sensibilidade']:.1%}]",
                   'Especificidade [IC95%]': f"{m['Especificidade']:.1%} [{inf['Especificidade']:.1%}–{sup['Especificidade']:.1%}]",
                   'VPP': f"{m['VPP (precisão)']:.1%}", 'VPN': f"{m['VPN']:.1%}", 'F1': f"{m['F1-Score']:.3f}",
                   'FN (Lyme perdidos)': m['FN'], 'FP (alarmes falsos)': m['FP']})
display(pd.DataFrame(linhas).style.hide(axis='index'))
''')

code(r'''
# ---------------------------------------------------------------
# 4.2 Calibracao das probabilidades (Platt) e efeito da prevalencia real no VPP
# ---------------------------------------------------------------
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
platt = LogisticRegression().fit(logit(P_OOF_FINAL).reshape(-1, 1), y_dev)      # ajustado so no desenvolvimento
P_TE_CAL = platt.predict_proba(logit(P_TE_FINAL).reshape(-1, 1))[:, 1]
print(f'Brier score no teste — antes da calibração: {brier_score_loss(y_te, P_TE_FINAL):.4f} | '
      f'após calibração de Platt: {brier_score_loss(y_te, P_TE_CAL):.4f} | modelo ingênuo (prevalência): '
      f'{brier_score_loss(y_te, np.full(len(y_te), y_dev.mean())):.4f}')
LIMIAR_TRIAGEM_CAL = platt.predict_proba(logit(np.array([[LIMIAR_TRIAGEM]])))[0, 1]
LIMIAR_CONFIRMACAO_CAL = platt.predict_proba(logit(np.array([[LIMIAR_CONFIRMACAO]])))[0, 1]
print(f'Limiares expressos como probabilidade calibrada: triagem = {LIMIAR_TRIAGEM_CAL:.1%} | confirmação = {LIMIAR_CONFIRMACAO_CAL:.1%}')

# Teorema de Bayes: VPP/VPN do ponto de triagem em diferentes cenarios de prevalencia
m_tri = metricas(y_te, P_TE_FINAL, LIMIAR_TRIAGEM)
sens, espec = m_tri['Sensibilidade'], m_tri['Especificidade']
cenarios = {'Este dataset (curado)': y_te.mean(), 'Ambulatório em área endêmica (EUA/Europa), lesão anelar': 0.30,
            'Atenção primária, qualquer lesão eritematosa': 0.05, 'Brasil (Baggio-Yoshinari), lesão eritematosa': 0.005}
tab_prev = pd.DataFrame([{'Cenário': k, 'Prevalência': f'{p:.1%}',
                          'VPP (triagem)': f'{sens * p / (sens * p + (1 - espec) * (1 - p)):.1%}',
                          'VPN (triagem)': f'{espec * (1 - p) / (espec * (1 - p) + (1 - sens) * p):.2%}'} for k, p in cenarios.items()])
display(tab_prev.style.hide(axis='index'))
''')

md("""
#### Gráfico 6 — Zonas de decisão do CDSS e confiabilidade das probabilidades
""")
code(r'''
# ---------------------------------------------------------------
# GRAFICO 6 - Distribuicao das probabilidades por classe com zonas de risco + curva de calibracao
# ---------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8), gridspec_kw={'width_ratios': [1.5, 1]})
bins = np.linspace(0, 1, 41)
ax1.axvspan(0, LIMIAR_TRIAGEM, color='#cde2fb', alpha=0.5, lw=0)
ax1.axvspan(LIMIAR_TRIAGEM, LIMIAR_CONFIRMACAO, color='#fde9b6', alpha=0.6, lw=0)
ax1.axvspan(LIMIAR_CONFIRMACAO, 1, color='#f7c9c4', alpha=0.6, lw=0)
for classe, cor in [(0, COR_NEG), (1, COR_POS)]:
    rot = f'{ROTULO[classe]} (n={(y_te == classe).sum()})'
    ax1.hist(P_TE_FINAL[y_te == classe], bins=bins, color=cor, alpha=0.30, density=True, histtype='stepfilled')
    ax1.hist(P_TE_FINAL[y_te == classe], bins=bins, color=cor, lw=2, density=True, histtype='step', label=rot)
topo = ax1.get_ylim()[1]
for x0, x1, txt in [(0, LIMIAR_TRIAGEM, 'BAIXO RISCO\nconduzir diferencial'), (LIMIAR_TRIAGEM, LIMIAR_CONFIRMACAO, 'INDETERMINADO\navaliação clínica dirigida'),
                    (LIMIAR_CONFIRMACAO, 1, 'ALTO RISCO\nEM provável')]:
    ax1.text((x0 + x1) / 2, topo * 0.93, txt, ha='center', va='top', fontsize=8.5, fontweight='bold', color='#0b0b0b')
ax1.set_xlabel(f'Escore do modelo final {NOME_FINAL} (probabilidade não calibrada de Lyme)')
ax1.set_ylabel('Densidade de fotografias')
ax1.set_title('A. Separação das classes no teste e zonas de decisão do CDSS')
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=8.5)

for p, nome, cor, mk in [(P_TE_FINAL, 'Modelo final (não calibrado)', COR_NEUTRA, 'o'), (P_TE_CAL, 'Modelo final + Platt (calibrado)', COR_POS, 's')]:
    frac, media = calibration_curve(y_te, p, n_bins=8, strategy='quantile')
    ax2.plot(media, frac, marker=mk, color=cor, lw=2, ms=6, label=f'{nome} — Brier {brier_score_loss(y_te, p):.3f}')
ax2.plot([0, 1], [0, 1], ls='--', color='#c3c2b7', lw=1, label='Calibração perfeita')
ax2.set_xlabel('Probabilidade prevista de Lyme')
ax2.set_ylabel('Fração observada de Lyme (por faixa)')
ax2.set_title('B. Curva de confiabilidade (calibração)')
ax2.legend(fontsize=8, loc='upper left')
ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
plt.tight_layout()
plt.show()
''')
interp('graf6')

code(r'''
# ---------------------------------------------------------------
# 4.3 Explicabilidade visual (CAM) do modelo B: onde a CNN "olha"?
# ---------------------------------------------------------------
# Para uma sonda linear sobre a media global dos mapas de ativacao, o mapa de ativacao de classe (CAM, Zhou et al., 2016)
# e exato: CAM(x,y) = soma_k (w_k / sigma_k) * A_k(x,y), onde w sao os pesos da regressao e sigma o desvio do StandardScaler.
rede_cam, transf_cam = carregar_backbone()
pesos_cam = torch.tensor(mod_B[-1].coef_[0] / mod_B[0].scale_, dtype=torch.float32)

def mapa_cam(linha):
    img = Image.open(caminho_imagem(linha)).convert('RGB')
    with torch.no_grad():
        A = rede_cam.features(transf_cam(img).unsqueeze(0).to(DEVICE))[0].cpu()
    cam = torch.einsum('k,khw->hw', pesos_cam, A).numpy()
    cam = np.asarray(Image.fromarray(cam.astype(np.float32)).resize((224, 224), Image.BILINEAR))
    return img.resize((224, 224)), (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)

teste_aval = teste.assign(prob=P_TE_FINAL)
casos = pd.concat([
    teste_aval[teste_aval.y == 1].nlargest(3, 'prob').assign(tipo='Verdadeiro positivo'),
    teste_aval[(teste_aval.y == 1)].nsmallest(2, 'prob').assign(tipo='Falso negativo'),
    teste_aval[teste_aval.y == 0].nlargest(3, 'prob').assign(tipo='Falso positivo'),
])
fig, eixos = plt.subplots(2, len(casos), figsize=(2.1 * len(casos), 4.8))
for j, (_, linha) in enumerate(casos.iterrows()):
    img, cam = mapa_cam(linha)
    eixos[0, j].imshow(img)
    eixos[1, j].imshow(img)
    eixos[1, j].imshow(cam, cmap='inferno', alpha=0.5)
    cor = {'Verdadeiro positivo': COR_3, 'Falso negativo': COR_POS, 'Falso positivo': COR_NEG}[linha['tipo']]
    eixos[0, j].set_title(f"{linha['tipo']}\n{linha['condition']}\nP = {linha['prob']:.2f}", fontsize=7.5, color=cor)
    for ax in eixos[:, j]:
        ax.axis('off')
fig.suptitle('Mapas de ativação de classe (CAM): regiões que aumentam a suspeita de Lyme (amarelo = maior peso)', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.show()
''')
interp('cam')

md(TEXTOS['cdss_texto'])

# =====================================================================================
# SECAO 05 - Analise critica e etica
# =====================================================================================
md(TEXTOS['etica_intro'])

code(r'''
# ---------------------------------------------------------------
# 5.1 Validacao visual do proxy ITA: o grupo "ITA <= 10" e realmente pele escura?
# ---------------------------------------------------------------
amostra_ita = df_limpo[df_limpo['ita_deg'] <= 10].sample(16, random_state=RANDOM_STATE)
fig, eixos = plt.subplots(2, 8, figsize=(15, 4.3))
for k, (ax, (_, linha)) in enumerate(zip(eixos.ravel(), amostra_ita.iterrows())):
    ax.imshow(Image.open(caminho_imagem(linha)).convert('RGB').resize((160, 160)))
    ax.set_title(f"#{k + 1} | ITA {linha['ita_deg']:.0f}°", fontsize=8)
    ax.axis('off')
fig.suptitle('Amostra aleatória de 16 fotos com ITA ≤ 10° (classificadas automaticamente como pele "escura")', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.show()
''')
interp('ita_validacao')

code(r'''
# ---------------------------------------------------------------
# 5.1 Equidade: desempenho do CDSS por tom de pele (ITA)
# ---------------------------------------------------------------
def ic_wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    centro = (p + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    meia = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
    return centro - meia, centro + meia

aval = pd.concat([dev.assign(prob=P_OOF_FINAL, origem='Desenvolvimento (fora-do-fold)'),
                  teste.assign(prob=P_TE_FINAL, origem='Teste')], ignore_index=True)
aval['grupo_tom_pele'] = aval['grupo_tom_pele'].cat.add_categories('ITA ausente').fillna('ITA ausente')
linhas = []
for g, sub in aval.groupby('grupo_tom_pele', observed=True):
    pos, neg = sub[sub.y == 1], sub[sub.y == 0]
    vp, vn = (pos.prob >= LIMIAR_TRIAGEM).sum(), (neg.prob < LIMIAR_TRIAGEM).sum()
    li, ls = ic_wilson(vp, len(pos))
    linhas.append({'Tom de pele (ITA)': g, 'Fotos': len(sub), '% da base': len(sub) / len(aval) * 100, 'Lyme (n)': len(pos),
                   'Sensibilidade': vp / len(pos) if len(pos) else np.nan, 'IC95% inf.': li, 'IC95% sup.': ls,
                   'Especificidade': vn / len(neg), 'AUC-ROC': roc_auc_score(sub.y, sub.prob) if sub.y.nunique() == 2 else np.nan})
tab_eq = pd.DataFrame(linhas)
display(tab_eq.style.format({'% da base': '{:.1f}%', 'Sensibilidade': '{:.1%}', 'IC95% inf.': '{:.1%}', 'IC95% sup.': '{:.1%}',
                             'Especificidade': '{:.1%}', 'AUC-ROC': '{:.3f}'}).hide(axis='index'))
''')

md("""
#### Gráfico 7 — Auditoria de equidade: representatividade e sensibilidade por tom de pele
""")
code(r'''
# ---------------------------------------------------------------
# GRAFICO 7 - Representatividade e sensibilidade por tom de pele
# ---------------------------------------------------------------
eq = tab_eq[tab_eq['Tom de pele (ITA)'] != 'ITA ausente'].reset_index(drop=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
cores_eq = ['#184f95', '#3987e5', '#9ec5f4']
ax1.barh(eq['Tom de pele (ITA)'], eq['% da base'], color=cores_eq, height=0.6)
ax1.invert_yaxis()
for i, (v, n) in enumerate(zip(eq['% da base'], eq['Fotos'])):
    ax1.text(v + 0.5, i, f'{v:.1f}%  (n={n})', va='center', fontsize=9, color=COR_TEXTO)
ax1.set_xlabel('Participação na base analítica (%)')
ax1.set_title('A. Representatividade por tom de pele aparente (ITA)')
ax1.set_xlim(0, eq['% da base'].max() * 1.3)
ax1.grid(axis='y', visible=False)

x = np.arange(len(eq))
ax2.bar(x, eq['Sensibilidade'] * 100, color=cores_eq, width=0.55)
ax2.errorbar(x, eq['Sensibilidade'] * 100, yerr=[(eq['Sensibilidade'] - eq['IC95% inf.']) * 100, (eq['IC95% sup.'] - eq['Sensibilidade']) * 100],
             fmt='none', ecolor='#0b0b0b', capsize=5, lw=1.2)
for i, (s, sup, n) in enumerate(zip(eq['Sensibilidade'], eq['IC95% sup.'], eq['Lyme (n)'])):
    ax2.text(i, sup * 100 + 2, f'{s:.1%}  (n Lyme = {n})', ha='center', fontsize=9, color='#0b0b0b', fontweight='bold')
ax2.axhline(90, ls='--', color=COR_NEUTRA, lw=1, label='Meta de sensibilidade da triagem (90%)')
ax2.set_xticks(x)
ax2.set_xticklabels(eq['Tom de pele (ITA)'], fontsize=8.5)
ax2.set_ylabel('Sensibilidade no limiar de triagem (%)')
ax2.set_ylim(0, 118)
ax2.set_title('B. Sensibilidade por tom de pele aparente (IC 95% de Wilson)')
ax2.legend(fontsize=8, loc='upper left')
ax2.grid(axis='x', visible=False)
fig.suptitle('Auditoria de viés algorítmico por tom de pele (proxy ITA, desenvolvimento fora-do-fold + teste)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()
''')
interp('graf7')

md(TEXTOS['etica_texto'])
md(TEXTOS['conclusao'])
md(TEXTOS['referencias'])

nb = nbf.v4.new_notebook()
nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python'},
    'colab': {'provenance': [], 'toc_visible': True},
}
os.makedirs(os.path.dirname(NB_PATH), exist_ok=True)
nbf.write(nb, NB_PATH)
print(f'Notebook gerado com {len(cells)} celulas em {os.path.normpath(NB_PATH)}')
