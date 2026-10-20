import os
import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage,
    KeepTogether, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        
        # We do not draw headers/footers on page 1 (cover / front page)
        if self._pageNumber > 1:
            # Header
            self.drawString(50, 800, "FESA - Faculdade Engenheiro Salvador Arena • Engenharia de Computação • Análise de Dados")
            self.drawRightString(545, 800, "Entrega 01: Doença de Lyme (Eritema Migratório)")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.6)
            self.line(50, 794, 545, 794)

            # Footer
            self.line(50, 45, 545, 45)
            self.drawString(50, 33, "Relatório Técnico e Plano do Problema de IA em Saúde")
            page_text = f"Página {self._pageNumber} de {page_count}"
            self.drawRightString(545, 33, page_text)
            
        self.restoreState()

def create_report():
    pdf_path = "/home/usuario/side-projects/analise-de-dados-fesa-doenca-de-lyme/reports/Relatorio_Entrega_01_Doenca_de_Lyme.pdf"
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=45,
        rightMargin=45,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    
    # Custom Palette
    primary_color = colors.HexColor("#0F172A")    # Deep Navy Slate
    secondary_color = colors.HexColor("#1E3A8A")  # Deep Royal Blue
    accent_color = colors.HexColor("#DC2626")     # Medical Crimson
    body_color = colors.HexColor("#1E293B")       # Dark Charcoal Text
    light_bg = colors.HexColor("#F8FAFC")         # Cool white background
    border_color = colors.HexColor("#E2E8F0")

    # Typography styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=21,
        leading=26,
        textColor=primary_color,
        alignment=0,
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=secondary_color,
        alignment=0,
        spaceAfter=14
    )

    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#475569")
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=primary_color,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=secondary_color,
        spaceBefore=10,
        spaceAfter=5,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.8,
        textColor=body_color,
        spaceAfter=6,
        alignment=4 # Justified
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=body_color,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
        alignment=1
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10.5,
        textColor=body_color
    )

    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10.5,
        textColor=primary_color
    )

    caption_style = ParagraphStyle(
        'Caption',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#64748B"),
        alignment=1,
        spaceAfter=8
    )

    story = []

    # -------------------------------------------------------------------------
    # HEADER / TITLE BLOCK (Bloco vermelho removido a pedido do usuário)
    # -------------------------------------------------------------------------
    story.append(Paragraph("Entrega 01: Relatório de Análise de Dados e Plano do Problema", title_style))
    story.append(Paragraph("Mapeamento Clínico, Curadoria do Dataset de Lesões Cutâneas e Formulação da Pipeline de Inteligência Artificial para Doença de Lyme", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=secondary_color, spaceAfter=8))

    meta_data = [
        [
            Paragraph("<b>Instituição:</b> FESA - Faculdade Engenheiro Salvador Arena", meta_style),
            Paragraph("<b>Curso:</b> Engenharia de Computação", meta_style)
        ],
        [
            Paragraph("<b>Disciplina:</b> Análise de Dados (N1 - 2º Bimestre)", meta_style),
            Paragraph("<b>Data da Entrega:</b> 19/10/2026", meta_style)
        ],
        [
            Paragraph("<b>Doença Selecionada:</b> Doença de Lyme (<i>Borrelia burgdorferi</i>)", meta_style),
            Paragraph("<b>Dataset:</b> Lyme Disease Rashes (5.059 imagens RGB)", meta_style)
        ]
    ]
    t_meta = Table(meta_data, colWidths=[270, 235])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F1F5F9")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 12))

    # -------------------------------------------------------------------------
    # SEÇÃO 1: CONTEXTO CLÍNICO E MAPEAMENTO DO DOMÍNIO
    # -------------------------------------------------------------------------
    story.append(Paragraph("1. Escolha da Doença e Mapeamento do Domínio Clínico", h1_style))
    story.append(Paragraph(
        "A <b>Doença de Lyme</b> é uma enfermidade infecciosa multissistêmica causada pela bactéria espiroqueta do complexo <i>Borrelia burgdorferi</i> "
        "sensu lato, transmitida primordialmente pela picada de carrapatos do gênero <i>Ixodes</i> (como <i>Ixodes scapularis</i> e <i>Ixodes ricinus</i>). "
        "A doença evolui tipicamente em três estágios clínicos progressivos:",
        body_style
    ))
    story.append(Paragraph("• <b>Estágio 1 - Infecção Localizada Precoce:</b> Caracteriza-se pelo surgimento, entre 3 a 30 dias após a inoculação, de uma lesão eritematosa expansiva no local da picada, denominada <b>Eritema Migratório (Erythema Migrans - EM)</b>. Frequentemente descrita como lesão em anel concêntrico ('em alvo' ou <i>bullseye</i>), mas que em 20% a 30% dos pacientes se apresenta como placa eritematosa homogênea, acompanhada de cefaleia, fadiga e febre baixa.", bullet_style))
    story.append(Paragraph("• <b>Estágio 2 - Infecção Disseminada Precoce:</b> Ocorre semanas a meses depois devido à disseminação hematogênica da espiroqueta, resultando em múltiplos eritemas secundários, neuropatias cranianas (como paralisia facial de Bell), meningite linfocítica e bloqueios atrioventriculares cardíacos (cardite de Lyme).", bullet_style))
    story.append(Paragraph("• <b>Estágio 3 - Infecção Tardia / Crônica:</b> Caracterizada por artrite inflamatória oligomarticular crônica (especialmente joelhos), encefalopatia crônica e síndrome persistente pós-tratamento da Doença de Lyme (PTLDS).", bullet_style))

    story.append(Spacer(1, 6))
    story.append(Paragraph("Gargalos e Limitações dos Métodos Diagnósticos Convencionais", h2_style))
    story.append(Paragraph(
        "A abordagem laboratorial padrão recomendada pelo <i>Centers for Disease Control and Prevention</i> (CDC) e pela <i>Infectious Diseases Society of America</i> (IDSA) "
        "consiste em um teste sorológico em duas etapas (triagem por imunoensaio enzimático ELISA/EIA seguida de confirmação por Western Blot / Immunoblotting). "
        "No entanto, este paradigma laboratorial enfrenta limitações críticas na janela clínica mais decisiva:",
        body_style
    ))
    story.append(Paragraph("1. <b>Janela Imunológica Tardia (Sensibilidade Insuficiente na Fase Aguda):</b> A resposta imune humoral (produção de anticorpos IgM e IgG específicos) demanda de 2 a 6 semanas para atingir níveis detectáveis. Nas primeiras 2 a 4 semanas de infecção – justamente quando o tratamento com antibióticos orais (doxiciclina) é capaz de curar a infecção e prevenir sequelas graves – a sorologia apresenta sensibilidade inferior a <b>30% a 50%</b>, gerando frequentes resultados falso-negativos.", bullet_style))
    story.append(Paragraph("2. <b>Dependência Crítica do Diagnóstico Dermatológico Visual:</b> Em razão da falha sorológica precoce, as diretrizes clínicas mundiais determinam que o diagnóstico da fase aguda deve ser puramente <b>clínico e dermatológico</b>, baseado na identificação do Eritema Migratório. Entretanto, estudos clínicos demonstram que médicos da atenção primária e emergencistas erram ou hesitam no diagnóstico em mais de <b>50% a 70% dos casos</b> atípicos.", bullet_style))
    story.append(Paragraph("3. <b>Mimetismo Dermatológico e Diagnósticos Diferenciais:</b> O eritema migratório compartilha características morfológicas, bordas e tonalidades com inúmeras patologias cutâneas frequentes, incluindo <i>Tinea corporis</i> (micose anular), eritema anular centrífugo, celulite bacteriana por <i>Staphylococcus/Streptococcus</i>, picadas necróticas de aranha e reações a drogas.", bullet_style))

    story.append(Spacer(1, 6))
    story.append(Paragraph("Justificativa e Relevância sob a Perspectiva de Análise de Dados e IA", h2_style))
    story.append(Paragraph(
        "A interseção entre a alta taxa de erro humano na fase aguda e o advento de redes neurais convolucionais (CNNs) e modelos de Visão Computacional "
        "torna a análise visual e classificação da Doença de Lyme um dos problemas de maior impacto em Inteligência Artificial aplicada à Saúde. "
        "Um sistema automatizado de triagem dermatológica é capaz de padronizar a detecção precoce no Ponto de Atendimento (<i>Point-of-Care</i>) e em plataformas de Telemedicina, "
        "reduzindo drasticamente o tempo até o início da antibioticoterapia de primeira linha, evitando o uso incorreto de antifúngicos ou corticoesteroides e impedindo a "
        "evolução debilitante para neuroborreliose e cardite.",
        body_style
    ))

    # Montage Figure
    story.append(Spacer(1, 8))
    fig6_path = "/home/usuario/side-projects/analise-de-dados-fesa-doenca-de-lyme/reports/figures/06_sample_montage.png"
    if os.path.exists(fig6_path):
        story.append(RLImage(fig6_path, width=500, height=250))
        story.append(Paragraph("<b>Figura 1:</b> Amostras representativas do dataset evidenciando o Eritema Migratório típico e em alvo (Lyme - Positivo, borda vermelha) em contraste direto com os diagnósticos diferenciais morfológicos (Negativo, borda azul).", caption_style))

    # -------------------------------------------------------------------------
    # SEÇÃO 2: CURADORIA DO DATASET
    # -------------------------------------------------------------------------
    story.append(Spacer(1, 8))
    story.append(Paragraph("2. Curadoria e Estrutura da Base de Dados Real", h1_style))
    story.append(Paragraph(
        "Para fundamentar o estudo com dados reais e clinicamente validados, realizou-se a curadoria do conjunto de dados "
        "<b>Lyme Disease Rashes (Full Stratified)</b>, disponibilizado na plataforma científica <b>Kaggle</b> (autor: <i>sshikamaru</i>). "
        "A base congrega um acervo com curadoria fotográfica de casos confirmados de Doença de Lyme e uma ampla gama de controles de patologias dermatológicas miméticas.",
        body_style
    ))
    story.append(Paragraph("• <b>Volume Total de Amostras:</b> O dataset completo totaliza <b>5.059 imagens dermatológicas em formato RGB</b>, subdivididas em duas partições primárias: <b>Positivo (941 casos de eritema de Lyme)</b> e <b>Negativo (4.118 imagens de controles dermatológicos)</b>.", bullet_style))
    story.append(Paragraph("• <b>Partições Estratificadas Prontas:</b> A base disponibiliza partições formais com divisão de <b>Treinamento (80.0% - 4.046 imagens)</b>, <b>Validação (10.0% - 506 imagens)</b> e <b>Teste Cego (10.0% - 507 imagens)</b>, mantendo estrita estratificação proporcional de classes.", bullet_style))
    story.append(Paragraph("• <b>Subconjunto Balanceado (Image Folder Balanced):</b> Para mitigar o impacto do desbalanceamento severo de classes em algoritmos sensíveis à proporção a priori, a curadoria do Kaggle inclui um subconjunto balanceado totalizando 1.896 imagens (Treino: 1.506 [752 Pos / 754 Neg], Val: 200 [94 Pos / 106 Neg], Teste: 190 [95 Pos / 95 Neg]).", bullet_style))
    story.append(Paragraph("• <b>Riqueza Taxonômica (15 Diagnósticos Diferenciados):</b> Cada amostra do dataset possui identificação nominal da lesão específica (prefixo taxonômico nos metadados), englobando 3 nomenclaturas de Lyme e 12 patologias dermatológicas diferenciais distintas.", bullet_style))

    # -------------------------------------------------------------------------
    # SEÇÃO 3: DICIONÁRIO COMPLETO DE DADOS
    # -------------------------------------------------------------------------
    story.append(Spacer(1, 8))
    story.append(Paragraph("3. Dicionário Completo de Dados da Base", h1_style))
    story.append(Paragraph(
        "A tabela a seguir apresenta a especificação estruturada de todos os atributos físicos, espectrais e computacionais "
        "extraídos pelo pipeline de processamento de imagem a partir do acervo fotográfico bruto.",
        body_style
    ))

    dict_rows = [
        [
            Paragraph("<b>Atributo</b>", table_header_style),
            Paragraph("<b>Tipo</b>", table_header_style),
            Paragraph("<b>Unidade</b>", table_header_style),
            Paragraph("<b>Faixa / Valores</b>", table_header_style),
            Paragraph("<b>Descrição Clínica e Técnica</b>", table_header_style)
        ],
        [
            Paragraph("<b>filename</b>", table_cell_bold),
            Paragraph("Texto", table_cell_style),
            Paragraph("Adimensional", table_cell_style),
            Paragraph("String única", table_cell_style),
            Paragraph("Nome de arquivo no disco contendo o identificador e rótulo diagnóstico específico da patologia.", table_cell_style)
        ],
        [
            Paragraph("<b>binary_class</b>", table_cell_bold),
            Paragraph("Categórico", table_cell_style),
            Paragraph("Classe", table_cell_style),
            Paragraph("{Positive, Negative}", table_cell_style),
            Paragraph("Rótulo clínico binário: 'Positive' indica Doença de Lyme (EM); 'Negative' indica outro quadro dermatológico.", table_cell_style)
        ],
        [
            Paragraph("<b>condition</b>", table_cell_bold),
            Paragraph("Categórico", table_cell_style),
            Paragraph("Patologia", table_cell_style),
            Paragraph("15 condições distintas", table_cell_style),
            Paragraph("Diagnóstico específico: Erythema Migrans, bullseye rash, tinea corporis, cellulitis, spider bite, etc.", table_cell_style)
        ],
        [
            Paragraph("<b>split</b>", table_cell_bold),
            Paragraph("Categórico", table_cell_style),
            Paragraph("Partição", table_cell_style),
            Paragraph("{train, val, test}", table_cell_style),
            Paragraph("Designação oficial da amostra no split de machine learning (80% treino, 10% validação, 10% teste).", table_cell_style)
        ],
        [
            Paragraph("<b>width</b>", table_cell_bold),
            Paragraph("Numérico (Int)", table_cell_style),
            Paragraph("Pixels (px)", table_cell_style),
            Paragraph("[80, 457] (Méd: 251.5)", table_cell_style),
            Paragraph("Dimensão horizontal da imagem dermatológica recortada.", table_cell_style)
        ],
        [
            Paragraph("<b>height</b>", table_cell_bold),
            Paragraph("Numérico (Int)", table_cell_style),
            Paragraph("Pixels (px)", table_cell_style),
            Paragraph("[66, 394] (Méd: 202.9)", table_cell_style),
            Paragraph("Dimensão vertical da imagem dermatológica recortada.", table_cell_style)
        ],
        [
            Paragraph("<b>aspect_ratio</b>", table_cell_bold),
            Paragraph("Numérico (Float)", table_cell_style),
            Paragraph("Razão (L/A)", table_cell_style),
            Paragraph("[0.43, 3.03] (Méd: 1.25)", table_cell_style),
            Paragraph("Proporção de aspecto (Largura / Altura), indicando se a imagem é quadrada, panorâmica ou vertical.", table_cell_style)
        ],
        [
            Paragraph("<b>color_mode</b>", table_cell_bold),
            Paragraph("Categórico", table_cell_style),
            Paragraph("Espaço de cor", table_cell_style),
            Paragraph("{RGB}", table_cell_style),
            Paragraph("Espaço de representação de cores tridimensional (100% das imagens são RGB de 3 canais).", table_cell_style)
        ],
        [
            Paragraph("<b>mean_r / g / b</b>", table_cell_bold),
            Paragraph("Numérico (Float)", table_cell_style),
            Paragraph("Intensidade (0-255)", table_cell_style),
            Paragraph("R: [42, 251] | G: [20, 248] | B: [6, 249]", table_cell_style),
            Paragraph("Média espectral dos canais Vermelho, Verde e Azul. Reflete a tonalidade da pele e o eritema inflamatório.", table_cell_style)
        ],
        [
            Paragraph("<b>brightness</b>", table_cell_bold),
            Paragraph("Numérico (Float)", table_cell_style),
            Paragraph("Luminância (0-255)", table_cell_style),
            Paragraph("[15.6, 249.1] (Méd: 139.5)", table_cell_style),
            Paragraph("Luminância perceptiva calculada por ITU-R 601 (0.299*R + 0.587*G + 0.114*B). Mede condições de iluminação.", table_cell_style)
        ],
        [
            Paragraph("<b>contrast</b>", table_cell_bold),
            Paragraph("Numérico (Float)", table_cell_style),
            Paragraph("Desvio Padrão", table_cell_style),
            Paragraph("[3.2, 106.0] (Méd: 45.9)", table_cell_style),
            Paragraph("Desvio padrão da luminância global da imagem. Quantifica a nitidez, desfoque e foco óptico.", table_cell_style)
        ],
        [
            Paragraph("<b>erythema_index</b>", table_cell_bold),
            Paragraph("Numérico (Float)", table_cell_style),
            Paragraph("Diferença Espectral", table_cell_style),
            Paragraph("[-33.2, 112.4] (Méd: 34.3)", table_cell_style),
            Paragraph("Índice dermatológico proxy de eritema cutâneo (R - G). Quantifica a rubefação / hiperemia na lesão.", table_cell_style)
        ],
        [
            Paragraph("<b>file_size_bytes</b>", table_cell_bold),
            Paragraph("Numérico (Int)", table_cell_style),
            Paragraph("Bytes", table_cell_style),
            Paragraph("[575, 15.369] (Méd: 4.860)", table_cell_style),
            Paragraph("Tamanho do arquivo binário comprimido em disco (indicador de ruído por compressão JPEG).", table_cell_style)
        ],
        [
            Paragraph("<b>corrupted</b>", table_cell_bold),
            Paragraph("Booleano", table_cell_style),
            Paragraph("Flag de Integridade", table_cell_style),
            Paragraph("{False} (0% corrompidos)", table_cell_style),
            Paragraph("Indicador de arquivo corrompido, truncated JPEG ou cabeçalho inválido. Verificado em 100% da base.", table_cell_style)
        ]
    ]

    t_dict = Table(dict_rows, colWidths=[80, 55, 60, 110, 200])
    t_dict.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), secondary_color),
        ('GRID', (0,0), (-1,-1), 0.5, border_color),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, light_bg]),
    ]))
    story.append(t_dict)

    story.append(Spacer(1, 6))
    story.append(Paragraph("Limitações Inerentes da Base de Dados Identificadas", h2_style))
    story.append(Paragraph(
        "Como prática rigorosa de auditoria em Análise de Dados em Saúde, documentam-se quatro limitações intrínsecas da base:",
        body_style
    ))
    story.append(Paragraph("1. <b>Ausência de Prontuários Tabulares e Demografia:</b> O repositório não acompanha variáveis estruturadas como idade, sexo biológico, comorbidades, tempo exato desde a picada do carrapato ou resposta imune confirmatória (Western Blot).", bullet_style))
    story.append(Paragraph("2. <b>Falta de Metadados de Fototipo de Fitzpatrick:</b> Não há rótulos explícitos de tonalidade cutânea (Escala de Fitzpatrick I a VI). Sabe-se na dermatologia clínica que o eritema migratório é frequentemente subdiagnosticado em peles de tons escuros (fototipos V e VI), sendo crucial avaliar viés algorítmico.", bullet_style))
    story.append(Paragraph("3. <b>Variabilidade de Resolução e Compressão JPEG:</b> As imagens variam entre 80x66 px e 457x394 px com forte compressão (tamanho médio de ~4.8 KB), exigindo etapas rigorosas de redimensionamento e interpolação bicúbica nas redes de visão.", bullet_style))
    story.append(Paragraph("4. <b>Variação Incontrolada de Ângulo e Flash:</b> A iluminação varia de fotos tiradas sob flash fotográfico intenso até ambientes sombreados, demandando técnicas robustas de color augmentation e normalização de luminância.", bullet_style))

    # -------------------------------------------------------------------------
    # SEÇÃO 4: AS 4 PERGUNTAS-CHAVE DO PROBLEMA
    # -------------------------------------------------------------------------
    story.append(Spacer(1, 8))
    story.append(Paragraph("4. Formulação Explícita das Quatro Perguntas-Chave do Projeto", h1_style))
    story.append(Paragraph(
        "Seguindo estritamente as diretrizes da comanda de avaliação acadêmica, a pipeline de modelagem do projeto é estruturada "
        "ao redor de quatro perguntas-chave multidisciplinares de Análise de Dados e Aprendizado de Máquina:",
        body_style
    ))

    # Pergunta 1: Agrupamento
    p1_box = [
        [Paragraph("<b>4.1 Pergunta de Agrupamento (Clustering Não Supervisionado)</b>", ParagraphStyle('P1T', parent=meta_style, fontName='Helvetica-Bold', textColor=secondary_color))],
        [Paragraph(
            "<b>Questão Central:</b> <i>\"Existem subgrupos, padrões morfológicos latentes ou fenótipos visuais não rotulados entre as lesões de eritema de pacientes com Lyme que permitam categorizar diferentes perfis de gravidade ou tempo de evolução clínica?\"</i>",
            body_style
        )],
        [Paragraph(
            "<b>Hipótese e Abordagem Metodológica:</b> O Eritema Migratório manifesta variações fenotípicas expressivas: (a) o padrão clássico anular com clareamento central (<i>bullseye</i>), (b) a placa eritematosa homogênea de bordas difusas, e (c) lesões atípicas purpúricas ou vesiculares centrais. "
            "Utilizando técnicas de aprendizado não supervisionado (extração de <i>embeddings</i> profundos via ResNet/EfficientNet pré-treinadas seguida de redução de dimensionalidade UMAP/t-SNE e algoritmos de agrupamento como HDBSCAN ou K-Means com análise de silhueta), "
            "investigaremos se o espaço latente de características visuais agrupa naturalmente subfenótipos clínicos distintos associados a diferentes durações de infecção ou virulência.",
            body_style
        )]
    ]
    t_p1 = Table(p1_box, colWidths=[505])
    t_p1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E1")),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_p1)
    story.append(Spacer(1, 6))

    # Pergunta 2: Classificação
    p2_box = [
        [Paragraph("<b>4.2 Pergunta de Classificação (Aprendizado Supervisionado)</b>", ParagraphStyle('P2T', parent=meta_style, fontName='Helvetica-Bold', textColor=secondary_color))],
        [Paragraph(
            "<b>Questão Central:</b> <i>\"É possível construir um modelo preditivo baseado em redes neurais convolucionais capaz de diagnosticar precocemente o eritema da Doença de Lyme e diferenciá-lo de outras 12 patologias dermatológicas similares em meio a imagens ruidosas?\"</i>",
            body_style
        )],
        [Paragraph(
            "<b>Hipótese e Abordagem Metodológica:</b> A diferenciação precoce entre o eritema de Lyme e patologias como <i>Tinea corporis</i>, celulite bacteriana e eritema anular centrífugo é a maior dificuldade clínica enfrentada por médicos generalistas. "
            "Serão treinados e comparados classificadores baseados em <i>Transfer Learning</i> (arquiteturas EfficientNet-B0/B2, ConvNeXt e Vision Transformers - ViT), "
            "utilizando funções de perda ponderadas (Focal Loss / Weighted Cross-Entropy) para contornar o desbalanceamento de 4.38:1 e avaliados rigorosamente pelas métricas "
            "de <b>Sensibilidade (Recall)</b>, <b>Especificidade</b>, <b>F1-Score</b> e <b>Área sob a Curva ROC (AUC-ROC)</b>.",
            body_style
        )]
    ]
    t_p2 = Table(p2_box, colWidths=[505])
    t_p2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E1")),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_p2)
    story.append(Spacer(1, 6))

    # Pergunta 3: Preditiva
    p3_box = [
        [Paragraph("<b>4.3 Pergunta Preditiva (Regressão / Modelagem Temporal)</b>", ParagraphStyle('P3T', parent=meta_style, fontName='Helvetica-Bold', textColor=secondary_color))],
        [Paragraph(
            "<b>Questão Central:</b> <i>\"Como modelar e prever a dinâmica geométrica de expansão temporal da lesão (taxa de crescimento diametral em cm/dia) e estimar a probabilidade contínua de disseminação sistêmica ou transição para fases crônicas?\"</i>",
            body_style
        )],
        [Paragraph(
            "<b>Hipótese e Abordagem Metodológica:</b> O Eritema Migratório expande-se centrifugamente de forma contínua a uma taxa média típica de 0.5 a 2 cm/dia, podendo atingir diâmetros superiores a 15 a 30 cm. "
            "Através de algoritmos de segmentação semântica e regressão não linear (modelos logísticos e polinomiais de crescimento radial acoplados a regressão ridge/gradient boosting sobre atributos morfológicos como índice de eritema cutâneo, uniformidade de borda e área estimada), "
            "modela-se a trajetória esperada da lesão em função do tempo decorrido, permitindo estimar a janela de risco para disseminação hematogênica bacteriana.",
            body_style
        )]
    ]
    t_p3 = Table(p3_box, colWidths=[505])
    t_p3.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E1")),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_p3)
    story.append(Spacer(1, 6))

    # Pergunta 4: Suporte à Decisão Clínica
    p4_box = [
        [Paragraph("<b>4.4 Pergunta de Suporte à Decisão (Aplicação Prescritiva / CDSS)</b>", ParagraphStyle('P4T', parent=meta_style, fontName='Helvetica-Bold', textColor=secondary_color))],
        [Paragraph(
            "<b>Questão Central:</b> <i>\"Como as predições probabilísticas calibradas e mapas de explicabilidade visual (Grad-CAM) podem ser integrados em um Sistema de Suporte à Decisão Clínica (CDSS) para auxiliar na recomendação de condutas terapêuticas precoces e otimização de tratamento?\"</i>",
            body_style
        )],
        [Paragraph(
            "<b>Hipótese e Abordagem Metodológica:</b> Um CDSS dermatológico não deve funcionar como 'caixa-preta'. "
            "A saída preditiva será combinada com técnicas de explicabilidade pós-hoc (<b>Grad-CAM - Gradient-weighted Class Activation Mapping</b>), "
            "evidenciando graficamente ao clínico as regiões anatômicas e bordas que determinaram a predição. "
            "O sistema fornecerá uma pontuação de risco calibrada (Calibrated Probability) associada a diretrizes clínicas prescritivas:",
            body_style
        )],
        [Paragraph("• <i>Risco Elevado (&gt; 80%):</i> Recomendação imediata de início empírico de antibioticoterapia de primeira linha (doxiciclina 100mg 2x/dia por 14 dias), sem necessidade de aguardar sorologia (conforme diretrizes CDC/IDSA).", bullet_style)],
        [Paragraph("• <i>Risco Intermediário (40% - 80%):</i> Solicitação de acompanhamento fotográfico em 48-72h para monitorar taxa de expansão diametral e agendamento de sorologia convalescente em 4 semanas.", bullet_style)],
        [Paragraph("• <i>Risco Baixo (&lt; 40%):</i> Orientação para investigação dos diagnósticos diferenciais prioritários destacados pelo modelo (tínea, celulite ou reação a picada).", bullet_style)]
    ]
    t_p4 = Table(p4_box, colWidths=[505])
    t_p4.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E1")),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_p4)

    # -------------------------------------------------------------------------
    # SEÇÃO 5: ANÁLISE EXPLORATÓRIA DE DADOS PRELIMINAR (EDA)
    # -------------------------------------------------------------------------
    story.append(Spacer(1, 10))
    story.append(Paragraph("5. Análise Exploratória Preliminar da Base (EDA)", h1_style))
    story.append(Paragraph(
        "A Análise Exploratória Preliminar foi conduzida sobre a totalidade das 5.059 instâncias fotográficas "
        "e seu acervo de metadados extraídos, cumprindo integralmente os quatro pilares de verificação exigidos pelo corpo docente:",
        body_style
    ))

    story.append(Paragraph("5.1 Diagnóstico de Dados Faltantes (Missing Values)", h2_style))
    story.append(Paragraph(
        "A integridade dos arquivos e dos atributos computacionais revelou uma taxa de <b>0,0% de dados faltantes (Null / NaN)</b> em todos os 20 atributos tabulares. "
        "Não foram detectados arquivos truncados ou com cabeçalhos binários corrompidos (0 imagens corrompidas entre as 5.059 verificadas). "
        "A totalidade dos dados é nativamente válida para ingestão nos pipelines de visão computacional.",
        body_style
    ))

    story.append(Spacer(1, 4))
    story.append(Paragraph("5.2 Análise de Desbalanceamento de Classes e Subpopulações Clínicas", h2_style))
    story.append(Paragraph(
        "Conforme ilustrado na <b>Figura 2</b>, a base completa possui forte desbalanceamento de classes: <b>4.118 amostras Negativas (81,4%)</b> contra "
        "<b>941 amostras Positivas (18,6%)</b>, correspondendo a uma razão de desbalanceamento de <b>4,38 : 1</b>. "
        "Esse fenômeno reflete com realismo o cenário de triagem ambulatorial, onde a vasta maioria das lesões eritematosas observadas no dia a dia decorre de outras causas dermatológicas.",
        body_style
    ))

    # Figure 1: Class Distribution
    fig1_path = "/home/usuario/side-projects/analise-de-dados-fesa-doenca-de-lyme/reports/figures/01_class_distribution.png"
    if os.path.exists(fig1_path):
        story.append(RLImage(fig1_path, width=490, height=190))
        story.append(Paragraph("<b>Figura 2:</b> Proporção do desbalanceamento severo de classes na base global e detalhamento por partições oficiais de treinamento, validação e teste.", caption_style))

    story.append(Paragraph(
        "A <b>Figura 3</b> apresenta a distribuição quantitativa individual das 15 patologias catalogadas. "
        "Nota-se que as 941 imagens de Doença de Lyme dividem-se equilibradamente entre <i>Erythema Migrans</i> (325), <i>bullseye rash</i> (310) e <i>EM rash</i> (306). "
        "Do lado dos controles negativos, o grupo de maior prevalência é o <i>eritema anular centrífugo</i> (447 amostras), seguido pela <i>tínea corporal</i> (430) e "
        "<i>picadas de aranha</i> (411), configurando um conjunto de testes altamente desafiador para qualquer classificador biomédico.",
        body_style
    ))

    # Figure 2: Conditions breakdown
    fig2_path = "/home/usuario/side-projects/analise-de-dados-fesa-doenca-de-lyme/reports/figures/02_conditions_breakdown.png"
    if os.path.exists(fig2_path):
        story.append(RLImage(fig2_path, width=490, height=260))
        story.append(Paragraph("<b>Figura 3:</b> Decomposição das 15 condições clínicas presentes no dataset. Em vermelho, as manifestações da Doença de Lyme; em azul, os 12 diagnósticos diferenciais dermatológicos de controle.", caption_style))

    story.append(Spacer(1, 6))
    story.append(Paragraph("5.3 Análise de Dimensões Espaciais, Resolução e Proporção de Tela", h2_style))
    story.append(Paragraph(
        "A análise geométrica das imagens (<b>Figura 4</b>) revelou grande heterogeneidade nas resoluções espaciais originais. "
        "A largura varia de 80 a 457 pixels (mediana de 259 px) e a altura varia de 66 a 394 pixels (mediana de 194 px). "
        "A proporção de tela (<i>Aspect Ratio = L / A</i>) apresenta média de 1,25, demonstrando predomínio de imagens horizontais retangulares, "
        "mas com presença de razões extremas entre 0,43 (recortes verticais estreitos) e 3,03 (panoramas alongados). "
        "Essa constatação exige que o pré-processamento utilize <i>letterboxing</i> com preenchimento ou <i>center crop</i> cuidadoso para evitar distorções anamórficas "
        "no formato geométrico do anel eritematoso ao alimentar redes neurais com entradas padronizadas (ex.: 224x224 ou 384x384).",
        body_style
    ))

    # Figure 3: Dimensions
    fig3_path = "/home/usuario/side-projects/analise-de-dados-fesa-doenca-de-lyme/reports/figures/03_dimensions_and_aspect_ratio.png"
    if os.path.exists(fig3_path):
        story.append(RLImage(fig3_path, width=490, height=195))
        story.append(Paragraph("<b>Figura 4:</b> Diagrama de dispersão de resolução espacial (largura vs altura) e distribuição da razão de aspecto das fotografias dermatológicas.", caption_style))

    story.append(Spacer(1, 6))
    story.append(Paragraph("5.4 Análise Espectral de Cor e Índice Clínico de Eritema (R - G)", h2_style))
    story.append(Paragraph(
        "Para avaliar se a cor pura é suficiente para separar os casos de Lyme dos diagnósticos diferenciais, calculou-se a média espectral dos canais RGB "
        "e o <b>Índice Clínico de Eritema (R - G)</b>, amplamente validado em dermatologia óptica para quantificar hiperemia capilar. "
        "Conforme revelado na <b>Figura 5</b>, a intensidade média do canal Vermelho foi de 160.3 nas lesões de Lyme contra 158.7 nos controles; o canal Verde foi de 126.5 contra 124.3; "
        "e o índice de eritema resultou em médias quase sobrepostas: <b>33.80 (Lyme) vs 34.38 (Controles)</b>.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Conclusão Científica Crítica:</b> <i>A assinatura global de cor (tonalidade média de vermelho) é incapaz, isoladamente, de discriminar o eritema da Doença de Lyme de suas patologias miméticas. "
        "Condições inflamatórias como celulite e dermatite produzem hiperemia tão ou mais intensa que o eritema migratório. "
        "Portanto, o aprendizado de máquina supervisionado não pode se apoiar em histogramas de cor simples, exigindo obrigatoriamente a extração de padrões morfológicos e espaciais "
        "(gradiente concêntrico, contraste de bordas e clareamento central) através de camadas convolucionais profundas.</i>",
        body_style
    ))

    # Figure 4: Color
    fig4_path = "/home/usuario/side-projects/analise-de-dados-fesa-doenca-de-lyme/reports/figures/04_color_and_erythema.png"
    if os.path.exists(fig4_path):
        story.append(RLImage(fig4_path, width=490, height=195))
        story.append(Paragraph("<b>Figura 5:</b> Médias de intensidade por canal RGB e curva de densidade do Índice de Eritema Cutâneo (R - G), comprovando sobreposição espectral entre as classes.", caption_style))

    story.append(Spacer(1, 6))
    story.append(Paragraph("5.5 Detecção de Outliers e Diagnóstico de Ruído na Base", h2_style))
    story.append(Paragraph(
        "Para identificar dados anômalos e ruídos experimentais, aplicou-se o método estatístico de Intervalo Interquartil (IQR) e limiares percentis (1% e 99%) "
        "sobre as métricas de <b>Brilho (Luminância)</b>, <b>Contraste (Desvio de Luminância)</b> e <b>Tamanho de Arquivo (KB)</b> (<b>Figura 6</b>):",
        body_style
    ))
    story.append(Paragraph("• <b>Outliers de Iluminação (Brilho Extremo):</b> O percentil 1% de luminância situa-se em 66.8 e o percentil 99% em 205.6. Detectaram-se 51 amostras com brilho inferior a 66.8 (fotos em penumbra com subexposição severa) e 51 amostras acima de 205.6 (imagens com reflexo especular de flash fotográfico estourado sobre a epiderme).", bullet_style))
    story.append(Paragraph("• <b>Outliers de Foco e Nitidez (Baixo Contraste):</b> Detectaram-se imagens com desvio de luminância inferior a 14.5 (percentil 1%), caracterizando fotografias fora de foco, com motion blur ou desfoque de lente macro no momento da captura ambulatorial.", bullet_style))
    story.append(Paragraph("• <b>Ruído de Compressão e Artefatos de JPEG:</b> O tamanho de arquivo varia de 575 bytes a 15.369 bytes (média de 4.8 KB). Arquivos abaixo de 1.5 KB exibem visíveis artefatos de blocos 8x8 típicos do algoritmo de quantização JPEG, constituindo ruído de alta frequência que será atenuado por filtros Gaussianos de suavização e técnicas de augmentação.", bullet_style))

    # Figure 5: Outliers
    fig5_path = "/home/usuario/side-projects/analise-de-dados-fesa-doenca-de-lyme/reports/figures/05_brightness_contrast_outliers.png"
    if os.path.exists(fig5_path):
        story.append(RLImage(fig5_path, width=490, height=195))
        story.append(Paragraph("<b>Figura 6:</b> Mapeamento de outliers de luminosidade e nitidez (Brilho vs Contraste com fronteiras percentis) e dispersão do ruído de compressão JPEG.", caption_style))

    # -------------------------------------------------------------------------
    # SEÇÃO 6: PLANO DE TRABALHO E PRÓXIMOS PASSOS
    # -------------------------------------------------------------------------
    story.append(Spacer(1, 10))
    story.append(Paragraph("6. Plano de Trabalho e Pipeline Analítica para as Próximas Entregas", h1_style))
    story.append(Paragraph(
        "Com base nos diagnósticos quantitativos da Entrega 01, o cronograma e a arquitetura das próximas etapas do projeto "
        "ficam estruturados da seguinte forma:",
        body_style
    ))

    roadmap_data = [
        [
            Paragraph("<b>Etapa</b>", table_header_style),
            Paragraph("<b>Foco / Módulo de IA</b>", table_header_style),
            Paragraph("<b>Ações Técnicas e Metodológicas</b>", table_header_style),
            Paragraph("<b>Entregáveis Esperados</b>", table_header_style)
        ],
        [
            Paragraph("<b>Fase 02</b>", table_cell_bold),
            Paragraph("Pré-processamento e Agrupamento", table_cell_style),
            Paragraph("Normalização de cor (CLAHE), redimensionamento preservando aspect ratio, extração de embeddings profundos e clustering (UMAP + HDBSCAN).", table_cell_style),
            Paragraph("Descoberta de subfenótipos clínicos de eritema de Lyme e caracterização morfológica.", table_cell_style)
        ],
        [
            Paragraph("<b>Fase 03</b>", table_cell_bold),
            Paragraph("Classificação Supervisionada", table_cell_style),
            Paragraph("Treinamento de CNNs (EfficientNet, ResNet50) e Vision Transformers com Focal Loss, class-weighting e data augmentation (rotacionamento, jitter de cor).", table_cell_style),
            Paragraph("Matriz de confusão, curvas ROC, AUC &gt; 0.90, Sensibilidade &gt; 92% na detecção de Lyme.", table_cell_style)
        ],
        [
            Paragraph("<b>Fase 04</b>", table_cell_bold),
            Paragraph("Modelagem Preditiva e CDSS", table_cell_style),
            Paragraph("Regressão da taxa de crescimento diametral, geração de mapas de calor explicáveis (Grad-CAM) e arquitetura de inferência clínica no ponto de atendimento.", table_cell_style),
            Paragraph("Módulo de decisão prescritiva com calibração de probabilidade e recomendação terapêutica.", table_cell_style)
        ]
    ]

    t_road = Table(roadmap_data, colWidths=[65, 115, 195, 130])
    t_road.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), secondary_color),
        ('GRID', (0,0), (-1,-1), 0.5, border_color),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, light_bg]),
    ]))
    story.append(t_road)

    # -------------------------------------------------------------------------
    # SEÇÃO 7: REFERÊNCIAS
    # -------------------------------------------------------------------------
    story.append(Spacer(1, 10))
    story.append(Paragraph("7. Referências Bibliográficas e Clínicas", h1_style))
    story.append(Paragraph("1. Centers for Disease Control and Prevention (CDC). <i>Lyme Disease: Clinical Guidance for Healthcare Providers and Erythema Migrans Diagnosis</i>. Atlanta: CDC, 2024.", bullet_style))
    story.append(Paragraph("2. Steere, A. C., Strle, F., Wormser, G. P., et al. <i>Lyme borreliosis</i>. Nature Reviews Disease Primers, 2(1), 16090, 2016.", bullet_style))
    story.append(Paragraph("3. Lantos, P. M., et al. <i>Clinical Practice Guidelines by the Infectious Diseases Society of America (IDSA), AAN, and ACR: 2020 Guidelines for the Prevention, Diagnosis, and Treatment of Lyme Disease</i>. Clinical Infectious Diseases, 72(1), e1-e48, 2021.", bullet_style))
    story.append(Paragraph("4. Burlina, P., Joshi, N., Billings, S. D., et al. <i>AI-based detection of erythema migrans: Toward automated decision support for early Lyme disease</i>. Computers in Biology and Medicine, 125, 103986, 2020.", bullet_style))
    story.append(Paragraph("5. Kaggle Dataset Repository. <i>Lyme Disease Rashes - Full Stratified Dataset</i>. Curadoria por sshikamaru, 2023.", bullet_style))

    # Build the document using NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF successfully generated: {pdf_path}")

if __name__ == '__main__':
    create_report()
