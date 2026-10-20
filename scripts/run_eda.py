import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from PIL import Image

def generate_eda_figures():
    csv_path = '/home/usuario/side-projects/analise-de-dados-fesa-doenca-de-lyme/data/metadata_lyme_rashes.csv'
    df = pd.read_csv(csv_path)
    
    out_dir = '/home/usuario/side-projects/analise-de-dados-fesa-doenca-de-lyme/reports/figures'
    os.makedirs(out_dir, exist_ok=True)
    
    # Palette configuration
    sns.set_theme(style='whitegrid', font='sans-serif')
    c_blue = '#1E3A8A'
    c_red = '#DC2626'
    c_green = '#059669'
    c_amber = '#D97706'
    palette_binary = {'Negative': '#4B5563', 'Positive': '#DC2626'}

    # ----------------------------------------------------
    # FIGURE 1: Class Distribution & Imbalance
    # ----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Bar chart
    class_counts = df['binary_class'].value_counts()
    sns.barplot(x=class_counts.index, y=class_counts.values, ax=axes[0], palette=['#6B7280', '#DC2626'])
    axes[0].set_title('Distribuição Absoluta de Classes (Dataset Completo)', fontsize=13, weight='bold')
    axes[0].set_xlabel('Classe', fontsize=11)
    axes[0].set_ylabel('Quantidade de Imagens', fontsize=11)
    for i, v in enumerate(class_counts.values):
        axes[0].text(i, v + 60, f'{v} ({v/len(df)*100:.1f}%)', ha='center', fontweight='bold', fontsize=11)
    axes[0].set_ylim(0, 4700)

    # Split breakdown
    split_counts = pd.crosstab(df['split'], df['binary_class'])
    split_counts.plot(kind='bar', stacked=True, ax=axes[1], color=['#6B7280', '#DC2626'])
    axes[1].set_title('Distribuição por Partição (Train / Val / Test)', fontsize=13, weight='bold')
    axes[1].set_xlabel('Partição', fontsize=11)
    axes[1].set_ylabel('Quantidade de Imagens', fontsize=11)
    axes[1].legend(['Negativo (Outras)', 'Positivo (Lyme EM)'])
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '01_class_distribution.png'), dpi=300)
    plt.close()

    # ----------------------------------------------------
    # FIGURE 2: Subcategories / Clinical Conditions Breakdown
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 7))
    cond_counts = df['condition'].value_counts()
    colors = ['#DC2626' if c in ['Erythema Migrans', 'bullseye rash', 'EM rash'] else '#3B82F6' for c in cond_counts.index]
    
    sns.barplot(x=cond_counts.values, y=cond_counts.index, palette=colors, ax=ax)
    ax.set_title('Detalhamento das Condições Dermatológicas e Diagnósticos Diferenciais', fontsize=14, weight='bold')
    ax.set_xlabel('Número de Amostras', fontsize=12)
    ax.set_ylabel('Condição / Diagnóstico Clínico', fontsize=12)
    
    for i, v in enumerate(cond_counts.values):
        ax.text(v + 5, i, f'{v}', va='center', fontsize=10, fontweight='bold')
        
    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#DC2626', label='Positivo: Doença de Lyme (Eritema Migratório)'),
        Patch(facecolor='#3B82F6', label='Negativo: Diagnósticos Diferenciais Dermatológicos')
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '02_conditions_breakdown.png'), dpi=300)
    plt.close()

    # ----------------------------------------------------
    # FIGURE 3: Image Dimensions, Resolution & Outliers
    # ----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    sns.scatterplot(data=df, x='width', y='height', hue='binary_class', alpha=0.5,
                    palette={'Negative': '#6B7280', 'Positive': '#DC2626'}, ax=axes[0], s=30)
    axes[0].set_title('Dispersão de Resoluções: Largura vs Altura (pixels)', fontsize=13, weight='bold')
    axes[0].set_xlabel('Largura (pixels)', fontsize=11)
    axes[0].set_ylabel('Altura (pixels)', fontsize=11)
    axes[0].axvline(df['width'].median(), color='blue', linestyle='--', label=f'Mediana Largura: {df["width"].median():.0f}px')
    axes[0].axhline(df['height'].median(), color='green', linestyle='--', label=f'Mediana Altura: {df["height"].median():.0f}px')
    axes[0].legend(loc='lower right')

    # Aspect Ratio Distribution
    sns.histplot(data=df, x='aspect_ratio', hue='binary_class', bins=40, kde=True,
                 palette={'Negative': '#6B7280', 'Positive': '#DC2626'}, ax=axes[1])
    axes[1].set_title('Distribuição da Proporção de Tela (Aspect Ratio = L / A)', fontsize=13, weight='bold')
    axes[1].set_xlabel('Razão de Aspecto', fontsize=11)
    axes[1].set_ylabel('Frequência', fontsize=11)
    axes[1].axvline(1.0, color='red', linestyle=':', label='Quadrada (1:1)')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '03_dimensions_and_aspect_ratio.png'), dpi=300)
    plt.close()

    # ----------------------------------------------------
    # FIGURE 4: Color Channels & Erythema Index (R - G)
    # ----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # RGB mean comparison
    rgb_means = pd.DataFrame({
        'Canal R': [df[df['binary_class']=='Positive']['mean_r'].mean(), df[df['binary_class']=='Negative']['mean_r'].mean()],
        'Canal G': [df[df['binary_class']=='Positive']['mean_g'].mean(), df[df['binary_class']=='Negative']['mean_g'].mean()],
        'Canal B': [df[df['binary_class']=='Positive']['mean_b'].mean(), df[df['binary_class']=='Negative']['mean_b'].mean()],
    }, index=['Positivo (Lyme)', 'Negativo (Outros)'])
    
    rgb_means.plot(kind='bar', ax=axes[0], color=['#EF4444', '#10B981', '#3B82F6'], width=0.6)
    axes[0].set_title('Médias de Intensidade por Canal de Cor (RGB)', fontsize=13, weight='bold')
    axes[0].set_ylabel('Intensidade Média (0-255)', fontsize=11)
    axes[0].set_ylim(0, 200)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_height():.1f}", (p.get_x() + p.get_width() / 2., p.get_height() + 2),
                         ha='center', va='bottom', fontsize=9, fontweight='bold')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)

    # Erythema Index KDE
    sns.kdeplot(data=df, x='erythema_index', hue='binary_class', fill=True, common_norm=False,
                palette={'Negative': '#6B7280', 'Positive': '#DC2626'}, ax=axes[1], alpha=0.4)
    axes[1].set_title('Distribuição do Índice Clínico de Eritema (R - G)', fontsize=13, weight='bold')
    axes[1].set_xlabel('Índice de Eritema (Diferença de Intensidade Vermelho - Verde)', fontsize=11)
    axes[1].set_ylabel('Densidade', fontsize=11)
    axes[1].legend(['Positivo (Lyme EM)', 'Negativo (Outros)'])

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '04_color_and_erythema.png'), dpi=300)
    plt.close()

    # ----------------------------------------------------
    # FIGURE 5: Brightness, Contrast & Noise/Outliers
    # ----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Brightness vs Contrast
    q_b_low, q_b_high = df['brightness'].quantile(0.01), df['brightness'].quantile(0.99)
    q_c_low, q_c_high = df['contrast'].quantile(0.01), df['contrast'].quantile(0.99)
    
    sns.scatterplot(data=df, x='brightness', y='contrast', hue='binary_class', alpha=0.5,
                    palette={'Negative': '#6B7280', 'Positive': '#DC2626'}, ax=axes[0], s=35)
    axes[0].axvline(q_b_low, color='red', linestyle='--', alpha=0.7, label=f'1% e 99% Brilho ({q_b_low:.1f}, {q_b_high:.1f})')
    axes[0].axvline(q_b_high, color='red', linestyle='--', alpha=0.7)
    axes[0].axhline(q_c_low, color='orange', linestyle='--', alpha=0.7, label=f'1% e 99% Contraste ({q_c_low:.1f}, {q_c_high:.1f})')
    axes[0].axhline(q_c_high, color='orange', linestyle='--', alpha=0.7)
    axes[0].set_title('Brilho vs Contraste com Fronteiras de Outliers', fontsize=13, weight='bold')
    axes[0].set_xlabel('Brilho Médio (Luminância)', fontsize=11)
    axes[0].set_ylabel('Contraste (Desvio Padrão Luminância)', fontsize=11)
    axes[0].legend(loc='lower left', fontsize=9)

    # Boxplot of File Sizes (JPEG compression noise)
    df['file_size_kb'] = df['file_size_bytes'] / 1024.0
    sns.boxplot(data=df, x='binary_class', y='file_size_kb', palette=['#6B7280', '#DC2626'], ax=axes[1], width=0.4)
    axes[1].set_title('Distribuição de Tamanho de Arquivo (KB) - Compressão/Artefatos', fontsize=13, weight='bold')
    axes[1].set_xlabel('Classe', fontsize=11)
    axes[1].set_ylabel('Tamanho do Arquivo (KB)', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '05_brightness_contrast_outliers.png'), dpi=300)
    plt.close()

    # ----------------------------------------------------
    # FIGURE 6: Clinical Montage Sample Grid
    # ----------------------------------------------------
    sample_conditions = [
        ('Positive', 'Erythema Migrans', 'Lyme: Eritema Migratório Típico'),
        ('Positive', 'bullseye rash', 'Lyme: Lesão em Alvo (Bullseye)'),
        ('Positive', 'EM rash', 'Lyme: Eritema em Expansão'),
        ('Negative', 'tinea corporis', 'Dif.: Tinea Corporis (Micose)'),
        ('Negative', 'Bacterial cellulitis', 'Dif.: Celulite Bacteriana'),
        ('Negative', 'spider bite rash', 'Dif.: Picada de Aranha'),
        ('Negative', 'superficial erythema annulare centrifugum', 'Dif.: Eritema Anular Centrífugo'),
        ('Negative', 'pityriasis rosea', 'Dif.: Pitiríase Rósea')
    ]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    base_datasplit = '/home/usuario/side-projects/analise-de-dados-fesa-doenca-de-lyme/data/DataSplit/image_folder 2'
    
    for i, (b_class, cond, title) in enumerate(sample_conditions):
        folder = os.path.join(base_datasplit, b_class)
        matching = [f for f in os.listdir(folder) if cond in f and f.lower().endswith(('.jpg', '.jpeg'))]
        if matching:
            img_p = os.path.join(folder, matching[0])
            img = Image.open(img_p)
            axes[i].imshow(img)
            border_color = 'red' if b_class == 'Positive' else 'blue'
            for spine in axes[i].spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3)
        axes[i].set_title(title, fontsize=10.5, weight='bold', pad=6)
        axes[i].axis('off')
        
    plt.suptitle('Amostras Representativas: Manifestações de Lyme vs Diagnósticos Diferenciais', fontsize=15, weight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '06_sample_montage.png'), dpi=300)
    plt.close()
    
    print('All 6 figures successfully generated in:', out_dir)

if __name__ == '__main__':
    generate_eda_figures()
