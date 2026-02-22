"""
Generate publication-quality figures for the Multi-Run Stability Filtering paper.

Produces 5 figures as PDF files in outputs/figures/:
  1. fig1_pipeline.pdf        - Pipeline flowchart (method overview)
  2. fig2_convergence.pdf     - Stability vs number of runs (R=1→5)
  3. fig3_temperature.pdf     - Temperature effect grouped bars by construct
  4. fig4_heatmap.pdf         - Cross-model pairwise agreement heatmap
  5. fig5_forest.pdf          - Alpha forest plot with CIs (T=0.5 only)

Usage:
  python generate_figures.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

# Paths
BASE = Path(__file__).parent
OUT = BASE / 'outputs' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

EXP12 = BASE / 'outputs' / 'experiment_12'
EXP14 = BASE / 'outputs' / 'experiment_14'

# Consistent style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Color palette (colorblind-friendly)
COLORS = {
    'T0': '#2166ac',    # dark blue
    'T05': '#b2182b',   # dark red
    'local': '#4393c3',
    'cloud': '#d6604d',
    'accent': '#f4a582',
    'grid': '#e0e0e0',
}

# Model display names
MODEL_NAMES = {
    'gemma3:27b': 'Gemma3 27B',
    'alibayram/medgemma:27b': 'MedGemma 27B',
    'gpt-oss:20b': 'GPT-OSS 20B',
    'phi4:latest': 'Phi-4 14B',
    'glm-4.7-flash': 'GLM-4.7-Flash',
    'qwen3:32b': 'Qwen3 32B',
    'deepseek-chat': 'DeepSeek-V3.2',
    'gpt-5-nano': 'GPT-5-nano',
    'minimax-m2.5': 'MiniMax-M2.5',
}

CONSTRUCT_NAMES = {
    'social_proof': 'Social Proof',
    'temporal_orientation': 'Temporal Orient.',
    'medical_authority': 'Medical Auth.',
    'agency_control': 'Agency/Control',
    'certainty_hedging': 'Certainty/Hedging',
    'symptom_concreteness': 'Symptom Concrete.',
}

# Construct order (easiest to hardest)
CONSTRUCT_ORDER = [
    'social_proof', 'medical_authority', 'temporal_orientation',
    'agency_control', 'certainty_hedging', 'symptom_concreteness'
]


def fig1_pipeline():
    """Figure 1: Multi-run stability filtering pipeline flowchart."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    def box(x, y, w, h, text, color='#e8f4fd', ec='#2166ac', fontsize=8, bold=False):
        rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                              boxstyle="round,pad=0.1", fc=color, ec=ec, lw=1.2)
        ax.add_patch(rect)
        weight = 'bold' if bold else 'normal'
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
                weight=weight, wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#333333', lw=1.2))

    # Row 1: Input
    box(1.2, 5, 2, 0.7, 'TikTok Video\nTranscripts', '#fff3cd', '#856404', bold=True)
    arrow(2.2, 5, 3.0, 5)
    box(3.8, 5, 1.4, 0.7, 'Chunking\n(150-500 chars)', '#e8f4fd')
    arrow(4.5, 5, 5.3, 5)

    # Row 1: Models fan-out
    box(6.5, 5, 2, 0.7, 'N Models\n(8 LLMs × 2 temps)', '#d4edda', '#155724', bold=True)

    # Arrow down from models
    arrow(6.5, 4.6, 6.5, 4.1)

    # Row 2: Multi-run
    box(6.5, 3.7, 2.4, 0.6, '5 Independent Runs\nper model × temp × construct', '#e8f4fd')

    # Arrow down
    arrow(6.5, 3.4, 6.5, 2.9)

    # Row 3: Within-model filter
    box(6.5, 2.5, 2.6, 0.7, 'Within-Model Stability Filter\n(≥4/5 agree or SD≤0.10)',
        '#f8d7da', '#721c24', bold=True)

    # Two arrows: pass and fail
    arrow(5.2, 2.5, 4.2, 2.5)  # fail left
    arrow(6.5, 2.1, 6.5, 1.5)  # pass down

    # Fail box
    box(3.2, 2.5, 1.8, 0.6, 'Unstable items\n→ flagged/excluded', '#f5f5f5', '#999999')

    # Row 4: Cross-model consensus
    box(6.5, 1.1, 2.6, 0.7, 'Cross-Model Consensus\n(majority vote across models)',
        '#cce5ff', '#004085', bold=True)

    # Arrow to output
    arrow(6.5, 0.7, 6.5, 0.3)
    box(6.5, 0.0, 2.2, 0.5, 'Reliable Annotations\n+ Confidence Metrics',
        '#d4edda', '#155724', fontsize=8, bold=True)

    # Side annotation
    ax.text(1.2, 3.7, '48,282 total\ninferences', ha='center', va='center',
            fontsize=8, style='italic', color='#666666')
    ax.text(1.2, 1.1, 'Two-stage filtering\nseparates measurement\nnoise from definitional\nambiguity',
            ha='center', va='center', fontsize=7, style='italic', color='#666666',
            bbox=dict(boxstyle='round,pad=0.3', fc='#f0f0f0', ec='#cccccc'))

    fig.suptitle('Figure 1. Multi-Run Stability Filtering Pipeline', fontsize=12, weight='bold', y=0.98)
    fig.savefig(OUT / 'fig1_pipeline.pdf')
    fig.savefig(OUT / 'fig1_pipeline.png', dpi=300)
    plt.close(fig)
    print(f"  Saved fig1_pipeline.pdf")


def fig2_convergence():
    """Figure 2: Stability rate vs number of runs (R=2→5), T=0.5 only."""
    df = pd.read_csv(EXP12 / 'run_convergence.csv')

    # Filter: T=0.5 only, R>=2 (R=1 has no stability), exclude qwen3:32b
    df = df[(df['temperature'] == 0.5) & (df['r_level'] >= 2)]
    df = df[df['model_name'] != 'qwen3:32b']

    # Average across models for each construct × R
    avg = df.groupby(['construct_name', 'r_level'])['stability_rate'].mean().reset_index()

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))

    markers = ['o', 's', '^', 'D', 'v', 'P']
    for i, construct in enumerate(CONSTRUCT_ORDER):
        cdata = avg[avg['construct_name'] == construct]
        ax.plot(cdata['r_level'], cdata['stability_rate'] * 100,
                marker=markers[i], label=CONSTRUCT_NAMES[construct],
                linewidth=1.8, markersize=6)

    ax.set_xlabel('Number of Runs (R)')
    ax.set_ylabel('Mean Stability Rate (%)')
    ax.set_xticks([2, 3, 4, 5])
    ax.set_ylim(40, 105)
    ax.legend(loc='lower right', framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_title('Figure 2. Stability Rate vs. Number of Runs (T=0.5)', weight='bold')

    fig.savefig(OUT / 'fig2_convergence.pdf')
    fig.savefig(OUT / 'fig2_convergence.png', dpi=300)
    plt.close(fig)
    print(f"  Saved fig2_convergence.pdf")


def fig3_temperature():
    """Figure 3: Temperature effect on stability — grouped bars by construct."""
    df = pd.read_csv(EXP12 / 'significance_temperature.csv')

    constructs = df['construct_name'].tolist()
    # Reorder by construct difficulty
    order = [c for c in CONSTRUCT_ORDER if c in constructs]
    df = df.set_index('construct_name').loc[order].reset_index()

    x = np.arange(len(order))
    width = 0.35

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))

    bars_t0 = ax.bar(x - width/2, df['stability_rate_t0'] * 100, width,
                      label='T = 0.0', color=COLORS['T0'], alpha=0.85)
    bars_t05 = ax.bar(x + width/2, df['stability_rate_t05'] * 100, width,
                       label='T = 0.5', color=COLORS['T05'], alpha=0.85)

    # Add Cohen's d annotations
    for i, row in df.iterrows():
        midx = x[i]
        t05_val = row['stability_rate_t05'] * 100
        ax.text(midx + width/2, t05_val + 1.5, f"d={row['cohens_d']:.2f}",
                ha='center', va='bottom', fontsize=7, color='#666666')

    ax.set_xlabel('Construct')
    ax.set_ylabel('Stability Rate (%)')
    ax.set_xticks(x)
    ax.set_xticklabels([CONSTRUCT_NAMES[c] for c in order], rotation=20, ha='right')
    ax.set_ylim(0, 115)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_title('Figure 3. Temperature Effect on Within-Model Stability (Local Models)',
                 weight='bold')

    fig.savefig(OUT / 'fig3_temperature.pdf')
    fig.savefig(OUT / 'fig3_temperature.png', dpi=300)
    plt.close(fig)
    print(f"  Saved fig3_temperature.pdf")


def fig4_heatmap():
    """Figure 4: Cross-model pairwise agreement heatmap (pooled temps, experiment 12)."""
    df = pd.read_csv(EXP12 / 'cross_model_agreement_matrix.csv')

    # Pooled (temperature is empty/NaN)
    df_pooled = df[df['temperature'].isna()]
    # Exclude qwen3:32b (too few data points)
    df_pooled = df_pooled[
        (~df_pooled['model_a'].str.contains('qwen3')) &
        (~df_pooled['model_b'].str.contains('qwen3'))
    ]

    # Average across constructs for each model pair
    pair_avg = df_pooled.groupby(['model_a', 'model_b'])['agreement_rate'].mean().reset_index()

    # Get unique models
    models = sorted(set(pair_avg['model_a'].tolist() + pair_avg['model_b'].tolist()))
    n = len(models)

    # Build matrix
    matrix = np.ones((n, n))  # diagonal = 1.0
    for _, row in pair_avg.iterrows():
        i = models.index(row['model_a'])
        j = models.index(row['model_b'])
        matrix[i][j] = row['agreement_rate']
        matrix[j][i] = row['agreement_rate']

    display_names = [MODEL_NAMES.get(m, m) for m in models]

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    im = ax.imshow(matrix, cmap='YlOrRd', vmin=0.4, vmax=1.0, aspect='auto')

    # Annotate cells
    for i in range(n):
        for j in range(n):
            color = 'white' if matrix[i][j] > 0.75 else 'black'
            ax.text(j, i, f'{matrix[i][j]:.2f}', ha='center', va='center',
                    fontsize=8, color=color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(display_names, fontsize=8)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Mean Pairwise Agreement Rate')

    ax.set_title('Figure 4. Cross-Model Pairwise Agreement (Local Models, Pooled Temps)',
                 weight='bold', pad=10)

    fig.savefig(OUT / 'fig4_heatmap.pdf')
    fig.savefig(OUT / 'fig4_heatmap.png', dpi=300)
    plt.close(fig)
    print(f"  Saved fig4_heatmap.pdf")


def fig5_forest():
    """Figure 5: Forest plot of Krippendorff's alpha with 95% CIs (T=0.5 only)."""
    # Load both experiments
    df12 = pd.read_csv(EXP12 / 'table1_group_reliability.csv')
    df14 = pd.read_csv(EXP14 / 'table1_group_reliability.csv')

    # T=0.5 only, exclude qwen3:32b
    df12 = df12[(df12['temperature'] == 0.5) & (df12['model_name'] != 'qwen3:32b')]
    df14 = df14[df14['temperature'] == 0.5]

    df = pd.concat([df12, df14], ignore_index=True)

    # Average alpha across constructs per model
    model_avg = df.groupby('model_name').agg(
        alpha=('krippendorff_alpha', 'mean'),
        ci_lo=('alpha_ci_lower', 'mean'),
        ci_hi=('alpha_ci_upper', 'mean'),
    ).reset_index()

    # Sort by alpha descending
    model_avg = model_avg.sort_values('alpha', ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    y_positions = range(len(model_avg))
    colors = []
    for _, row in model_avg.iterrows():
        is_cloud = row['model_name'] in ('deepseek-chat', 'gpt-5-nano', 'minimax-m2.5')
        colors.append(COLORS['cloud'] if is_cloud else COLORS['local'])

    for i, (_, row) in enumerate(model_avg.iterrows()):
        is_cloud = row['model_name'] in ('deepseek-chat', 'gpt-5-nano', 'minimax-m2.5')
        color = COLORS['cloud'] if is_cloud else COLORS['local']
        ax.errorbar(row['alpha'], i,
                    xerr=[[row['alpha'] - row['ci_lo']], [row['ci_hi'] - row['alpha']]],
                    fmt='o', color=color, markersize=7, capsize=4, capthick=1.2,
                    elinewidth=1.2)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([MODEL_NAMES.get(m, m) for m in model_avg['model_name']])

    # Reference lines
    ax.axvline(x=0.667, color='#999999', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(x=0.800, color='#666666', linestyle='--', linewidth=1, alpha=0.7)
    ax.text(0.667, len(model_avg) - 0.3, 'α=0.667\n(tentative)', fontsize=7,
            ha='center', color='#999999')
    ax.text(0.800, len(model_avg) - 0.3, 'α=0.800\n(good)', fontsize=7,
            ha='center', color='#666666')

    ax.set_xlabel("Mean Krippendorff's α (T=0.5, averaged across constructs)")
    ax.set_xlim(0.3, 1.05)
    ax.grid(True, axis='x', alpha=0.3)

    # Legend
    local_patch = mpatches.Patch(color=COLORS['local'], label='Local (Ollama)')
    cloud_patch = mpatches.Patch(color=COLORS['cloud'], label='Cloud API')
    ax.legend(handles=[local_patch, cloud_patch], loc='lower right')

    ax.set_title("Figure 5. Within-Model Reliability at T=0.5 (Stochastic Inference)",
                 weight='bold')

    fig.savefig(OUT / 'fig5_forest.pdf')
    fig.savefig(OUT / 'fig5_forest.png', dpi=300)
    plt.close(fig)
    print(f"  Saved fig5_forest.pdf")


if __name__ == '__main__':
    print("Generating paper figures...")
    print()

    print("Figure 1: Pipeline flowchart")
    fig1_pipeline()

    print("Figure 2: Stability vs runs convergence")
    fig2_convergence()

    print("Figure 3: Temperature effect bars")
    fig3_temperature()

    print("Figure 4: Cross-model agreement heatmap")
    fig4_heatmap()

    print("Figure 5: Alpha forest plot (T=0.5)")
    fig5_forest()

    print(f"\nAll figures saved to {OUT}")
