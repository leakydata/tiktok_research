"""
Generate publication-quality figures for the Multi-Run Stability Filtering paper.

Main text figures (outputs/figures/):
  1. fig1_pipeline.pdf          - Pipeline flowchart (method overview)
  2. fig2_convergence.pdf       - Stability vs number of runs (R=2->5)
  3. fig3_temperature.pdf       - Temperature effect grouped bars by construct
  4. fig4_cross_model_alpha.pdf - Cross-model alpha forest (Local vs Cloud, T=0.5)
  5. fig5_heatmap.pdf           - Cross-model pairwise agreement heatmap
  6. fig6_forest.pdf            - Within-model alpha forest plot (T=0.5 only)
  7. fig7_retention.pdf         - Stability retention curve (yield vs threshold)

Appendix figures:
  A1. figA1_baserate.pdf        - Label base-rate distributions by model
  A2. figA2_density.pdf         - Continuous construct bin distributions
  A3. figA3_cost_reliability.pdf - Reliability vs runtime scatter

Usage:
  python generate_figures.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

# Paths
BASE = Path(__file__).parent
OUT = BASE / 'outputs' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

EXP12 = BASE / 'outputs' / 'experiment_12'
EXP14 = BASE / 'outputs' / 'experiment_14'

# ── Journal-polished style ────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '#cccccc',
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
    'axes.linewidth': 0.6,
    'grid.alpha': 0.2,
    'grid.linewidth': 0.5,
    'grid.color': '#cccccc',
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
})

# ── Color palette (colorblind-friendly) ──────────────────────────────────
COLORS = {
    'T0': '#2166ac',       # dark blue
    'T05': '#b2182b',      # dark red
    'local': '#4393c3',    # muted blue
    'cloud': '#d6604d',    # muted red
    'accent': '#f4a582',
    'categorical': '#4393c3',
    'continuous': '#92c5de',
}

CLOUD_MODELS = {'deepseek-chat', 'gpt-5-nano', 'minimax-m2.5', 'claude-haiku-4.5'}

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
    'claude-haiku-4.5': 'Claude Haiku 4.5',
}

MODEL_PARAMS_B = {
    'gemma3:27b': 27, 'alibayram/medgemma:27b': 27, 'gpt-oss:20b': 20,
    'phi4:latest': 14, 'glm-4.7-flash': 9, 'qwen3:32b': 32,
    'deepseek-chat': 685, 'gpt-5-nano': 8, 'minimax-m2.5': 456,
    'claude-haiku-4.5': 8,
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

CATEGORICAL_CONSTRUCTS = {'social_proof', 'medical_authority', 'temporal_orientation', 'agency_control'}
CONTINUOUS_CONSTRUCTS = {'certainty_hedging', 'symptom_concreteness'}


def _save(fig, name):
    """Save figure as PDF + PNG and close."""
    fig.savefig(OUT / f'{name}.pdf')
    fig.savefig(OUT / f'{name}.png', dpi=300)
    plt.close(fig)
    print(f"  Saved {name}.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN TEXT FIGURES
# ═══════════════════════════════════════════════════════════════════════════

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
    box(6.5, 5, 2, 0.7, 'N Models\n(8 LLMs x 2 temps)', '#d4edda', '#155724', bold=True)

    # Arrow down from models
    arrow(6.5, 4.6, 6.5, 4.1)

    # Row 2: Multi-run
    box(6.5, 3.7, 2.4, 0.6, '5 Independent Runs\nper model x temp x construct', '#e8f4fd')

    # Arrow down
    arrow(6.5, 3.4, 6.5, 2.9)

    # Row 3: Within-model filter
    box(6.5, 2.5, 2.6, 0.7, 'Within-Model Stability Filter\n(>=4/5 agree or SD<=0.10)',
        '#f8d7da', '#721c24', bold=True)

    # Two arrows: pass and fail
    arrow(5.2, 2.5, 4.2, 2.5)  # fail left
    arrow(6.5, 2.1, 6.5, 1.5)  # pass down

    # Fail box
    box(3.2, 2.5, 1.8, 0.6, 'Unstable items\n-> flagged/excluded', '#f5f5f5', '#999999')

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

    fig.suptitle('Figure 1. Multi-Run Stability Filtering Pipeline',
                 fontsize=12, weight='bold', y=0.98)
    _save(fig, 'fig1_pipeline')


def fig2_convergence():
    """Figure 2: Stability rate vs number of runs (R=2->5), T=0.5 only."""
    df = pd.read_csv(EXP12 / 'run_convergence.csv')

    # Filter: T=0.5 only, R>=2, exclude qwen3:32b
    df = df[(df['temperature'] == 0.5) & (df['r_level'] >= 2)]
    df = df[df['model_name'] != 'qwen3:32b']

    # Average across models for each construct x R
    avg = df.groupby(['construct_name', 'r_level'])['stability_rate'].mean().reset_index()

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))

    markers = ['o', 's', '^', 'D', 'v', 'P']
    for i, construct in enumerate(CONSTRUCT_ORDER):
        cdata = avg[avg['construct_name'] == construct]
        ax.plot(cdata['r_level'], cdata['stability_rate'] * 100,
                marker=markers[i], label=CONSTRUCT_NAMES[construct],
                linewidth=1.5, markersize=5)

    ax.set_xlabel('Number of Runs (R)')
    ax.set_ylabel('Mean Stability Rate (%)')
    ax.set_xticks([2, 3, 4, 5])
    ax.set_ylim(40, 105)
    ax.legend(loc='lower right', ncol=2)
    ax.grid(True, axis='both')
    ax.set_title('Figure 2. Stability Rate vs. Number of Runs (T = 0.5)')

    _save(fig, 'fig2_convergence')


def fig3_temperature():
    """Figure 3: Temperature effect on stability -- grouped bars by construct."""
    df = pd.read_csv(EXP12 / 'significance_temperature.csv')

    constructs = df['construct_name'].tolist()
    order = [c for c in CONSTRUCT_ORDER if c in constructs]
    df = df.set_index('construct_name').loc[order].reset_index()

    x = np.arange(len(order))
    width = 0.35

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))

    ax.bar(x - width/2, df['stability_rate_t0'] * 100, width,
           label='T = 0.0', color=COLORS['T0'], alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.bar(x + width/2, df['stability_rate_t05'] * 100, width,
           label='T = 0.5', color=COLORS['T05'], alpha=0.85, edgecolor='white', linewidth=0.5)

    # Cohen's d annotations
    for i, row in df.iterrows():
        midx = x[i]
        t05_val = row['stability_rate_t05'] * 100
        ax.text(midx + width/2, t05_val + 1.5, f"d={row['cohens_d']:.2f}",
                ha='center', va='bottom', fontsize=7, color='#555555')

    ax.set_xlabel('Construct')
    ax.set_ylabel('Stability Rate (%)')
    ax.set_xticks(x)
    ax.set_xticklabels([CONSTRUCT_NAMES[c] for c in order], rotation=20, ha='right')
    ax.set_ylim(0, 115)
    ax.legend()
    ax.grid(True, axis='y')
    ax.set_title('Figure 3. Temperature Effect on Within-Model Stability (Local Models)')

    _save(fig, 'fig3_temperature')


def fig4_cross_model_alpha():
    """Figure 4: Cross-model alpha forest plot -- Local vs Cloud, T=0.5, per construct.

    This is the centerpiece figure: shows that within-model reliability is high
    but cross-model alpha is lower, with construct difficulty clearly visible.
    """
    # Load cross-model alpha (T=0.5 only)
    df12 = pd.read_csv(EXP12 / 'cross_model_krippendorff.csv')
    df14 = pd.read_csv(EXP14 / 'cross_model_krippendorff.csv')

    df12_t05 = df12[df12['temperature'] == 0.5].copy()
    df14_t05 = df14[df14['temperature'] == 0.5].copy()
    df12_t05['group'] = f'Local (N={df12_t05["num_models"].iloc[0]})'
    df14_t05['group'] = f'Cloud (N={df14_t05["num_models"].iloc[0]})'

    # Load consensus for secondary annotation
    cons12 = pd.read_csv(EXP12 / 'cross_model_consensus.csv')
    cons14 = pd.read_csv(EXP14 / 'cross_model_consensus.csv')
    cons12_t05 = cons12[cons12['temperature'] == 0.5].set_index('construct_name')
    cons14_t05 = cons14[cons14['temperature'] == 0.5].set_index('construct_name')

    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))

    constructs = [c for c in CONSTRUCT_ORDER if c in df12_t05['construct_name'].values]
    y_positions = np.arange(len(constructs))
    bar_height = 0.35

    for i, construct in enumerate(constructs):
        # Local
        row12 = df12_t05[df12_t05['construct_name'] == construct].iloc[0]
        ax.errorbar(row12['cross_model_alpha'], i + bar_height/2,
                    xerr=[[row12['cross_model_alpha'] - row12['alpha_ci_lower']],
                          [row12['alpha_ci_upper'] - row12['cross_model_alpha']]],
                    fmt='s', color=COLORS['local'], markersize=6, capsize=3,
                    capthick=1.0, elinewidth=1.0, zorder=3)
        # Majority consensus label
        if construct in cons12_t05.index:
            maj = cons12_t05.loc[construct, 'majority_rate']
            ax.text(row12['alpha_ci_upper'] + 0.02, i + bar_height/2,
                    f'{maj:.0%}', fontsize=7, va='center', color=COLORS['local'], alpha=0.8)

        # Cloud
        row14 = df14_t05[df14_t05['construct_name'] == construct].iloc[0]
        ax.errorbar(row14['cross_model_alpha'], i - bar_height/2,
                    xerr=[[row14['cross_model_alpha'] - row14['alpha_ci_lower']],
                          [row14['alpha_ci_upper'] - row14['cross_model_alpha']]],
                    fmt='D', color=COLORS['cloud'], markersize=5, capsize=3,
                    capthick=1.0, elinewidth=1.0, zorder=3)
        if construct in cons14_t05.index:
            maj = cons14_t05.loc[construct, 'majority_rate']
            ax.text(row14['alpha_ci_upper'] + 0.02, i - bar_height/2,
                    f'{maj:.0%}', fontsize=7, va='center', color=COLORS['cloud'], alpha=0.8)

    # Reference lines
    ax.axvline(x=0.667, color='#aaaaaa', linestyle='--', linewidth=0.8, zorder=1)
    ax.axvline(x=0.800, color='#888888', linestyle='--', linewidth=0.8, zorder=1)
    ax.text(0.667, len(constructs) - 0.1, 'tentative', fontsize=6.5,
            ha='center', color='#999999', style='italic')
    ax.text(0.800, len(constructs) - 0.1, 'good', fontsize=6.5,
            ha='center', color='#777777', style='italic')

    ax.set_yticks(y_positions)
    ax.set_yticklabels([CONSTRUCT_NAMES[c] for c in constructs])
    ax.set_xlabel("Cross-Model Krippendorff's alpha (T = 0.5)")
    ax.set_xlim(-0.1, 1.05)
    ax.grid(True, axis='x')

    # Legend
    local_h = ax.errorbar([], [], xerr=[], fmt='s', color=COLORS['local'],
                           markersize=6, capsize=3, label=df12_t05['group'].iloc[0])
    cloud_h = ax.errorbar([], [], xerr=[], fmt='D', color=COLORS['cloud'],
                           markersize=5, capsize=3, label=df14_t05['group'].iloc[0])
    ax.legend(handles=[local_h, cloud_h], loc='upper left',
              title='Percentages = majority consensus rate')

    ax.set_title('Figure 4. Cross-Model Convergent Validity (alpha with 95% CI)')
    fig.tight_layout()
    _save(fig, 'fig4_cross_model_alpha')


def fig5_heatmap():
    """Figure 5: Cross-model pairwise agreement heatmap (pooled temps, experiment 12)."""
    df = pd.read_csv(EXP12 / 'cross_model_agreement_matrix.csv')

    # Pooled (temperature is NaN)
    df_pooled = df[df['temperature'].isna()]
    df_pooled = df_pooled[
        (~df_pooled['model_a'].str.contains('qwen3')) &
        (~df_pooled['model_b'].str.contains('qwen3'))
    ]

    pair_avg = df_pooled.groupby(['model_a', 'model_b'])['agreement_rate'].mean().reset_index()
    models = sorted(set(pair_avg['model_a'].tolist() + pair_avg['model_b'].tolist()))
    n = len(models)

    matrix = np.ones((n, n))
    for _, row in pair_avg.iterrows():
        i = models.index(row['model_a'])
        j = models.index(row['model_b'])
        matrix[i][j] = row['agreement_rate']
        matrix[j][i] = row['agreement_rate']

    display_names = [MODEL_NAMES.get(m, m) for m in models]

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.5))
    # Re-enable spines for heatmap
    for spine in ax.spines.values():
        spine.set_visible(True)

    im = ax.imshow(matrix, cmap='YlOrRd', vmin=0.4, vmax=1.0, aspect='auto')

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

    ax.set_title('Figure 5. Cross-Model Pairwise Agreement\n(Local Models, Pooled Temps)',
                 pad=10)
    fig.tight_layout()
    _save(fig, 'fig5_heatmap')


def fig6_forest():
    """Figure 6: Within-model alpha forest plot with 95% CIs (T=0.5 only)."""
    df12 = pd.read_csv(EXP12 / 'table1_group_reliability.csv')
    df14 = pd.read_csv(EXP14 / 'table1_group_reliability.csv')

    df12 = df12[(df12['temperature'] == 0.5) & (df12['model_name'] != 'qwen3:32b')]
    df14 = df14[df14['temperature'] == 0.5]

    df = pd.concat([df12, df14], ignore_index=True)

    model_avg = df.groupby('model_name').agg(
        alpha=('krippendorff_alpha', 'mean'),
        ci_lo=('alpha_ci_lower', 'mean'),
        ci_hi=('alpha_ci_upper', 'mean'),
    ).reset_index()

    model_avg = model_avg.sort_values('alpha', ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for i, (_, row) in enumerate(model_avg.iterrows()):
        is_cloud = row['model_name'] in CLOUD_MODELS
        color = COLORS['cloud'] if is_cloud else COLORS['local']
        ax.errorbar(row['alpha'], i,
                    xerr=[[row['alpha'] - row['ci_lo']], [row['ci_hi'] - row['alpha']]],
                    fmt='o', color=color, markersize=6, capsize=3, capthick=1.0,
                    elinewidth=1.0)

    ax.set_yticks(range(len(model_avg)))
    ax.set_yticklabels([MODEL_NAMES.get(m, m) for m in model_avg['model_name']])

    ax.axvline(x=0.667, color='#aaaaaa', linestyle='--', linewidth=0.8)
    ax.axvline(x=0.800, color='#888888', linestyle='--', linewidth=0.8)
    ax.text(0.667, len(model_avg) - 0.3, 'tentative', fontsize=6.5,
            ha='center', color='#999999', style='italic')
    ax.text(0.800, len(model_avg) - 0.3, 'good', fontsize=6.5,
            ha='center', color='#777777', style='italic')

    ax.set_xlabel("Mean Within-Model Krippendorff's alpha (T = 0.5)")
    ax.set_xlim(0.3, 1.05)
    ax.grid(True, axis='x')

    local_patch = mpatches.Patch(color=COLORS['local'], label='Local (Ollama)')
    cloud_patch = mpatches.Patch(color=COLORS['cloud'], label='Cloud API')
    ax.legend(handles=[local_patch, cloud_patch], loc='lower right')

    ax.set_title("Figure 6. Within-Model Reliability at T = 0.5 (Stochastic Inference)")
    fig.tight_layout()
    _save(fig, 'fig6_forest')


def fig7_retention():
    """Figure 7: Stability retention curve -- % items surviving vs threshold.

    Two panels: categorical (agreement ratio thresholds) and continuous (SD thresholds).
    Answers: is 4/5 arbitrary? Is the pipeline too exclusionary?
    """
    df = pd.read_csv(EXP12 / 'figure_stability_threshold_curve.csv')
    # Exclude qwen3:32b
    df = df[df['model_name'] != 'qwen3:32b']

    # --- Categorical panel: agreement ratio thresholds ---
    cat_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    cat_constructs = sorted(CATEGORICAL_CONSTRUCTS & set(df['construct_name']))

    # Remap thresholds to human-readable labels (out of 5 runs)
    # 0.6 = 3/5, 0.8 = 4/5, 1.0 = 5/5
    thresh_labels = {0.5: '>=2/5\n(0.5)', 0.6: '>=3/5\n(0.6)', 0.7: '>=3.5/5\n(0.7)',
                     0.8: '>=4/5\n(0.8)', 0.9: '>=4.5/5\n(0.9)', 1.0: '5/5\n(1.0)'}

    # Average stability rate across models for each construct x threshold
    cat_df = df[df['threshold'].isin(cat_thresholds) & df['construct_name'].isin(cat_constructs)]
    cat_avg = cat_df.groupby(['construct_name', 'threshold'])['stability_rate'].mean().reset_index()

    # --- Continuous panel: SD thresholds ---
    cont_thresholds = [0.05, 0.1, 0.15, 0.2, 0.3]
    cont_constructs = sorted(CONTINUOUS_CONSTRUCTS & set(df['construct_name']))
    # For continuous, threshold 0.5 is also present (agreement-based) -- skip it
    cont_df = df[df['threshold'].isin(cont_thresholds) & df['construct_name'].isin(cont_constructs)]
    cont_avg = cont_df.groupby(['construct_name', 'threshold'])['stability_rate'].mean().reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    # Panel A: Categorical
    markers = ['o', 's', '^', 'D']
    for i, construct in enumerate([c for c in CONSTRUCT_ORDER if c in cat_constructs]):
        cdata = cat_avg[cat_avg['construct_name'] == construct].sort_values('threshold')
        ax1.plot(cdata['threshold'], cdata['stability_rate'] * 100,
                 marker=markers[i % len(markers)], label=CONSTRUCT_NAMES[construct],
                 linewidth=1.5, markersize=5)

    # Highlight the 0.8 = 4/5 threshold used in practice
    ax1.axvline(x=0.8, color='#b2182b', linestyle=':', linewidth=1.2, alpha=0.6)
    ax1.text(0.81, 52, '4/5 threshold\n(used)', fontsize=7, color='#b2182b',
             va='bottom', style='italic')

    ax1.set_xlabel('Agreement Ratio Threshold')
    ax1.set_ylabel('Mean Retention Rate (%)')
    ax1.set_xticks(cat_thresholds)
    ax1.set_xticklabels([f'{t:.1f}' for t in cat_thresholds])
    ax1.set_ylim(45, 105)
    ax1.legend(loc='lower left', fontsize=7)
    ax1.grid(True, axis='both')
    ax1.set_title('A. Categorical Constructs')

    # Panel B: Continuous
    for i, construct in enumerate([c for c in CONSTRUCT_ORDER if c in cont_constructs]):
        cdata = cont_avg[cont_avg['construct_name'] == construct].sort_values('threshold')
        ax2.plot(cdata['threshold'], cdata['stability_rate'] * 100,
                 marker=markers[i % len(markers)], label=CONSTRUCT_NAMES[construct],
                 linewidth=1.5, markersize=5)

    ax2.axvline(x=0.10, color='#b2182b', linestyle=':', linewidth=1.2, alpha=0.6)
    ax2.text(0.105, 52, 'SD<=0.10\n(used)', fontsize=7, color='#b2182b',
             va='bottom', style='italic')

    ax2.set_xlabel('SD Threshold')
    ax2.set_xticks(cont_thresholds)
    ax2.set_xticklabels([f'{t:.2f}' for t in cont_thresholds])
    ax2.legend(loc='lower left', fontsize=7)
    ax2.grid(True, axis='both')
    ax2.set_title('B. Continuous Constructs')

    fig.suptitle('Figure 7. Stability Retention: Proportion of Items Surviving the Filter',
                 fontsize=11, weight='bold', y=1.02)
    fig.tight_layout()
    _save(fig, 'fig7_retention')


# ═══════════════════════════════════════════════════════════════════════════
# APPENDIX FIGURES
# ═══════════════════════════════════════════════════════════════════════════

def figA1_baserate():
    """Appendix Figure A1: Label base-rate distribution by model.

    Shows stacked horizontal bars for two constructs where marginal distributions
    differ most: agency_control (4-category) and medical_authority (4-category).
    Visually explains why cross-model alpha is low despite high pairwise agreement.
    """
    df12 = pd.read_csv(EXP12 / 'table5_label_distributions.csv')
    df14 = pd.read_csv(EXP14 / 'table5_label_distributions.csv')
    df = pd.concat([df12, df14], ignore_index=True)

    # Exclude qwen3:32b (too few data points)
    df = df[df['model_name'] != 'qwen3:32b']

    constructs_to_show = ['agency_control', 'medical_authority']
    label_colors = {
        # agency_control
        'active': '#2166ac', 'passive': '#92c5de', 'mixed': '#fddbc7', 'helpless': '#b2182b',
        # medical_authority
        'none_observed': '#d1e5f0', 'professional': '#2166ac',
        'self_research': '#ef8a62', 'mixed': '#fddbc7',
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    for panel_idx, construct in enumerate(constructs_to_show):
        ax = axes[panel_idx]
        cdf = df[df['construct_name'] == construct].copy()

        # Get all labels and models
        label_col = 'final_label_text' if construct in CATEGORICAL_CONSTRUCTS else 'final_label_bin'
        all_labels = cdf[label_col].dropna().unique()

        # Define label order (most common first across all models)
        label_order = (cdf.groupby(label_col)['proportion']
                       .mean().sort_values(ascending=False).index.tolist())

        models = sorted(cdf['model_name'].unique(),
                        key=lambda m: m in CLOUD_MODELS)  # local first

        y_pos = np.arange(len(models))

        lefts = np.zeros(len(models))
        for label in label_order:
            widths = []
            for model in models:
                row = cdf[(cdf['model_name'] == model) & (cdf[label_col] == label)]
                widths.append(row['proportion'].values[0] * 100 if len(row) > 0 else 0)
            widths = np.array(widths)

            color = label_colors.get(label, '#cccccc')
            bars = ax.barh(y_pos, widths, left=lefts, height=0.6,
                           label=label.replace('_', ' ').title(),
                           color=color, edgecolor='white', linewidth=0.3)

            # Label inside bars if wide enough
            for j, w in enumerate(widths):
                if w > 8:
                    ax.text(lefts[j] + w/2, y_pos[j], f'{w:.0f}%',
                            ha='center', va='center', fontsize=6.5, color='black')

            lefts += widths

        display = [MODEL_NAMES.get(m, m) for m in models]
        # Mark cloud models
        display = [f'{d} *' if m in CLOUD_MODELS else d
                   for d, m in zip(display, models)]

        ax.set_yticks(y_pos)
        ax.set_yticklabels(display, fontsize=8)
        ax.set_xlabel('Label Distribution (%)')
        ax.set_xlim(0, 105)
        ax.legend(loc='upper right', fontsize=7, title=CONSTRUCT_NAMES[construct],
                  title_fontsize=8)
        ax.set_title(f'{"A" if panel_idx == 0 else "B"}. {CONSTRUCT_NAMES[construct]}')
        # Re-enable top/right spines for this chart
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.4)

    fig.suptitle('Figure A1. Label Base-Rate Distributions by Model\n'
                 '(* = cloud API models)',
                 fontsize=11, weight='bold', y=1.04)
    fig.tight_layout()
    _save(fig, 'figA1_baserate')


def figA2_density():
    """Appendix Figure A2: Continuous construct bin distributions per model.

    Shows grouped bar charts for certainty_hedging and symptom_concreteness,
    confirming models aren't collapsing to midrange and bins aren't inflating reliability.
    """
    df12 = pd.read_csv(EXP12 / 'table5_label_distributions.csv')
    df14 = pd.read_csv(EXP14 / 'table5_label_distributions.csv')
    df = pd.concat([df12, df14], ignore_index=True)
    df = df[df['model_name'] != 'qwen3:32b']

    constructs = ['certainty_hedging', 'symptom_concreteness']
    bin_order = {'low': 0, 'moderate': 1, 'high': 2,
                 'abstract': 0, 'moderate': 1, 'concrete': 2}
    bin_colors = {'low': '#92c5de', 'moderate': '#f4a582', 'high': '#b2182b',
                  'abstract': '#92c5de', 'concrete': '#b2182b'}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for panel_idx, construct in enumerate(constructs):
        ax = axes[panel_idx]
        cdf = df[df['construct_name'] == construct].copy()
        bins = cdf['final_label_bin'].dropna().unique()
        bins = sorted(bins, key=lambda b: bin_order.get(b, 1))

        models = sorted(cdf['model_name'].unique(),
                        key=lambda m: m in CLOUD_MODELS)

        x = np.arange(len(models))
        n_bins = len(bins)
        width = 0.8 / n_bins

        for j, bin_label in enumerate(bins):
            heights = []
            for model in models:
                row = cdf[(cdf['model_name'] == model) & (cdf['final_label_bin'] == bin_label)]
                heights.append(row['proportion'].values[0] * 100 if len(row) > 0 else 0)

            offset = (j - n_bins/2 + 0.5) * width
            color = bin_colors.get(bin_label, '#cccccc')
            ax.bar(x + offset, heights, width * 0.9,
                   label=bin_label.title(), color=color, edgecolor='white', linewidth=0.3)

        display = [MODEL_NAMES.get(m, m) for m in models]
        display = [f'{d} *' if m in CLOUD_MODELS else d
                   for d, m in zip(display, models)]

        ax.set_xticks(x)
        ax.set_xticklabels(display, rotation=35, ha='right', fontsize=7)
        ax.set_ylabel('Proportion (%)')
        ax.set_ylim(0, 100)
        ax.legend(fontsize=7)
        ax.grid(True, axis='y')
        ax.set_title(f'{"A" if panel_idx == 0 else "B"}. {CONSTRUCT_NAMES[construct]}')

    fig.suptitle('Figure A2. Continuous Construct Bin Distributions\n'
                 '(* = cloud API models)',
                 fontsize=11, weight='bold', y=1.04)
    fig.tight_layout()
    _save(fig, 'figA2_density')


def figA3_cost_reliability():
    """Appendix Figure A3: Reliability vs runtime scatter.

    X = runtime (hours for 6000 tasks), Y = mean within-model alpha (T=0.5),
    color = local vs cloud, size = parameter count.
    Supports: "Model size is not destiny."
    """
    # Cost data
    cost12 = pd.read_csv(EXP12 / 'table6_cost_efficiency.csv')
    cost14 = pd.read_csv(EXP14 / 'table6_cost_efficiency.csv')
    cost = pd.concat([cost12, cost14], ignore_index=True)
    cost = cost[cost['model_name'] != 'qwen3:32b']

    # Reliability data
    rel12 = pd.read_csv(EXP12 / 'table1_group_reliability.csv')
    rel14 = pd.read_csv(EXP14 / 'table1_group_reliability.csv')
    rel12 = rel12[(rel12['temperature'] == 0.5) & (rel12['model_name'] != 'qwen3:32b')]
    rel14 = rel14[rel14['temperature'] == 0.5]
    rel = pd.concat([rel12, rel14], ignore_index=True)

    model_alpha = rel.groupby('model_name')['krippendorff_alpha'].mean().reset_index()
    model_alpha.columns = ['model_name', 'mean_alpha']

    merged = cost.merge(model_alpha, on='model_name')

    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))

    for _, row in merged.iterrows():
        is_cloud = row['model_name'] in CLOUD_MODELS
        color = COLORS['cloud'] if is_cloud else COLORS['local']
        params = MODEL_PARAMS_B.get(row['model_name'], 20)
        # Scale marker size: log scale for huge cloud models
        size = max(30, min(250, np.log2(params + 1) * 25))

        ax.scatter(row['total_hours'], row['mean_alpha'],
                   s=size, color=color, alpha=0.8, edgecolors='white', linewidth=0.5,
                   zorder=3)
        # Label
        name = MODEL_NAMES.get(row['model_name'], row['model_name'])
        ax.annotate(name, (row['total_hours'], row['mean_alpha']),
                    textcoords='offset points', xytext=(8, 4), fontsize=7,
                    color='#333333')

    ax.axhline(y=0.667, color='#aaaaaa', linestyle='--', linewidth=0.8)
    ax.axhline(y=0.800, color='#888888', linestyle='--', linewidth=0.8)
    ax.text(0.1, 0.672, 'tentative', fontsize=6.5, color='#999999', style='italic')
    ax.text(0.1, 0.805, 'good', fontsize=6.5, color='#777777', style='italic')

    ax.set_xlabel('Total Runtime (hours, 6000 annotation tasks)')
    ax.set_ylabel("Mean Within-Model Krippendorff's alpha (T = 0.5)")
    ax.set_xscale('log')
    ax.grid(True, axis='both')

    local_patch = mpatches.Patch(color=COLORS['local'], label='Local (Ollama)')
    cloud_patch = mpatches.Patch(color=COLORS['cloud'], label='Cloud API')
    ax.legend(handles=[local_patch, cloud_patch], loc='lower left',
              title='Marker size ~ log(params)')

    ax.set_title("Figure A3. Reliability vs. Runtime\n(Model Size Is Not Destiny)")
    fig.tight_layout()
    _save(fig, 'figA3_cost_reliability')


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("Generating paper figures...")
    print()

    print("=== Main Text ===")
    print("Figure 1: Pipeline flowchart")
    fig1_pipeline()

    print("Figure 2: Stability vs runs convergence")
    fig2_convergence()

    print("Figure 3: Temperature effect bars")
    fig3_temperature()

    print("Figure 4: Cross-model alpha forest (NEW)")
    fig4_cross_model_alpha()

    print("Figure 5: Cross-model pairwise heatmap")
    fig5_heatmap()

    print("Figure 6: Within-model alpha forest (T=0.5)")
    fig6_forest()

    print("Figure 7: Stability retention curve (NEW)")
    fig7_retention()

    print()
    print("=== Appendix ===")
    print("Figure A1: Label base-rate distributions (NEW)")
    figA1_baserate()

    print("Figure A2: Continuous construct bin distributions (NEW)")
    figA2_density()

    print("Figure A3: Reliability vs runtime scatter (NEW)")
    figA3_cost_reliability()

    print(f"\nAll figures saved to {OUT}")
