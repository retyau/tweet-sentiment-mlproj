import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.patches import Patch

def create_academic_chart():
    """
    Generates and saves an academic-style bar chart comparing model performance.
    """
    # --- Data from your project results, including Random Guess ---
    model_data = [
        { 'label': 'Fine-tuned DistilBERT', 'accuracy': 88.07, 'type': 'transformer_tuned', 'n': None, 'k': None },
        { 'label': 'Simple Neural Network', 'accuracy': 85.27, 'type': 'neural_network', 'n': None, 'k': None },
        { 'label': 'Transformer (from scratch)', 'accuracy': 83.30, 'type': 'transformer_scratch', 'n': None, 'k': None },
        { 'label': 'Word2Vec (custom) + LogReg', 'accuracy': 75.72, 'type': 'classic_ml', 'n': None, 'k': None },
        { 'label': 'Gemini 2.5 Flash (05-20) - Prompt 1', 'accuracy': 72.00, 'type': 'llm', 'n': 100, 'k': 72 },
        { 'label': 'GloVe (100d) + LogReg', 'accuracy': 71.39, 'type': 'classic_ml', 'n': None, 'k': None },
        { 'label': 'gpt-4o (2024-11-20) - Prompt 2', 'accuracy': 71.18, 'type': 'llm', 'n': 1100, 'k': 783 },
        { 'label': 'gpt-4o (2024-11-20) - Prompt 1', 'accuracy': 69.37, 'type': 'llm', 'n': 1923, 'k': 1334 },
        { 'label': 'DeepSeek v3 (0324)', 'accuracy': 69.00, 'type': 'llm', 'n': 100, 'k': 69 },
        { 'label': 'GloVe (50d) + LogReg', 'accuracy': 66.41, 'type': 'classic_ml', 'n': None, 'k': None },
        { 'label': 'Gemini 2.5 Flash (05-20) - Prompt 2', 'accuracy': 65.00, 'type': 'llm', 'n': 100, 'k': 65 },
        { 'label': 'Random Guess', 'accuracy': 50.00, 'type': 'classic_ml', 'n': None, 'k': None }
    ]

    # Reverse data for horizontal bar chart (so best is at the top)
    model_data.reverse()

    # --- Academic Color Palette ---
    legend_info = {
        'transformer_tuned': { 'color': (37/255, 99/255, 235/255, 0.8), 'label': 'Fine-tuned Transformer (Proposed)' },
        'transformer_scratch': { 'color': (96/255, 165/255, 250/255, 0.8), 'label': 'Transformer (from Scratch)' },
        'neural_network': { 'color': (5/255, 150/255, 105/255, 0.8), 'label': 'Simple Neural Network' },
        'llm': { 'color': (217/255, 119/255, 6/255, 0.8), 'label': 'LLM (Zero-Shot)' },
        'classic_ml': { 'color': (107/255, 114/255, 128/255, 0.8), 'label': 'Classic ML Baselines' },
    }

    # --- Wilson Score Interval Calculation ---
    def wilson_score_interval(p, n, z=1.96):
        if n == 0: return (0, 0)
        z2 = z * z
        denominator = 1 + z2 / n
        center = p + z2 / (2 * n)
        term = math.sqrt((p * (1 - p) / n) + (z2 / (4 * n * n)))
        margin = z * term
        return ((center - margin) / denominator, (center + margin) / denominator)

    # --- Prepare data for plotting ---
    labels = [d['label'] for d in model_data]
    accuracies = [d['accuracy'] for d in model_data]
    colors = [legend_info[d['type']]['color'] for d in model_data]
    
    errors = np.zeros((2, len(model_data)))
    for i, d in enumerate(model_data):
        if d['type'] == 'llm' and d['n'] and d['k']:
            p = d['k'] / d['n']
            ci_lower, ci_upper = wilson_score_interval(p, d['n'])
            errors[0, i] = d['accuracy'] - ci_lower * 100
            errors[1, i] = ci_upper * 100 - d['accuracy']

    # --- Plotting ---
    try:
        plt.rcParams['font.family'] = 'Georgia'
    except:
        print("Warning: 'Georgia' font not found. Using default sans-serif font.")
        plt.rcParams['font.family'] = 'sans-serif'

    fig, ax = plt.subplots(figsize=(12, 8.5))

    y_pos = np.arange(len(labels))
    ax.barh(y_pos, accuracies, color=colors, align='center', edgecolor='black', linewidth=0.7,
            xerr=errors, capsize=4, ecolor='firebrick')

    # --- Style the chart ---
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Figure 1: Comparison of Model Performance on Tweet Sentiment Classification', 
                 fontsize=16, fontweight='bold', pad=20)
    
    ax.set_xlim(45, 90)
    ax.xaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.6)
    ax.set_axisbelow(True)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # --- Create Custom Legend ---
    legend_elements = [Patch(facecolor=info['color'], edgecolor='black', linewidth=0.7, label=info['label']) 
                       for info in legend_info.values()]
    
    ax.legend(handles=legend_elements, 
              loc='lower right', 
              bbox_to_anchor=(0.99, 0.01),
              fontsize=10, 
              title="Model Categories",
              title_fontsize=11,
              frameon=True,
              edgecolor='black')

    fig.text(0.5, 0.01, '*LLM results are shown with 95% confidence intervals (Wilson score).', 
             ha='center', fontsize=9, style='italic', color='gray')

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    # --- Save and show the plot ---
    plt.savefig("model_performance_comparison.png", dpi=300, bbox_inches='tight')
    print("Chart saved as 'model_performance_comparison.png'")
    plt.show()


if __name__ == '__main__':
    create_academic_chart()