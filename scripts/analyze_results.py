#!/usr/bin/env python
"""
Pilot Results Analysis: Generate visualizations and CSV exports
"""

import json
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

# Setup
ROOT_DIR = Path(__file__).resolve().parents[1]
# Try pilot_results.json first, fall back to comparison_results.json
PILOT_FILE = ROOT_DIR / "results" / "pilot_results.json"
COMPARISON_FILE = ROOT_DIR / "results" / "comparison_results.json"
RESULTS_FILE = PILOT_FILE if PILOT_FILE.exists() else COMPARISON_FILE
OUTPUT_DIR = ROOT_DIR / "results"

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def _normalize_to_pilot_format(results):
    """Convert comparison_results.json format to pilot_results.json format."""
    # If already in pilot format, return as-is
    if results and 'query_index' in results[0]:
        return results
    
    # Convert from comparison format
    normalized = []
    for idx, item in enumerate(results, 1):
        normalized.append({
            'query_index': idx,
            'query': item.get('query', ''),
            'llm_model': item.get('llm_model', 'unknown'),
            'embedding_model': item.get('embedding_model', 'unknown'),
            'sage_response': item.get('sage_response', {}),
            'traditional_response': item.get('traditional_response', {}),
            # Map similarity_scores to similarity
            'similarity': item.get('similarity_scores', {}),
            # Map rouge_scores to rouge
            'rouge': item.get('rouge_scores', {}),
            'llm_evaluation': item.get('llm_evaluation', {}),
        })
    return normalized

def load_results():
    """Load pilot results from JSON"""
    with open(RESULTS_FILE, 'r') as f:
        data = json.load(f)
        return _normalize_to_pilot_format(data)


def _nested_get(value, path):
    """Get a nested value from dicts using a list of keys."""
    current = value
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _score_from_metric(value):
    """Pick a representative numeric score from a metric payload."""
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, dict):
        # Try the most specific paths first, prefer non-zero values
        result = None
        # Try direct f1 first (comparison_results format)
        result = _nested_get(value, ['f1'])
        if isinstance(result, (int, float)) and result > 0:
            return result
        # Try top-level ROUGE f1 values (comparison_results format)
        for path in (['rouge-1', 'f'], ['rouge-l', 'f'], ['rouge-2', 'f']):
            result = _nested_get(value, path)
            if isinstance(result, (int, float)) and result > 0:
                return result
        
        # Try system1_f1 and other system-specific paths (pilot_results format)
        for path in (
            ['system1_f1'],
            ['score'],
            ['system1', 'rouge-l', 'f'],
            ['system1', 'rouge-1', 'f'],
        ):
            candidate = _nested_get(value, path)
            if isinstance(candidate, (int, float)) and candidate > 0:
                return candidate
        
        # Fall back to any zero value if nothing else found
        for path in (['f1'], ['system1_f1'], ['score']):
            candidate = _nested_get(value, path)
            if isinstance(candidate, (int, float)):
                return candidate
    return None


def _extract_metric_columns(item: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested metrics into CSV-friendly columns."""
    similarity = item.get('similarity') or {}
    rouge = item.get('rouge') or {}
    llm_eval = item.get('llm_evaluation') or {}

    # Handle both pilot_results and comparison_results formats
    # comparison_results has: similarity['f1'], similarity['precision'], similarity['recall']
    # plus similarity['system1_f1'], etc. (which we boosted to 0)
    # pilot_results has: similarity['system1_f1'], etc.
    
    return {
        'similarity_system1_precision': similarity.get('system1_precision') or similarity.get('precision') if isinstance(similarity, dict) else None,
        'similarity_system1_recall': similarity.get('system1_recall') or similarity.get('recall') if isinstance(similarity, dict) else None,
        'similarity_system1_f1': similarity.get('system1_f1') or similarity.get('f1') if isinstance(similarity, dict) else None,
        'similarity_system2_precision': similarity.get('system2_precision') if isinstance(similarity, dict) else None,
        'similarity_system2_recall': similarity.get('system2_recall') if isinstance(similarity, dict) else None,
        'similarity_system2_f1': similarity.get('system2_f1') if isinstance(similarity, dict) else None,
        'rouge_system1_rouge1_f': _nested_get(rouge, ['system1', 'rouge-1', 'f']) or _nested_get(rouge, ['rouge-1', 'f']),
        'rouge_system1_rouge2_f': _nested_get(rouge, ['system1', 'rouge-2', 'f']) or _nested_get(rouge, ['rouge-2', 'f']),
        'rouge_system1_rougeL_f': _nested_get(rouge, ['system1', 'rouge-l', 'f']) or _nested_get(rouge, ['rouge-l', 'f']),
        'rouge_system2_rouge1_f': _nested_get(rouge, ['system2', 'rouge-1', 'f']),
        'rouge_system2_rouge2_f': _nested_get(rouge, ['system2', 'rouge-2', 'f']),
        'rouge_system2_rougeL_f': _nested_get(rouge, ['system2', 'rouge-l', 'f']),
        'llm_system1_score': llm_eval.get('system1_score') if isinstance(llm_eval, dict) else None,
        'llm_system2_score': llm_eval.get('system2_score') if isinstance(llm_eval, dict) else None,
        'llm_better_system': llm_eval.get('better_system') if isinstance(llm_eval, dict) else None,
    }


def plot_all_metrics(results: List[Dict[str, Any]]):
    """Generate comprehensive metric visualizations from available fields."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    similarity_rows = []
    rouge_rows = []
    llm_rows = []
    win_counts = {'system1': 0, 'system2': 0, 'tie': 0}

    for idx, r in enumerate(results, 1):
        sim = r.get('similarity') or {}
        rouge = r.get('rouge') or {}
        llm = r.get('llm_evaluation') or {}

        precision = sim.get('precision') if isinstance(sim, dict) else None
        recall = sim.get('recall') if isinstance(sim, dict) else None
        f1 = sim.get('f1') if isinstance(sim, dict) else None
        if isinstance(precision, (int, float)) or isinstance(recall, (int, float)) or isinstance(f1, (int, float)):
            similarity_rows.append((idx, precision, recall, f1))

        rouge1 = _nested_get(rouge, ['rouge-1', 'f']) or _nested_get(rouge, ['system1', 'rouge-1', 'f'])
        rouge2 = _nested_get(rouge, ['rouge-2', 'f']) or _nested_get(rouge, ['system1', 'rouge-2', 'f'])
        rougel = _nested_get(rouge, ['rouge-l', 'f']) or _nested_get(rouge, ['system1', 'rouge-l', 'f'])
        if isinstance(rouge1, (int, float)) or isinstance(rouge2, (int, float)) or isinstance(rougel, (int, float)):
            rouge_rows.append((idx, rouge1, rouge2, rougel))

        s1 = llm.get('system1_score') if isinstance(llm, dict) else None
        s2 = llm.get('system2_score') if isinstance(llm, dict) else None
        if isinstance(s1, (int, float)) and isinstance(s2, (int, float)):
            llm_rows.append((idx, s1, s2))
            if s1 > s2:
                win_counts['system1'] += 1
            elif s2 > s1:
                win_counts['system2'] += 1
            else:
                win_counts['tie'] += 1
        elif isinstance(llm, dict):
            better = llm.get('better_system')
            if better in ('system1', 'system2'):
                win_counts[better] += 1

    if similarity_rows:
        arr = np.array(similarity_rows, dtype=object)
        x = arr[:, 0].astype(float)
        if any(v is not None for v in arr[:, 1]):
            axes[0, 0].plot(x, [np.nan if v is None else float(v) for v in arr[:, 1]], label='Precision', color='#2E86AB')
        if any(v is not None for v in arr[:, 2]):
            axes[0, 0].plot(x, [np.nan if v is None else float(v) for v in arr[:, 2]], label='Recall', color='#F18F01')
        if any(v is not None for v in arr[:, 3]):
            axes[0, 0].plot(x, [np.nan if v is None else float(v) for v in arr[:, 3]], label='F1', color='#A23B72', linewidth=2)
        axes[0, 0].set_title('Similarity Components by Query', fontweight='bold')
        axes[0, 0].set_xlabel('Query Index', fontweight='bold')
        axes[0, 0].set_ylabel('Score', fontweight='bold')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

    if rouge_rows:
        arr = np.array(rouge_rows, dtype=object)
        x = arr[:, 0].astype(float)
        if any(v is not None for v in arr[:, 1]):
            axes[0, 1].plot(x, [np.nan if v is None else float(v) for v in arr[:, 1]], label='ROUGE-1 F', color='#2E86AB')
        if any(v is not None for v in arr[:, 2]):
            axes[0, 1].plot(x, [np.nan if v is None else float(v) for v in arr[:, 2]], label='ROUGE-2 F', color='#F18F01')
        if any(v is not None for v in arr[:, 3]):
            axes[0, 1].plot(x, [np.nan if v is None else float(v) for v in arr[:, 3]], label='ROUGE-L F', color='#A23B72', linewidth=2)
        axes[0, 1].set_title('ROUGE Components by Query', fontweight='bold')
        axes[0, 1].set_xlabel('Query Index', fontweight='bold')
        axes[0, 1].set_ylabel('Score', fontweight='bold')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

    if llm_rows:
        arr = np.array(llm_rows, dtype=float)
        x = arr[:, 0]
        axes[1, 0].plot(x, arr[:, 1], label='SAGE (System1)', color='#2E86AB', linewidth=2)
        axes[1, 0].plot(x, arr[:, 2], label='Traditional (System2)', color='#A23B72', linewidth=2)
        axes[1, 0].set_title('LLM Evaluation Scores by Query', fontweight='bold')
        axes[1, 0].set_xlabel('Query Index', fontweight='bold')
        axes[1, 0].set_ylabel('Score', fontweight='bold')
        axes[1, 0].set_ylim(0, 10)
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

    labels = ['SAGE Wins', 'Traditional Wins', 'Ties']
    values = [win_counts['system1'], win_counts['system2'], win_counts['tie']]
    axes[1, 1].bar(labels, values, color=['#2E86AB', '#A23B72', '#CFCFCF'], edgecolor='black', alpha=0.85)
    axes[1, 1].set_title('LLM Winner Counts', fontweight='bold')
    axes[1, 1].set_ylabel('Count', fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.suptitle('All Metrics Overview', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'all_metrics_overview.png', dpi=300, bbox_inches='tight')
    print(f"✓ Chart: all_metrics_overview.png")
    plt.close()

def create_performance_csv(results: List[Dict[str, Any]]):
    """Export detailed performance metrics to CSV"""
    csv_file = OUTPUT_DIR / "pilot_performance.csv"
    
    rows = []
    for item in results:
        metrics = _extract_metric_columns(item)
        rows.append({
            'query_index': item.get('query_index'),
            'query': item.get('query', '')[:100],  # First 100 chars
            'llm_model': item.get('llm_model'),
            'embedding_model': item.get('embedding_model'),
            'sage_answer': item.get('sage_response', {}).get('answer', '')[:100],
            'traditional_answer': item.get('traditional_response', {}).get('answer', '')[:100],
            'sage_latency': item.get('sage_response', {}).get('latency'),
            'traditional_latency': item.get('traditional_response', {}).get('latency'),
            **metrics,
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(csv_file, index=False)
    print(f"✓ Performance CSV: {csv_file.name}")
    return df

def create_summary_csv(results: List[Dict[str, Any]]):
    """Export aggregated summary statistics to CSV"""
    csv_file = OUTPUT_DIR / "pilot_summary.csv"
    
    # Group by model combinations
    summary_data = []
    
    for llm in set(r['llm_model'] for r in results):
        for emb in set(r['embedding_model'] for r in results):
            subset = [r for r in results if r['llm_model'] == llm and r['embedding_model'] == emb]
            
            sage_latencies = [r.get('sage_response', {}).get('latency', 0) for r in subset if r.get('sage_response')]
            trad_latencies = [r.get('traditional_response', {}).get('latency', 0) for r in subset if r.get('traditional_response')]
            
            similarities = [
                _score_from_metric(r.get('similarity'))
                for r in subset
            ]
            similarities = [value for value in similarities if value is not None]

            rouges = [
                _score_from_metric(r.get('rouge'))
                for r in subset
            ]
            rouges = [value for value in rouges if value is not None]
            
            summary_data.append({
                'LLM_Model': llm,
                'Embedding_Model': emb,
                'Total_Queries': len(subset),
                'Avg_SAGE_Latency_s': np.mean(sage_latencies) if sage_latencies else 0,
                'Avg_Traditional_Latency_s': np.mean(trad_latencies) if trad_latencies else 0,
                'Avg_Similarity': np.mean(similarities) if similarities else 0,
                'Avg_ROUGE': np.mean(rouges) if rouges else 0,
                'SAGE_Faster_%': (1 - np.mean(sage_latencies) / np.mean(trad_latencies)) * 100 if (trad_latencies and np.mean(trad_latencies) > 0) else 0,
            })
    
    df = pd.DataFrame(summary_data)
    df.to_csv(csv_file, index=False)
    print(f"✓ Summary CSV: {csv_file.name}")
    return df

def plot_latency_comparison(results: List[Dict[str, Any]]):
    """Plot SAGE vs Traditional latency"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sage_latencies = []
    trad_latencies = []
    
    for r in results:
        sage = r.get('sage_response', {}).get('latency')
        trad = r.get('traditional_response', {}).get('latency')
        if sage and trad:
            sage_latencies.append(sage)
            trad_latencies.append(trad)
    
    x = np.arange(len(sage_latencies[:20]))  # First 20 for clarity
    width = 0.35
    
    ax.bar(x - width/2, sage_latencies[:20], width, label='SAGE', alpha=0.8, color='#2E86AB')
    ax.bar(x + width/2, trad_latencies[:20], width, label='Traditional', alpha=0.8, color='#A23B72')
    
    ax.set_xlabel('Query Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latency (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('SAGE vs Traditional RAG: Latency Comparison (First 20 Queries)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'latency_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Chart: latency_comparison.png")
    plt.close()

def plot_performance_metrics(results: List[Dict[str, Any]]):
    """Plot overall performance metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Collect metrics
    similarities = [
        _score_from_metric(r.get('similarity'))
        for r in results
    ]
    similarities = [value for value in similarities if value is not None]

    rouges = [
        _score_from_metric(r.get('rouge'))
        for r in results
    ]
    rouges = [value for value in rouges if value is not None]
    
    sage_latencies = [r.get('sage_response', {}).get('latency') for r in results if r.get('sage_response', {}).get('latency')]
    trad_latencies = [r.get('traditional_response', {}).get('latency') for r in results if r.get('traditional_response', {}).get('latency')]
    
    # Plot 1: Similarity distribution
    if similarities:
        axes[0, 0].hist(similarities, bins=20, alpha=0.7, color='#2E86AB', edgecolor='black')
        axes[0, 0].axvline(np.mean(similarities), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(similarities):.2f}')
        axes[0, 0].set_xlabel('Similarity Score', fontweight='bold')
        axes[0, 0].set_ylabel('Frequency', fontweight='bold')
        axes[0, 0].set_title('Similarity Score Distribution', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
    
    # Plot 2: ROUGE distribution
    if rouges:
        axes[0, 1].hist(rouges, bins=20, alpha=0.7, color='#A23B72', edgecolor='black')
        axes[0, 1].axvline(np.mean(rouges), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rouges):.2f}')
        axes[0, 1].set_xlabel('ROUGE Score', fontweight='bold')
        axes[0, 1].set_ylabel('Frequency', fontweight='bold')
        axes[0, 1].set_title('ROUGE Score Distribution', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
    
    # Plot 3: Latency comparison
    if sage_latencies and trad_latencies:
        data = [sage_latencies, trad_latencies]
        box = axes[1, 0].boxplot(data, tick_labels=['SAGE', 'Traditional'], patch_artist=True)
        for patch, color in zip(box['boxes'], ['#2E86AB', '#A23B72']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[1, 0].set_ylabel('Latency (seconds)', fontweight='bold')
        axes[1, 0].set_title('Latency Distribution', fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Plot 4: Summary statistics table
    if similarities or rouges or sage_latencies or trad_latencies:
        summary_stats = {
            'Metric': ['Similarity', 'ROUGE', 'SAGE Latency', 'Trad Latency'],
            'Mean': [
                f'{np.mean(similarities):.3f}' if similarities else 'N/A',
                f'{np.mean(rouges):.3f}' if rouges else 'N/A',
                f'{np.mean(sage_latencies):.3f}s' if sage_latencies else 'N/A',
                f'{np.mean(trad_latencies):.3f}s' if trad_latencies else 'N/A',
            ],
            'Min': [
                f'{min(similarities):.3f}' if similarities else 'N/A',
                f'{min(rouges):.3f}' if rouges else 'N/A',
                f'{min(sage_latencies):.3f}s' if sage_latencies else 'N/A',
                f'{min(trad_latencies):.3f}s' if trad_latencies else 'N/A',
            ],
            'Max': [
                f'{max(similarities):.3f}' if similarities else 'N/A',
                f'{max(rouges):.3f}' if rouges else 'N/A',
                f'{max(sage_latencies):.3f}s' if sage_latencies else 'N/A',
                f'{max(trad_latencies):.3f}s' if trad_latencies else 'N/A',
            ],
        }
        
        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')
        table = axes[1, 1].table(cellText=[[summary_stats['Metric'][i], summary_stats['Mean'][i], 
                                             summary_stats['Min'][i], summary_stats['Max'][i]] 
                                            for i in range(len(summary_stats['Metric']))],
                                 colLabels=['Metric', 'Mean', 'Min', 'Max'],
                                 cellLoc='center',
                                 loc='center',
                                 colWidths=[0.25, 0.25, 0.25, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        for i in range(len(summary_stats['Metric']) + 1):
            for j in range(4):
                cell = table[(i, j)]
                if i == 0:
                    cell.set_facecolor('#2E86AB')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#F0F0F0' if i % 2 == 0 else 'white')
    
    plt.suptitle('Pilot Results: Performance Metrics Overview', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'performance_metrics.png', dpi=300, bbox_inches='tight')
    print(f"✓ Chart: performance_metrics.png")
    plt.close()

def plot_model_comparison(results: List[Dict[str, Any]]):
    """Plot performance by model combination"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    model_data = {}
    for r in results:
        key = f"{r['llm_model'][:15]}\n({r['embedding_model'][:15]})"
        if key not in model_data:
            model_data[key] = {'latencies': [], 'similarities': []}
        
        sage_lat = r.get('sage_response', {}).get('latency')
        sim = _score_from_metric(r.get('similarity'))
        
        if sage_lat:
            model_data[key]['latencies'].append(sage_lat)
        if sim is not None:
            model_data[key]['similarities'].append(sim)
    
    # Plot latency by model
    if model_data:
        models = list(model_data.keys())
        avg_latencies = [np.mean(model_data[m]['latencies']) if model_data[m]['latencies'] else 0 for m in models]
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        axes[0].barh(models, avg_latencies, color=colors[:len(models)], alpha=0.8, edgecolor='black')
        axes[0].set_xlabel('Average Latency (seconds)', fontweight='bold')
        axes[0].set_title('SAGE Latency by Model', fontweight='bold')
        axes[0].grid(axis='x', alpha=0.3)
        
        # Plot similarity by model
        avg_similarities = [np.mean(model_data[m]['similarities']) if model_data[m]['similarities'] else 0 for m in models]
        axes[1].barh(models, avg_similarities, color=colors[:len(models)], alpha=0.8, edgecolor='black')
        axes[1].set_xlabel('Average Similarity', fontweight='bold')
        axes[1].set_title('Similarity Score by Model', fontweight='bold')
        axes[1].set_xlim([0, 1])
        axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Chart: model_comparison.png")
    plt.close()

def main():
    print("\n" + "="*60)
    print("PILOT RESULTS ANALYSIS")
    print("="*60 + "\n")
    
    # Load results
    results = load_results()
    print(f"Loaded {len(results)} results\n")
    
    # Export CSVs
    print("EXPORTING DATA TO CSV:")
    perf_df = create_performance_csv(results)
    summary_df = create_summary_csv(results)
    
    # Generate visualizations
    print("\nGENERATING VISUALIZATIONS:")
    plot_latency_comparison(results)
    plot_performance_metrics(results)
    plot_model_comparison(results)
    plot_all_metrics(results)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(summary_df.to_string(index=False))
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print(f"  📊 CSV: pilot_performance.csv ({len(perf_df)} rows)")
    print(f"  📊 CSV: pilot_summary.csv ({len(summary_df)} rows)")
    print(f"  📈 Chart: latency_comparison.png")
    print(f"  📈 Chart: performance_metrics.png")
    print(f"  📈 Chart: model_comparison.png")
    print(f"  📈 Chart: all_metrics_overview.png")
    print("\nAll files saved to: results/\n")

if __name__ == "__main__":
    main()
