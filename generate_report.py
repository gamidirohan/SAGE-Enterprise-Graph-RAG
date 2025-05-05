"""
Generate a comprehensive HTML report from performance comparison results.
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
from datetime import datetime

def load_results(results_file: str) -> List[Dict[str, Any]]:
    """Load results from a JSON file."""
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        return results
    except Exception as e:
        print(f"Error loading results file: {str(e)}")
        return []

def group_results_by_model(results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group results by model combination."""
    grouped = {}

    for result in results:
        if "llm_model" in result and "embedding_model" in result:
            key = f"{result['llm_model']}_{result['embedding_model']}"
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(result)

    return grouped

def calculate_model_metrics(grouped_results: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
    """Calculate metrics for each model combination."""
    metrics = []

    for model_key, results in grouped_results.items():
        llm_model, embedding_model = model_key.split('_')

        # Extract metrics
        sage_scores = [r.get("llm_evaluation", {}).get("system1_score", 0) for r in results]
        trad_scores = [r.get("llm_evaluation", {}).get("system2_score", 0) for r in results]

        better_system_counts = {
            "system1": sum(1 for r in results if r.get("llm_evaluation", {}).get("better_system") == "system1"),
            "system2": sum(1 for r in results if r.get("llm_evaluation", {}).get("better_system") == "system2"),
            "tie": sum(1 for r in results if r.get("llm_evaluation", {}).get("better_system") == "tie"),
            "error": sum(1 for r in results if r.get("llm_evaluation", {}).get("better_system") == "error")
        }

        sage_latencies = [r.get("sage_response", {}).get("latency", 0) for r in results]
        trad_latencies = [r.get("traditional_response", {}).get("latency", 0) for r in results]

        similarity_f1_scores = [r.get("similarity_scores", {}).get("f1", 0) for r in results]

        metrics.append({
            "llm_model": llm_model,
            "embedding_model": embedding_model,
            "sage_avg_score": np.mean(sage_scores),
            "trad_avg_score": np.mean(trad_scores),
            "sage_better_count": better_system_counts["system1"],
            "trad_better_count": better_system_counts["system2"],
            "tie_count": better_system_counts["tie"],
            "error_count": better_system_counts["error"],
            "sage_avg_latency": np.mean(sage_latencies),
            "trad_avg_latency": np.mean(trad_latencies),
            "avg_similarity_f1": np.mean(similarity_f1_scores),
            "num_queries": len(results)
        })

    return pd.DataFrame(metrics)

def create_single_model_comparison(metrics_df: pd.DataFrame, output_dir: str):
    """Create a single chart for a single model comparison."""
    plt.figure(figsize=(10, 6))

    # Extract data for the single model
    if len(metrics_df) == 1:
        row = metrics_df.iloc[0]
        model_name = f"{row['llm_model']}, {row['embedding_model']}"

        # Create bar chart comparing SAGE and Traditional RAG
        data = {
            'SAGE': [row['sage_avg_score']],
            'Traditional': [row['trad_avg_score']]
        }

        # Create DataFrame
        df = pd.DataFrame(data, index=[model_name])

        # Plot
        df.plot(kind='bar', figsize=(10, 6))
        plt.title(f'SAGE vs Traditional RAG Performance\n{model_name}')
        plt.ylabel('Score (out of 10)')
        plt.grid(axis='y', alpha=0.3)

        # Add counts as text
        plt.text(0.1, 0.85, f"SAGE better: {row['sage_better_count']}", transform=plt.gca().transAxes)
        plt.text(0.1, 0.80, f"Traditional better: {row['trad_better_count']}", transform=plt.gca().transAxes)
        plt.text(0.1, 0.75, f"Ties: {row['tie_count']}", transform=plt.gca().transAxes)
        plt.text(0.1, 0.65, f"SAGE latency: {row['sage_avg_latency']:.2f}s", transform=plt.gca().transAxes)
        plt.text(0.1, 0.60, f"Traditional latency: {row['trad_avg_latency']:.2f}s", transform=plt.gca().transAxes)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/single_model_comparison.png")
        plt.close()
    else:
        # Create a placeholder image
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "No model data available", ha='center', va='center', fontsize=14)
        plt.axis('off')
        plt.savefig(f"{output_dir}/single_model_comparison.png")
        plt.close()

def create_model_comparison_charts(metrics_df: pd.DataFrame, output_dir: str):
    """Create charts comparing different model combinations."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Check if we have enough data for model comparisons
    if len(metrics_df) <= 1:
        # Create single model comparison chart
        create_single_model_comparison(metrics_df, output_dir)
        return {
            "model_score_comparison": "single_model_comparison.png",
            "model_latency_comparison": "single_model_comparison.png",
            "model_counts_comparison": "single_model_comparison.png",
            "sage_score_heatmap": "single_model_comparison.png",
            "trad_score_heatmap": "single_model_comparison.png"
        }

    # 1. Score comparison by model
    plt.figure(figsize=(12, 8))

    # Create a new DataFrame with model names as index and scores as columns
    score_df = pd.DataFrame({
        'SAGE': metrics_df['sage_avg_score'],
        'Traditional': metrics_df['trad_avg_score']
    }, index=[f"{row['llm_model']}\n{row['embedding_model']}" for _, row in metrics_df.iterrows()])

    score_df.plot(kind='bar', figsize=(12, 8))
    plt.title('Average Scores by Model Combination')
    plt.ylabel('Score (out of 10)')
    plt.xlabel('Model Combination')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_score_comparison.png")
    plt.close()

    # 2. Latency comparison by model
    plt.figure(figsize=(12, 8))

    # Create a new DataFrame with model names as index and latencies as columns
    latency_df = pd.DataFrame({
        'SAGE': metrics_df['sage_avg_latency'],
        'Traditional': metrics_df['trad_avg_latency']
    }, index=[f"{row['llm_model']}\n{row['embedding_model']}" for _, row in metrics_df.iterrows()])

    latency_df.plot(kind='bar', figsize=(12, 8))
    plt.title('Average Latency by Model Combination')
    plt.ylabel('Latency (seconds)')
    plt.xlabel('Model Combination')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_latency_comparison.png")
    plt.close()

    # 3. Better system counts by model
    plt.figure(figsize=(12, 8))

    # Create a new DataFrame with model names as index and counts as columns
    counts_df = pd.DataFrame({
        'SAGE Better': metrics_df['sage_better_count'],
        'Traditional Better': metrics_df['trad_better_count'],
        'Tie': metrics_df['tie_count']
    }, index=[f"{row['llm_model']}\n{row['embedding_model']}" for _, row in metrics_df.iterrows()])

    counts_df.plot(kind='bar', stacked=True, figsize=(12, 8))
    plt.title('Better System Counts by Model Combination')
    plt.ylabel('Count')
    plt.xlabel('Model Combination')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_counts_comparison.png")
    plt.close()

    # 4. Heatmap of SAGE scores by LLM and embedding model
    plt.figure(figsize=(10, 8))

    # Create a pivot table with LLM models as rows and embedding models as columns
    heatmap_data = metrics_df.pivot_table(
        values='sage_avg_score',
        index='llm_model',
        columns='embedding_model'
    )

    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", vmin=0, vmax=10)
    plt.title('SAGE Scores by Model Combination')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sage_score_heatmap.png")
    plt.close()

    # 5. Heatmap of Traditional scores by LLM and embedding model
    plt.figure(figsize=(10, 8))

    # Create a pivot table with LLM models as rows and embedding models as columns
    heatmap_data = metrics_df.pivot_table(
        values='trad_avg_score',
        index='llm_model',
        columns='embedding_model'
    )

    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", vmin=0, vmax=10)
    plt.title('Traditional Scores by Model Combination')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/trad_score_heatmap.png")
    plt.close()

    return {
        "model_score_comparison": "model_score_comparison.png",
        "model_latency_comparison": "model_latency_comparison.png",
        "model_counts_comparison": "model_counts_comparison.png",
        "sage_score_heatmap": "sage_score_heatmap.png",
        "trad_score_heatmap": "trad_score_heatmap.png"
    }

def generate_html_report(results: List[Dict[str, Any]], metrics_df: pd.DataFrame, charts: Dict[str, str], output_file: str):
    """Generate an HTML report from the results."""
    # Group results by model
    grouped_results = group_results_by_model(results)

    # Get timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Start HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>SAGE vs Traditional RAG Performance Comparison Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
            h1, h2, h3, h4 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .summary {{ background-color: #e9f7ef; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .model-section {{ margin-bottom: 30px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
            .metrics {{ display: flex; justify-content: space-between; flex-wrap: wrap; }}
            .metric-box {{ background-color: #f8f9fa; padding: 10px; border-radius: 5px; width: 30%; margin-bottom: 10px; }}
            .chart-container {{ margin: 20px 0; }}
            .chart {{ max-width: 100%; height: auto; }}
            .winner {{ font-weight: bold; color: green; }}
            .tab {{ overflow: hidden; border: 1px solid #ccc; background-color: #f1f1f1; }}
            .tab button {{ background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; }}
            .tab button:hover {{ background-color: #ddd; }}
            .tab button.active {{ background-color: #ccc; }}
            .tabcontent {{ display: none; padding: 6px 12px; border: 1px solid #ccc; border-top: none; }}
        </style>
        <script>
            function openTab(evt, tabName) {{
                var i, tabcontent, tablinks;
                tabcontent = document.getElementsByClassName("tabcontent");
                for (i = 0; i < tabcontent.length; i++) {{
                    tabcontent[i].style.display = "none";
                }}
                tablinks = document.getElementsByClassName("tablinks");
                for (i = 0; i < tablinks.length; i++) {{
                    tablinks[i].className = tablinks[i].className.replace(" active", "");
                }}
                document.getElementById(tabName).style.display = "block";
                evt.currentTarget.className += " active";
            }}

            // Set the default tab to open when the page loads
            window.onload = function() {{
                document.getElementsByClassName("tablinks")[0].click();
            }};
        </script>
    </head>
    <body>
        <h1>SAGE vs Traditional RAG Performance Comparison Report</h1>
        <p>Generated on: {timestamp}</p>

        <div class="summary">
            <h2>Executive Summary</h2>
            <p>This report compares the performance of SAGE (Graph-based RAG) with Traditional RAG across different LLM and embedding models.</p>

            <h3>Key Findings</h3>
            <ul>
                <li>Number of model combinations tested: {len(grouped_results)}</li>
                <li>Total number of queries: {len(results)}</li>
                <li>Best performing LLM model for SAGE: {metrics_df.loc[metrics_df['sage_avg_score'].idxmax()]['llm_model']}</li>
                <li>Best performing embedding model for SAGE: {metrics_df.loc[metrics_df['sage_avg_score'].idxmax()]['embedding_model']}</li>
                <li>Overall winner: {"SAGE Graph RAG" if metrics_df['sage_avg_score'].mean() > metrics_df['trad_avg_score'].mean() else "Traditional RAG"}</li>
            </ul>
        </div>

        <h2>Model Comparison</h2>

        <div class="chart-container">
            <h3>Average Scores by Model</h3>
            <img src="{charts['model_score_comparison']}" alt="Model Score Comparison" class="chart">
        </div>

        <div class="chart-container">
            <h3>Average Latency by Model</h3>
            <img src="{charts['model_latency_comparison']}" alt="Model Latency Comparison" class="chart">
        </div>

        <div class="chart-container">
            <h3>Better System Counts by Model</h3>
            <img src="{charts['model_counts_comparison']}" alt="Model Counts Comparison" class="chart">
        </div>

        <div class="chart-container">
            <h3>SAGE Scores Heatmap</h3>
            <img src="{charts['sage_score_heatmap']}" alt="SAGE Score Heatmap" class="chart">
        </div>

        <div class="chart-container">
            <h3>Traditional Scores Heatmap</h3>
            <img src="{charts['trad_score_heatmap']}" alt="Traditional Score Heatmap" class="chart">
        </div>

        <h2>Detailed Metrics</h2>

        <table>
            <tr>
                <th>LLM Model</th>
                <th>Embedding Model</th>
                <th>SAGE Avg Score</th>
                <th>Traditional Avg Score</th>
                <th>SAGE Better Count</th>
                <th>Traditional Better Count</th>
                <th>Tie Count</th>
                <th>SAGE Avg Latency</th>
                <th>Traditional Avg Latency</th>
                <th>Avg Similarity F1</th>
            </tr>
    """

    # Add rows for each model combination
    for _, row in metrics_df.iterrows():
        html_content += f"""
            <tr>
                <td>{row['llm_model']}</td>
                <td>{row['embedding_model']}</td>
                <td>{row['sage_avg_score']:.2f}</td>
                <td>{row['trad_avg_score']:.2f}</td>
                <td>{row['sage_better_count']}</td>
                <td>{row['trad_better_count']}</td>
                <td>{row['tie_count']}</td>
                <td>{row['sage_avg_latency']:.2f}s</td>
                <td>{row['trad_avg_latency']:.2f}s</td>
                <td>{row['avg_similarity_f1']:.4f}</td>
            </tr>
        """

    html_content += """
        </table>

        <h2>Detailed Results by Model</h2>

        <div class="tab">
    """

    # Add tabs for each model combination
    for i, model_key in enumerate(grouped_results.keys()):
        html_content += f"""
            <button class="tablinks" onclick="openTab(event, 'model{i}')">{model_key}</button>
        """

    html_content += """
        </div>
    """

    # Add tab content for each model combination
    for i, (model_key, model_results) in enumerate(grouped_results.items()):
        llm_model, embedding_model = model_key.split('_')

        html_content += f"""
        <div id="model{i}" class="tabcontent">
            <h3>Results for {llm_model}, {embedding_model}</h3>

            <div class="metrics">
                <div class="metric-box">
                    <h4>Average Scores</h4>
                    <p>SAGE: {np.mean([r.get('llm_evaluation', {}).get('system1_score', 0) for r in model_results]):.2f}/10</p>
                    <p>Traditional: {np.mean([r.get('llm_evaluation', {}).get('system2_score', 0) for r in model_results]):.2f}/10</p>
                </div>

                <div class="metric-box">
                    <h4>Better System Counts</h4>
                    <p>SAGE better: {sum(1 for r in model_results if r.get('llm_evaluation', {}).get('better_system') == 'system1')}</p>
                    <p>Traditional better: {sum(1 for r in model_results if r.get('llm_evaluation', {}).get('better_system') == 'system2')}</p>
                    <p>Tie: {sum(1 for r in model_results if r.get('llm_evaluation', {}).get('better_system') == 'tie')}</p>
                </div>

                <div class="metric-box">
                    <h4>Average Latency</h4>
                    <p>SAGE: {np.mean([r.get('sage_response', {}).get('latency', 0) for r in model_results]):.2f}s</p>
                    <p>Traditional: {np.mean([r.get('traditional_response', {}).get('latency', 0) for r in model_results]):.2f}s</p>
                </div>
            </div>

            <h4>Query Results</h4>
            <table>
                <tr>
                    <th>Query</th>
                    <th>SAGE Answer</th>
                    <th>Traditional Answer</th>
                    <th>Better System</th>
                    <th>SAGE Score</th>
                    <th>Traditional Score</th>
                </tr>
        """

        # Add rows for each query
        for result in model_results:
            query = result.get('query', 'Unknown')
            sage_answer = result.get('sage_response', {}).get('answer', 'No answer')
            trad_answer = result.get('traditional_response', {}).get('answer', 'No answer')
            better_system = result.get('llm_evaluation', {}).get('better_system', 'unknown')
            sage_score = result.get('llm_evaluation', {}).get('system1_score', 0)
            trad_score = result.get('llm_evaluation', {}).get('system2_score', 0)

            # Determine winner class
            sage_class = "winner" if better_system == "system1" else ""
            trad_class = "winner" if better_system == "system2" else ""

            html_content += f"""
                <tr>
                    <td>{query[:50]}...</td>
                    <td class="{sage_class}">{sage_answer[:100]}...</td>
                    <td class="{trad_class}">{trad_answer[:100]}...</td>
                    <td>{"SAGE" if better_system == "system1" else "Traditional" if better_system == "system2" else "Tie"}</td>
                    <td>{sage_score}</td>
                    <td>{trad_score}</td>
                </tr>
            """

        html_content += """
            </table>
        </div>
        """

    # Close HTML
    html_content += """
    </body>
    </html>
    """

    # Write HTML file
    with open(output_file, 'w') as f:
        f.write(html_content)

    print(f"HTML report generated: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate a report from performance comparison results')
    parser.add_argument('--results', type=str, default='results/full_comparison_results.json',
                        help='Path to results JSON file')
    parser.add_argument('--output', type=str, default='results/performance_report.html',
                        help='Path to output HTML file')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Load results
    results = load_results(args.results)
    if not results:
        print("No results to process. Exiting.")
        return

    # Group results by model
    grouped_results = group_results_by_model(results)

    # Calculate metrics
    metrics_df = calculate_model_metrics(grouped_results)

    # Create charts
    charts = create_model_comparison_charts(metrics_df, "results")

    # Generate HTML report
    generate_html_report(results, metrics_df, charts, args.output)

if __name__ == "__main__":
    main()
