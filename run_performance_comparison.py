"""
Script to run the performance comparison and generate a report.
"""

import os
import sys
import json
import argparse
import subprocess
from datetime import datetime

def run_tests():
    """Run the test script to check if everything is set up correctly"""
    print("Running tests to check if everything is set up correctly...")
    result = subprocess.run(
        [sys.executable, "test_performance_comparison.py"],
        capture_output=True,
        text=True
    )
    
    if "All tests passed!" in result.stdout:
        print("All tests passed! Proceeding with performance comparison.")
        return True
    else:
        print("Some tests failed. Please fix the issues before running the performance comparison.")
        print(result.stdout)
        return False

def run_comparison(queries_file=None, output_file=None, skip_tests=False):
    """Run the performance comparison"""
    # Create output directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Generate default filenames based on timestamp if not provided
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if not output_file:
        output_file = f"results/comparison_results_{timestamp}.json"
    
    # Run tests if not skipped
    if not skip_tests:
        if not run_tests():
            return False
    
    # Build command
    cmd = [sys.executable, "performance_comparison.py"]
    
    if queries_file:
        cmd.extend(["--queries", queries_file])
    
    cmd.extend(["--output", output_file])
    
    # Run comparison
    print(f"Running performance comparison...")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print(f"Performance comparison completed successfully!")
        print(f"Results saved to {output_file}")
        print(f"Visualization saved to performance_comparison_results.png")
        return True
    else:
        print(f"Performance comparison failed with exit code {result.returncode}")
        return False

def generate_html_report(results_file):
    """Generate an HTML report from the JSON results file"""
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Create HTML report filename
        html_file = results_file.replace('.json', '.html')
        
        # Calculate summary statistics
        sage_scores = [r["llm_evaluation"].get("system1_score", 0) for r in results]
        trad_scores = [r["llm_evaluation"].get("system2_score", 0) for r in results]
        
        better_system_counts = {
            "system1": sum(1 for r in results if r["llm_evaluation"].get("better_system") == "system1"),
            "system2": sum(1 for r in results if r["llm_evaluation"].get("better_system") == "system2"),
            "tie": sum(1 for r in results if r["llm_evaluation"].get("better_system") == "tie")
        }
        
        # Generate HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>SAGE vs Traditional RAG Performance Comparison</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .summary {{ background-color: #e9f7ef; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .query-section {{ margin-bottom: 30px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
                .metrics {{ display: flex; justify-content: space-between; }}
                .metric-box {{ background-color: #f8f9fa; padding: 10px; border-radius: 5px; width: 30%; }}
                .answer {{ background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 10px; }}
                .winner {{ font-weight: bold; color: green; }}
            </style>
        </head>
        <body>
            <h1>SAGE vs Traditional RAG Performance Comparison</h1>
            
            <div class="summary">
                <h2>Summary</h2>
                <p>Number of queries: {len(results)}</p>
                
                <h3>Average Scores</h3>
                <p>SAGE Graph RAG: {sum(sage_scores)/len(sage_scores):.2f}/10</p>
                <p>Traditional RAG: {sum(trad_scores)/len(trad_scores):.2f}/10</p>
                
                <h3>Better System Counts</h3>
                <p>SAGE Graph RAG better: {better_system_counts['system1']}</p>
                <p>Traditional RAG better: {better_system_counts['system2']}</p>
                <p>Tie: {better_system_counts['tie']}</p>
                
                <h3>Visualization</h3>
                <img src="performance_comparison_results.png" alt="Performance Comparison Results" style="max-width: 100%;">
            </div>
            
            <h2>Detailed Results</h2>
        """
        
        # Add each query result
        for i, result in enumerate(results):
            query = result["query"]
            sage_answer = result["sage_response"]["answer"]
            trad_answer = result["traditional_response"]["answer"]
            bert_f1 = result["bert_scores"].get("f1", 0)
            llm_eval = result["llm_evaluation"]
            
            winner = ""
            if llm_eval.get("better_system") == "system1":
                winner = "SAGE Graph RAG"
            elif llm_eval.get("better_system") == "system2":
                winner = "Traditional RAG"
            else:
                winner = "Tie"
            
            html_content += f"""
            <div class="query-section">
                <h3>Query {i+1}: {query}</h3>
                
                <div class="metrics">
                    <div class="metric-box">
                        <h4>LLM Evaluation</h4>
                        <p>SAGE Score: {llm_eval.get("system1_score", 0)}/10</p>
                        <p>Traditional Score: {llm_eval.get("system2_score", 0)}/10</p>
                        <p>Winner: <span class="winner">{winner}</span></p>
                    </div>
                    
                    <div class="metric-box">
                        <h4>BERT Score</h4>
                        <p>F1: {bert_f1:.4f}</p>
                    </div>
                    
                    <div class="metric-box">
                        <h4>Response Time</h4>
                        <p>SAGE: {result["sage_response"]["latency"]:.2f}s</p>
                        <p>Traditional: {result["traditional_response"]["latency"]:.2f}s</p>
                    </div>
                </div>
                
                <h4>SAGE Graph RAG Answer:</h4>
                <div class="answer">{sage_answer}</div>
                
                <h4>Traditional RAG Answer:</h4>
                <div class="answer">{trad_answer}</div>
                
                <h4>LLM Reasoning:</h4>
                <div class="answer">{llm_eval.get("reasoning", "No reasoning provided.")}</div>
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        # Write HTML file
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        print(f"HTML report generated: {html_file}")
        return html_file
    
    except Exception as e:
        print(f"Error generating HTML report: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Run performance comparison and generate report')
    parser.add_argument('--queries', type=str, help='Path to JSON file with test queries')
    parser.add_argument('--output', type=str, help='Path to output JSON file for results')
    parser.add_argument('--skip-tests', action='store_true', help='Skip running tests')
    parser.add_argument('--html', action='store_true', help='Generate HTML report')
    args = parser.parse_args()
    
    # Run comparison
    success = run_comparison(args.queries, args.output, args.skip_tests)
    
    if success and args.html:
        # Generate HTML report
        output_file = args.output or f"results/comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        generate_html_report(output_file)

if __name__ == "__main__":
    main()
