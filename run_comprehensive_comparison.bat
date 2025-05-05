@echo off
echo Running Comprehensive SAGE vs Traditional RAG Performance Comparison

REM Activate virtual environment
call myenv\Scripts\activate

REM Create results directory if it doesn't exist
if not exist results mkdir results

REM Process messages and generate QA pairs
echo Step 1: Processing messages from uploads folder...
python message_processor.py --directory uploads --num-pairs 30 --output qa_pairs.json

REM Run the performance comparison with a subset of models
echo Step 2: Running initial performance comparison with a subset of models...
python performance_comparison.py --queries qa_pairs.json --output results/initial_comparison_results.json --llm-models "gemma2-9b-it,llama3-8b-8192" --embedding-models "all-mpnet-base-v2,multi-qa-mpnet-base-dot-v1"

REM Generate initial report
echo Step 3: Generating initial report...
python generate_report.py --results results/initial_comparison_results.json --output results/initial_performance_report.html

REM Open the initial report
start results/initial_performance_report.html

echo Done with initial comparison! Check the report for results.
echo.
echo Press any key to continue with the full model comparison (this will take longer)...
pause

REM Run the performance comparison with all models
echo Step 4: Running full performance comparison with all models...
python performance_comparison.py --queries qa_pairs.json --output results/full_comparison_results.json --llm-models "gemma2-9b-it,llama-guard-3-8b,mistral-saba-24b,llama3-8b-8192,compound-beta-mini,deepseek-r1-distill-llama-70b,llama-3.3-70b-versatile,llama3-70b-8192,llama-3.1-8b-instant" --embedding-models "all-mpnet-base-v2,all-MiniLM-L6-v2,multi-qa-mpnet-base-dot-v1,all-distilroberta-v1,paraphrase-multilingual-mpnet-base-v2"

REM Generate comprehensive report
echo Step 5: Generating comprehensive report...
python generate_report.py --results results/full_comparison_results.json --output results/full_performance_report.html

echo Done! Full report generated at results/full_performance_report.html
echo Opening report in browser...
start results/full_performance_report.html

pause
