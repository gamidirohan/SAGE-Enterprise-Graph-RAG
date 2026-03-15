@echo off
echo Running Quick SAGE vs Traditional RAG Performance Comparison

set ROOT_DIR=%~dp0..
pushd "%ROOT_DIR%"

REM Activate virtual environment
call .venv\Scripts\activate

REM Process messages and generate QA pairs
echo Step 1: Processing messages from data/uploads folder...
python scripts\message_processor.py --directory data\uploads --num-pairs 10 --output data\eval\qa_pairs_quick.json

REM Run the performance comparison with a single model
echo Step 2: Running quick performance comparison...
python scripts\performance_comparison.py --queries data\eval\qa_pairs_quick.json --output results\quick_comparison_results.json --llm-models "llama3-8b-8192" --embedding-models "all-mpnet-base-v2"

REM Generate report
echo Step 3: Generating report...
python scripts\generate_report.py --results results\quick_comparison_results.json --output results\quick_performance_report.html

echo Done! Report generated at results\quick_performance_report.html
echo Opening report in browser...
start results\quick_performance_report.html

popd
pause
