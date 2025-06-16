@echo off
echo Running Quick SAGE vs Traditional RAG Performance Comparison

REM Activate virtual environment
call myenv\Scripts\activate

REM Process messages and generate QA pairs
echo Step 1: Processing messages from uploads folder...
python message_processor.py --directory uploads --num-pairs 10 --output qa_pairs_quick.json

REM Run the performance comparison with a single model
echo Step 2: Running quick performance comparison...
python performance_comparison.py --queries qa_pairs_quick.json --output results/quick_comparison_results.json --llm-models "llama3-8b-8192" --embedding-models "all-mpnet-base-v2"

REM Generate report
echo Step 3: Generating report...
python generate_report.py --results results/quick_comparison_results.json --output results/quick_performance_report.html

echo Done! Report generated at results/quick_performance_report.html
echo Opening report in browser...
start results/quick_performance_report.html

pause
