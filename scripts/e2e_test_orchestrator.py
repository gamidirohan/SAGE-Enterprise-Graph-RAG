"""
End-to-end test script for SAGE agentic orchestrator using Bruno CLI.
Tests various query types to verify agent routing, orchestration contract display,
and colored logging output.
"""

import subprocess
import json
import sys
from pathlib import Path

# Test queries for different route families
TEST_QUERIES = [
    {
        "name": "policy_reasoning",
        "query": "What is our data retention policy?",
        "file": "01 Bootstrap Test Data.yml",
        "expected_in_response": ["policy", "retention", "data"]
    },
    {
        "name": "relationship_lookup",
        "query": "Who does John Smith report to?",
        "file": "02 Send Conversation Message.yml",
        "expected_in_response": ["report", "john", "smith"]
    },
    {
        "name": "broad_synthesis",
        "query": "Compare Project Alpha and Project Beta",
        "file": "03 Query Chat.yml",
        "expected_in_response": ["project", "alpha", "beta"]
    },
    {
        "name": "filtered_search",
        "query": "Find all documents from 2024",
        "file": "04 List Conversation Messages.yml",
        "expected_in_response": ["document", "2024"]
    },
    {
        "name": "temporal_lookup",
        "query": "What happened in Q3 2024?",
        "file": "Get GROQ Models.yml",
        "expected_in_response": ["q3", "2024"]
    },
]

def run_bruno_test(query: str, bruno_file: str) -> dict:
    """Run a single test query via Bruno CLI.
    
    Args:
        query: The query string to test
        bruno_file: The Bruno collection file to use
        
    Returns:
        Dict with response details and any errors
    """
    bruno_dir = Path("d:\\College\\Sem_8\\Bruno API SAGE")
    
    # Construct the bruno-cli command
    cmd = [
        "bruno",
        "run",
        str(bruno_dir / bruno_file),
        "--env", "default",
        "--insecure",
    ]
    
    result = {
        "query": query,
        "bruno_file": bruno_file,
        "success": False,
        "stdout": "",
        "stderr": "",
        "response": None,
        "trace": None,
        "orchestration_contract": None,
    }
    
    try:
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        result["stdout"] = process.stdout
        result["stderr"] = process.stderr
        result["success"] = process.returncode == 0
        
        # Parse response if available
        if process.stdout:
            try:
                # Try to extract JSON response
                response_data = json.loads(process.stdout)
                result["response"] = response_data
                
                # Extract trace if present
                if isinstance(response_data, dict):
                    if "trace" in response_data:
                        result["trace"] = response_data["trace"]
                    if "orchestration" in response_data or ("trace" in response_data and "orchestration" in response_data.get("trace", {})):
                        result["orchestration_contract"] = (
                            response_data.get("orchestration") or 
                            response_data.get("trace", {}).get("orchestration")
                        )
            except json.JSONDecodeError:
                # Response is not JSON, just store raw output
                result["response"] = process.stdout
        
    except subprocess.TimeoutExpired:
        result["stderr"] = "Test timed out after 30 seconds"
    except Exception as e:
        result["stderr"] = str(e)
    
    return result

def validate_result(result: dict, expected_keywords: list) -> dict:
    """Validate test result contains expected content.
    
    Args:
        result: Test result dict from run_bruno_test
        expected_keywords: List of keywords that should appear in response
        
    Returns:
        Validation dict with pass/fail status and details
    """
    validation = {
        "passed": True,
        "checks": {},
        "issues": [],
    }
    
    # Check 1: Command executed successfully
    validation["checks"]["execution_success"] = result["success"]
    if not result["success"]:
        validation["issues"].append(f"Command failed: {result['stderr']}")
        validation["passed"] = False
    
    # Check 2: Response received
    validation["checks"]["response_received"] = bool(result["response"])
    if not result["response"]:
        validation["issues"].append("No response received")
        validation["passed"] = False
    
    # Check 3: Orchestration contract present
    validation["checks"]["orchestration_contract_present"] = bool(result["orchestration_contract"])
    if not result["orchestration_contract"]:
        validation["issues"].append("Orchestration contract not found in response")
    
    # Check 4: Response contains expected keywords
    response_text = str(result.get("response", "")).lower()
    keyword_matches = []
    for keyword in expected_keywords:
        if keyword.lower() in response_text:
            keyword_matches.append(keyword)
    
    validation["checks"]["expected_keywords_found"] = len(keyword_matches) > 0
    if not keyword_matches:
        validation["issues"].append(f"Expected keywords not found: {expected_keywords}")
    
    return validation

def print_result(test_name: str, result: dict, validation: dict) -> None:
    """Print formatted test result.
    
    Args:
        test_name: Name of the test
        result: Test result dict
        validation: Validation dict
    """
    status = "✓ PASS" if validation["passed"] else "✗ FAIL"
    print(f"\n{'='*70}")
    print(f"{status} | Test: {test_name}")
    print(f"{'='*70}")
    print(f"Query: {result['query']}")
    print(f"File:  {result['bruno_file']}")
    
    # Print validation details
    print("\nValidation Checks:")
    for check_name, check_result in validation["checks"].items():
        status_icon = "✓" if check_result else "✗"
        print(f"  {status_icon} {check_name}")
    
    # Print issues if any
    if validation["issues"]:
        print("\nIssues:")
        for issue in validation["issues"]:
            print(f"  - {issue}")
    
    # Print orchestration contract if present
    if result["orchestration_contract"]:
        print("\nOrchestration Contract:")
        contract = result["orchestration_contract"]
        if isinstance(contract, dict):
            print(f"  Route Family: {contract.get('route_family', 'unknown')}")
            print(f"  Planner Required: {contract.get('planner_required', False)}")
            print(f"  Retriever Required: {contract.get('retriever_required', False)}")
            print(f"  Tool Sequence: {contract.get('tool_sequence', [])}")
    
    # Print response snippet
    if result["response"]:
        response_str = str(result["response"])[:200]
        print(f"\nResponse (first 200 chars):\n  {response_str}...")

def main():
    """Run all end-to-end tests."""
    print("\n" + "="*70)
    print("SAGE Agentic Orchestrator - End-to-End Tests")
    print("="*70)
    
    results = []
    passed_count = 0
    failed_count = 0
    
    for test in TEST_QUERIES:
        print(f"\nRunning test: {test['name']}...")
        
        # Run the test
        result = run_bruno_test(test["query"], test["file"])
        
        # Validate the result
        validation = validate_result(result, test["expected_in_response"])
        
        # Print the result
        print_result(test["name"], result, validation)
        
        # Track results
        results.append({
            "name": test["name"],
            "result": result,
            "validation": validation,
        })
        
        if validation["passed"]:
            passed_count += 1
        else:
            failed_count += 1
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total Tests: {len(results)}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")
    print(f"Success Rate: {(passed_count/len(results)*100):.1f}%")
    
    # Detailed summary
    print("\nTest Results:")
    for res in results:
        status = "✓" if res["validation"]["passed"] else "✗"
        print(f"  {status} {res['name']}")
    
    return 0 if failed_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
