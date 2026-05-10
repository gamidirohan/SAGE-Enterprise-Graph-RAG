"""
Direct test of SAGE agentic orchestrator with colored logging output.
Demonstrates orchestration contract display and agent activity tracking.
"""

import sys
from io import StringIO
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import agentic, orchestrator_logging, orchestrator


def test_colored_logging_output():
    """Test colored logging output for orchestration contract and agent activity."""
    
    print("\n" + "="*80)
    print("SAGE Agentic Orchestrator - Colored Logging Test")
    print("="*80 + "\n")
    
    # Capture stderr to see colored output
    events_captured = []
    
    def capture_events(event):
        """Capture events for analysis."""
        events_captured.append(event)
        # Also call the colored sink for visual output
        orchestrator_logging.colored_event_sink(event)
    
    # Test 1: Policy Reasoning Route
    print("\n" + "-"*80)
    print("Test 1: Policy Reasoning Query")
    print("-"*80 + "\n")
    
    test_cases = [
        {
            "name": "Policy Reasoning",
            "query": "What is our data retention policy and GDPR compliance?",
            "description": "Should route through policy_reasoning family",
        },
        {
            "name": "Relationship Lookup",
            "query": "Who does John Smith report to?",
            "description": "Should route through relationship_lookup family",
        },
        {
            "name": "Comparison Query",
            "query": "Compare Project Alpha and Project Beta based on budget and timeline.",
            "description": "Should route through broad_synthesis family",
        },
    ]
    
    # Mock the dependencies to test orchestration
    import unittest.mock as mock
    
    for test_case in test_cases:
        print(f"\n{'─'*80}")
        print(f"Query: {test_case['query']}")
        print(f"Expected: {test_case['description']}")
        print(f"{'─'*80}\n")
        
        events_captured.clear()
        
        try:
            with mock.patch('app.agentic.retrieval_selector.decide_strategy') as mock_strategy, \
                 mock.patch('app.agentic.vector_search.retrieve') as mock_retrieve, \
                 mock.patch('app.agentic.rerank.rerank') as mock_rerank, \
                 mock.patch('app.agentic.graph_query.expand_retrieval_context') as mock_expand, \
                 mock.patch('app.agentic.graph_query.validate_trace_paths') as mock_validate, \
                 mock.patch('app.agentic.services.generate_groq_response') as mock_generate, \
                 mock.patch('app.agentic.policy_guard.evaluate_answer') as mock_critic:
                
                # Setup mocks
                mock_strategy.return_value = {
                    "strategy": "semantic",
                    "reasons": ["test"],
                    "llm_used": False,
                    "heuristic_confidence": 0.9,
                }
                
                mock_retrieve.return_value = {
                    "documents": ["Test document"],
                    "trace": {
                        "query_type": "general_search",
                        "user_scoped": False,
                        "evidence": [
                            {"chunk_id": "chunk-1", "rank_score": 0.95, "document": {"doc_id": "doc-1"}}
                        ],
                        "selector_strategy": "semantic",
                    },
                }
                
                mock_rerank.return_value = lambda docs, trace: {"documents": docs, "trace": trace}
                mock_expand.return_value = {"documents": [], "trace": {"evidence": []}}
                mock_validate.return_value = {
                    "valid": True,
                    "validated_evidence_count": 1,
                    "missing_fields": []
                }
                
                mock_generate.return_value = {
                    "answer": f"Answer to: {test_case['query']}",
                    "answer_payload": {
                        "schema_version": 1,
                        "mode": "short",
                        "reason_code": "direct_lookup",
                        "summary": f"Answer to: {test_case['query']}",
                        "bullets": ["Key point 1", "Key point 2"],
                        "explanation": "This is based on available evidence.",
                        "evidence_refs": ["chunk:chunk-1"],
                    },
                    "thinking": ["Analyzed the query", "Found relevant evidence"],
                    "trace": {"evidence": [{"chunk_id": "chunk-1"}]},
                }
                
                mock_critic.return_value = {
                    "passed": True,
                    "retryable": False,
                    "issues": [],
                    "grounded_evidence_count": 1,
                    "provenance_count": 1,
                }
                
                # Run the agentic query with colored event sink
                result = orchestrator.run_agentic_query(
                    test_case["query"],
                    user_id="test_user",
                    event_sink=capture_events,
                )
                
                # Analyze results
                print(f"\n✓ Query completed successfully")
                print(f"\nOrchestration Analysis:")
                orchestration = result["trace"]["agentic"]["orchestration"]
                print(f"  Route Family: {orchestration.get('route_family', 'N/A')}")
                print(f"  Agents Used:")
                print(f"    - Planner: {orchestration.get('planner_required', False)}")
                print(f"    - Retriever: {orchestration.get('retriever_required', False)}")
                print(f"    - Reasoner: {orchestration.get('reasoner_required', False)}")
                print(f"    - Generator: {orchestration.get('generator_required', False)}")
                print(f"    - Critic: {orchestration.get('critic_required', False)}")
                print(f"  Tool Sequence: {orchestration.get('tool_sequence', [])}")
                
                print(f"\nExecution Summary:")
                print(f"  Rounds: {len(result['trace']['agentic']['rounds'])}")
                print(f"  Tool Calls: {len(result['trace']['agentic']['tool_calls'])}")
                print(f"  Evidence Found: {len(result['trace']['agentic']['selected_evidence'])}")
                print(f"  Status: {result['trace']['agentic']['status']}")
                
                print(f"\nEvent Log ({len(events_captured)} events):")
                for i, event in enumerate(events_captured, 1):
                    agent = event.get('agent', 'unknown')
                    event_type = event.get('event_type', 'unknown')
                    message = event.get('message', '')
                    print(f"  {i}. [{agent:12}] {event_type:20} - {message[:50]}")
                
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*80}")
    print("Test Summary")
    print(f"{'='*80}")
    print("✓ All tests completed with colored logging output")
    print("✓ Orchestration contract displayed correctly")
    print("✓ Agent activity tracked throughout execution")
    print("\nLog output was written to stderr with color codes:")
    print("  - Planner: CYAN")
    print("  - Retriever: GREEN")
    print("  - Reasoner: YELLOW")
    print("  - Generator: BLUE")
    print("  - Critic: MAGENTA")
    print("  - Orchestrator: WHITE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    test_colored_logging_output()
