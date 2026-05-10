from app import query_shape


def test_analyze_query_marks_various_plural_lookup_as_multi_item():
    profile = query_shape.analyze_query("Who are the various managers we have?")

    assert profile["expects_multiple_items"] is True
    assert profile["requires_broad_coverage"] is True
    assert profile["wants_list_format"] is True
    assert profile["minimum_unique_evidence"] == 2
    assert profile["minimum_tool_rounds"] == 2


def test_analyze_query_keeps_single_lookup_as_single_item():
    profile = query_shape.analyze_query("Who is my manager?")

    assert profile["expects_multiple_items"] is False
    assert profile["minimum_unique_evidence"] == 1


def test_analyze_query_marks_comparison_as_broad_multi_item():
    profile = query_shape.analyze_query(
        "Compare Project Beta and Project Gamma based on the chat history."
    )

    assert profile["expects_multiple_items"] is True
    assert profile["requires_broad_coverage"] is True
    assert profile["minimum_unique_evidence"] == 2
    assert profile["minimum_tool_rounds"] == 2


def test_recommend_graph_depth_keeps_direct_document_lookup_shallow():
    depth = query_shape.recommend_graph_depth(
        "What is the office address for HQ?",
        query_type="general_search",
    )

    assert depth["seed_hops"] == 0
    assert depth["expand_hops"] == 1
    assert depth["max_hops"] == 2


def test_recommend_graph_depth_marks_relationship_lookup_as_medium():
    depth = query_shape.recommend_graph_depth(
        "Who does Rohan report to?",
        query_type="person_lookup",
    )

    assert depth["seed_hops"] == 1
    assert depth["expand_hops"] == 2
    assert depth["max_hops"] == 3


def test_recommend_graph_depth_marks_broad_and_policy_queries_as_deep():
    broad_depth = query_shape.recommend_graph_depth(
        "Compare Project Beta and Project Gamma based on the chat history.",
        query_type="general_search",
    )
    policy_depth = query_shape.recommend_graph_depth(
        "Is the API we use compliant with policy A?",
        query_type="general_search",
    )

    assert broad_depth["expand_hops"] == 3
    assert policy_depth["expand_hops"] == 3
