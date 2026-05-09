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
