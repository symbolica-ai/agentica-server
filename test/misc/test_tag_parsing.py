from agentic.monads.common import text_between, text_not_between


class TestTextBetween:
    def test_simple(self):
        assert list(text_between("a<t>b</t>c", "<t>", "</t>")) == ["b"]

    def test_multiple(self):
        assert list(text_between("<t>one</t> <t>two</t>", "<t>", "</t>")) == ["one", "two"]

    def test_nested(self):
        text = "before <tag>outer <tag>inner</tag> content</tag> after"
        assert list(text_between(text, "<tag>", "</tag>")) == ["outer <tag>inner</tag> content"]

    def test_deeply_nested(self):
        text = "<x><x><x>deep</x></x></x>"
        assert list(text_between(text, "<x>", "</x>")) == ["<x><x>deep</x></x>"]

    def test_nested_multiple(self):
        text = "<t><t>a</t></t> <t>b</t>"
        assert list(text_between(text, "<t>", "</t>")) == ["<t>a</t>", "b"]

    def test_no_matches(self):
        assert list(text_between("no tags here", "<t>", "</t>")) == []

    def test_empty_content(self):
        assert list(text_between("<t></t>", "<t>", "</t>")) == [""]

    def test_unmatched_start(self):
        assert list(text_between("<t>no end", "<t>", "</t>")) == []

    def test_empty_text(self):
        assert list(text_between("", "<t>", "</t>")) == []


class TestTextNotBetween:
    def test_simple(self):
        assert list(text_not_between("a<t>b</t>c", "<t>", "</t>")) == ["a", "c"]

    def test_multiple(self):
        assert list(text_not_between("x<t>one</t>y<t>two</t>z", "<t>", "</t>")) == ["x", "y", "z"]

    def test_nested(self):
        text = "before <tag>outer <tag>inner</tag> content</tag> after"
        assert list(text_not_between(text, "<tag>", "</tag>")) == ["before ", " after"]

    def test_deeply_nested(self):
        text = "a<x><x><x>deep</x></x></x>b"
        assert list(text_not_between(text, "<x>", "</x>")) == ["a", "b"]

    def test_no_tags(self):
        assert list(text_not_between("just text", "<t>", "</t>")) == ["just text"]

    def test_empty_outside(self):
        assert list(text_not_between("<t>all inside</t>", "<t>", "</t>")) == ["", ""]

    def test_empty_text(self):
        assert list(text_not_between("", "<t>", "</t>")) == [""]

    def test_unmatched_start(self):
        # Unmatched start means everything from ptr onwards is "not between"
        assert list(text_not_between("before <t>no end", "<t>", "</t>")) == ["before <t>no end"]


class TestMalformed:
    """Test behavior on malformed/edge-case inputs."""

    def test_only_end_tag(self):
        # End tag without start - should find nothing between
        assert list(text_between("</t>", "<t>", "</t>")) == []
        assert list(text_not_between("</t>", "<t>", "</t>")) == ["</t>"]

    def test_end_before_start(self):
        assert list(text_between("</t><t>content</t>", "<t>", "</t>")) == ["content"]
        assert list(text_not_between("</t><t>content</t>", "<t>", "</t>")) == ["</t>", ""]

    def test_extra_end_tags(self):
        # More ends than starts
        assert list(text_between("<t>a</t></t></t>", "<t>", "</t>")) == ["a"]
        assert list(text_not_between("<t>a</t></t></t>", "<t>", "</t>")) == ["", "</t></t>"]

    def test_adjacent_tags(self):
        assert list(text_between("<t></t><t></t>", "<t>", "</t>")) == ["", ""]

    def test_partial_tag(self):
        # Incomplete tags shouldn't match
        assert list(text_between("<t>content</t", "<t>", "</t>")) == []
        assert list(text_between("<tcontent</t>", "<t>", "</t>")) == []


class TestOverlappingTagPatterns:
    """Test when end tag is a prefix/suffix of start tag (like markdown fences)."""

    def test_markdown_fence_simple(self):
        # ```python ... ``` where ``` is prefix of ```python
        text = "```python\nprint('hi')\n```"
        result = list(text_between(text, "```python", "```"))
        # This finds ``` at position 0 which is part of ```python - tricky!
        assert result == ["\nprint('hi')\n"]

    def test_markdown_fence_nested(self):
        # Nested code blocks - the inner ```python looks like end tag
        text = "```python\nouter\n```python\ninner\n```\nmore\n```"
        result = list(text_between(text, "```python", "```"))
        # The ``` in ```python might be found as end - let's see what happens
        # This documents current behavior (may not be "correct")
        assert result == ["\nouter\n```python\ninner\n```\nmore\n"]

    def test_markdown_fence_not_between(self):
        text = "before\n```python\ncode\n```\nafter"
        result = list(text_not_between(text, "```python", "```"))
        assert result == ["before\n", "\nafter"]

    def test_end_is_prefix_of_start(self):
        # More explicit test: end="[" start="[tag]"
        text = "x[tag]content[y"
        result = list(text_between(text, "[tag]", "["))
        assert result == ["content"]

    def test_start_ends_with_end(self):
        # start="<<END" end="END"
        text = "a<<ENDcontentENDb"
        result = list(text_between(text, "<<END", "END"))
        assert result == ["content"]

    def test_identical_start_end(self):
        # Edge case: same delimiter (like | for tables)
        # LIMITATION: when start == end, each end also looks like a start,
        # so nesting depth never decreases and we only get first match
        text = "|cell1|cell2|"
        result = list(text_between(text, "|", "|"))
        assert result == ["cell1"]  # only first cell, second | is consumed as "end"

    def test_identical_start_end_not_between(self):
        text = "|cell1|cell2|"
        result = list(text_not_between(text, "|", "|"))
        # "" before first |, then "cell2|" (trailing | is unmatched, included in output)
        assert result == ["", "cell2|"]

    def test_identical_start_end_not_shared(self):
        text = "|cell1||cell2|"
        result = list(text_between(text, "|", "|"))
        assert result == ["cell1", "cell2"]
