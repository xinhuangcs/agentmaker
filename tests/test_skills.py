"""Skill loading and frontmatter parsing regression (hermetic).

Covers frontmatter parsing edge cases (normal / none / unclosed / a key:value in the body not misparsed / multiline scalar), SkillLoader discover / catalog / load with fail-loud on duplicate names, and progressive disclosure (discover does not read the body, load does).
"""

import pytest

from agentmaker.skills.loader import SkillLoader, _parse_frontmatter, _parse_meta


# ---------- frontmatter parsing ----------

def test_parse_frontmatter_normal():
    """Standard frontmatter -> (metadata, body); everything after the closing --- is the body."""
    meta, body = _parse_frontmatter("---\nname: daily\ndescription: 整理待办\n---\n步骤一\n步骤二\n")
    assert meta == {"name": "daily", "description": "整理待办"}
    assert body == "步骤一\n步骤二\n"


def test_parse_frontmatter_none():
    """First line is not --- -> empty metadata, whole text is the body (a key:value in the body is not mistaken for frontmatter)."""
    meta, body = _parse_frontmatter("# 标题\nname: 不是 frontmatter\n正文\n")
    assert meta == {} and body.startswith("# 标题")


def test_parse_frontmatter_unclosed():
    """Missing closing --- -> empty metadata, whole text is the body (does not swallow everything after as frontmatter)."""
    meta, body = _parse_frontmatter("---\nname: x\n正文没有闭合\n")
    assert meta == {} and "正文没有闭合" in body


def test_parse_meta_strips_quotes_and_skips_comments():
    """Single-line scalars: strip quotes, skip blank lines and full-line comments."""
    assert _parse_meta('name: "daily"\n# 注释\n\ndescription: \'整理\'') == {"name": "daily", "description": "整理"}


def test_parse_meta_multiline_folded_scalar():
    """Multiline / folded scalar (description: >-): parses correctly when pyyaml is installed; otherwise raises a clear ValueError with the path (never silently garbled)."""
    import importlib.util
    front = "name: daily\ndescription: >-\n  把零散待办\n  整理成计划\n"
    if importlib.util.find_spec("yaml") is None:           # no pyyaml: falls back to the handwritten subset, multiline scalar raises a clear ValueError
        with pytest.raises(ValueError, match="pyyaml"):
            _parse_meta(front, path="x/SKILL.md")
    else:                                                  # pyyaml present: folded scalar parses correctly
        meta = _parse_meta(front)
        assert meta["name"] == "daily" and "整理成计划" in meta["description"]


# ---------- SkillLoader ----------

def _make_skill(root, name, *, front_name=None, description="做某事", body="正文内容"):
    """Create a skill directory with a SKILL.md under root."""
    d = root / name
    d.mkdir()
    fn = front_name if front_name is not None else name
    (d / "SKILL.md").write_text(f"---\nname: {fn}\ndescription: {description}\n---\n{body}\n", encoding="utf-8")
    return d


def test_discover_and_load(tmp_path):
    """discover lists skills (sorted by directory name, body left empty = progressive-disclosure layer 1); load reads the body (layer 2)."""
    _make_skill(tmp_path, "b-skill", description="第二个")
    _make_skill(tmp_path, "a-skill", description="第一个", body="A 的步骤")
    (tmp_path / "not-a-skill").mkdir()                      # a directory without SKILL.md is skipped
    loader = SkillLoader(str(tmp_path))
    skills = loader.discover()
    assert [s.name for s in skills] == ["a-skill", "b-skill"]   # sorted
    assert all(s.body == "" for s in skills)                    # discover does not read the body
    assert loader.load("a-skill") == "A 的步骤\n"               # load does
    assert loader.load("不存在") is None


def test_catalog_format(tmp_path):
    """catalog renders "- name: description" listing text (placed in the system prompt for the model to choose from)."""
    _make_skill(tmp_path, "plan", description="规划今天")
    assert SkillLoader(str(tmp_path)).catalog() == "- plan: 规划今天"


def test_duplicate_name_fail_loud(tmp_path):
    """Two skills with the same name (explicit duplicate in frontmatter) -> discover raises ValueError (no silent ambiguity)."""
    _make_skill(tmp_path, "dir1", front_name="same")
    _make_skill(tmp_path, "dir2", front_name="same")
    with pytest.raises(ValueError, match="Duplicate skill name"):
        SkillLoader(str(tmp_path)).discover()


def test_missing_dir_returns_empty(tmp_path):
    """Missing skills root -> discover returns an empty list (does not raise)."""
    assert SkillLoader(str(tmp_path / "nope")).discover() == []
