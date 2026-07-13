"""Skill loading and frontmatter parsing tests (hermetic).

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


def test_parse_meta_rejects_aliases_and_non_string_fields():
    with pytest.raises(ValueError, match="must not contain YAML aliases"):
        _parse_meta("name: &name daily\ndescription: *name\n", path="x/SKILL.md")
    with pytest.raises(ValueError, match="'description' must be a string"):
        _parse_meta("name: daily\ndescription: [one, two]\n", path="x/SKILL.md")


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


def test_skill_loader_skips_symlinked_entries_without_aborting_discovery(tmp_path):
    """Symlinked skill directories / SKILL.md files are never followed: each is skipped while sibling skills keep loading."""
    _make_skill(tmp_path, "good")
    outside = tmp_path / "outside.md"
    outside.write_text("secret", encoding="utf-8")
    file_skill = tmp_path / "file-link"
    file_skill.mkdir()
    try:
        (file_skill / "SKILL.md").symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable on this platform")
    real = tmp_path / "real"
    real.mkdir()
    _make_skill(real, "nested")
    try:
        (tmp_path / "dir-link").symlink_to(real / "nested", target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink creation is unavailable on this platform")

    loader = SkillLoader(str(tmp_path))
    assert [s.name for s in loader.discover()] == ["good"]   # symlinked entries skipped, discovery unaffected
    assert loader.load("good") is not None


def test_skill_loader_enforces_frontmatter_and_body_byte_limits(tmp_path):
    _make_skill(tmp_path, "large-front", description="x" * 100)
    with pytest.raises(ValueError, match="frontmatter exceeds"):
        SkillLoader(str(tmp_path), max_frontmatter_bytes=32).discover()

    (tmp_path / "large-front" / "SKILL.md").unlink()
    (tmp_path / "large-front").rmdir()
    _make_skill(tmp_path, "large-body", body="正文" * 20)
    loader = SkillLoader(str(tmp_path), max_body_bytes=16)
    assert loader.discover()[0].name == "large-body"
    with pytest.raises(ValueError, match="body exceeds"):
        loader.load("large-body")


@pytest.mark.skipif(not hasattr(__import__("os"), "mkfifo"), reason="FIFO unavailable")
def test_skill_loader_never_reads_non_regular_skill_file(tmp_path):
    import os
    skill = tmp_path / "pipe"
    skill.mkdir()
    os.mkfifo(skill / "SKILL.md")
    assert SkillLoader(str(tmp_path)).discover() == []


def test_skill_load_reads_the_descriptor_opened_before_path_swap(tmp_path, monkeypatch):
    import agentmaker.skills.loader as loader_module

    skill = _make_skill(tmp_path, "safe", body="trusted body")
    outside = tmp_path / "outside.md"
    outside.write_text("---\nname: safe\n---\nuntrusted\n", encoding="utf-8")
    original = loader_module._read_frontmatter_fd
    swapped = False

    def read_and_swap(fd, path, limit):
        nonlocal swapped
        meta = original(fd, path, limit)
        if not swapped:
            swapped = True
            (skill / "SKILL.md").unlink()
            (skill / "SKILL.md").symlink_to(outside)
        return meta

    monkeypatch.setattr(loader_module, "_read_frontmatter_fd", read_and_swap)
    assert SkillLoader(str(tmp_path)).load("safe") == "trusted body\n"


def test_skill_loader_portable_open_rechecks_file_identity(tmp_path, monkeypatch):
    import agentmaker.skills.loader as loader_module

    _make_skill(tmp_path, "safe", body="trusted body")
    monkeypatch.setattr(loader_module.os, "supports_dir_fd", set())

    assert SkillLoader(str(tmp_path)).load("safe") == "trusted body\n"


def test_skill_frontmatter_numeric_scalars_coerce_to_text(tmp_path):
    """Unquoted YAML scalars like `name: 2048` load as their text instead of aborting discovery."""
    skill_dir = tmp_path / "game"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: 2048\ndescription: 42\n---\n正文", encoding="utf-8")
    skills = SkillLoader(str(tmp_path)).discover()
    assert [(s.name, s.description) for s in skills] == [("2048", "42")]


def test_skill_open_failure_is_skipped_not_raised(tmp_path, monkeypatch):
    """A SKILL.md open that fails (e.g. ELOOP from a symlink swapped in after the stat check) skips that skill instead of aborting discovery."""
    import errno
    import os
    import agentmaker.skills.loader as loader_module

    _make_skill(tmp_path, "good")
    _make_skill(tmp_path, "racy")
    if os.open not in os.supports_dir_fd or not getattr(os, "O_NOFOLLOW", 0):
        pytest.skip("POSIX dirfd open path is required for this race")

    real_open = os.open

    def flaky_open(path, flags, *args, **kwargs):
        if path == "SKILL.md" and kwargs.get("dir_fd") is not None:
            raise OSError(errno.ELOOP, "symlink swapped in after the check")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(loader_module.os, "open", flaky_open)
    monkeypatch.setattr(loader_module.os, "supports_dir_fd", {flaky_open})   # keep the dirfd branch selected after patching os.open
    assert SkillLoader(str(tmp_path)).discover() == []   # both skills skipped, no exception escapes
