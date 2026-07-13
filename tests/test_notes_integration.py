import os
import stat
import threading
import time

import pytest

from agentmaker.tools.integrations.notes import NotesTool


@pytest.mark.skipif(os.name != "posix", reason="POSIX dirfd behavior")
def test_notes_rejects_parent_symlinks_for_read_and_append(tmp_path):
    root = tmp_path / "notes"
    outside = tmp_path / "outside"
    root.mkdir(mode=0o700)
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    (root / "linked").symlink_to(outside, target_is_directory=True)
    tool = NotesTool(str(root))

    assert tool.run({"action": "read", "path": "linked/secret.txt"}).status == "error"
    assert tool.run({"action": "append", "path": "linked/new.txt", "content": "x"}).status == "error"
    assert not (outside / "new.txt").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX O_NOFOLLOW behavior")
def test_notes_rejects_final_symlink_for_read_and_append(tmp_path):
    root = tmp_path / "notes"
    root.mkdir(mode=0o700)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (root / "linked.txt").symlink_to(outside)
    tool = NotesTool(str(root))

    assert tool.run({"action": "read", "path": "linked.txt"}).status == "error"
    assert tool.run({"action": "append", "path": "linked.txt", "content": "x"}).status == "error"
    assert outside.read_text(encoding="utf-8") == "secret"


@pytest.mark.skipif(os.name != "posix", reason="POSIX file locking behavior")
def test_notes_file_limit_is_atomic_across_instances(tmp_path):
    first = NotesTool(str(tmp_path), max_file_bytes=6)
    second = NotesTool(str(tmp_path), max_file_bytes=6)
    barrier = threading.Barrier(3)
    responses = []

    def append(tool):
        barrier.wait()
        responses.append(tool.run({"action": "append", "path": "shared.md", "content": "12345"}))

    threads = [threading.Thread(target=append, args=(tool,)) for tool in (first, second)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert sorted(response.status for response in responses) == ["error", "success"]
    assert (tmp_path / "shared.md").read_text(encoding="utf-8") == "12345\n"


def test_notes_invalid_utf8_is_a_tool_error(tmp_path):
    (tmp_path / "binary.md").write_bytes(b"\xff")
    response = NotesTool(str(tmp_path)).run({"action": "read", "path": "binary.md"})
    assert response.status == "error"


@pytest.mark.skipif(os.name != "posix", reason="POSIX FIFO behavior")
def test_notes_rejects_fifo_without_blocking(tmp_path):
    os.mkfifo(tmp_path / "pipe")
    tool = NotesTool(str(tmp_path))
    assert tool.run({"action": "read", "path": "pipe"}).status == "error"
    assert tool.run({"action": "append", "path": "pipe", "content": "x"}).status == "error"


@pytest.mark.skipif(os.name != "posix", reason="POSIX hard-link behavior")
def test_notes_rejects_hard_links(tmp_path):
    root = tmp_path / "notes"
    root.mkdir(mode=0o700)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    os.link(outside, root / "linked.txt")
    tool = NotesTool(str(root))
    assert tool.run({"action": "read", "path": "linked.txt"}).status == "error"
    assert tool.run({"action": "append", "path": "linked.txt", "content": "x"}).status == "error"
    assert outside.read_text(encoding="utf-8") == "secret"


def test_notes_oversized_first_append_leaves_no_file(tmp_path):
    response = NotesTool(str(tmp_path), max_file_bytes=1).run(
        {"action": "append", "path": "note.md", "content": "x"})
    assert response.status == "error"
    assert not (tmp_path / "note.md").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX flock behavior")
def test_notes_reports_lock_contention_without_waiting(tmp_path):
    import fcntl

    path = tmp_path / "note.md"
    path.write_text("", encoding="utf-8")
    fd = os.open(path, os.O_WRONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        response = NotesTool(str(tmp_path)).run(
            {"action": "append", "path": "note.md", "content": "x"})
        assert time.monotonic() - started < 0.5
        assert response.status == "error"
    finally:
        os.close(fd)


def test_notes_requires_dirfd_support(monkeypatch, tmp_path):
    monkeypatch.setattr(os, "supports_dir_fd", set())
    with pytest.raises(OSError, match="dirfd"):
        NotesTool(str(tmp_path))


def test_notes_root_must_be_private_owned_directory(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="non-empty"):
        NotesTool("")

    created = tmp_path / "created"
    NotesTool(str(created))
    assert stat.S_IMODE(created.stat().st_mode) == 0o700

    public = tmp_path / "public"
    public.mkdir(mode=0o755)
    with pytest.raises(PermissionError, match="group or others"):
        NotesTool(str(public))

    group_readable = tmp_path / "group"
    group_readable.mkdir(mode=0o750)
    with pytest.raises(PermissionError, match="group or others"):
        NotesTool(str(group_readable))

    # Special bits and stricter owner modes grant nothing to group/others, so they are accepted.
    sticky = tmp_path / "sticky"
    sticky.mkdir(mode=0o700)
    sticky.chmod(0o1700)
    NotesTool(str(sticky))

    read_only = tmp_path / "readonly"
    read_only.mkdir(mode=0o700)
    read_only.chmod(0o500)
    NotesTool(str(read_only))
    read_only.chmod(0o700)

    regular = tmp_path / "file"
    regular.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="directory"):
        NotesTool(str(regular))

    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    monkeypatch.setattr(os, "geteuid", lambda: private.stat().st_uid + 1)
    with pytest.raises(PermissionError, match="owned"):
        NotesTool(str(private))
