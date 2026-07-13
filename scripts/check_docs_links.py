"""Validate links and anchors in the generated documentation site."""

from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


class PageScan(HTMLParser):
    """Collect anchors and links from one generated HTML page."""

    def __init__(self) -> None:
        """Initialize empty anchor and link collections."""
        super().__init__()
        self.ids: set[str] = set()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Collect id and href attributes from a start tag."""
        values = dict(attrs)
        if value := values.get("id"):
            self.ids.add(value)
        if tag == "a" and (href := values.get("href")):
            self.links.append(href)


def _site_prefix(config_path: Path) -> str:
    """Return the normalized path prefix declared by site_url."""
    config_text = config_path.read_text(encoding="utf-8")
    site_url = next(
        line.split(":", 1)[1].strip()
        for line in config_text.splitlines()
        if line.startswith("site_url:")
    )
    return urlsplit(site_url).path.rstrip("/") + "/"


def main() -> None:
    """Scan generated pages and fail when an internal target is missing."""
    root = Path("site").resolve()
    if not root.is_dir():
        raise SystemExit("Generated site directory does not exist: site")
    prefix = _site_prefix(Path("mkdocs.yml"))
    pages: dict[Path, PageScan] = {}
    for page in root.rglob("*.html"):
        scan = PageScan()
        scan.feed(page.read_text(encoding="utf-8", errors="replace"))
        pages[page.resolve()] = scan
    if not pages:
        raise SystemExit("Generated site contains no HTML pages")

    failures: list[str] = []
    for page, scan in pages.items():
        for href in scan.links:
            parsed = urlsplit(href)
            if parsed.scheme or parsed.netloc:
                continue
            path = unquote(parsed.path)
            if not path:
                target = page
            elif path.startswith(prefix):
                target = root / path.removeprefix(prefix)
            elif path.startswith("/"):
                failures.append(f"{page.relative_to(root)} -> {href} (outside site_url path)")
                continue
            else:
                target = page.parent / path
            if path.endswith("/") or target.is_dir() or (path and not target.suffix):
                target /= "index.html"
            target = target.resolve()
            try:
                target.relative_to(root)
            except ValueError:
                failures.append(f"{page.relative_to(root)} -> {href} (outside built site)")
                continue
            if not target.exists():
                failures.append(f"{page.relative_to(root)} -> {href} (missing target)")
                continue
            fragment = unquote(parsed.fragment)
            if fragment and target.suffix == ".html":
                target_scan = pages.get(target)
                if target_scan is None or fragment not in target_scan.ids:
                    failures.append(f"{page.relative_to(root)} -> {href} (missing anchor)")

    if failures:
        raise SystemExit("Broken generated links:\n" + "\n".join(sorted(set(failures))))
    print(f"checked {sum(len(scan.links) for scan in pages.values())} generated links")


if __name__ == "__main__":
    main()
