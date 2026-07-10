/* Two link-usability fixes, both re-run after every instant-navigation page swap
   via material's document$ observable.

   1. External links open in a new tab (rel=noopener).
   2. The header language toggle (overrides/partials/alternate.html) is rendered
      server-side per page, but material's instant loading swaps page content
      WITHOUT re-rendering the header — so after in-site navigation the toggle
      would still point at the PREVIOUS page's translation. We recompute its href
      from the current path. Structure is symmetric (mkdocs-static-i18n with
      fallback_to_default builds every en page under /zh/ too), so a pure path
      rewrite is always valid — no probing needed. */

(function () {
  function externalLinks() {
    document.querySelectorAll(".md-content a[href^='http']").forEach(function (a) {
      if (a.host && a.host !== location.host) {
        a.target = "_blank";
        a.rel = "noopener";
      }
    });
  }

  var root = null; // site root path, e.g. "/agentmaker/", derived once
  function syncLangToggle() {
    var btn = document.querySelector(".md-header .lang-toggle");
    if (!btn) return;
    if (root === null) {
      var h = new URL(btn.getAttribute("href"), location.origin).pathname;
      var p = location.pathname;
      if (h.indexOf("zh/") !== -1) root = h.slice(0, h.indexOf("zh/"));
      else if (p.indexOf("zh/") !== -1) root = p.slice(0, p.indexOf("zh/"));
      else root = ""; // neither side is zh — leave the server-rendered href alone
    }
    if (!root) return;
    var p2 = location.pathname;
    var zhRoot = root + "zh/";
    var isZh = p2.indexOf(zhRoot) === 0;
    var rest = isZh ? p2.slice(zhRoot.length) : p2.slice(root.length);
    btn.setAttribute("href", isZh ? root + rest : zhRoot + rest);
  }

  function run() { externalLinks(); syncLangToggle(); }

  if (window.document$ && window.document$.subscribe) {
    window.document$.subscribe(run);
  } else {
    document.addEventListener("DOMContentLoaded", run);
  }
})();
