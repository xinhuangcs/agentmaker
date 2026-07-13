/* Page-level fixes, re-run after every instant-navigation page swap via
   material's document$ observable.
 */

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

  function parseGood(raw) {
    try { var d = JSON.parse(raw); return d && typeof d.stars === "number" ? d : null; }
    catch (e) { return null; }
  }

  function fixSourceFacts() {
    var factsEl = document.querySelector(".md-header .md-source__facts");
    if (!factsEl) return;
    var key = Object.keys(sessionStorage).filter(function (k) {
      return k.slice(-9) === ".__source";
    })[0];
    var cached = key ? parseGood(sessionStorage.getItem(key)) : null;
    if (cached) {
      try { localStorage.setItem("__source_backup", JSON.stringify(cached)); } catch (e) {}
      return;
    }
    var backup = parseGood(localStorage.getItem("__source_backup"));
    if (backup) {
      var stars = factsEl.querySelector(".md-source__fact--stars");
      var forks = factsEl.querySelector(".md-source__fact--forks");
      if (stars) stars.textContent = backup.stars.toLocaleString();
      if (forks) forks.textContent = backup.forks.toLocaleString();
      if (key) { try { sessionStorage.setItem(key, JSON.stringify(backup)); } catch (e) {} }
    } else {
      factsEl.style.display = "none";
    }
  }

  function run() {
    externalLinks();
    syncLangToggle();
    fixSourceFacts();
    setTimeout(fixSourceFacts, 2500);
  }

  if (window.document$ && window.document$.subscribe) {
    window.document$.subscribe(run);
  } else {
    document.addEventListener("DOMContentLoaded", run);
  }
})();
