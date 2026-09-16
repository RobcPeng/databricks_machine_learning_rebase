// Mobile nav toggle — progressive enhancement; nav links work without JS.
(function () {
  var toggle = document.querySelector(".nav__toggle");
  var links = document.getElementById("navlinks");
  if (!toggle || !links) return;
  toggle.addEventListener("click", function () {
    var open = links.classList.toggle("open");
    toggle.setAttribute("aria-expanded", String(open));
  });
  // Collapse the menu after following a link on small screens.
  links.addEventListener("click", function (e) {
    if (e.target.tagName === "A" && links.classList.contains("open")) {
      links.classList.remove("open");
      toggle.setAttribute("aria-expanded", "false");
    }
  });
})();
