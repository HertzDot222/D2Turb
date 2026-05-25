(function () {
  "use strict";

  const lightbox = document.querySelector(".lightbox");
  const lightboxImage = document.querySelector(".lightbox-image");
  const closeButton = document.querySelector(".lightbox-close");
  const zoomButtons = document.querySelectorAll("[data-lightbox-src]");
  let lastTrigger = null;

  function closeLightbox() {
    lightbox.hidden = true;
    lightboxImage.src = "";
    document.body.classList.remove("modal-open");
    if (lastTrigger) {
      lastTrigger.focus();
    }
  }

  function openLightbox(trigger) {
    lastTrigger = trigger;
    lightboxImage.src = trigger.dataset.lightboxSrc;
    lightboxImage.alt = trigger.dataset.lightboxAlt || "";
    lightbox.hidden = false;
    document.body.classList.add("modal-open");
    closeButton.focus();
  }

  zoomButtons.forEach(function (button) {
    button.addEventListener("click", function () {
      openLightbox(button);
    });
  });

  closeButton.addEventListener("click", closeLightbox);
  lightbox.addEventListener("click", function (event) {
    if (event.target === lightbox) {
      closeLightbox();
    }
  });
  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape" && !lightbox.hidden) {
      closeLightbox();
    }
  });

  const links = document.querySelectorAll(".nav-links a[href^='#']");
  const sections = Array.from(links)
    .map(function (link) {
      return document.querySelector(link.getAttribute("href"));
    })
    .filter(Boolean);

  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) {
          return;
        }
        links.forEach(function (link) {
          link.classList.toggle("active", link.getAttribute("href") === "#" + entry.target.id);
        });
      });
    }, { rootMargin: "-30% 0px -62% 0px", threshold: 0 });

    sections.forEach(function (section) {
      observer.observe(section);
    });
  }
}());
