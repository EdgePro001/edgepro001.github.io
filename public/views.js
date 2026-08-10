(() => {
  const counterBase = "https://edgep.goatcounter.com/counter/";

  const loadCount = async (element) => {
    const key = element.dataset.viewKey;
    if (!key) return;

    try {
      const response = await fetch(
        `${counterBase}${encodeURIComponent(key)}.json`,
        { cache: "no-store" },
      );

      // GoatCounter returns a JSON count of zero with HTTP 404 until a new path has
      // its first persisted visit. Treat that as a valid public counter response.
      if (!response.ok && response.status !== 404) return;

      const data = await response.json();
      if (!data || typeof data.count !== "string") return;

      element.textContent = data.count === "1" ? "1 view" : `${data.count} views`;
      element.hidden = false;
    } catch {
      // Analytics must never affect navigation or article rendering.
    }
  };

  const loadAllCounts = () => {
    document.querySelectorAll("[data-view-key]").forEach(loadCount);
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", loadAllCounts, { once: true });
  } else {
    loadAllCounts();
  }
})();
