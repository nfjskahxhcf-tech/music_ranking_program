/* RunRank Service Worker (offline cache)
   - Cache the shell (index + manifest + icons)
   - Network-first for API (always try fresh)
*/
const CACHE_NAME = "runrank-shell-v101";

const SHELL = [
  "/",                    // index.html (html=True라 /가 index 역할)
  "/index.html",
  "/manifest.webmanifest",
  "/favicon.ico",
  "/icon-192.png",
  "/icon-512.png",

  // iOS touch icons (너 static에 실제 있음)
  "/apple-touch-icon.png",
  "/apple-touch-icon-precomposed.png",
  "/apple-touch-icon-120x120.png",
  "/apple-touch-icon-120x120-precomposed.png"
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => cache.addAll(SHELL))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.map((k) => (k === CACHE_NAME ? null : caches.delete(k))))
    ).then(() => self.clients.claim())
  );
});

function isApi(url) {
  try {
    const u = new URL(url);
    return u.pathname.startsWith("/api/");
  } catch {
    return false;
  }
}

self.addEventListener("fetch", (event) => {
  const req = event.request;

  if (req.method !== "GET") return;

  // ✅ SPA 네비게이션(/, /index.html 등)은 shell로
  if (req.mode === "navigate") {
    event.respondWith(
      fetch(req).catch(() => caches.match("/"))
    );
    return;
  }

  // ✅ API: 네트워크 우선, 실패 시 캐시
  if (isApi(req.url)) {
    event.respondWith(
      fetch(req).then((res) => {
        const copy = res.clone();
        caches.open(CACHE_NAME).then((cache) => cache.put(req, copy)).catch(() => {});
        return res;
      }).catch(() => caches.match(req))
    );
    return;
  }

  // ✅ 나머지: 캐시 우선
  event.respondWith(
    caches.match(req).then((cached) => cached || fetch(req).then((res) => {
      const copy = res.clone();
      caches.open(CACHE_NAME).then((cache) => cache.put(req, copy)).catch(() => {});
      return res;
    }))
  );
});

