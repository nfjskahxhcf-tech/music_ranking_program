#셀b, manifest.webmanifest 생성
%%writefile static/manifest.webmanifest
{
  "name": "RunRank",
  "short_name": "RunRank",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#0f0f12",
  "theme_color": "#0f0f12",
  "icons": [
    { "src": "/icon-192.png", "sizes": "192x192", "type": "image/png" },
    { "src": "/icon-512.png", "sizes": "512x512", "type": "image/png" }
  ]
}