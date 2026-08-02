# Shashlik format tiles v1

Shashlik native tile implementation from scratch

## Key features

* **Payload Format**: Zero-copy arrays, super cheap to decode with no memory overhead, good for porting to bare metal vector renderers. Less space overhead than Flat-buffers. Coordinates are quantized and delta encoded relative to local tile coordinates.
* **Storage Format**: PMTiles, very compact on disk; does not require server side database and can be served over static CDN or from local file; can be ported to bare metal applications for local storage for smaller regional maps.
* **Tile edge handling**: No tile margins, geometry cuts are densily encoded in the tiles for clean and efficient rendering.
* **Caching**: Smart spacial LRU caching for smooth tile scene (pre/re)loading
