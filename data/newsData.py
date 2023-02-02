import zstandard as zstd

your_filename = './data/npr.ndjson.zst'
with open(your_filename, "rb") as f:
    data = f.read()

dctx = zstd.ZstdDecompressor()
decompressed = dctx.decompress(data)
print(decompressed)