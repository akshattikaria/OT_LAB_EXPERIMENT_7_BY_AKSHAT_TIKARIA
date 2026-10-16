import re, zlib

with open('report.pdf', 'rb') as f:
    b = f.read()

streams = re.findall(rb'stream[\r\n]+(.*?)[\r\n]+endstream', b, re.DOTALL)
print(f"Total streams: {len(streams)}")
for i, s in enumerate(streams):
    try:
        decomp = zlib.decompress(s)
        # Search for any text operations like Tj or TJ
        if b"Tj" in decomp or b"TJ" in decomp:
            sample = decomp[:300]
            print(f"Stream {i} sample: {sample}")
            if b"Akshat" in decomp:
                print(f"Found Akshat in stream {i}!")
    except Exception as e:
        pass
