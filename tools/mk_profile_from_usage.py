import struct, sys
src, dst, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
u = open(src, 'rb').read()
assert u[:4] == b'STRU'
L, E = struct.unpack_from('<2i', u, 4)
cnt = struct.unpack_from('<%dQ' % (L * E), u, 12)
order = sorted(range(L * E), key=lambda i: (-cnt[i], i))[:N]
tot = sum(cnt) or 1
with open(dst, 'wb') as f:
    f.write(b'STRP')
    f.write(struct.pack('<5I', 1, L, E, N, N))
    for i in order:
        f.write(struct.pack('<2H', i // E, i % E))
    f.write(struct.pack('<%df' % (L * E), *[c / tot for c in cnt]))
print('записано', dst, 'пар', N, 'покрытие маршрутов %.2f%%' % (100.0 * sum(cnt[i] for i in order) / tot))
