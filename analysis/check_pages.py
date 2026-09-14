import sys, fitz
doc = fitz.open(sys.argv[1]); print('pages:', len(doc))
def body_blocks(p):  # drop the centred page-number footer
    return [b for b in p.get_text('blocks') if b[4].strip() and not (b[4].strip().isdigit() and b[1] > 700)]
p9 = body_blocks(doc[8]); bottom = max(b[3] for b in p9)
print('page 9 body bottom: %.1f pt of %.1f; slack to 721.3 reference: %.1f pt; last words: %r' % (bottom, doc[8].rect.height, 721.3 - bottom, p9[-1][4].strip()[-70:]))
for key in ('ai use statement', 'discussion and limitations', 'reproducibility statement', 'references'):
    pages = [i + 1 for i, p in enumerate(doc) if key in p.get_text().lower()]
    print(f'{key!r} on pages {pages[:3]}')
