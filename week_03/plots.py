"""Export a standalone reliability diagram without an additional plotting dependency."""
from html import escape


def reliability_svg(records):
    colors = ['#2563eb', '#dc2626', '#16a34a', '#9333ea', '#d97706']
    names = list(dict.fromkeys(r['backend'] for r in records))
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="620" height="390" viewBox="0 0 620 390">',
             '<rect width="620" height="390" fill="white"/>',
             '<path d="M50 35 V325 H350 M50 325 L350 35" stroke="#888" fill="none"/>',
             '<text x="80" y="375">Predicted-class probability</text>',
             '<text x="5" y="18">Observed accuracy</text>']
    for value in (0, 0.5, 1):
        parts.append(f'<text x="{45 + 300 * value}" y="345">{value}</text>')
        parts.append(f'<text x="10" y="{330 - 290 * value}">{value}</text>')
    for index, name in enumerate(names):
        rows = [r for r in records if r['backend'] == name and r['ok']]
        color = colors[index % len(colors)]
        parts.append(f'<text x="375" y="{55 + index * 25}" fill="{color}">{escape(name)}</text>')
        points = []
        for b in range(10):
            selected = [r for r in rows if min(9, int(r['prob'] * 10)) == b]
            if not selected:
                continue
            probability = sum(r['prob'] for r in selected) / len(selected)
            accuracy = sum(r['pred'] == r['label'] for r in selected) / len(selected)
            x, y = 50 + 300 * probability, 325 - 290 * accuracy
            points.append((x, y, len(selected)))
        coordinates = ' '.join(f'{x},{y}' for x, y, _ in points)
        parts.append(f'<polyline points="{coordinates}" stroke="{color}" fill="none"/>')
        for x, y, n in points:
            parts.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{color}"><title>{escape(name)}: n={n}</title></circle>')
    return ''.join(parts + ['</svg>'])
