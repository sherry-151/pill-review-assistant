"""Build labeled review sheets; thumbnails never enter model inference."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--derived', type=Path, required=True)
    args = p.parse_args()
    rows = json.loads((args.run/'inference/result.json').read_text())['images']
    font = ImageFont.truetype('/System/Library/Fonts/Menlo.ttc', 15)
    for kind, source in [('crops', args.derived),
                         ('predictions', args.run/'inference/annotated-all')]:
        target = args.run/f'review-{kind}.jpg'
        if target.exists():
            p.error(f'Refusing to overwrite {target}')
        sheet = Image.new('RGB', (1800, 5*400), 'white')
        draw = ImageDraw.Draw(sheet)
        for i, row in enumerate(rows):
            x, y = (i % 3)*600, (i//3)*400
            with Image.open(source/row['image']) as image:
                image.thumbnail((600, 338))
                sheet.paste(image, (x, y+60))
            draw.text((x+8, y+4), row['image'], font=font, fill='black')
            pred = ', '.join(f"{b['class_name']} {b['confidence']:.6f}"
                             for b in row.get('predictions', [])) or 'zero boxes'
            draw.text((x+8, y+26), pred, font=font,
                      fill='darkgreen' if row.get('counts_match_reference') else 'red')
        sheet.save(target, quality=95)
        print(target)


if __name__ == '__main__':
    main()
