"""Make one prespecified, lossless central crop for every manifest case."""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageOps
from evaluate_camera_minimal import validate_cases

SOURCE_SIZE = (4608, 2592)
ROI = (1152, 648, 3456, 1944)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args, *, case_validator=validate_cases):
    parser = argparse.ArgumentParser(description=__doc__)
    if args.output.exists() or args.record.exists():
        parser.error('Output/record exists. Do not overwrite; choose a new version.')
    cases = json.loads(args.cases.read_text())
    case_validator(cases)
    files = {p.name for p in args.images.iterdir() if p.is_file() and p.name != '.DS_Store'}
    if files != set(cases):
        parser.error('Source directory does not match the full case manifest')
    inputs = {name: sha(args.images/name) for name in cases}
    # Validate all sources before writing any crops.
    for name in cases:
        with Image.open(args.images/name) as raw:
            if raw.size != SOURCE_SIZE or raw.getexif().get(274, 1) != 1:
                parser.error(f'Unexpected size/orientation: {name}')
    args.output.mkdir(parents=True, exist_ok=False)
    record = dict(source_dir=str(args.images.resolve()), source_size=list(SOURCE_SIZE),
                  roi_xyxy=list(ROI), derived_size=[2304, 1296],
                  method='Same fixed central half-width/half-height crop for every image, '
                         'including empty. EXIF transpose then RGB, no resampling or color enhancement. '
                         'Lossless PNG preserves decoded source ROI pixels.',
                  script_sha256=sha(Path(__file__)), cases_sha256=sha(args.cases), images=[])
    for name, reference in cases.items():
        output_name = Path(name).stem+'.png'
        with Image.open(args.images/name) as raw:
            oriented = ImageOps.exif_transpose(raw).convert('RGB')
            crop = oriented.crop(ROI)
            crop.save(args.output/output_name, format='PNG')
            with Image.open(args.output/output_name) as saved:
                if saved.size != (2304, 1296) or saved.convert('RGB').tobytes() != crop.tobytes():
                    raise RuntimeError('Derived pixel check failed')
        record['images'].append(dict(source_image=name, source_sha256=inputs[name],
                                     derived_image=output_name,
                                     derived_sha256=sha(args.output/output_name), reference_class=reference))
    if any(sha(args.images/n) != h for n, h in inputs.items()):
        raise RuntimeError('Original image changed during preparation')
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    print(f'All {len(cases)} originals unchanged; derived PNG pixels match the fixed source ROI.')
    print('Saved:', args.output, args.record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--record', type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
