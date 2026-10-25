"""Exactly one development candidate: ROI, conf=0.10, imgsz=640, IoU=0.7, CPU."""
from pathlib import Path
from diagnose_camera_low_conf import build_parser, run


if __name__ == '__main__':
    parser = build_parser()
    parser.description = __doc__
    raise SystemExit(run(parser.parse_args(), conf=0.10,
                         purpose='DEVELOPMENT_CANDIDATE_CONF010_NOT_INDEPENDENT_VALIDATION',
                         entrypoint=Path(__file__)))
