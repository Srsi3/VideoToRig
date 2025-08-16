#!/usr/bin/env python3
"""
Video2Rigify - Test Suite
=========================================

Purpose
-------
A fast, self-contained test runner for the Multi-Person Video-2-Rigify
pipeline that validates:

1) Environment & versions (torch/mmcv/mmdet/mmpose)
2) MMPose demo on an image (produces predictions)
3) MotionBERT on a synthetic H36M track (produces BVH)
4) (Optional) End-to-end: video → MMPose → MotionBERT (tiny clip)

It writes rich logs and returns non-zero on failure. Designed for quick
smoke testing and debugging.

Usage
-----
python v2r_test_suite.py \
  --python "C:/Users/you/AppData/Roaming/Blender Foundation/Blender/4.5/config/video2rigify_env/Scripts/python.exe" \
  --mmpose "C:/Users/you/AppData/Roaming/Blender Foundation/Blender/4.5/config/mmpose" \
  --motionbert "C:/Users/you/AppData/Roaming/Blender Foundation/Blender/4.5/config/MotionBERT" \
  --video path/to/short.mp4  --device auto  --quick

Flags
-----
--quick            Run a minimal but useful subset (1,2,3 + tiny E2E if --video given)
--skip-e2e         Skip the end-to-end test even if --video is provided
--device {cpu,cuda:0,auto}  Default: auto (detect via torch)
--framecap N       Hard cap on frames per track for E2E (default 120)
--workdir DIR      Where to write outputs/logs (default: system temp)

Notes
-----
• Requires that you already installed deps via the Blender add-on (or equivalent).
• If ffmpeg is available on PATH, the suite will downsample the provided video to 5fps/shorter cut.
• No network fetches except when MMPose pulls model weights on first run.
"""

from __future__ import annotations
import argparse, json, logging, os, shutil, subprocess, sys, tempfile, textwrap, time
from dataclasses import dataclass
from pathlib import Path

LOG = logging.getLogger("v2r_tests")

# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def setup_logging(workdir: Path) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    LOG.setLevel(logging.DEBUG)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S")
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO); ch.setFormatter(fmt)
    fh = logging.FileHandler(workdir/"v2r_tests.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG); fh.setFormatter(fmt)
    LOG.addHandler(ch); LOG.addHandler(fh)


def run(cmd, *, env=None, cwd=None, desc="") -> subprocess.CompletedProcess:
    LOG.info("▶ %s", " ".join(map(str, cmd)))
    cp = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=cwd)
    if cp.stdout:
        LOG.debug(cp.stdout)
    if cp.stderr:
        # many ML stacks write INFO to stderr; keep as WARNING
        LOG.warning(cp.stderr)
    if cp.returncode != 0:
        raise RuntimeError(f"Command failed: {desc or cmd[0]} (code {cp.returncode})")
    return cp


def which(exe: str) -> str | None:
    return shutil.which(exe)


def detect_device(py: Path) -> str:
    try:
        out = subprocess.check_output([str(py), "-c", "import torch;print('cuda:0' if torch.cuda.is_available() else 'cpu')"], text=True)
        return out.strip()
    except Exception:
        return "cpu"



# Embedded pipeline stub (same logic as the add‑on; multi‑person + chunks)

PIPELINE_STUB = r"""#!/usr/bin/env python3
import json, argparse, subprocess, tempfile, sys, os, math, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger("V2RStub")

MB_FRAME_CAP = 243

parser = argparse.ArgumentParser()
parser.add_argument('video', type=Path)
parser.add_argument('--outdir',     type=Path, required=True)
parser.add_argument('--mmpose',     type=Path, required=True)
parser.add_argument('--motionbert', type=Path, required=True)
parser.add_argument('--device', default='cpu')
parser.add_argument('--cap', type=int, default=243)
args = parser.parse_args()

MB_FRAME_CAP = max(4, int(args.cap))
work = Path(tempfile.mkdtemp(prefix='v2r_'))
log.info("Workdir: %s", work)

# --- paths
demo_py = args.mmpose / 'demo' / 'topdown_demo_with_mmdet.py'
if not demo_py.exists():
    sys.exit('[V2R] demo script not found: ' + str(demo_py))

det_cfg  = args.mmpose / 'demo' / 'mmdetection_cfg' / 'faster_rcnn_r50_fpn_coco.py'
det_ckpt = ('https://download.openmmlab.com/mmdetection/v3.0/'
            'faster_rcnn/faster_rcnn_r50_fpn_1x_coco/'
            'faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth')

pose_cfg  = args.mmpose / 'configs/body/2d_kpt_sview_rgb_img' / 'topdown_heatmap/coco/rtmpose_m_8xb256-210e_coco-256x192.py'
pose_ckpt = ('https://download.openmmlab.com/mmpose/v1/'
             'rtmpose/rtmpose_m_8xb256-210e_coco-256x192-a24f2126_20230323.pth')

out_dir = work / 'mmpose_out'
out_dir.mkdir(parents=True, exist_ok=True)

# --- run demo
cmd = [sys.executable, str(demo_py), str(det_cfg), det_ckpt, str(pose_cfg), pose_ckpt,
       '--video-path', str(args.video), '--out-video-root', str(out_dir),
       '--save-predictions', '--device', args.device]
log.info("Run MMPose demo: %s", " ".join(map(str, cmd)))
proc = subprocess.run(cmd, capture_output=True, text=True)
if proc.stdout: log.info(proc.stdout)
if proc.stderr: log.warning(proc.stderr)
if proc.returncode:
    sys.exit('[V2R] pose demo failed')

# --- locate predictions
pred_file = None
for cand in out_dir.rglob('*'):
    if cand.suffix.lower() in {'.json', '.pkl', '.npz'} and 'pred' in cand.stem.lower():
        pred_file = cand; break
if pred_file is None:
    for cand in out_dir.rglob('*.json'):
        pred_file = cand; break
if pred_file is None:
    sys.exit('[V2R] could not find predictions output')

# --- load preds
def _load_preds(p):
    import numpy as _np, json as _json, pickle as _pkl
    if p.suffix.lower() == '.json':
        return _json.loads(p.read_text(encoding='utf8'))
    if p.suffix.lower() == '.pkl':
        with open(p, 'rb') as f: return _pkl.load(f)
    if p.suffix.lower() == '.npz':
        return _np.load(p, allow_pickle=True).item()
    return None
preds = _load_preds(pred_file)

# --- helpers
import numpy as np

def _frame_instances(frame):
    if isinstance(frame, dict):
        return frame.get('pred_instances') or frame.get('instances') or frame.get('preds') or []
    return []

def _inst_center(inst):
    kpts = inst.get('keypoints') or inst.get('keypoints_2d') or inst.get('coordinates')
    if kpts is None: return None
    k = np.asarray(kpts).reshape(-1,3)
    valid = k[k[:,2] > 0]
    if len(valid) == 0:
        valid = k
    return valid[:,:2].mean(axis=0)

def coco_to_h36m(kpts):
    kp = np.asarray(kpts).reshape(-1, 3)  # 17x3
    def mid(a,b): return (kp[a,:2] + kp[b,:2]) / 2.0
    def conf(a,b): return (kp[a,2] + kp[b,2]) / 2.0
    pelvis_xy = mid(11,12); pelvis_c = conf(11,12)
    thorax_xy = mid(5,6);   thorax_c = conf(5,6)
    neck_xy   = mid(1,2);   neck_c   = conf(1,2)
    head_xy   = kp[0,:2];   head_c   = kp[0,2]
    spine_xy  = (pelvis_xy + thorax_xy) / 2.0; spine_c = (pelvis_c + thorax_c)/2.0
    out = np.zeros((17,3), dtype=np.float32)
    out[0,:2]=pelvis_xy; out[0,2]=pelvis_c
    out[1],out[2],out[3] = kp[11], kp[13], kp[15]
    out[4],out[5],out[6] = kp[12], kp[14], kp[16]
    out[7,:2]=spine_xy;  out[7,2]=spine_c
    out[8,:2]=thorax_xy; out[8,2]=thorax_c
    out[9,:2]=neck_xy;   out[9,2]=neck_c
    out[10,:2]=head_xy;  out[10,2]=head_c
    out[11],out[12],out[13] = kp[5], kp[7], kp[9]
    out[14],out[15],out[16] = kp[6], kp[8], kp[10]
    return out

frames = preds if isinstance(preds, list) else preds.get('predictions', [])
if not isinstance(frames, list) or not frames:
    sys.exit('[V2R] predictions structure unrecognized or empty')

# --- basic multi-person association
tracks = []
next_tid = 0
max_dist = 80.0
for fi, frm in enumerate(frames):
    insts = _frame_instances(frm)
    cur = []
    for inst in insts:
        kpts = inst.get('keypoints') or inst.get('keypoints_2d') or inst.get('coordinates')
        if kpts is None: continue
        center = _inst_center(inst)
        score = float(inst.get('bbox_score') or inst.get('score') or 0.0)
        cur.append((kpts, center, score))

    used = set()
    for tr in tracks:
        if tr['last_frame'] != fi-1:
            continue
        best_j, best_d = None, 1e9
        for j,(kpts, center, _score) in enumerate(cur):
            if j in used or center is None or tr['last_center'] is None:
                continue
            d = float(np.linalg.norm(center - tr['last_center']))
            if d < best_d:
                best_d, best_j = d, j
        if best_j is not None and best_d <= max_dist:
            kpts, center, _score = cur[best_j]
            used.add(best_j)
            h36m = coco_to_h36m(kpts).tolist()
            tr['items'].append({'frame_id': fi, 'keypoints': h36m})
            tr['last_center'] = center
            tr['last_frame']  = fi

    for j,(kpts, center, _score) in enumerate(cur):
        if j in used:
            continue
        h36m = coco_to_h36m(kpts).tolist()
        tracks.append({'id': next_tid, 'last_center': center, 'last_frame': fi, 'items': [{'frame_id': fi, 'keypoints': h36m}]})
        next_tid += 1

tracks = [t for t in tracks if len(t['items']) >= 8]
args.outdir.mkdir(parents=True, exist_ok=True)

all_track_outputs = []
mb_env = dict(os.environ)
mb_env['PYTHONPATH'] = os.pathsep.join([str(args.motionbert), mb_env.get('PYTHONPATH','')])

for tr in tracks:
    items = tr['items']
    items.sort(key=lambda x: x['frame_id'])
    bvh_paths = []
    if not items:
        continue

    num_chunks = math.ceil(len(items) / MB_FRAME_CAP)
    for ci in range(num_chunks):
        start = ci * MB_FRAME_CAP
        end   = min((ci+1)*MB_FRAME_CAP, len(items))
        chunk = items[start:end]
        tjson = args.outdir / f'track{tr['id']}_part{ci:03d}.json'
        with tjson.open('w', encoding='utf8') as f: json.dump(chunk, f)
        out_bvh = args.outdir / f'track{tr['id']}_part{start:06d}_{end:06d}.bvh'
        cmd = [sys.executable, str(args.motionbert / 'apps/demo_pose3d.py'),
               '--pose2d_json', str(tjson), '--save_bvh', str(out_bvh), '--device', args.device]
        log.info("Run MotionBERT: %s", " ".join(map(str, cmd)))
        proc = subprocess.run(cmd, capture_output=True, text=True, env=mb_env, cwd=str(args.motionbert))
        if proc.returncode:
            log.error("MotionBERT failed on %s", tjson)
            continue
        bvh_paths.append(str(out_bvh))
    if bvh_paths:
        all_track_outputs.append({'id': tr['id'], 'parts': bvh_paths})

print(json.dumps({"tracks": all_track_outputs}))
"""

# ------------------------------------------------------------------
# Test Cases
# ------------------------------------------------------------------
@dataclass
class TestOutcome:
    name: str
    ok: bool
    detail: str = ""
    artifact: Path | None = None


def test_versions(py: Path) -> TestOutcome:
    try:
        code = (
            "import torch,mmcv,mmdet,mmpose,mmengine;"
            "print('torch',torch.__version__,'cuda',torch.cuda.is_available());"
            "print('mmcv',mmcv.__version__);"
            "print('mmdet',mmdet.__version__);"
            "print('mmpose',mmpose.__version__)"
        )
        cp = run([str(py), "-c", code], desc="versions")
        LOG.info(cp.stdout)
        return TestOutcome("versions", True, cp.stdout.strip())
    except Exception as e:
        return TestOutcome("versions", False, str(e))


def test_mmpose_image(py: Path, mmpose: Path, outdir: Path, device: str) -> TestOutcome:
    try:
        img = mmpose/"demo"/"resources"/"human-pose.jpg"
        if not img.exists():
            return TestOutcome("mmpose_image", False, f"missing demo image: {img}")
        pred_out = outdir/"mmpose_image"
        pred_out.mkdir(parents=True, exist_ok=True)
        det_cfg  = mmpose/"demo"/"mmdetection_cfg"/"faster_rcnn_r50_fpn_coco.py"
        det_ckpt = ("https://download.openmmlab.com/mmdetection/v3.0/"
                    "faster_rcnn/faster_rcnn_r50_fpn_1x_coco/"
                    "faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth")
        pose_cfg = mmpose/"configs"/"body"/"2d_kpt_sview_rgb_img"/"topdown_heatmap"/"coco"/"rtmpose_m_8xb256-210e_coco-256x192.py"
        pose_ckpt=("https://download.openmmlab.com/mmpose/v1/rtmpose/"
                   "rtmpose_m_8xb256-210e_coco-256x192-a24f2126_20230323.pth")
        env = dict(os.environ)
        # help dynamic imports
        env["PYTHONPATH"] = os.pathsep.join([str(mmpose), env.get("PYTHONPATH","")])
        cp = run([str(py), str(mmpose/"demo"/"topdown_demo_with_mmdet.py"),
                  str(det_cfg), det_ckpt, str(pose_cfg), pose_ckpt,
                  "--input", str(img), "--out-img-root", str(pred_out),
                  "--save-predictions", "--device", device], env=env,
                 desc="mmpose image")
        # find a predictions file
        pred_file = None
        for cand in pred_out.rglob("*.json"):
            pred_file = cand; break
        if not pred_file:
            return TestOutcome("mmpose_image", False, "no predictions json produced")
        return TestOutcome("mmpose_image", True, "predictions ok", pred_file)
    except Exception as e:
        return TestOutcome("mmpose_image", False, str(e))


def test_motionbert_synth(py: Path, motionbert: Path, outdir: Path, device: str) -> TestOutcome:
    try:
        work = outdir/"mb_synth"; work.mkdir(parents=True, exist_ok=True)
        tjson = work/"t.json"; bvh = work/"out.bvh"
        # 32 frames of constant H36M 17x3 points with conf=1
        frames = [{"frame_id":i, "keypoints":[[0,0,1]]*17} for i in range(32)]
        tjson.write_text(json.dumps(frames), encoding="utf-8")
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(motionbert), env.get("PYTHONPATH","")])
        cp = run([str(py), str(motionbert/"apps"/"demo_pose3d.py"),
                  "--pose2d_json", str(tjson), "--save_bvh", str(bvh), "--device", device],
                 env=env, cwd=str(motionbert), desc="motionbert synth")
        if not bvh.exists():
            return TestOutcome("motionbert_synth", False, "no BVH emitted")
        return TestOutcome("motionbert_synth", True, "BVH ok", bvh)
    except Exception as e:
        return TestOutcome("motionbert_synth", False, str(e))


def maybe_downsample_video(src: Path, dst: Path) -> Path:
    ff = which("ffmpeg")
    if not ff:
        LOG.info("ffmpeg not found; using original video (no downsample)")
        return src
    try:
        # 5 fps, first 4 seconds, no audio
        run([ff, "-y", "-i", str(src), "-vf", "fps=5", "-t", "4", "-an", str(dst)], desc="ffmpeg downsample")
        return dst if dst.exists() else src
    except Exception:
        return src


def test_end_to_end(py: Path, mmpose: Path, motionbert: Path, video: Path, outdir: Path, device: str, framecap: int) -> TestOutcome:
    try:
        work = outdir/"e2e"; work.mkdir(parents=True, exist_ok=True)
        stub = work/"pipeline_stub.py"
        stub.write_text(PIPELINE_STUB, encoding="utf-8"); stub.chmod(0o755)
        # try to reduce video size for speed
        tiny = work/"tiny.mp4"
        v_in = maybe_downsample_video(video, tiny)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(mmpose), str(motionbert), env.get("PYTHONPATH","")])
        cp = run([str(py), str(stub), str(v_in), "--outdir", str(work), "--mmpose", str(mmpose),
                  "--motionbert", str(motionbert), "--device", device, "--cap", str(framecap)],
                 env=env, desc="end-to-end pipeline")
        # parse tracks json (last json object printed)
        tracks = []
        for line in cp.stdout.splitlines():
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and 'tracks' in obj:
                    tracks = obj['tracks']
            except Exception:
                pass
        parts = sum(len(t.get('parts', [])) for t in tracks)
        if parts <= 0:
            return TestOutcome("e2e", False, "no BVH parts produced")
        return TestOutcome("e2e", True, f"tracks={len(tracks)} parts={parts}")
    except Exception as e:
        return TestOutcome("e2e", False, str(e))


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--python", required=True, type=Path, help="Path to external Python (venv) created by the add‑on")
    p.add_argument("--mmpose", required=True, type=Path, help="Path to local mmpose repo")
    p.add_argument("--motionbert", required=True, type=Path, help="Path to local MotionBERT repo")
    p.add_argument("--video", type=Path, help="Path to a short test video (optional but recommended)")
    p.add_argument("--device", default="auto", help="cpu | cuda:0 | auto")
    p.add_argument("--quick", action="store_true", help="Run a faster subset of tests")
    p.add_argument("--skip-e2e", action="store_true", help="Skip end‑to‑end test even if video provided")
    p.add_argument("--framecap", type=int, default=120, help="Max frames per track for the E2E stub")
    p.add_argument("--workdir", type=Path, default=Path(tempfile.gettempdir())/"v2r_tests", help="Output directory")

    args = p.parse_args(argv)
    setup_logging(args.workdir)

    py = args.python
    if not py.exists():
        LOG.error("Python not found: %s", py); return 2
    if args.device == "auto":
        args.device = detect_device(py)
    LOG.info("Using device: %s", args.device)

    results: list[TestOutcome] = []

    # 1) versions
    results.append(test_versions(py))

    # 2) mmpose image demo
    results.append(test_mmpose_image(py, args.mmpose, args.workdir, args.device))

    # 3) motionbert synth
    results.append(test_motionbert_synth(py, args.motionbert, args.workdir, args.device))

    # 4) end‑to‑end (optional)
    if args.video and not args.skip_e2e:
        results.append(test_end_to_end(py, args.mmpose, args.motionbert, args.video, args.workdir, args.device, args.framecap))
    else:
        LOG.info("Skipping E2E (no --video or --skip-e2e)")

    # Summary
    LOG.info("\n==== SUMMARY ====")
    failures = 0
    for r in results:
        status = "PASS" if r.ok else "FAIL"
        LOG.info("%-18s %s", f"[{status}]", r.name)
        if r.detail:
            LOG.info("  → %s", textwrap.shorten(r.detail.replace('\n',' '), width=160))
        if r.artifact:
            LOG.info("  • artifact: %s", r.artifact)
        failures += 0 if r.ok else 1

    LOG.info("=================\nArtifacts and logs in: %s", args.workdir)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
