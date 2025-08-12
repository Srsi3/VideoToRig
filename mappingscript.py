#!/usr/bin/env python3
# ------------------------------------------------------------------
#  Multi-Person Video-2-Rigify  –  one-click ML pipeline for Blender
# ------------------------------------------------------------------
from __future__ import annotations

bl_info = {
    "name":        "Multi-Person Video-2-Rigify",
    "author":      "Samir Saldanha",
    "version":     (0, 5, 1),
    "blender":     (4, 0, 0),
    "location":    "View3D ▸ Sidebar ▸ Video 2 Rigify",
    "description": "Video → MMPose → MotionBERT → Rigify (multi-person)",
    "category":    "Animation",
}

# ------------------------------------------------------------------
#  Imports
# ------------------------------------------------------------------
import bpy, os, sys, shutil, subprocess, tempfile, platform, venv, logging, importlib, importlib.util, json
from pathlib import Path
from bpy.types   import AddonPreferences, Operator, Panel, PropertyGroup
from bpy.props   import (StringProperty, BoolProperty, IntProperty,
                         FloatProperty, PointerProperty)

# ------------------------------------------------------------------
#  Constants
# ------------------------------------------------------------------
ADDON_ID      = __package__ or __name__
REQ_MODULES   = ("torch", "mmcv", "mmpose", "numpy")
ENV_DIRNAME   = "video2rigify_env"
LOG_NAME      = "video2rigify.log"
MB_FRAME_CAP  = 243  # motionbert hard cap

# ------------------------------------------------------------------
#  Logging
# ------------------------------------------------------------------
def get_logger() -> logging.Logger:
    """Log to both Blender console and to a file in the Blender config dir."""
    logger = logging.getLogger("Video2Rigify")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    try:
        cfg_dir = Path(bpy.utils.user_resource('CONFIG'))
    except Exception:
        cfg_dir = Path(tempfile.gettempdir())
    log_file = cfg_dir / LOG_NAME

    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S")
    fh.setFormatter(fmt); ch.setFormatter(fmt)

    logger.addHandler(fh); logger.addHandler(ch)
    logger.debug("Logger initialized → %s", log_file)
    return logger

log = get_logger()

# ------------------------------------------------------------------
#  Helpers
# ------------------------------------------------------------------
def missing_modules():
    return [m for m in REQ_MODULES if importlib.util.find_spec(m) is None]

def get_venv_python(env: Path) -> Path:
    exe = "Scripts/python.exe" if platform.system() == "Windows" else "bin/python"
    return env / exe

def addon_prefs():
    ad = bpy.context.preferences.addons.get(ADDON_ID)
    return ad.preferences if ad else None

def run_cmd(args, cwd=None, env=None, desc=""):
    """Run a subprocess, log stdout/stderr, raise on non-zero."""
    log.info("▶ %s", " ".join(map(str, args)))
    proc = subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True)
    if proc.stdout:
        log.debug(proc.stdout)
    if proc.stderr:
        log.warning(proc.stderr)
    if proc.returncode != 0:
        msg = f"Command failed ({desc or args[0]}), code={proc.returncode}"
        log.error(msg)
        raise RuntimeError(msg)
    return proc

# ------------------------------------------------------------------
#  Scene-level settings
# ------------------------------------------------------------------
class V2R_Props(PropertyGroup):
    video_path: StringProperty(name="Video", subtype='FILE_PATH')
    rig_name:   StringProperty(name="Rigify Armature", default="rig")
    bake_step:  IntProperty(  name="Bake Step", default=1, min=1, soft_max=6)
    err_tol:    FloatProperty(name="Extreme Error Tol.", default=0.03,
                              min=0.0, soft_max=0.1)
    duplicate_rigs: BoolProperty(name="Duplicate Rig per Person", default=True)

# ------------------------------------------------------------------
#  Preferences
# ------------------------------------------------------------------
class V2R_Prefs(AddonPreferences):
    bl_idname = ADDON_ID
    python_exe:      StringProperty(name="External Python", subtype='FILE_PATH')
    mmpose_repo:     StringProperty(name="MMPose Repo",  subtype='DIR_PATH')
    motionbert_repo: StringProperty(name="MotionBERT Repo", subtype='DIR_PATH')

    def draw(self, ctx):
        col = self.layout.column()
        col.label(text="Environment:")
        col.prop(self, "python_exe")
        col.prop(self, "mmpose_repo")
        col.prop(self, "motionbert_repo")

        row = col.row(align=True)
        op  = row.operator("v2r.install_deps", text="Install Deps (CPU)",
                           icon='CONSOLE');   op.gpu = False
        op  = row.operator("v2r.install_deps", text="Install Deps (GPU)",
                           icon='SHADERFX');  op.gpu = True

        col.separator()
        col.label(text="Maintenance", icon='TRASH')
        col.operator("v2r.uninstall", text="Remove Video2Rigify Data",
                     icon='TRASH')

# ------------------------------------------------------------------
#  Dependency-installer  –  robust (OpenMIM, NumPy pin, checkpoint copy)
# ------------------------------------------------------------------
class V2R_OT_InstallDeps(Operator):
    bl_idname = "v2r.install_deps"
    bl_label  = "Install Dependencies"

    gpu: BoolProperty(name="GPU (CUDA 12.1)", default=False)

    def execute(self, ctx):
        prefs   = ctx.preferences.addons[ADDON_ID].preferences
        cfg_dir = Path(bpy.utils.user_resource('CONFIG'))
        env_dir = cfg_dir / ENV_DIRNAME
        env_dir.mkdir(parents=True, exist_ok=True)

        # ---------- venv ----------
        py = get_venv_python(env_dir)
        if not py.exists():
            self.report({'INFO'}, f"Creating virtual-env at {env_dir}")
            log.info("Creating virtual-env at %s", env_dir)
            venv.create(env_dir, with_pip=True)

        # ---------- bootstrap ----------
        try:
            run_cmd([str(py), "-m", "pip", "install", "-U",
                     "pip", "wheel", "setuptools==60.2.0"], desc="bootstrap pip")
            run_cmd([str(py), "-m", "pip", "install", "-U", "openmim"], desc="install openmim")
        except Exception as e:
            self.report({'ERROR'}, f"Pip bootstrap failed: {e}")
            return {'CANCELLED'}

        # ---------- PyTorch ----------
        if self.gpu and platform.system() in {"Windows", "Linux"}:
            torch_pkgs = [
                "torch==2.3.0+cu121",
                "torchvision==0.18.0+cu121",
                "torchaudio==2.3.0+cu121",
                "--extra-index-url", "https://download.pytorch.org/whl/cu121",
            ]
        else:
            torch_pkgs = ["torch==2.3.0", "torchvision==0.18.0", "torchaudio==2.3.0"]
        try:
            run_cmd([str(py), "-m", "pip", "install", *torch_pkgs], desc="install torch stack")
        except Exception as e:
            self.report({'ERROR'}, f"PyTorch install failed: {e}")
            return {'CANCELLED'}

        # ---------- base deps via pip ----------
        base = [
            "numpy==1.26.4",  # avoid NumPy 2.x ABI break
            "mmengine",
            "scipy",
            "opencv-python",
            "pycocotools==2.0.7",
            "--no-build-isolation", "--no-binary=pymo", "pymo==0.2.0",
            "chumpy-fork==0.71",
            "--no-build-isolation", "chumpy==0.70",
        ]
        try:
            run_cmd([str(py), "-m", "pip", "install", *base], desc="install base deps")
        except Exception as e:
            self.report({'ERROR'}, f"Base deps install failed: {e}")
            return {'CANCELLED'}

        # ---------- mmcv/mmdet/mmpose via mim (pin correct wheel index) ----------
        try:
            run_cmd([str(py), "-m", "pip", "uninstall", "-y", "mmcv", "mmcv-lite"], desc="uninstall mmcv/mmcv-lite")
            run_cmd([str(py), "-m", "pip", "install", "-U", "openmim"], desc="ensure openmim")

            # pick wheel index matching torch 2.3.0 (+ cu121 or cpu)
            if self.gpu and platform.system() in {"Windows", "Linux"}:
                mmcv_index = "https://download.openmmlab.com/mmcv/dist/cu121/torch2.3/index.html"
            else:
                mmcv_index = "https://download.openmmlab.com/mmcv/dist/cpu/torch2.3/index.html"

            run_cmd([str(py), "-m", "mim", "install", "mmcv==2.0.1", "-f", mmcv_index], desc="mim mmcv full")
            run_cmd([str(py), "-m", "mim", "install", "mmdet==3.3.0"], desc="mim mmdet")
            run_cmd([str(py), "-m", "mim", "install", "mmpose==1.3.1"], desc="mim mmpose")
        except Exception as e:
            self.report({'ERROR'}, f"MIM install failed: {e}")
            return {'CANCELLED'}

        # ---------- clone repos if absent or missing ----------
        try:
            if not getattr(prefs, "mmpose_repo", "") or not Path(prefs.mmpose_repo).exists():
                prefs.mmpose_repo = str(cfg_dir / "mmpose")
            if not Path(prefs.mmpose_repo).exists():
                run_cmd(["git", "clone", "https://github.com/open-mmlab/mmpose", prefs.mmpose_repo],
                        desc="clone mmpose")

            if not getattr(prefs, "motionbert_repo", "") or not Path(prefs.motionbert_repo).exists():
                prefs.motionbert_repo = str(cfg_dir / "MotionBERT")
            if not Path(prefs.motionbert_repo).exists():
                run_cmd(["git", "clone", "https://github.com/Walter0807/MotionBERT",
                         prefs.motionbert_repo], desc="clone MotionBERT")
        except Exception as e:
            self.report({'ERROR'}, f"Git clone failed: {e}")
            return {'CANCELLED'}

        # ---------- MotionBERT checkpoint copy ----------
        ckpt_dir = Path(prefs.motionbert_repo) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        candidate_names = ["mb_ft_h36m.bin", "latest_epoch.bin", "mb_lite.bin"]

        def _already_there():
            return any((ckpt_dir / n).exists() for n in candidate_names)

        def _try_copy_from_resources():
            # zip-safe
            try:
                import importlib.resources as ires
                for n in candidate_names:
                    res = ires.files(__name__).joinpath("resources", n)
                    if res.is_file():
                        with res.open("rb") as src, open(ckpt_dir / n, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        return ckpt_dir / n
            except Exception:
                pass
            # filesystem fallback
            for n in candidate_names:
                fs = Path(__file__).parent / "resources" / n
                if fs.is_file():
                    shutil.copy2(fs, ckpt_dir / n)
                    return ckpt_dir / n
            return None

        if not _already_there():
            copied = _try_copy_from_resources()
            if copied:
                log.info("Copied bundled MotionBERT checkpoint → %s", copied)
                if copied.name != "mb_ft_h36m.bin":
                    try: shutil.copy2(copied, ckpt_dir / "mb_ft_h36m.bin")
                    except Exception: pass
            else:
                self.report({'WARNING'},
                            f"MotionBERT checkpoint missing — place one of "
                            f"{candidate_names} in {ckpt_dir}")

        prefs.python_exe = str(py)
        self.report({'INFO'}, "Dependencies installed ✔")
        log.info("Dependencies installed ✔")
        return {'FINISHED'}

# ------------------------------------------------------------------
#  Pipeline stub  – multi-person, tracks + ≤243-frame chunks per track
# ------------------------------------------------------------------
PIPELINE_STUB = r"""#!/usr/bin/env python3
import json, argparse, subprocess, tempfile, shutil, sys, os, math, logging
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
args = parser.parse_args()

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

pose_cfg  = args.mmpose / 'configs/body/2d_kpt_sview_rgb_img'
pose_cfg /= 'topdown_heatmap/coco/rtmpose_m_8xb256-210e_coco-256x192.py'
pose_ckpt = ('https://download.openmmlab.com/mmpose/v1/'
             'rtmpose/rtmpose_m_8xb256-210e_coco-256x192-a24f2126_20230323.pth')

out_dir = work / 'mmpose_out'
out_dir.mkdir(parents=True, exist_ok=True)

# --- run demo
cmd = [
    sys.executable, str(demo_py),
    str(det_cfg), det_ckpt,
    str(pose_cfg), pose_ckpt,
    '--video-path',     str(args.video),
    '--out-video-root', str(out_dir),
    '--save-predictions',
    '--device',         args.device
]
log.info("Run MMPose demo: %s", " ".join(map(str, cmd)))
proc = subprocess.run(cmd, capture_output=True, text=True)
if proc.stdout: log.info(proc.stdout)
if proc.stderr: log.warning(proc.stderr)
if proc.returncode:
    sys.exit('[V2R] pose demo failed')

# --- pick predictions file
pred_file = None
for cand in out_dir.rglob('*'):
    if cand.suffix.lower() in {'.json', '.pkl', '.npz'} and 'pred' in cand.stem.lower():
        pred_file = cand; break
if pred_file is None:
    for cand in out_dir.rglob('*.json'):
        pred_file = cand; break
if pred_file is None:
    sys.exit('[V2R] could not find predictions output')
log.info("Predictions: %s", pred_file)

# --- load predictions
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

# --- collect frames
frames = preds if isinstance(preds, list) else preds.get('predictions', [])
if not isinstance(frames, list) or not frames:
    sys.exit('[V2R] predictions structure unrecognized or empty')

# --- simple multi-person tracking (nearest-center, 1-frame memory)
tracks = []           # each: { 'id': int, 'last_center': np.array([x,y]), 'last_frame': int, 'items': [ {frame_id, keypoints} ] }
next_tid = 0
max_dist = 80.0       # px threshold for matching
for fi, frm in enumerate(frames):
    insts = _frame_instances(frm)
    cur = []
    for inst in insts:
        kpts = inst.get('keypoints') or inst.get('keypoints_2d') or inst.get('coordinates')
        if kpts is None: continue
        center = _inst_center(inst)
        score = float(inst.get('bbox_score') or inst.get('score') or 0.0)
        cur.append((kpts, center, score))

    # try match to existing (only those updated at fi-1)
    used = set()
    for ti, tr in enumerate(tracks):
        if tr['last_frame'] != fi-1:
            continue
        best_j, best_d = None, 1e9
        for j,(kpts, center, score) in enumerate(cur):
            if j in used or center is None or tr['last_center'] is None:
                continue
            d = float(np.linalg.norm(center - tr['last_center']))
            if d < best_d:
                best_d, best_j = d, j
        if best_j is not None and best_d <= max_dist:
            kpts, center, score = cur[best_j]
            used.add(best_j)
            h36m = coco_to_h36m(kpts).tolist()
            tr['items'].append({'frame_id': fi, 'keypoints': h36m})
            tr['last_center'] = center
            tr['last_frame']  = fi

    # spawn new tracks for unmatched
    for j,(kpts, center, score) in enumerate(cur):
        if j in used:
            continue
        h36m = coco_to_h36m(kpts).tolist()
        tracks.append({'id': next_tid, 'last_center': center, 'last_frame': fi, 'items': [{'frame_id': fi, 'keypoints': h36m}]})
        next_tid += 1

# drop very short tracks
tracks = [t for t in tracks if len(t['items']) >= 8]

args.outdir.mkdir(parents=True, exist_ok=True)

# --- run MotionBERT per track in chunks
all_track_outputs = []
mb_env = dict(os.environ)
# Help MotionBERT resolve local imports
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
        tjson = work / f'track{tr['id']}_part{ci:03d}.json'
        with tjson.open('w', encoding='utf8') as f: json.dump(chunk, f)
        out_bvh = args.outdir / f'track{tr['id']}_part{start:06d}_{end:06d}.bvh'
        cmd = [
            sys.executable,
            str(args.motionbert / 'apps/demo_pose3d.py'),
            '--pose2d_json', str(tjson),
            '--save_bvh',    str(out_bvh),
            '--device',      args.device
        ]
        log.info("Run MotionBERT: %s", " ".join(map(str, cmd)))
        proc = subprocess.run(cmd, capture_output=True, text=True, env=mb_env, cwd=str(args.motionbert))
        if proc.stdout: log.info(proc.stdout)
        if proc.stderr: log.warning(proc.stderr)
        if proc.returncode:
            log.error("MotionBERT failed on %s", tjson)
            continue
        bvh_paths.append(str(out_bvh))
    if bvh_paths:
        all_track_outputs.append({'id': tr['id'], 'parts': bvh_paths})

print(json.dumps({"tracks": all_track_outputs}))
"""

# ------------------------------------------------------------------
#  Key-reducer helper – extremes + in-between keepers
# ------------------------------------------------------------------
def reduce_keys_extremes(action, err=0.02, inter_keep=10):
    """
    Keep:
      - first/last
      - local extrema (velocity sign changes)
      - a few evenly spaced in-between keys (every `inter_keep` keys)
      - then drop keys whose value is within +/-err of linear interp.
    """
    import numpy as np
    for fc in action.fcurves:
        kps = fc.keyframe_points
        if len(kps) < 4:
            continue
        xs = np.array([kp.co.x for kp in kps]);  ys = np.array([kp.co.y for kp in kps])
        grad = np.gradient(ys)
        extrema = np.where(((grad[:-1] > 0) & (grad[1:] < 0)) |
                           ((grad[:-1] < 0) & (grad[1:] > 0)))[0] + 1
        keep = set(extrema.tolist() + [0, len(kps)-1])
        # evenly spaced in-between keepers
        keep.update(list(range(0, len(kps), max(2, inter_keep))))
        # cull by error tolerance
        for i in reversed(range(1, len(kps)-1)):
            if i in keep:
                continue
            y_pred = np.interp(kps[i].co.x, [xs[k] for k in sorted(keep)], [ys[k] for k in sorted(keep)])
            if abs(y_pred - ys[i]) < err:
                kps.remove(kps[i])

# ------------------------------------------------------------------
#  Main operator
# ------------------------------------------------------------------
class V2R_OT_Run(Operator):
    bl_idname = "v2r.run_pipeline"
    bl_label  = "Run Video → Rigify"

    def _choose_device(self, py_exe: str) -> str:
        """If torch cuda available in venv → cuda:0 else cpu."""
        try:
            out = subprocess.check_output([py_exe, "-c",
                "import torch;print('cuda:0' if torch.cuda.is_available() else 'cpu')"],
                text=True)
            return out.strip()
        except Exception:
            return "cpu"

    def execute(self, ctx):
        s = ctx.scene.v2r_settings
        video = Path(bpy.path.abspath(s.video_path))
        if not video.is_file():
            msg = f"Video not found: {video}"
            self.report({'ERROR'}, msg); log.error(msg)
            return {'CANCELLED'}

        prefs = addon_prefs()
        if not prefs or not Path(prefs.python_exe).exists():
            msg = "Install dependencies first (Prefs → Install Deps)"
            self.report({'ERROR'}, msg); log.error(msg)
            return {'CANCELLED'}

        device = self._choose_device(prefs.python_exe)
        log.info("Selected device: %s", device)

        with tempfile.TemporaryDirectory(prefix="v2r_") as tmp:
            outdir  = Path(tmp) / "bvhs"
            stub_py = Path(tmp) / "pipeline_stub.py"
            stub_py.write_text(PIPELINE_STUB, encoding='utf8'); stub_py.chmod(0o755)

            cmd = [prefs.python_exe, str(stub_py), str(video),
                   '--outdir', str(outdir),
                   '--mmpose', str(prefs.mmpose_repo),
                   '--motionbert', str(prefs.motionbert_repo),
                   '--device', device]
            self.report({'INFO'}, "Running external pose pipeline…")
            log.info("Run pipeline stub: %s", " ".join(map(str, cmd)))
            proc = subprocess.run(list(map(str, cmd)), capture_output=True, text=True)
            if proc.stdout: log.info(proc.stdout)
            if proc.stderr: log.warning(proc.stderr)
            if proc.returncode:
                self.report({'ERROR'}, "External pipeline failed")
                return {'CANCELLED'}

            # Parse JSON of BVH parts from stub's stdout (support old/new schema)
            tracks = []  # list of { 'id': int, 'parts': [path, ...] }
            for line in proc.stdout.splitlines():
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        if 'tracks' in obj:
                            tracks = obj['tracks']
                        elif 'bvh_parts' in obj:
                            tracks = [{'id': 0, 'parts': obj['bvh_parts']}]
                except Exception:
                    continue
            if not tracks:
                msg = "No BVH files produced by MotionBERT"
                self.report({'ERROR'}, msg); log.error(msg)
                return {'CANCELLED'}

            # -------- Blender-side retarget --------------------------------
            target_orig = ctx.scene.objects.get(s.rig_name)
            if not target_orig:
                msg = f'Rig "{s.rig_name}" not found'
                self.report({'ERROR'}, msg); log.error(msg)
                return {'CANCELLED'}

            # Try to enable & import the Animation-Retargeting add-on
            addon_mod = "animation_retargeting"
            if addon_mod not in ctx.preferences.addons:
                try: bpy.ops.preferences.addon_enable(module=addon_mod)
                except Exception: pass

            try:
                ar = importlib.import_module('animation_retargeting')
                has_ui = hasattr(ar, "ui") and hasattr(ar.ui, "build_bone_list")
            except Exception:
                ar = None; has_ui = False
                log.warning("Animation-Retargeting add-on not found; using fallback retarget")

            coll = (target_orig.users_collection[0]
                    if target_orig.users_collection else ctx.scene.collection)

            start_frame_global = ctx.scene.frame_start
            scene_end = start_frame_global

            for t_idx, tr in enumerate(sorted(tracks, key=lambda t: t.get('id', 0))):
                # Create/choose target armature
                if t_idx == 0 or not s.duplicate_rigs:
                    target = target_orig
                else:
                    target = target_orig.copy()
                    target.data = target_orig.data.copy()
                    target.animation_data_clear()
                    target.name = f"{s.rig_name}_{t_idx+1}"
                    coll.objects.link(target)

                cur_offset = 0
                for part_path in tr.get('parts', []):
                    bvh = Path(part_path)
                    if not bvh.exists():
                        log.error("Missing BVH part: %s", bvh)
                        continue

                    log.info("Import BVH: %s", bvh)
                    # Import BVH -> creates a new armature with action
                    try:
                        bpy.ops.object.select_all(action='DESELECT')
                    except Exception:
                        pass
                    bpy.ops.import_anim.bvh(filepath=str(bvh), axis_forward='-Z', axis_up='Y')

                    # pick the imported armature (selected & type ARMATURE)
                    source_arm = None
                    for o in bpy.context.selected_objects:
                        if o.type == 'ARMATURE':
                            source_arm = o; break
                    if source_arm is None:
                        self.report({'ERROR'}, "BVH import failed")
                        log.error("BVH import yielded no armature selection")
                        return {'CANCELLED'}

                    src_action = source_arm.animation_data.action if source_arm.animation_data else None
                    src_len = int(src_action.frame_range[1] - src_action.frame_range[0]) + 1 if src_action else 0

                    # Retarget
                    if has_ui:
                        try:
                            ar.ui.build_bone_list(source_arm, target)
                            ar.ui.retarget(target)
                        except Exception as e:
                            log.error("Animation-Retargeting error: %s", e)
                            has_ui = False

                    if not has_ui:
                        # naive per-frame paste of world matrices
                        fr_start = start_frame_global + cur_offset
                        fr_end   = fr_start + max(src_len, 0)
                        for f in range(fr_start, fr_end, max(1, s.bake_step)):
                            ctx.scene.frame_set(f)
                            # best-effort name-based mapping
                            for b_s in source_arm.pose.bones:
                                b_t = target.pose.bones.get(b_s.name)
                                if b_t is not None:
                                    b_t.matrix = b_s.matrix

                    # Bake to keyframes on target
                    fr_start = start_frame_global + cur_offset
                    fr_end   = fr_start + (src_len if src_len else ctx.scene.frame_end - fr_start)
                    bpy.ops.nla.bake(frame_start=fr_start,
                                     frame_end=fr_end,
                                     step=max(1, s.bake_step),
                                     visual_keying=True,
                                     clear_constraints=True,
                                     use_current_action=True,
                                     bake_types={'POSE'})

                    # Clean up source armature to avoid clutter
                    try:
                        bpy.data.objects.remove(source_arm, do_unlink=True)
                    except Exception:
                        pass

                    # next chunk starts after this
                    cur_offset += max(src_len, 0)
                    scene_end = max(scene_end, start_frame_global + cur_offset)

                # Key reduction per track (on the last baked action)
                if target.animation_data and target.animation_data.action:
                    try:
                        reduce_keys_extremes(target.animation_data.action, err=s.err_tol, inter_keep=10)
                    except Exception as e:
                        log.error("Key reduction failed: %s", e)

            # Expand scene end to fit all baked keys
            try:
                ctx.scene.frame_end = max(ctx.scene.frame_end, scene_end)
            except Exception:
                pass

        self.report({'INFO'}, "Retarget finished ✔ (see log for details)")
        log.info("Retarget finished ✔")
        return {'FINISHED'}

# ------------------------------------------------------------------
#  UI panel
# ------------------------------------------------------------------
class V2R_PT_Panel(Panel):
    bl_idname      = "V2R_PT_panel"
    bl_label       = "Video → Rigify"
    bl_space_type  = 'VIEW_3D';  bl_region_type = 'UI'
    bl_category    = "Video 2 Rigify"

    def draw(self, ctx):
        s  = ctx.scene.v2r_settings
        col = self.layout.column()
        col.prop(s, "video_path")
        col.prop(s, "rig_name")
        col.prop(s, "bake_step")
        col.prop(s, "err_tol")
        col.prop(s, "duplicate_rigs")
        col.separator()
        col.operator("v2r.run_pipeline", text="Run Pipeline", icon='PLAY')
        col.separator()
        col.label(text="Install / Update deps in Add-on Prefs →")

# ------------------------------------------------------------------
#  Un-installer
# ------------------------------------------------------------------
class V2R_OT_Uninstall(Operator):
    bl_idname = "v2r.uninstall"
    bl_label  = "Remove Video2Rigify Data"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        prefs = addon_prefs()
        cfg   = Path(bpy.utils.user_resource('CONFIG'))
        def nuke(p: Path):
            try:
                if p.is_dir(): shutil.rmtree(p, ignore_errors=True)
                else: p.unlink(missing_ok=True)
                log.info("Removed %s", p)
            except Exception as e:
                log.error("Couldn't remove %s: %s", p, e)

        for p in (cfg/ENV_DIRNAME,
                  Path(getattr(prefs, "mmpose_repo", "")),
                  Path(getattr(prefs, "motionbert_repo", ""))):
            if p and p.exists():
                nuke(p)
        if prefs:
            prefs.python_exe = prefs.mmpose_repo = prefs.motionbert_repo = ""
        self.report({'INFO'}, "Video2Rigify data removed – disable add-on to finish")
        return {'FINISHED'}

# ------------------------------------------------------------------
#  Registration
# ------------------------------------------------------------------
classes = (
    V2R_Props, V2R_Prefs, V2R_OT_InstallDeps, V2R_OT_Run,
    V2R_OT_Uninstall, V2R_PT_Panel
)

def register():
    for c in classes: bpy.utils.register_class(c)
    bpy.types.Scene.v2r_settings = PointerProperty(type=V2R_Props)
    # Friendly popup if Blender's own Python misses libs (we use venv anyway)
    if missing_modules():
        def _msg(self, _): self.layout.label(text="Video2Rigify: install deps in Add-on Prefs")
        try: bpy.context.window_manager.popup_menu(_msg, title="Setup Required", icon='ERROR')
        except Exception: pass
    log.info("Add-on registered")

def unregister():
    for c in reversed(classes):
        try: bpy.utils.unregister_class(c)
        except Exception: pass
    if hasattr(bpy.types.Scene, "v2r_settings"):
        del bpy.types.Scene.v2r_settings
    log.info("Add-on unregistered")
