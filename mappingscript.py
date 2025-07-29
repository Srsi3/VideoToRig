#!/usr/bin/env python3
#Todo
#testing
#check how good rigify motionBERT conversion works
#update UI
#see if video liks could work


# ------------------------------------------------------------------
#  Imports
# ------------------------------------------------------------------
from __future__ import annotations

bl_info = {
    "name":        "Multi-Person Video-2-Rigify",
    "author":      "Samir Saldanha",
    "version":     (0, 4, 0),
    "blender":     (4, 0, 0),
    "location":    "View3D ▸ Sidebar ▸ Video 2 Rigify",
    "description": "Video → MMPose (tracking) → MotionBERT → Rigify (multi-person)",
    "category":    "Animation",
}

import bpy, shutil, subprocess, sys, tempfile, importlib.util, platform, venv, urllib.request, json, os
from pathlib import Path
from typing import Annotated
from bpy.types   import AddonPreferences, Operator, Panel
from bpy.props   import (StringProperty, BoolProperty,
                          IntProperty, FloatProperty)

# ------------------------------------------------------------------
#  Constants & helpers
# ------------------------------------------------------------------
REQ_MODULES   = ("torch", "mmcv", "mmpose", "numpy")
MB_CKPT_URL   = "https://huggingface.co/walter0807/MotionBERT/resolve/main/mb_ft_h36m.bin"
ENV_DIRNAME   = "video2rigify_env"

def missing_modules():
    return [m for m in REQ_MODULES if importlib.util.find_spec(m) is None]

def get_venv_python(env_path: Path) -> Path:
    return env_path / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python")

# ------------------------------------------------------------------
#  Preferences
# ------------------------------------------------------------------
class V2R_Prefs(AddonPreferences):
    bl_idname = __name__

    python_exe:    Annotated[str, StringProperty(name="External Python",  subtype='FILE_PATH')]
    mmpose_repo:   Annotated[str, StringProperty(name="MMPose Repo",      subtype='DIR_PATH', description="Auto-cloned if empty")]
    motionbert_repo: Annotated[str, StringProperty(name="MotionBERT Repo", subtype='DIR_PATH', description="Auto-cloned if empty")]

    def draw(self, ctx):
        col = self.layout.column()
        col.label(text="Environment:")
        col.prop(self, "python_exe");  col.prop(self, "mmpose_repo"); col.prop(self, "motionbert_repo")
        col.operator("v2r.install_deps", icon="CONSOLE")
        col.separator(); col.label(text="Maintenance", icon='TRASH')
        col.operator("v2r.uninstall", text="Remove Video2Rigify Data", icon='TRASH')

# ------------------------------------------------------------------
#  Dependency-installer
# ------------------------------------------------------------------
class V2R_OT_InstallDeps(Operator):
    bl_idname = "v2r.install_deps"
    bl_label  = "Install Dependencies"
    gpu: BoolProperty(name="GPU (CUDA 12.1)", default=False)

    def execute(self, ctx):
        prefs      = ctx.preferences.addons[__name__].preferences
        cfg_dir    = Path(bpy.utils.user_resource('CONFIG'))
        env_path   = cfg_dir / ENV_DIRNAME
        env_path.mkdir(parents=True, exist_ok=True)

        # 1 — create venv
        if not get_venv_python(env_path).exists():
            self.report({'INFO'}, f"Creating virtual-env at {env_path}")
            venv.create(env_path, with_pip=True, clear=False)

        py = get_venv_python(env_path)

        # 2 — pip wheels
        pkgs = ["openmim", "mmengine", "numpy", "scipy", "pymo",
                "chumpy @ git+https://github.com/vchoutas/chumpy.git#egg=chumpy",
                "mmpose==1.3.1"]
        if self.gpu and platform.system() in {"Linux", "Windows"}:
            pkgs += ["torch==2.3.0+cu121", "torchvision==0.18.0+cu121", "torchaudio==2.3.0+cu121",
                     "--extra-index-url", "https://download.pytorch.org/whl/cu121", "mmcv==2.0.1"]
        else:
            pkgs += ["torch==2.3.0", "torchvision==0.18.0", "torchaudio==2.3.0", "mmcv==2.0.1"]

        self.report({'INFO'}, "Installing wheels …")
        if subprocess.call([str(py), "-m", "pip", "install", "--upgrade", *pkgs]) != 0:
            self.report({'ERROR'}, "pip install failed — see console")
            return {'CANCELLED'}

        # 3 — clone repos if absent
        if not prefs.mmpose_repo:
            prefs.mmpose_repo = str(cfg_dir / "mmpose")
            subprocess.call(["git", "clone", "https://github.com/open-mmlab/mmpose", prefs.mmpose_repo])
        if not prefs.motionbert_repo:
            prefs.motionbert_repo = str(cfg_dir / "MotionBERT")
            subprocess.call(["git", "clone", "https://github.com/Walter0807/MotionBERT", prefs.motionbert_repo])

        # 4 — download checkpoint
        ckpt = Path(prefs.motionbert_repo) / "checkpoints" / "mb_ft_h36m.bin"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        if not ckpt.exists():
            self.report({'INFO'}, "Downloading MotionBERT weights (200 MB)…")
            urllib.request.urlretrieve(MB_CKPT_URL, ckpt)

        prefs.python_exe = str(py)
        self.report({'INFO'}, "Dependencies installed ✔")
        return {'FINISHED'}

# ------------------------------------------------------------------
#  External stub — multi-person
# ------------------------------------------------------------------
PIPELINE_STUB = r"""#!/usr/bin/env python3
import json, argparse, subprocess, tempfile, shutil, os
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('video', type=Path)
parser.add_argument('--outdir', type=Path, required=True)
parser.add_argument('--mmpose', type=Path, required=True)
parser.add_argument('--motionbert', type=Path, required=True)
parser.add_argument('--device', default='cuda:0')
args = parser.parse_args()

work = Path(tempfile.mkdtemp(prefix='v2r_'))
pose_json = work / 'pose2d_all.json'
# 1 — multi-person 2-D detection + tracking
subprocess.check_call([
    'python', args.mmpose / 'tools/inferencer/video_demo.py',
    '--input', args.video, '--vis_out', pose_json,
    '--tracking', '1', '--det-cat-id', '0', '--device', args.device
])

# 2 — split by track-id
with open(pose_json, 'r', encoding='utf8') as f: frames = json.load(f)
tracks = {}
for frm in frames:
    for inst in frm['instances']:
        tid = inst.get('track_id', 0)
        tracks.setdefault(tid, []).append({'keypoints': inst['keypoints'], 'frame_id': frm['frame_id']})

# 3 — run MotionBERT per track
args.outdir.mkdir(parents=True, exist_ok=True)
for tid, data in tracks.items():
    tjson = work / f'track{tid}.json'
    with open(tjson, 'w') as f: json.dump(data, f)
    out_bvh = args.outdir / f'track{tid}.bvh'
    subprocess.check_call([
        'python', args.motionbert / 'apps/demo_pose3d.py',
        '--pose2d_json', tjson, '--save_bvh', out_bvh,
        '--device', args.device
    ])
print('[V2R] BVHs ready in', args.outdir)
"""

# ------------------------------------------------------------------
#  Extremes-aware reducer (unchanged)
# ------------------------------------------------------------------
def reduce_keys_extremes(action, err=0.02):
    try: import numpy as np
    except ImportError:
        print("[V2R] numpy missing – reduction skipped");  return
    for fc in action.fcurves:
        kps = fc.keyframe_points
        if len(kps) < 4:  continue
        xs = np.array([kp.co.x for kp in kps]);  ys = np.array([kp.co.y for kp in kps])
        grad = np.gradient(ys)
        extrema = np.where((grad[:-1] > 0) & (grad[1:] < 0) | (grad[:-1] < 0) & (grad[1:] > 0))[0] + 1
        keep = set(extrema.tolist() + [0, len(kps)-1])
        for i in reversed(range(1, len(kps)-1)):
            if i in keep: continue
            y_pred = np.interp(kps[i].co.x, xs[list(keep)], ys[list(keep)])
            if abs(y_pred - ys[i]) < err:  kps.remove(kps[i])

# ------------------------------------------------------------------
#  Main operator
# ------------------------------------------------------------------
class V2R_OT_Run(Operator):
    bl_idname = "v2r.run_pipeline"
    bl_label  = "Run Video → Rigify"

    video_path:      Annotated[str,  StringProperty(name="Video", subtype='FILE_PATH')]
    rig_name:        Annotated[str,  StringProperty(name="Rigify Armature", default="rig")]
    bake_step:       IntProperty(name="Bake Step", default=1, min=1, soft_max=6)
    err_tol:         FloatProperty(name="Extreme Error Tol.", default=0.03, min=0.0, soft_max=0.1)
    duplicate_rigs:  BoolProperty(name="Duplicate Rig per Person", default=True)

    def execute(self, ctx):
        prefs = ctx.preferences.addons[__name__].preferences
        if not prefs.python_exe or not Path(prefs.python_exe).exists():
            self.report({'ERROR'}, "Install dependencies first (Add-ons Prefs)");  return {'CANCELLED'}

        video = Path(bpy.path.abspath(self.video_path))
        if not video.is_file():
            self.report({'ERROR'}, f"Video not found: {video}");  return {'CANCELLED'}

        with tempfile.TemporaryDirectory(prefix="v2r_") as tmp:
            outdir  = Path(tmp) / "bvhs"
            stub_py = Path(tmp) / "pipeline_stub.py";  stub_py.write_text(PIPELINE_STUB);  stub_py.chmod(0o755)

            cmd = [prefs.python_exe, stub_py, video,
                   '--outdir', outdir,
                   '--mmpose', prefs.mmpose_repo,
                   '--motionbert', prefs.motionbert_repo]
            self.report({'INFO'}, "Running external pose pipeline…")
            if subprocess.call([str(c) for c in cmd]) != 0:
                self.report({'ERROR'}, "External pipeline failed");  return {'CANCELLED'}

            # -------------------------------------------------- Blender side
            target_orig = ctx.scene.objects.get(self.rig_name)
            if target_orig is None:
                self.report({'ERROR'}, f'Rig "{self.rig_name}" not found');  return {'CANCELLED'}

            if 'retarget_animation' not in ctx.preferences.addons:
                bpy.ops.preferences.addon_enable(module='retarget_animation')
            import retarget_animation

            coll = target_orig.users_collection[0] if target_orig.users_collection else ctx.scene.collection
            for idx, bvh in enumerate(sorted(outdir.glob("*.bvh"))):
                bpy.ops.import_anim.bvh(filepath=str(bvh), axis_forward='-Z', axis_up='Y')
                source_arm = ctx.selected_objects[0]

                if idx == 0:
                    target = target_orig
                else:
                    if self.duplicate_rigs:
                        target = target_orig.copy()
                        target.data = target_orig.data.copy()
                        target.name = f"{self.rig_name}_{idx+1}"
                        coll.objects.link(target)
                    else:
                        target = target_orig

                retarget_animation.ui.build_bone_list(source_arm, target)
                retarget_animation.ui.retarget(target)

                bpy.ops.nla.bake(frame_start=ctx.scene.frame_start,
                                 frame_end=ctx.scene.frame_end,
                                 step=self.bake_step,
                                 visual_keying=True, clear_constraints=True,
                                 use_current_action=True, bake_types={'POSE'})
                if self.err_tol > 0:
                    reduce_keys_extremes(target.animation_data.action, err=self.err_tol)

        self.report({'INFO'}, "Retarget finished ✔")
        return {'FINISHED'}

# ------------------------------------------------------------------
#  UI panel
# ------------------------------------------------------------------
class V2R_PT_Panel(Panel):
    bl_label       = "Video → Rigify"
    bl_idname      = "V2R_PT_panel"
    bl_space_type  = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category    = "Video 2 Rigify"

    def draw(self, ctx):
        op = self.layout.operator(V2R_OT_Run.bl_idname, text="Run Pipeline", icon="PLAY")
        self.layout.prop(op, "video_path"); self.layout.prop(op, "rig_name")
        self.layout.prop(op, "bake_step");  self.layout.prop(op, "err_tol")
        self.layout.prop(op, "duplicate_rigs")
        self.layout.separator(); self.layout.label(text="Install / Update deps in Add-on Prefs →")

# ------------------------------------------------------------------
#  Un-installer
# ------------------------------------------------------------------
class V2R_OT_Uninstall(Operator):
    bl_idname = "v2r.uninstall"
    bl_label  = "Remove Video2Rigify Data"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        prefs   = ctx.preferences.addons[__name__].preferences
        cfg_dir = Path(bpy.utils.user_resource('CONFIG'))
        nuke(cfg_dir / ENV_DIRNAME)
        for repo in (prefs.mmpose_repo, prefs.motionbert_repo):
            p = Path(repo) if repo else None
            if p and p.exists() and p.is_relative_to(cfg_dir):  nuke(p)
        prefs.python_exe = prefs.mmpose_repo = prefs.motionbert_repo = ""
        self.report({'INFO'}, "Video2Rigify data removed — disable add-on to finish")
        return {'FINISHED'}

def nuke(path: Path):
    try:
        shutil.rmtree(path) if path.is_dir() else path.unlink(missing_ok=True)
    except Exception as e:
        print(f"[V2R] Couldn't remove {path}: {e}")

# ------------------------------------------------------------------
#  Registration
# ------------------------------------------------------------------
classes = (V2R_Prefs, V2R_OT_InstallDeps, V2R_OT_Run, V2R_OT_Uninstall, V2R_PT_Panel)
def register():
    for c in classes: bpy.utils.register_class(c)
    if missing_modules():
        def _msg(self, _): self.layout.label(text="Video2Rigify: install deps in Add-on Prefs")
        bpy.context.window_manager.popup_menu(_msg, title="Setup Required", icon='ERROR')

def unregister():
    for c in reversed(classes): bpy.utils.unregister_class(c)
