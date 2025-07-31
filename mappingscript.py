#!/usr/bin/env python3
# Todo
# - testing
# - check how good rigify MotionBERT conversion works
# - update UI
# - see if video links could work

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
from bpy.types   import AddonPreferences, Operator, Panel
from bpy.props   import (StringProperty, BoolProperty, IntProperty, FloatProperty)

# A stable module/add-on id that works whether installed as a package or single file.
ADDON_ID = __package__ or __name__

# ------------------------------------------------------------------
#  Constants & helpers
# ------------------------------------------------------------------
REQ_MODULES   = ("torch", "mmcv", "mmpose", "numpy")
MB_CKPT_URL   = "https://huggingface.co/walter0807/MotionBERT/resolve/main/mb_ft_h36m.bin"
ENV_DIRNAME   = "video2rigify_env"

def missing_modules():
    # This checks Blender's embedded Python, not the external venv—used only for a friendly popup.
    return [m for m in REQ_MODULES if importlib.util.find_spec(m) is None]

def get_venv_python(env_path: Path) -> Path:
    return env_path / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python")

def addon_prefs():
    ad = bpy.context.preferences.addons.get(ADDON_ID)
    return getattr(ad, "preferences", None) if ad else None

# ------------------------------------------------------------------
#  Preferences
# ------------------------------------------------------------------
class V2R_Prefs(AddonPreferences):
    bl_idname = ADDON_ID  # IMPORTANT: tie prefs to this module id

    python_exe: StringProperty(
        name="External Python",
        subtype='FILE_PATH',
        default=""
    )
    mmpose_repo: StringProperty(
        name="MMPose Repo",
        subtype='DIR_PATH',
        description="Auto-cloned if empty",
        default=""
    )
    motionbert_repo: StringProperty(
        name="MotionBERT Repo",
        subtype='DIR_PATH',
        description="Auto-cloned if empty",
        default=""
    )

    def draw(self, ctx):
        col = self.layout.column()
        col.label(text="Environment:")
        col.prop(self, "python_exe")
        col.prop(self, "mmpose_repo")
        col.prop(self, "motionbert_repo")
        col.operator("v2r.install_deps", icon="CONSOLE")
        col.separator()
        col.label(text="Maintenance", icon='TRASH')
        col.operator("v2r.uninstall", text="Remove Video2Rigify Data", icon='TRASH')

# ------------------------------------------------------------------
#  Dependency-installer
# ------------------------------------------------------------------
class V2R_OT_InstallDeps(Operator):
    bl_idname = "v2r.install_deps"
    bl_label  = "Install Dependencies"
    gpu: BoolProperty(name="GPU (CUDA 12.1)", default=False)

    def execute(self, ctx):
        prefs = ctx.preferences.addons[ADDON_ID].preferences

        cfg_dir    = Path(bpy.utils.user_resource('CONFIG'))
        env_path   = cfg_dir / ENV_DIRNAME
        env_path.mkdir(parents=True, exist_ok=True)

        # 1 — create venv
        py = get_venv_python(env_path)
        if not py.exists():
            self.report({'INFO'}, f"Creating virtual-env at {env_path}")
            venv.create(env_path, with_pip=True, clear=False)

        # 0 — unify build tools *without* breaking openxlab’s pin
        subprocess.check_call([str(py), "-m", "pip", "install", "-U", "pip", "wheel"])
        subprocess.check_call([str(py), "-m", "pip", "install", "setuptools==60.2.0"])

        # 2 — pip wheels
        pkgs = [
            "openmim", "mmengine", "numpy", "scipy",

            # We need pymo wheel; try source build without isolation
            "--no-build-isolation", "--no-binary=pymo", "pymo==0.2.0",

            # chumpy fixes (some deps still pull this)
            "chumpy-fork==0.71",
            "--no-build-isolation", "chumpy==0.70",

            "mmpose==1.3.1",
        ]

        if self.gpu and platform.system() in {"Linux", "Windows"}:
            pkgs += [
                "torch==2.3.0+cu121", "torchvision==0.18.0+cu121", "torchaudio==2.3.0+cu121",
                "--extra-index-url", "https://download.pytorch.org/whl/cu121", "mmcv==2.0.1"
            ]
        else:
            pkgs += ["torch==2.3.0", "torchvision==0.18.0", "torchaudio==2.3.0", "mmcv==2.0.1"]

        self.report({'INFO'}, "Installing wheels …")
        if subprocess.call([str(py), "-m", "pip", "install", "--upgrade", *pkgs]) != 0:
            self.report({'ERROR'}, "pip install failed — see console")
            return {'CANCELLED'}

        # 3 — clone repos if absent or not set
        if not getattr(prefs, "mmpose_repo", ""):
            prefs.mmpose_repo = str(cfg_dir / "mmpose")
            subprocess.call(["git", "clone", "https://github.com/open-mmlab/mmpose", prefs.mmpose_repo])

        if not getattr(prefs, "motionbert_repo", ""):
            prefs.motionbert_repo = str(cfg_dir / "MotionBERT")
            subprocess.call(["git", "clone", "https://github.com/Walter0807/MotionBERT", prefs.motionbert_repo])

        # 4 — download checkpoint
        ckpt = Path(prefs.motionbert_repo) / "checkpoints" / "mb_ft_h36m.bin"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        if not ckpt.exists():
            try:
                self.report({'INFO'}, "Downloading MotionBERT weights (≈200 MB)…")
                urllib.request.urlretrieve(MB_CKPT_URL, ckpt)
            except Exception as e:
                self.report({'WARNING'}, f"Could not download MotionBERT weights automatically: {e}")

        # Save python path
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
    'python', str(args.mmpose / 'tools/inferencer/video_demo.py'),
    '--input', str(args.video),
    '--vis_out', str(pose_json),
    '--tracking', '1', '--det-cat-id', '0',
    '--device', args.device
])

# 2 — split by track-id
with open(pose_json, 'r', encoding='utf8') as f:
    frames = json.load(f)

tracks = {}
for frm in frames:
    for inst in frm.get('instances', []):
        tid = inst.get('track_id', 0)
        tracks.setdefault(tid, []).append({
            'keypoints': inst.get('keypoints', []),
            'frame_id': frm.get('frame_id', 0)
        })

# 3 — run MotionBERT per track
args.outdir.mkdir(parents=True, exist_ok=True)
for tid, data in tracks.items():
    tjson = work / f'track{tid}.json'
    with open(tjson, 'w', encoding='utf8') as f:
        json.dump(data, f)

    out_bvh = args.outdir / f'track{tid}.bvh'
    subprocess.check_call([
        'python', str(args.motionbert / 'apps/demo_pose3d.py'),
        '--pose2d_json', str(tjson),
        '--save_bvh', str(out_bvh),
        '--device', args.device
    ])

print('[V2R] BVHs ready in', args.outdir)
"""

# ------------------------------------------------------------------
#  Extremes-aware reducer (unchanged)
# ------------------------------------------------------------------
def reduce_keys_extremes(action, err=0.02):
    try:
        import numpy as np
    except ImportError:
        print("[V2R] numpy missing – reduction skipped")
        return
    for fc in action.fcurves:
        kps = fc.keyframe_points
        if len(kps) < 4:
            continue
        xs = np.array([kp.co.x for kp in kps]);  ys = np.array([kp.co.y for kp in kps])
        grad = np.gradient(ys)
        extrema = np.where(
            ((grad[:-1] > 0) & (grad[1:] < 0)) | ((grad[:-1] < 0) & (grad[1:] > 0))
        )[0] + 1
        keep = set(extrema.tolist() + [0, len(kps)-1])
        for i in reversed(range(1, len(kps)-1)):
            if i in keep:
                continue
            y_pred = np.interp(kps[i].co.x, xs[list(keep)], ys[list(keep)])
            if abs(y_pred - ys[i]) < err:
                kps.remove(kps[i])

# ------------------------------------------------------------------
#  Main operator
# ------------------------------------------------------------------
class V2R_OT_Run(Operator):
    bl_idname = "v2r.run_pipeline"
    bl_label  = "Run Video → Rigify"

    video_path:      StringProperty(name="Video", subtype='FILE_PATH', default="")
    rig_name:        StringProperty(name="Rigify Armature", default="rig")
    bake_step:       IntProperty(name="Bake Step", default=1, min=1, soft_max=6)
    err_tol:         FloatProperty(name="Extreme Error Tol.", default=0.03, min=0.0, soft_max=0.1)
    duplicate_rigs:  BoolProperty(name="Duplicate Rig per Person", default=True)

    def execute(self, ctx):
        prefs = addon_prefs()
        if not prefs or not getattr(prefs, "python_exe", "") or not Path(prefs.python_exe).exists():
            self.report({'ERROR'}, "Install dependencies first (Add-ons Prefs)")
            return {'CANCELLED'}

        video = Path(bpy.path.abspath(self.video_path))
        if not video.is_file():
            self.report({'ERROR'}, f"Video not found: {video}")
            return {'CANCELLED'}

        with tempfile.TemporaryDirectory(prefix="v2r_") as tmp:
            outdir  = Path(tmp) / "bvhs"
            stub_py = Path(tmp) / "pipeline_stub.py"
            stub_py.write_text(PIPELINE_STUB, encoding='utf8')
            stub_py.chmod(0o755)

            cmd = [
                str(prefs.python_exe), str(stub_py), str(video),
                '--outdir', str(outdir),
                '--mmpose', str(prefs.mmpose_repo),
                '--motionbert', str(prefs.motionbert_repo)
            ]
            self.report({'INFO'}, "Running external pose pipeline…")
            if subprocess.call(cmd) != 0:
                self.report({'ERROR'}, "External pipeline failed")
                return {'CANCELLED'}

            # -------------------------------------------------- Blender side
            target_orig = ctx.scene.objects.get(self.rig_name)
            if target_orig is None:
                self.report({'ERROR'}, f'Rig "{self.rig_name}" not found')
                return {'CANCELLED'}

            # Ensure retarget add-on is enabled
            if 'retarget_animation' not in ctx.preferences.addons:
                try:
                    bpy.ops.preferences.addon_enable(module='retarget_animation')
                except Exception:
                    pass
            try:
                import retarget_animation
            except Exception as e:
                self.report({'ERROR'}, f"retarget_animation add-on not available: {e}")
                return {'CANCELLED'}

            coll = target_orig.users_collection[0] if target_orig.users_collection else ctx.scene.collection
            for idx, bvh in enumerate(sorted(outdir.glob("*.bvh"))):
                bpy.ops.import_anim.bvh(filepath=str(bvh), axis_forward='-Z', axis_up='Y')
                if not ctx.selected_objects:
                    self.report({'ERROR'}, "BVH import failed (no selection)")
                    return {'CANCELLED'}

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

                # Retarget using the add-on's UI helpers
                retarget_animation.ui.build_bone_list(source_arm, target)
                retarget_animation.ui.retarget(target)

                bpy.ops.nla.bake(
                    frame_start=ctx.scene.frame_start,
                    frame_end=ctx.scene.frame_end,
                    step=self.bake_step,
                    visual_keying=True,
                    clear_constraints=True,
                    use_current_action=True,
                    bake_types={'POSE'}
                )

                if self.err_tol > 0 and target.animation_data and target.animation_data.action:
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
        self.layout.prop(op, "video_path")
        self.layout.prop(op, "rig_name")
        self.layout.prop(op, "bake_step")
        self.layout.prop(op, "err_tol")
        self.layout.prop(op, "duplicate_rigs")
        self.layout.separator()
        self.layout.label(text="Install / Update deps in Add-on Prefs →")

# ------------------------------------------------------------------
#  Un-installer
# ------------------------------------------------------------------
class V2R_OT_Uninstall(Operator):
    bl_idname = "v2r.uninstall"
    bl_label  = "Remove Video2Rigify Data"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        addon   = ctx.preferences.addons.get(ADDON_ID)
        prefs   = addon.preferences if addon and hasattr(addon, "preferences") else None
        cfg_dir = Path(bpy.utils.user_resource('CONFIG'))

        nuke(cfg_dir / ENV_DIRNAME)

        for repo in (getattr(prefs, "mmpose_repo", ""),
                     getattr(prefs, "motionbert_repo", "")):
            p = Path(repo) if repo else None
            try:
                if p and p.exists():
                    # Only delete if under the Blender config dir (safety)
                    if hasattr(p, "is_relative_to"):
                        should_delete = p.is_relative_to(cfg_dir)
                    else:
                        should_delete = str(p).startswith(str(cfg_dir))
                    if should_delete:
                        nuke(p)
            except Exception as e:
                print(f"[V2R] Couldn't evaluate removal for {p}: {e}")

        if prefs:
            prefs.python_exe = ""
            prefs.mmpose_repo = ""
            prefs.motionbert_repo = ""

        self.report({'INFO'}, "Video2Rigify data removed — disable add-on to finish")
        return {'FINISHED'}

def nuke(path: Path):
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
    except Exception as e:
        print(f"[V2R] Couldn't remove {path}: {e}")

# ------------------------------------------------------------------
#  Registration
# ------------------------------------------------------------------
classes = (V2R_Prefs, V2R_OT_InstallDeps, V2R_OT_Run, V2R_OT_Uninstall, V2R_PT_Panel)

def register():
    for c in classes:
        bpy.utils.register_class(c)
    # Friendly hint if Blender Python is missing libs (we use external venv anyway)
    if missing_modules():
        def _msg(self, _): self.layout.label(text="Video2Rigify: install deps in Add-on Prefs")
        try:
            bpy.context.window_manager.popup_menu(_msg, title="Setup Required", icon='ERROR')
        except Exception:
            # In certain headless contexts popup may fail—ignore
            pass

def unregister():
    for c in reversed(classes):
        try:
            bpy.utils.unregister_class(c)
        except Exception:
            pass
