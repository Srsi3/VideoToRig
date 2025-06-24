#!/usr/bin/env python3
"""
Video-To-Rigify Pipeline — self-installing add-on
=================================================
This version adds **one-click dependency setup** *without polluting Blender own
Python*.  When the user presses *Install Dependencies* we:

1. Create a **virtual-env** inside Blender's config folder (`…/config/video2rigify_env`).
2. `pip install` the required wheels (CPU-only by default; CUDA 12 wheels if the
   user checks “GPU”).
3. Auto-download the MotionBERT checkpoint.
4. Save the venv's *python* path in the add-on preferences.

▶  After that, the main operator always calls that external python, so heavy ML
   libs stay isolated from Blender.

Compatible with Blender 4? → YES (uses built-in venv & subprocess).
"""

bl_info = {
    "name": "Video 2 Rigify Pipeline (auto-setup)",
    "author": "ChatGPT + You",
    "version": (0, 3, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Video 2 Rigify",
    "description": "Video → MMPose → MotionBERT → Rigify with one-click dependency install",
    "category": "Animation",
}

from pathlib import Path
import bpy, subprocess, sys, tempfile, importlib.util, platform, venv, urllib.request, json, os
from bpy.props import (StringProperty, BoolProperty, IntProperty, FloatProperty)
from bpy.types import (AddonPreferences, Operator, Panel)

# -----------------------------------------------------------------------------
#  Settings & helpers
# -----------------------------------------------------------------------------

REQ_MODULES = ("torch", "mmcv", "mmpose", "numpy")
MB_CKPT_URL = (
    "https://huggingface.co/walter0807/MotionBERT/resolve/main/"
    "mb_ft_h36m.bin"
)
ENV_DIRNAME = "video2rigify_env"


def missing_modules():
    return [m for m in REQ_MODULES if importlib.util.find_spec(m) is None]


def get_venv_python(env_path: Path) -> Path:
    if platform.system() == "Windows":
        return env_path / "Scripts" / "python.exe"
    return env_path / "bin" / "python"

# -----------------------------------------------------------------------------
#  Add‑on Preferences (stores env path & repo clones)
# -----------------------------------------------------------------------------

class V2R_Prefs(AddonPreferences):
    bl_idname = __name__

    python_exe: StringProperty(
        name="External Python",
        subtype='FILE_PATH',
        description="Automatically filled after Install Dependencies",
    )
    mmpose_repo: StringProperty(
        name="MMPose Repo", subtype='DIR_PATH', default="", description="Auto‑cloned if empty"
    )
    motionbert_repo: StringProperty(
        name="MotionBERT Repo", subtype='DIR_PATH', default="", description="Auto‑cloned if empty"
    )

    def draw(self, context):
        layout = self.layout
        col = layout.column()
        col.label(text="Environment:")
        col.prop(self, "python_exe")
        col.prop(self, "mmpose_repo")
        col.prop(self, "motionbert_repo")
        col.operator("v2r.install_deps", icon="CONSOLE")

# -----------------------------------------------------------------------------
#  Operator: install dependencies into isolated venv
# -----------------------------------------------------------------------------

class V2R_OT_InstallDeps(Operator):
    bl_idname = "v2r.install_deps"
    bl_label = "Install Dependencies"

    gpu: BoolProperty(name="GPU (CUDA 12.1)", default=False)

    def execute(self, context):
        prefs = context.preferences.addons[__name__].preferences
        config_dir = Path(bpy.utils.user_resource('CONFIG'))
        env_path = config_dir / ENV_DIRNAME
        env_path.mkdir(parents=True, exist_ok=True)

        # 1 — Create venv if missing
        if not get_venv_python(env_path).exists():
            self.report({'INFO'}, f"Creating virtual env at {env_path}")
            venv.create(env_path, with_pip=True, clear=False)

        py = get_venv_python(env_path)

        # 2 — Build pip install cmd list
        base_pkgs = [
            "openmim", "mmengine", "numpy", "scipy", "pymo", "mmpose==1.3.1",
        ]
        if self.gpu and platform.system() in {"Linux", "Windows"}:
            base_pkgs += [
                "torch==2.3.0+cu121", "torchvision==0.18.0+cu121",
                "torchaudio==2.3.0+cu121",
                "--extra-index-url", "https://download.pytorch.org/whl/cu121",
                "mmcv==2.0.1",
            ]

        else:
            base_pkgs += [
                "torch==2.3.0+cpu", "torchvision==0.18.0+cpu", "torchaudio==2.3.0+cpu",
                "mmcv==2.0.1",
            ]

        self.report({'INFO'}, "Installing wheels… this can take several minutes")
        cmd = [str(py), "-m", "pip", "install", "--upgrade", *base_pkgs]
        if subprocess.call(cmd) != 0:
            self.report({'ERROR'}, "pip install failed — check console")
            return {'CANCELLED'}

        # 3 — Auto‑clone repos if needed
        if not prefs.mmpose_repo:
            prefs.mmpose_repo = str(config_dir / "mmpose")
            subprocess.call(["git", "clone", "https://github.com/open-mmlab/mmpose", prefs.mmpose_repo])
        if not prefs.motionbert_repo:
            prefs.motionbert_repo = str(config_dir / "MotionBERT")
            subprocess.call(["git", "clone", "https://github.com/Walter0807/MotionBERT", prefs.motionbert_repo])

        # 4 — Download MotionBERT checkpoint
        ckpt_path = Path(prefs.motionbert_repo) / "checkpoints" / "mb_ft_h36m.bin"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        if not ckpt_path.exists():
            self.report({'INFO'}, "Downloading MotionBERT weights (~200 MB)…")
            urllib.request.urlretrieve(MB_CKPT_URL, ckpt_path)

        # 5 — Save python path
        prefs.python_exe = str(py)
        self.report({'INFO'}, "Dependencies installed ✔  — ready to run")
        return {'FINISHED'}

# -----------------------------------------------------------------------------
#  External stub script (unchanged, uses prefs.python_exe)
# -----------------------------------------------------------------------------

PIPELINE_STUB = """#!/usr/bin/env python3
import argparse, subprocess, tempfile
from pathlib import Path
parser = argparse.ArgumentParser(); parser.add_argument('video', type=Path); parser.add_argument('--out', type=Path, required=True); parser.add_argument('--mmpose', type=Path, required=True); parser.add_argument('--motionbert', type=Path, required=True); parser.add_argument('--device', default='cuda:0'); args = parser.parse_args()
work = Path(tempfile.mkdtemp(prefix='v2r_'))
pose_json = work / 'pose2d.json'; raw_bvh = work / 'raw.bvh'
subprocess.check_call(['python', args.mmpose / 'tools/inferencer/webcam_demo.py', '--input', args.video, '--vis-out', pose_json, '--device', args.device])
subprocess.check_call(['python', args.motionbert / 'apps/demo_pose3d.py', '--pose2d_json', pose_json, '--save_bvh', raw_bvh, '--device', args.device])
raw_bvh.replace(args.out)
print('BVH ready:', args.out)
"""

# -----------------------------------------------------------------------------
#  Extremes‑aware reduction (same as before)
# -----------------------------------------------------------------------------

def reduce_keys_extremes(action, err=0.02):
    try:
        import numpy as np
    except ImportError:
        print("[V2R] numpy missing – extremes reduction skipped")
        return
    for fc in action.fcurves:
        kps = fc.keyframe_points
        if len(kps) < 4:
            continue
        xs = np.array([kp.co.x for kp in kps]); ys = np.array([kp.co.y for kp in kps])
        grad = np.gradient(ys)
        extrema = np.where((grad[:-1] > 0) & (grad[1:] < 0) | (grad[:-1] < 0) & (grad[1:] > 0))[0] + 1
        keep = set(extrema.tolist() + [0, len(kps)-1])
        for i in reversed(range(1, len(kps)-1)):
            if i in keep: continue
            y_pred = np.interp(kps[i].co.x, xs[list(keep)], ys[list(keep)])
            if abs(y_pred - ys[i]) < err:
                kps.remove(kps[i])

# -----------------------------------------------------------------------------
#  Main operator — unchanged except it now uses prefs.python_exe by default
# -----------------------------------------------------------------------------

class V2R_OT_Run(Operator):
    bl_idname = "v2r.run_pipeline"
    bl_label = "Run Video → Rigify"

    video_path: StringProperty(name="Video", subtype='FILE_PATH')
    rig_name:   StringProperty(name="Rigify Armature", default="rig")
    bake_step:  IntProperty(name="Bake Step (frames)", default=1, min=1, soft_max=6)
    err_tol:    FloatProperty(name="Extreme Error Tol.", default=0.03, min=0.0, soft_max=0.1)

    def execute(self, context):
        prefs = context.preferences.addons[__name__].preferences
        if not prefs.python_exe or not Path(prefs.python_exe).exists():
            self.report({'ERROR'}, "Dependencies not installed — go to Preferences > Install Dependencies")
            return {'CANCELLED'}
        missing = missing_modules()
        if missing:
            self.report({'ERROR'}, f"Missing Python modules: {', '.join(missing)}")
            return {'CANCELLED'}

        video = Path(bpy.path.abspath(self.video_path))
        if not video.is_file():
            self.report({'ERROR'}, f"Video not found: {video}")
            return {'CANCELLED'}

        with tempfile.TemporaryDirectory(prefix="v2r_") as tmp:
            bvh_path = Path(tmp) / "anim.bvh"
            stub_py = Path(tmp) / "pipeline_stub.py"; stub_py.write_text(PIPELINE_STUB); stub_py.chmod(0o755)
            cmd = [prefs.python_exe, stub_py, video, '--out', bvh_path, '--mmpose', prefs.mmpose_repo, '--motionbert', prefs.motionbert_repo]
            self.report({'INFO'}, 'Running external pose pipeline…')
            if subprocess.call([str(c) for c in cmd]) != 0:
                self.report({'ERROR'}, 'External pipeline failed')
                return {'CANCELLED'}

            bpy.ops.import_anim.bvh(filepath=str(bvh_path), axis_forward='-Z', axis_up='Y')
            source_arm = context.selected_objects[0]
            target = context.scene.objects.get(self.rig_name)
            if target is None:
                self.report({'ERROR'}, f'Rigify armature "{self.rig_name}" not found')
                return {'CANCELLED'}
            if 'retarget_animation' not in context.preferences.addons:
                bpy.ops.preferences.addon_enable(module='retarget_animation')
            import retarget_animation
            retarget_animation.ui.build_bone_list(source_arm, target)
            retarget_animation.ui.retarget(target)

            bpy.ops.nla.bake(frame_start=context.scene.frame_start, frame_end=context.scene.frame_end, step=self.bake_step, visual_keying=True, clear_constraints=True, use_current_action=True, bake_types={'POSE'})
            if self.err_tol > 0:
                reduce_keys_extremes(target.animation_data.action, err=self.err_tol)

        self.report({'INFO'}, 'Motion retargeted ✔')
        return {'FINISHED'}

# -----------------------------------------------------------------------------
#  UI panel
# -----------------------------------------------------------------------------

class V2R_PT_Panel(Panel):
    bl_label = "Video → Rigify"
    bl_idname = "V2R_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Video 2 Rigify"

    def draw(self, context):
        layout = self.layout
        op = layout.operator(V2R_OT_Run.bl_idname, text="Run Pipeline", icon="PLAY")
        layout.prop(op, 'video_path'); layout.prop(op, 'rig_name'); layout.prop(op, 'bake_step'); layout.prop(op, 'err_tol')
        layout.separator(); layout.label(text="Install/Update dependencies in Add‑ons Prefs →")

# -----------------------------------------------------------------------------
#  Registration helpers
# -----------------------------------------------------------------------------

classes = (V2R_Prefs, V2R_OT_InstallDeps, V2R_OT_Run, V2R_PT_Panel)

def register():
    for c in classes: bpy.utils.register_class(c)
    # first‑run checker
    if missing_modules():
        def _msg(self, ctx): self.layout.label(text="Video2Rigify: dependencies missing — open Prefs > Install Deps")
        bpy.context.window_manager.popup_menu(_msg, title="Setup Required", icon='ERROR')

def unregister():
    for c in reversed(classes): bpy.utils.unregister
