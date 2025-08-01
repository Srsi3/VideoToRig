#!/usr/bin/env python3
# ------------------------------------------------------------------
#  Multi-Person Video-2-Rigify  –  one-click ML pipeline for Blender
# ------------------------------------------------------------------
from __future__ import annotations

bl_info = {
    "name":        "Multi-Person Video-2-Rigify",
    "author":      "Samir Saldanha",
    "version":     (0, 4, 2),
    "blender":     (4, 0, 0),
    "location":    "View3D ▸ Sidebar ▸ Video 2 Rigify",
    "description": "Video → MMPose → MotionBERT → Rigify (multi-person)",
    "category":    "Animation",
}

# ------------------------------------------------------------------
#  Imports
# ------------------------------------------------------------------
import bpy, shutil, subprocess, tempfile, importlib.util, platform, venv
import urllib.request, json, os, importlib
from pathlib import Path
from bpy.types   import AddonPreferences, Operator, Panel, PropertyGroup
from bpy.props   import (StringProperty, BoolProperty, IntProperty,
                          FloatProperty, PointerProperty)

ADDON_ID      = __package__ or __name__
REQ_MODULES   = ("torch", "mmcv", "mmpose", "numpy")
ENV_DIRNAME   = "video2rigify_env"

# ------------------------------------------------------------------
#  Helper utils
# ------------------------------------------------------------------
def missing_modules():
    return [m for m in REQ_MODULES if importlib.util.find_spec(m) is None]

def get_venv_python(env: Path) -> Path:
    exe = "Scripts/python.exe" if platform.system() == "Windows" else "bin/python"
    return env / exe

def addon_prefs():
    ad = bpy.context.preferences.addons.get(ADDON_ID)
    return ad.preferences if ad else None

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
#  Dependency-installer  (unchanged from your last file)
# ------------------------------------------------------------------
class V2R_OT_InstallDeps(Operator):
    bl_idname = "v2r.install_deps"
    bl_label  = "Install Dependencies"

    gpu: BoolProperty(name="GPU (CUDA 12.1)", default=False)

    def execute(self, ctx):
        prefs   = ctx.preferences.addons[ADDON_ID].preferences
        cfg_dir = Path(bpy.utils.user_resource('CONFIG'))
        env_dir = cfg_dir / ENV_DIRNAME;  env_dir.mkdir(parents=True, exist_ok=True)

        py = get_venv_python(env_dir)
        if not py.exists():
            self.report({'INFO'}, f"Creating virtual-env at {env_dir}")
            venv.create(env_dir, with_pip=True)

        base = [
            "openmim", "mmengine", "numpy", "scipy",
            "--no-build-isolation", "--no-binary=pymo", "pymo==0.2.0",
            "chumpy-fork==0.71", "--no-build-isolation", "chumpy==0.70",
            "mmpose==1.3.1"
        ]
        if self.gpu and platform.system() in {"Linux", "Windows"}:
            base += ["torch==2.3.0+cu121", "torchvision==0.18.0+cu121",
                     "torchaudio==2.3.0+cu121",
                     "--extra-index-url", "https://download.pytorch.org/whl/cu121",
                     "mmcv==2.0.1"]
        else:
            base += ["torch==2.3.0", "torchvision==0.18.0",
                     "torchaudio==2.3.0", "mmcv==2.0.1"]

        subprocess.check_call([str(py), "-m", "pip", "install", "-U",
                               "pip", "wheel", "setuptools==60.2.0"])
        if subprocess.call([str(py), "-m", "pip", "install", "--upgrade", *base]):
            self.report({'ERROR'}, "pip install failed"); return {'CANCELLED'}

        if not getattr(prefs, "mmpose_repo", ""):
            prefs.mmpose_repo = str(cfg_dir / "mmpose")
            subprocess.call(["git", "clone",
                             "https://github.com/open-mmlab/mmpose",
                             prefs.mmpose_repo])
        if not getattr(prefs, "motionbert_repo", ""):
            prefs.motionbert_repo = str(cfg_dir / "MotionBERT")
            subprocess.call(["git", "clone",
                             "https://github.com/Walter0807/MotionBERT",
                             prefs.motionbert_repo])

        ckpt = Path(prefs.motionbert_repo) / "checkpoints" / "mb_ft_h36m.bin"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        if not ckpt.exists():
            self.report({'WARNING'}, "MotionBERT checkpoint missing – "
                                     "place mb_ft_h36m.bin in "
                                     "MotionBERT/checkpoints/")
        prefs.python_exe = str(py)
        self.report({'INFO'}, "Dependencies installed ✔")
        return {'FINISHED'}

# ------------------------------------------------------------------
#  Pipeline stub  (unchanged placeholder)
# ------------------------------------------------------------------
PIPELINE_STUB = r"""#!/usr/bin/env python3
# your existing stub content …
"""

# ------------------------------------------------------------------
#  Key-reducer helper (unchanged)
# ------------------------------------------------------------------
def reduce_keys_extremes(action, err=0.02):
    try: import numpy as np
    except ImportError: print("[V2R] numpy missing – reduction skipped"); return
    for fc in action.fcurves:
        kps = fc.keyframe_points
        if len(kps) < 4: continue
        xs = np.array([kp.co.x for kp in kps]); ys = np.array([kp.co.y for kp in kps])
        grad = np.gradient(ys)
        extrema = np.where(((grad[:-1] > 0) & (grad[1:] < 0)) |
                           ((grad[:-1] < 0) & (grad[1:] > 0)))[0] + 1
        keep = set(extrema.tolist() + [0, len(kps)-1])
        for i in reversed(range(1, len(kps)-1)):
            if i in keep: continue
            y_pred = np.interp(kps[i].co.x, xs[list(keep)], ys[list(keep)])
            if abs(y_pred - ys[i]) < err: kps.remove(kps[i])

# ------------------------------------------------------------------
#  Main operator
# ------------------------------------------------------------------
class V2R_OT_Run(Operator):
    bl_idname = "v2r.run_pipeline"
    bl_label  = "Run Video → Rigify"

    def execute(self, ctx):
        s = ctx.scene.v2r_settings
        video = Path(bpy.path.abspath(s.video_path))
        if not video.is_file():
            self.report({'ERROR'}, f"Video not found: {video}"); return {'CANCELLED'}

        prefs = addon_prefs()
        if not prefs or not Path(prefs.python_exe).exists():
            self.report({'ERROR'}, "Install dependencies first"); return {'CANCELLED'}

        with tempfile.TemporaryDirectory(prefix="v2r_") as tmp:
            outdir  = Path(tmp) / "bvhs"
            stub_py = Path(tmp) / "pipeline_stub.py"
            stub_py.write_text(PIPELINE_STUB, encoding='utf8'); stub_py.chmod(0o755)

            cmd = [prefs.python_exe, stub_py, video,
                   '--outdir', outdir,
                   '--mmpose', prefs.mmpose_repo,
                   '--motionbert', prefs.motionbert_repo]
            self.report({'INFO'}, "Running external pose pipeline…")
            if subprocess.call(list(map(str, cmd))):
                self.report({'ERROR'}, "External pipeline failed"); return {'CANCELLED'}

            # -------- Blender-side retarget --------------------------------
            target_orig = ctx.scene.objects.get(s.rig_name)
            if not target_orig:
                self.report({'ERROR'}, f'Rig "{s.rig_name}" not found'); return {'CANCELLED'}

            # Try to enable & import the Animation-Retargeting add-on
            addon_mod = "animation_retargeting"
            if addon_mod not in ctx.preferences.addons:
                try: bpy.ops.preferences.addon_enable(module=addon_mod)
                except Exception: pass
            try:
                ar        = importlib.import_module(addon_mod)
                has_ui    = hasattr(ar, "ui") and hasattr(ar.ui, "build_bone_list")
            except ModuleNotFoundError:
                ar = None; has_ui = False

            if not has_ui:
                self.report({'WARNING'}, "Animation-Retargeting add-on missing "
                                         "or outdated – using simple fallback")

            coll = (target_orig.users_collection[0]
                    if target_orig.users_collection else ctx.scene.collection)

            for idx, bvh in enumerate(sorted(outdir.glob("*.bvh"))):
                bpy.ops.import_anim.bvh(filepath=str(bvh),
                                        axis_forward='-Z', axis_up='Y')
                if not ctx.selected_objects:
                    self.report({'ERROR'}, "BVH import failed"); return {'CANCELLED'}
                source_arm = ctx.selected_objects[0]

                if idx == 0 or not s.duplicate_rigs:
                    target = target_orig
                else:
                    target = target_orig.copy()
                    target.data = target_orig.data.copy()
                    target.name = f"{s.rig_name}_{idx+1}"
                    coll.objects.link(target)

                if has_ui:
                    ar.ui.build_bone_list(source_arm, target)
                    ar.ui.retarget(target)
                else:
                    # naive world-space copy
                    for f in range(ctx.scene.frame_start,
                                   ctx.scene.frame_end + 1, s.bake_step):
                        ctx.scene.frame_set(f)
                        for b_t, b_s in zip(target.pose.bones,
                                            source_arm.pose.bones):
                            b_t.matrix = b_s.matrix

                bpy.ops.nla.bake(frame_start=ctx.scene.frame_start,
                                 frame_end=ctx.scene.frame_end,
                                 step=s.bake_step,
                                 visual_keying=True,
                                 clear_constraints=True,
                                 use_current_action=True,
                                 bake_types={'POSE'})

                if (s.err_tol > 0 and target.animation_data
                        and target.animation_data.action):
                    reduce_keys_extremes(target.animation_data.action,
                                         err=s.err_tol)

        self.report({'INFO'}, "Retarget finished ✔")
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
#  Un-installer (unchanged)
# ------------------------------------------------------------------
class V2R_OT_Uninstall(Operator):
    bl_idname = "v2r.uninstall"
    bl_label  = "Remove Video2Rigify Data"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        prefs = addon_prefs()
        cfg   = Path(bpy.utils.user_resource('CONFIG'))
        def nuke(p: Path): shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
        for p in (cfg/ENV_DIRNAME,
                  Path(getattr(prefs, "mmpose_repo", "")),
                  Path(getattr(prefs, "motionbert_repo", ""))):
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
    if missing_modules():
        def _msg(self, _): self.layout.label(text="Video2Rigify: install deps in Add-on Prefs")
        try: bpy.context.window_manager.popup_menu(_msg,
                                                   title="Setup Required",
                                                   icon='ERROR')
        except Exception: pass

def unregister():
    for c in reversed(classes):
        try: bpy.utils.unregister_class(c)
        except Exception: pass
    if hasattr(bpy.types.Scene, "v2r_settings"):
        del bpy.types.Scene.v2r_settings
