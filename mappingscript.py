#!/usr/bin/env python3
"""
Video–To–Rigify Pipeline Blender Add‑On
======================================
A *single‑file* Blender add‑on that takes a video clip, runs 2‑D pose detection
(MMPose), 3‑D lifting (MotionBERT), optionally smooths the motion, and retargets
it onto a Rigify armature in the current scene.

Install: `Edit ▸ Preferences ▸ Add‑ons ▸ Install…`, pick this file, enable it.

UI:  *3D‑View ⟶ Sidebar (N) ⟶ «Video 2 Rigify» tab*

All heavy dependencies live **outside** Blender.  The add‑on shells out to your
system Python, so you can keep Torch, MMPose, MotionBERT, etc., in Conda without
trying to compile them inside Blender’s Python.

‖—————————————————————————————————————————————————————————————————————————‖
"""

bl_info = {
    "name": "Video 2 Rigify Pipeline",
    "author": "ChatGPT + You",
    "version": (0, 1, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Video 2 Rigify",
    "description": "Run MMPose + MotionBERT on a video and retarget to a Rigify rig",
    "category": "Animation",
}

from pathlib import Path
import subprocess, sys, tempfile, json, shutil, bpy, os
from bpy.props import (
    StringProperty,
    BoolProperty,
    IntProperty,
    FloatProperty,
)
from bpy.types import (AddonPreferences, Operator, Panel)

def _quote(p: Path | str) -> str:
    return str(p).replace("\\", "/")

# -----------------------------------------------------------------------------
#  ADD‑ON PREFERENCES — user sets repo paths & python interpreter
# -----------------------------------------------------------------------------

class V2R_Prefs(AddonPreferences):
    bl_idname = __name__

    system_python: StringProperty(
        name="Python Executable",
        subtype='FILE_PATH',
        description="External Python with Torch, MMPose, MotionBERT installed",
        default=sys.executable,
    )
    mmpose_repo: StringProperty(
        name="MMPose Repo",
        subtype='DIR_PATH',
        default="~/src/mmpose",
    )
    motionbert_repo: StringProperty(
        name="MotionBERT Repo",
        subtype='DIR_PATH',
        default="~/src/MotionBERT",
    )

    def draw(self, context):
        layout = self.layout
        layout.label(text="External Environment")
        layout.prop(self, "system_python")
        layout.prop(self, "mmpose_repo")
        layout.prop(self, "motionbert_repo")

# -----------------------------------------------------------------------------
#  STAGE SCRIPTS (calculated relative to this add‑on file at runtime)
# -----------------------------------------------------------------------------

PIPELINE_STUB = """#!/usr/bin/env python3
import argparse, subprocess, json, tempfile, os, sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("video", type=Path)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--mmpose", type=Path, required=True)
parser.add_argument("--motionbert", type=Path, required=True)
parser.add_argument("--device", default="cuda:0")
args = parser.parse_args()

work = Path(tempfile.mkdtemp(prefix="v2r_"))
pose_json = work / "pose2d.json"
raw_bvh   = work / "raw.bvh"

# 1 — MMPose (very thin wrapper)
subprocess.check_call([
    "python", args.mmpose / "tools/inferencer/webcam_demo.py", "--input", args.video,
    "--vis-out", pose_json, "--device", args.device,
])

# 2 — MotionBERT
subprocess.check_call([
    "python", args.motionbert / "apps/demo_pose3d.py",
    "--pose2d_json", pose_json, "--save_bvh", raw_bvh,
    "--device", args.device,
])

# Copy result to caller
raw_bvh.replace(args.out)
print("BVH ready:", args.out)
"""

# -----------------------------------------------------------------------------
#  OPERATOR — runs the pipeline then retargets
# -----------------------------------------------------------------------------

class V2R_OT_Run(Operator):
    bl_idname = "v2r.run_pipeline"
    bl_label = "Run Video → Rigify"

    video_path: StringProperty(name="Video", subtype='FILE_PATH')
    rig_name:   StringProperty(name="Rigify Armature", default="rig")
    key_step:   IntProperty(name="Bake Step", default=1, min=1, soft_max=6)
    smooth_sigma: FloatProperty(name="Smooth σ", default=2.0, min=0.0, soft_max=5.0)

    def execute(self, context):
        prefs = context.preferences.addons[__name__].preferences
        python = Path(bpy.path.abspath(prefs.system_python)).expanduser()
        mmpose = Path(bpy.path.abspath(prefs.mmpose_repo)).expanduser()
        motionbert = Path(bpy.path.abspath(prefs.motionbert_repo)).expanduser()
        video = Path(bpy.path.abspath(self.video_path))
        if not video.is_file():
            self.report({'ERROR'}, f"Video not found: {video}")
            return {'CANCELLED'}

        # 1. run external pipeline → BVH
        with tempfile.TemporaryDirectory(prefix="v2r_") as tmp:
            bvh_path = Path(tmp) / "anim.bvh"
            stub_py  = Path(tmp) / "pipeline_stub.py"
            stub_py.write_text(PIPELINE_STUB)
            stub_py.chmod(0o755)
            cmd = [python, stub_py, video, "--out", bvh_path,
                   "--mmpose", mmpose, "--motionbert", motionbert]
            self.report({'INFO'}, "Running external pipeline… this may take a while")
            try:
                subprocess.check_call([_quote(c) for c in cmd])
            except subprocess.CalledProcessError as e:
                self.report({'ERROR'}, f"Pipeline failed: {e}")
                return {'CANCELLED'}

            # 2. import BVH
            bpy.ops.import_anim.bvh(filepath=_quote(bvh_path), axis_forward='-Z', axis_up='Y')
            source_arm = context.selected_objects[0]
            target_arm = context.scene.objects.get(self.rig_name)
            if target_arm is None:
                self.report({'ERROR'}, f"Rigify armature '{self.rig_name}' not found")
                return {'CANCELLED'}

            # 3. Retarget with built‑in add‑on
            if 'retarget_animation' not in context.preferences.addons:
                bpy.ops.preferences.addon_enable(module='retarget_animation')
            import retarget_animation
            retarget_animation.ui.build_bone_list(source_arm, target_arm)
            retarget_animation.ui.retarget(target_arm)

            # 4. Bake to keyframes with step
            bpy.ops.nla.bake(frame_start=context.scene.frame_start,
                             frame_end=context.scene.frame_end,
                             step=self.key_step,
                             visual_keying=True,
                             clear_constraints=True,
                             use_current_action=True,
                             bake_types={'POSE'})

            # 5. Optional F‑Curve Decimate for extra sparsity
            if self.key_step == 1 and self.smooth_sigma > 0:
                bpy.ops.graph.decimate(mode='ERROR', remove_error=self.smooth_sigma)

        self.report({'INFO'}, "Video motion retargeted to Rigify rig ✔")
        return {'FINISHED'}

# -----------------------------------------------------------------------------
#  UI PANEL
# -----------------------------------------------------------------------------

class V2R_PT_Panel(Panel):
    bl_label = "Video → Rigify"
    bl_idname = "V2R_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Video 2 Rigify"

    def draw(self, context):
        layout = self.layout
        op = layout.operator(V2R_OT_Run.bl_idname, text="Run Pipeline")
        row = layout.row()
        row.prop(op, "video_path")
        row = layout.row()
        row.prop(op, "rig_name")
        layout.prop(op, "key_step")
        layout.prop(op, "smooth_sigma")
        layout.separator()
        layout.label(text="Preferences (set paths in 👆 Add‑ons)")

# -----------------------------------------------------------------------------
#  REGISTRATION
# -----------------------------------------------------------------------------

classes = (V2R_Prefs, V2R_OT_Run, V2R_PT_Panel)

def register():
    for c in classes:
        bpy.utils.register_class(c)

def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)

if __name__ == "__main__":
    register()
