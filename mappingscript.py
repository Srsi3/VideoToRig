#!/usr/bin/env python3
"""
Video–To–Rigify Pipeline Blender Add‑On  (extremes‑aware edition)
================================================================
A single‑file add‑on that runs MMPose → MotionBERT on a video and retargets the
result to a Rigify armature **with an extremes‑aware key‑reduction pass** so you
get sparse, editable keyframes that always keep important pose extremes.

Install → *Edit ▸ Preferences ▸ Add‑ons ▸ Install…* then enable
"Video 2 Rigify Pipeline".

UI lives in *3D‑View ▸ Sidebar (N) ▸ «Video 2 Rigify»*.
"""

bl_info = {
    "name": "Video 2 Rigify Pipeline",
    "author": "ChatGPT + You",
    "version": (0, 2, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Video 2 Rigify",
    "description": "Run MMPose + MotionBERT and retarget to Rigify with extremes‑aware key reduction",
    "category": "Animation",
}

from pathlib import Path
import subprocess, sys, tempfile, bpy, os
from bpy.props import (StringProperty, BoolProperty, IntProperty, FloatProperty)
from bpy.types import (AddonPreferences, Operator, Panel)
import importlib

# -----------------------------------------------------------------------------
#  Add‑on preferences — tell Blender where your external env lives
# -----------------------------------------------------------------------------

class V2R_Prefs(AddonPreferences):
    bl_idname = __name__

    system_python: StringProperty(
        name="Python Executable",
        subtype='FILE_PATH',
        default=sys.executable,
    )
    mmpose_repo: StringProperty(
        name="MMPose Repo", subtype='DIR_PATH', default="~/src/mmpose",
    )
    motionbert_repo: StringProperty(
        name="MotionBERT Repo", subtype='DIR_PATH', default="~/src/MotionBERT",
    )

    def draw(self, context):
        layout = self.layout
        layout.label(text="External Environment")
        layout.prop(self, "system_python")
        layout.prop(self, "mmpose_repo")
        layout.prop(self, "motionbert_repo")

# -----------------------------------------------------------------------------
#  Minimal external stub (runs outside Blender)
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
#  Extremes‑aware key‑reduction helper
# -----------------------------------------------------------------------------

def reduce_keys_extremes(action, err=0.02):
    """Keep local extrema in every F‑Curve; drop intermediates if error < *err*."""
    try:
        import numpy as np
    except ImportError:
        print("[V2R] numpy not found; skipping extremes‑aware reduction")
        return

    for fc in action.fcurves:
        kps = fc.keyframe_points
        if len(kps) < 4:
            continue
        xs = np.array([kp.co.x for kp in kps])
        ys = np.array([kp.co.y for kp in kps])
        grad = np.gradient(ys)
        extrema = np.where((grad[:-1] > 0) & (grad[1:] < 0) | (grad[:-1] < 0) & (grad[1:] > 0))[0] + 1
        keep = set(extrema.tolist() + [0, len(kps)-1])  # always first & last
        # iterate backwards to keep indices valid
        for i in reversed(range(1, len(kps)-1)):
            if i in keep:
                continue
            y_pred = np.interp(kps[i].co.x, xs[list(keep)], ys[list(keep)])
            if abs(y_pred - ys[i]) < err:
                kps.remove(kps[i])

# -----------------------------------------------------------------------------
#  Main operator
# -----------------------------------------------------------------------------

class V2R_OT_Run(Operator):
    bl_idname = "v2r.run_pipeline"
    bl_label = "Run Video → Rigify"

    video_path: StringProperty(name="Video", subtype='FILE_PATH')
    rig_name:   StringProperty(name="Rigify Armature", default="rig")
    bake_step:  IntProperty(name="Bake Step (frames)", default=1, min=1, soft_max=6)
    err_tol:    FloatProperty(name="Extreme Error Tolerance", default=0.03, min=0.0, soft_max=0.1, description="Max F‑Curve deviation when dropping keys (lower = more keys)")

    def execute(self, context):
        prefs = context.preferences.addons[__name__].preferences
        python = Path(bpy.path.abspath(prefs.system_python)).expanduser()
        mmpose = Path(bpy.path.abspath(prefs.mmpose_repo)).expanduser()
        motionbert = Path(bpy.path.abspath(prefs.motionbert_repo)).expanduser()
        video = Path(bpy.path.abspath(self.video_path))
        if not video.is_file():
            self.report({'ERROR'}, f"Video not found: {video}")
            return {'CANCELLED'}

        with tempfile.TemporaryDirectory(prefix="v2r_") as tmp:
            bvh_path = Path(tmp) / "anim.bvh"
            stub_py = Path(tmp) / "pipeline_stub.py"; stub_py.write_text(PIPELINE_STUB); stub_py.chmod(0o755)
            cmd = [python, stub_py, video, '--out', bvh_path, '--mmpose', mmpose, '--motionbert', motionbert]
            self.report({'INFO'}, 'Running external pose pipeline…')
            if subprocess.call([str(c) for c in cmd]) != 0:
                self.report({'ERROR'}, 'External pipeline failed')
                return {'CANCELLED'}

            # import BVH
            bpy.ops.import_anim.bvh(filepath=str(bvh_path), axis_forward='-Z', axis_up='Y')
            source_arm = context.selected_objects[0]
            target = context.scene.objects.get(self.rig_name)
            if target is None:
                self.report({'ERROR'}, f'Rigify armature "{self.rig_name}" not found')
                return {'CANCELLED'}

            # Retarget
            if 'retarget_animation' not in context.preferences.addons:
                bpy.ops.preferences.addon_enable(module='retarget_animation')
            import retarget_animation
            retarget_animation.ui.build_bone_list(source_arm, target)
            retarget_animation.ui.retarget(target)

            # Bake to keyframes
            bpy.ops.nla.bake(frame_start=context.scene.frame_start, frame_end=context.scene.frame_end, step=self.bake_step, visual_keying=True, clear_constraints=True, use_current_action=True, bake_types={'POSE'})

            # Extremes‑aware reduction
            action = target.animation_data.action
            if action and self.err_tol > 0:
                reduce_keys_extremes(action, err=self.err_tol)

        self.report({'INFO'}, 'Motion retargeted and reduced ✔')
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
        op = layout.operator(V2R_OT_Run.bl_idname, text="Run Pipeline")
        row = layout.row(); row.prop(op, 'video_path')
        row = layout.row(); row.prop(op, 'rig_name')
        layout.prop(op, 'bake_step')
        layout.prop(op, 'err_tol')
        layout.separator(); layout.label(text="Set repo paths in Add‑on Preferences →")

# -----------------------------------------------------------------------------
#  Registration helpers
# -----------------------------------------------------------------------------

classes = (V2R_Prefs, V2R_OT_Run, V2R_PT_Panel)

def register():
    for c in classes: bpy.utils.register_class(c)

def unregister():
    for c in reversed(classes): bpy.utils.unregister_class(c)

if __name__ == "__main__":
    register()
