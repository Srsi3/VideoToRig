# Multi‑Person Video‑2‑Rigify

A Blender 4.x add‑on that converts any **video of multiple people** into fully‑rigged **Rigify** animations, powered by **MMPose** for 2‑D pose tracking and **MotionBERT** for 3‑D pose lifting. Press one button, get cleaned keyframes on your own Rigify armatures.

## Key Features

* STILL IN DEVELOPMENT - CURRENTLY HAS ISSUES WITH MMPOSE NOT WORKING PROPERLY

* Supports **multiple people per video** with per‑track BVH export
* Extreme‑aware keyframe reduction (smooths curves while preserving peaks)
* Dependency installer in isolated virtual‑env keeps Blender Python clean
* CPU **or** CUDA 12.1 GPU wheels
* Automatic cloning of **MMPose** & **MotionBERT**, plus checkpoint download
* Optional duplication of the target Rigify rig for each person
* Fully integrated into the **View3D ▸ Sidebar ▸ Video 2 Rigify** panel

## Quick‑Start

### 1. Install the Add‑on

1. Download `video2rigify.py` (or the ZIP produced by Blender’s “Export Add‑on”).
2. In **Blender 4.x** open **Edit ▸ Preferences ▸ Add‑ons ▸ Install…** and pick the file.
3. Enable **Multi‑Person Video‑2‑Rigify**.

### 2. Build the External Environment

1. Still in Preferences, open the add‑on’s panel and press **Install Dependencies**.
2. When prompted, choose **GPU** if you have a CUDA 12.1‑compatible NVIDIA card, otherwise leave it off for a pure CPU build.

   * The tool will:

     * Create `<Blender Config>/video2rigify_env/`
     * Upgrade **pip / wheel / setuptools**
     * Install `torch`, `torchvision`, `torchaudio`, `mmcv`, `mmpose`, **etc.**
     * Clone **MMPose** and **MotionBERT**
     * Download the 200 MB MotionBERT checkpoint
3. Wait until “Dependencies installed ✔” appears.
   *(Subsequent Blender sessions reuse the same env; rerun to upgrade.)*

### 3. Prepare a Rigify Armature

* Generate a standard Rigify rig or import an existing one.
* Make sure it is **in T‑pose** and named **`rig`** (or adjust the field later).

### 4. Run the Pipeline

1. Switch to **View3D ▸ Sidebar ▸ Video 2 Rigify**.
2. Press **Run Pipeline** and fill in:

   * **Video** – the source file (`.mp4`, `.mov`, …).
   * **Rigify Armature** – name of your rig (default `rig`).
   * **Bake Step** – frame step when baking (1 = every frame).
   * **Extreme Error Tol.** – 0–0.1, lower = more keys kept.
   * **Duplicate Rigs** – create a copy per detected person.
3. Click **OK**.
   Internally the add‑on will:

   ```
   video → MMPose (2‑D & tracking)
         → per‑track JSON
         → MotionBERT (3‑D lift) → BVH
         → Retarget Animation add‑on → Rigify armature(s)
   ```
4. When finished you’ll see baked actions on your rig(s).

## Command‑Line Stub

The temporary script below is auto‑generated inside `TMPDIR` and can be reused standalone:

```bash
python pipeline_stub.py input.mp4 --outdir out/ \
  --mmpose /path/to/mmpose \
  --motionbert /path/to/MotionBERT \
  --device cuda:0
```

## Advanced Options

| Field                  | Description                                          |
| ---------------------- | ---------------------------------------------------- |
| **Bake Step**          | Skip frames when baking to reduce key count.         |
| **Extreme Error Tol.** | Threshold for the numpy‑based key reducer.           |
| **Duplicate Rigs**     | If disabled, all tracks are baked onto the same rig. |

## Troubleshooting

| Symptom                                                             | Fix                                                                           |
| ------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| *“Install dependencies first”*                                      | Open Preferences ▸ Add‑ons ▸ Video 2 Rigify ▸ **Install Dependencies**.       |
| `AttributeError: 'V2R_Prefs' object has no attribute 'mmpose_repo'` | Re‑run **Install Dependencies** in **v0.4.0** or later; prefs schema changed. |
| CUDA‐related import errors                                          | Reinstall deps with **GPU** unchecked ➔ CPU wheels.                           |
| BVH imports but rig does not move                                   | Ensure Rigify rig is selected as **Target** before baking.                    |

## Development

```bash
# run tests
pytest

# lint
ruff check .
```

### TODO
* fix mmpose step
* fix test suite
* Improve MotionBERT ↔ Rigify bone‑map
* UI polish & live progress bar
* URL video inputs
* Attempt to make it work with non rigify rigs

## Credits

* **OpenMMLab** - MMPose
* **Walter0807** - MotionBERT
* **Blender Foundation** - Rigify & Python API

