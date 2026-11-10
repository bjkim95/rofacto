# DexYCB Human → Robot Video Generation — Full Pipeline

End-to-end: DexYCB human hand-object demo → robot retargeting → URDF rendering →
world-model conditioning → WAN video generation → comparison.

Robot: **Panda arm + Inspire (right) hand**. Default subject `20200709-subject-01`.
Resolution **480×640**, **81 frames** (4k+1), 16 fps. Main camera `840412060917`;
8 cameras per session.

---

## 0. Source data (READ-ONLY)
Set `DEX_YCB_DIR=/path/to/dexycb` to your downloaded DexYCB dataset. Keep the raw data read-only.
Per session/cam: `color_*.jpg`, `aligned_depth_to_color_*.png`, `labels_*.npz`
(MANO `pose_m`, `joint_3d` 21-kpt, object `pose_y`), `calibration/` (intrinsics, extrinsics, per-subject MANO betas).

## 1. Hand retargeting — AnyTeleop / dex-retargeting
`precompute_retarget.py` (env: `spider` — name only; algo = dex-retargeting)
- 5 fingertip kpts (idx 4,8,12,16,20) → Inspire hand 6 proximal joints, position-based opt.
- config: `third_party/dex-retargeting/.../offline/inspire_hand_right.yml`
- → `outputs/dexycb/retarget/{SESSION}_{CAM}_retarget_rq.npz` (rq, valid, j3)

## 2. Arm IK — pyroki (sequence-level, preferred)
`precompute_panda_ik_traj_dexycb.py` (env: `pyroki`, CPU)
- **Trajectory-level** jaxls solve over all frames: per-frame `pose_cost` (wrist target)
  + `smoothness_cost` + `five_point_acceleration_cost` + limit/velocity constraints.
- Reads `panda_base.yml` (interactive base, §2b) + `retarget_rq.npz`.
- → `outputs/dexycb/panda_ik_traj/{SESSION}_{CAM}_panda_ik_joints.npz`  (target_link `panda_hand`)
- **Why trajectory not per-frame:** old `precompute_panda_ik_dexycb.py` solved each frame
  independently → joint-branch flips (max step ~3.5 rad). Trajopt → ~0.1 rad (smooth).
- ⚠️ Re-run IK whenever `panda_base.yml` changes, else render snaps hand to a stale EEF.

### 2b. Interactive base placement
`interactive_panda_base.py` (env: `pyroki`, viser @ :7979) — drag Panda base, real-time
IK, MANO mesh / skeleton, record modes (human_hand / robot_hand / arm_and_hand),
Save view, Capture trail (motion overlay), Base height cut, Render scale.
Writes `outputs/dexycb/panda_base/panda_base.yml`.

## 3. URDF rendering — pyrender (EGL, GPU)
`render_dexycb.py` (env: `rofacto`). Reads retarget_rq + panda_ik_joints + panda_base.
Env vars: `SESSION`, `CAMS=<serial>`, `HAND_BLACK=1` (black hand mesh),
`OUT_TAG`, `BG_VIDEO` (composite robot onto bg → rgb_overlay.mp4).
Outputs per cam in `outputs/dexycb/render/renders/{OUT_TAG}_{cam}/`:
`robot_mesh.mp4` (arm+hand), `eef_depth.mp4` (hand-only depth, **same run = aligned with mesh**),
`scene_depth.mp4` (real sensor depth f0), `robot_mask.npy`.
- Each entrypoint has a step-specific default root; pass a shared `--output-root` (or
  `DEXYCB_OUTPUT_ROOT`) when chaining retarget, base placement, IK, render, and assembly.

- Hand is snapped to arm panda_hand FK → mesh-hand and eef-depth share `T_hand_root`
  (aligned **only if rendered in the same run**; mixing old eef + new mesh → misalignment).

## 4. Conditioning streams (per KEY = `s01_{session}_{cam}`)
Collected into `data/dexycb_retarget/{obj}/`:
- `inpaint_videos/` — human hand removed (minimax + SAM3) — static scene source
- `inspire_arm_hand_black_mesh_videos/` — black robot mesh (from §3 HAND_BLACK=1)
- `inspire_arm_hand_eef_depth_videos/` (legacy) / `..._newbase_videos/` (matched to current IK)
- `scene_depth_realdep_videos/` — real sensor depth
- `videos/` — original DexYCB RGB (comparison reference)

Depth options:
- **realdep** (default): real sensor depth, first frame repeated.
- **moge**: `make_moge_static_depth.py` runs MoGe on the **inpainted first frame** (same frame
  as static RGB), turbo-colorized identically to realdep (`depth_to_turbo_bgr`, p2/p98).
  (`assemble_dexycb_conditioning.py` `DEPTH_MODE=moge` is the original single-frame variant.)

### Frame padding → 81
- static RGB / static depth = first frame × 81.
- dynamic (inpaint, mesh, eef) = `ffmpeg tpad=stop=9:stop_mode=clone` (pads to 81 **only if
  source is 72**; for ≠72 sessions, force exactly 81 with `tpad...,trim=end_frame=81`).

## 5. Inference — WAN 2.1 14B Fun-InP
`python -m training.wan.inference` (Wan environment from `setup_env.sh`, GPU)
- ckpt: `models/rofacto-wan/depth-droid-robocasa` from the [public release](../../docs/model_cards/wan.md).
  Pass its `inference.yaml` with `--experiment_config` and download the Wan base separately.
- 5-stream cond: `--video_path`(inpaint) `--static_video_path`(static rgb)
  `--static_disparity_video_path`(static depth) `--hand_video_path`(black mesh)
  `--hand_disparity_video_path`(eef depth)
- `--num_frames 81 --height 480 --width 640 --num_inference_steps 4 --guidance_scale 1.0`
  LightX2V 4-step LoRA (`--lightx2v_compat_mode force_compat`).
- prompt: Qwen3-VL scene-only caption (`clip_scene_only.txt`), `--prompt_mode custom`.

### Cosmos3 inference from prepared RGB conditions

The same retargeted robot mesh and static RGB can be passed to the additional
Cosmos3 backend. Prepare matched 832×480 inputs, preserve their common FPS and
use a `4n+1` frame count. Supply a JSON/JSONL manifest to
`python -m training.cosmos3.inference` in its separate environment.
See [the Cosmos3 inference guide](../../docs/cosmos3/inference.md).
This path uses static RGB + mesh RGB; the depth streams above belong to the Wan
recipe. Retargeting, base placement and IK are unchanged.

## 6. Comparison + flat folders
6-panel hstack `[original | inpaint | black mesh | eef depth | scene depth | generated]`,
all forced to 81 frames (`tpad...,trim=end_frame=81,setpts=N/(16*TB)`, `-frames:v 81`).
Flat symlink folders `_all_generated/` and `_all_comparison/` for easy browsing.

---

Optional physics refinement scripts are experimental and require a separate
[SPIDER](https://github.com/facebookresearch/spider) installation. They are not
needed for the inference quickstarts.
