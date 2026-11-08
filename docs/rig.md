# Characters you can pose and animate

There are two ways to get a toon character with a skeleton.

| | From scratch (`blender/character.py`) | TRELLIS.2 model (`toon_scene.py --rig`) |
|---|---|---|
| Geometry | Modelled in code: clean, smooth, closed meshes | The AI mesh (lumpy under its texture) |
| Parts | 17 objects: Head, Face, Hair, Torso, Jacket, and for each side UpperArm, Forearm, Hand, Thigh, Shin, Foot | One object, or 14 with `--parts` (cut from the skin weights) |
| Joints | Round capsule ends meet at every joint, so bends never open a gap | Skin weights blend across the joints |
| Look | Colours from a spec (`CharacterSpec`), painted anime face | The Anima image's colours and face |
| File | `character.blend`, about 0.6 MB | `toon.blend`, about 50 MB |
| Time (RTX 2070 SUPER) | 38 s for stills, turntable and six animations | about 1 min for the rig, stills and one animation |

Both use the same skeleton, controls, pose words and animations. The from-scratch character is the main path; `--rig` on TRELLIS.2 models is opt-in.

## Make one

```bash
# from scratch, with all six animations (Colab, or local with --blender)
python -m maker3d.pipeline --character out/girl
python -m maker3d.pipeline --character out/girl --animations wave,jump \
    --blender "C:/Program Files/Blender Foundation/Blender 4.5/blender.exe"

# a TRELLIS.2 job: ToonSettings(rig=True, parts=True, pose="cheer", animations="wave,walk")
```

Colours and the hair length come from a JSON file of `CharacterSpec` fields:

```json
{"hair": "#2b2b38", "eyes": "#c0392b", "jacket": "#2e5aa8", "hair_length": 0.75}
```

Outputs: `stills/`, `turntable.mp4`, `animations/<name>.mp4` (24 fps on twos), `character.blend` (or `toon.blend`), `spec.json`. Without ffmpeg, Blender encodes the videos itself.

## In Blender

Open the `.blend`. The armature is `ToonRig`. Its bone colours: red = the character's left, blue = right, yellow = the middle, purple = IK controls.

| Control | What it does |
|---|---|
| `root` (big circle with an arrow, on the floor) | Moves and turns the whole character |
| `torso` (circle around the hips) | Moves the body; with IK legs the feet stay planted, so lowering it crouches |
| `foot_ik.L/R` (boxes under the feet), `knee_pole.L/R` (diamonds in front of the knees) | Legs, IK on by default |
| `hand_ik.L/R`, `elbow_pole.L/R` | Arms, IK off by default (rotate the arm bones instead) |
| Every other circle | One bone, rotate it (`R`) |

**IK/FK switches:** select `ToonRig`, then Object properties › Custom Properties: `ik_leg.L`, `ik_leg.R`, `ik_arm.L` and `ik_arm.R`. 1 = IK, 0 = FK. They are keyable.

**Animations:** each one is an action. Pick it in the Action Editor (Dope Sheet › Action Editor). `Turntable` is the pivot's spin.

**Parts:** every part is its own object under the `Character` collection. Select one to hide it, move it, scale it or recolour it. Its colour is a material per role, such as Skin, Hair or Jacket.

## The pose language

A pose names bones and says what they do, in degrees from the rest pose (the A-pose):

```json
{"upper_arm.R": {"raise": 120, "twist": -90}, "forearm.R": {"bend": 80}, "head": {"turn": 15},
 "torso": {"move": [0, 0, -0.05]}, "ik_leg.L": 0}
```

| Bones | Words |
|---|---|
| `root`, `torso`, `hips`, `spine`, `chest`, `neck`, `head` | `bend` (+ forward), `lean` (+ to the character's left), `turn` (+ to the character's left) |
| `shoulder.*`, `upper_arm.*` | `raise` (upper arm: **absolute**, 0 = down at the side, 90 = straight out, 180 = straight up), `forward` (+ forward), `twist` (+ front of the arm turns down) |
| `forearm.*` | `bend` (+ elbow bends), `twist` |
| `hand.*`, `hand_ik.*` | `bend` (+ forward), `side` (+ up), `twist` |
| `thigh.*` | `forward` (+ leg forward), `out` (+ away from the other leg), `twist` |
| `shin.*` | `bend` (+ knee bends) |
| `foot.*`, `foot_ik.*` | `point` (+ toes down), `tilt`, `out` (+ toes out) |
| `toe.*` | `curl` |
| `root`, `torso`, `foot_ik.*`, `hand_ik.*`, `knee_pole.*`, `elbow_pole.*` | `move: [left, forward, up]` in model heights |
| `ik_leg.L/R`, `ik_arm.L/R` | a number from 0 (FK) to 1 (IK) |

`.L`/`.R` are the character's own left and right. Left and right use the same signs, so `"raise": 90` lifts either arm. A pose can also be a name from `blender/poses.json`, `"mirror:<name>"`, or a list that is merged left to right: `["idle", {"head": {"turn": 20}}]`.

Built-in poses: `rest`, `t_pose`, `relaxed`, `idle`, `cheer`, `victory`, `wave_out`, `wave_in`, `hands_on_hips`, `think`, `point`, `bow`, `crouch`, `jump`, `kick`, `walk_contact`, `walk_pass`, `dance_a`, `dance_b`.

## Animations

One JSON file per animation in `blender/animations/`:

```json
{
  "description": "A walk cycle in place (two steps).",
  "frames": 32, "loop": true, "hold": 2,
  "interpolation": "smooth",
  "camera": {"view": 70},
  "keys": [
    {"frame": 1, "pose": "walk_contact"},
    {"frame": 9, "pose": "walk_pass"},
    {"frame": 17, "pose": "mirror:walk_contact"},
    {"frame": 25, "pose": "mirror:walk_pass"}
  ]
}
```

- Every key is a full pose: bones that a key leaves out return to rest.
- `loop` repeats the first key one frame after the end.
- `hold` renders on twos (2) or threes (3), like limited anime animation.
- `interpolation` is `smooth`, `linear` or `constant` (pure key poses).
- `camera.view` is the turn angle in degrees.

Built in: `idle`, `wave`, `walk`, `jump`, `cheer`, `dance`. `tests/test_rig.py` checks that every pose and animation is valid.

## From Python inside Blender, or through blender-mcp

The `.blend` carries the pose code and the library as text blocks (`toon_rig.py`, `toon_library.json`), so this works without the repository:

```python
import bpy

exec(bpy.data.texts["toon_rig.py"].as_string())  # once per session; registers "toon_rig"
import toon_rig

toon_rig.pose("cheer")  # pose now
toon_rig.pose(["idle", {"head": {"turn": 25}}], frame=12)  # and key it on frame 12
toon_rig.animate("jump")  # a library animation as a new action
toon_rig.animate({"frames": 24, "loop": True, "keys": [...]})  # or a new one
```

When the file is trusted (Preferences › Save & Load › Auto Run Python Scripts), `toon_rig.py` registers itself on load.
