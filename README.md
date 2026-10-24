# Fusion2USD

Export Autodesk Fusion assemblies to USD articulations that load directly in **Isaac Lab / Isaac Sim**
(PhysX), including closed kinematic loops, multi-DOF joints, motion links (gears, screws, mirrored
fingers), joint metadata (torque, velocity, gains, ...), and optional URDF output.

## Install

1. Copy or symlink the `Fusion2USD/` folder into Fusion's add-in directory:
   - macOS: `~/Library/Application Support/Autodesk/Autodesk Fusion 360/API/AddIns/`
   - Windows: `%APPDATA%\Autodesk\Autodesk Fusion 360\API\AddIns\`
2. In Fusion: **Utilities → Add-Ins → Scripts and Add-Ins → Add-Ins → Fusion2USD → Run**.

No third-party packages are needed inside Fusion. The add-in writes `.usda` text directly. For binary
`.usd`/`.usdc` output, `usd-core` must be importable; if it isn't, the add-in falls back to `.usda` and
prints a warning.

## Use

- **Export to USD**: pick the output file and options, then export. The add-in writes:
  - `robot.usda`: the articulation (links, joints, drives, mimic joints, materials, collisions)
  - `robot_cfg.py`: an Isaac Lab `ArticulationCfg` with actuators and initial joint positions
  - `robot.urdf` + `robot_meshes/`: optional URDF output
  - `robot.model.json`: optional raw extraction, so you can re-export offline without Fusion
- **Joint Properties**: select a joint and set effort/torque, velocity, stiffness, damping,
  armature, friction, actuated, loop closure, and any extra metadata. The values are stored as Fusion
  attributes, so they persist in the design.

```python
from my_robot_cfg import MY_ROBOT_CFG
robot_cfg = MY_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
```

## How the CAD maps to USD

| Fusion | USD / PhysX |
|---|---|
| Occurrences that move together (rigid joints, rigid groups, ground-to-parent, unjointed children) | one rigid-body link (`PhysicsRigidBodyAPI`, `PhysicsMassAPI` from Fusion's physical properties) |
| Grounded occurrences / root bodies | base link; fixed base = `root_joint` to the world |
| Revolute / Slider | `PhysicsRevoluteJoint` / `PhysicsPrismaticJoint` with limits, `DriveAPI`, `JointStateAPI` |
| Cylindrical, Pin-slot, Planar, Ball | chain of single-DOF joints through tiny helper links (Isaac Lab handles 1-DOF joints best) |
| Closed loops | spanning tree; one joint per loop is `excludeFromArticulation = true` (multi-DOF loop joints become D6 joints) |
| Motion links | `PhysxMimicJointAPI` on the follower (`q_f + gearing·q_l + offset = 0`) |
| Bodies' appearances | `UsdPreviewSurface` materials under the default prim |
| Collision | per-body meshes (coarser tessellation), `convexDecomposition` by default |

Joint axes and signs are **measured**: with "probe joints" enabled, the exporter nudges every joint in
Fusion, records how the parts actually move, and restores the pose. Fusion's documented axis vectors
are the fallback.

### Asset origin

The USD's origin is the base link, kept axis-aligned with Z up, as in the Isaac Sim URDF importer. The
original placement in the design is written to the Isaac Lab cfg's `init_state.pos`. The base
component's original CAD orientation is kept in the base link's `customData` (`cad_frame_rotation`,
w x y z), which you need when sensor offsets are defined in that frame.

### Joint zero position

- `fusion` (default): USD joint position 0 is Fusion's joint value 0, and limits are exactly as set in
  Fusion. The generated Isaac Lab cfg sets `init_state.joint_pos` to the exported pose. If you
  override it with zeros, closed loops may start violated.
- `current`: the exported pose becomes position 0 everywhere, and limits are shifted to match. This is
  the safest choice for closed loops.

## Metadata: torque, velocity, gains, custom data

The exporter reads properties from three sources, in increasing priority:

1. `defaults` in the overrides file
2. Fusion attributes in group `fusion2usd` on joints and occurrences (set them with the Joint
   Properties command or the API)
3. Per-name entries in the overrides JSON (glob patterns allowed; exact names beat patterns)

```json
{
  "defaults": {"joint": {"damping": 0.5}},
  "joints": {
    "Revolute 1": {"effort": "2.5 Nm", "velocity": "180 deg/s", "stiffness": 40},
    "finger_*":   {"effort": "15 kgf*cm", "actuated": true},
    "Revolute 7": {"loop_closure": true}
  },
  "links": {"base:1": {"mass": "3.2 kg", "collision": "convexHull", "base": true}}
}
```

Joint keys: `effort` (aliases `torque`, `force`, `max_torque`), `velocity`, `stiffness` (`kp`),
`damping` (`kd`), `armature`, `friction`, `actuated`, `drive_type`, `target`, `lower`, `upper`,
`name`, `loop_closure`, `enabled`. Link keys: `name`, `mass`, `collision`, `base`. Values without a
unit are SI (N·m, N, rad/s, m/s, per-rad gains). Units such as `deg`, `deg/s`, `rpm`, `mm`, `Nm`,
`Ncm`, `kgf*cm` and `g` are converted. Unrecognized keys are kept and written to the prim's
`customData`. Fusion information (component, part number, description, material) is exported there
too.

Two scoping rules apply to Fusion attributes:

- A link is a rigid group of occurrences. Link properties set on *any* occurrence in the group apply
  to the whole link; if occurrences disagree, the link's anchor occurrence wins.
- Attributes are stored on the native (component-level) entity, so properties set on one instance
  of a reused component apply to *every* instance. For per-instance values, use the overrides JSON,
  which is matched by occurrence or joint name.

| Property | USD attribute |
|---|---|
| effort | `drive:*:physics:maxForce` |
| velocity | `physxJoint:maxJointVelocity` |
| stiffness / damping | `drive:*:physics:stiffness/damping` (converted to per-degree for revolute joints) |
| armature / friction | `physxJoint:armature` / `physxJoint:jointFriction` |

### Warnings

- **Continuous joints**: rotational DOFs without both limits produce a warning unless the joint matches
  `ALLOW_CONTINUOUS` (comma-separated names or globs, matched against the export name or the Fusion
  name). Set `DISABLE_CONTINUOUS_WARN=1` to silence these warnings entirely. macOS GUI apps don't
  inherit shell variables, so for Fusion use `launchctl setenv ALLOW_CONTINUOUS "wheel_*"` and
  restart Fusion. For the CLI, plain environment variables work.
- **Default values**: every driven joint whose effort, velocity, stiffness or damping falls back to
  the export defaults is listed.
- Loop-closing joints, dropped motion links, disconnected parts, missing mass, and probe deviations are
  reported as well.

## Sensors and cameras

1. Add a Joint Origin on the part that carries the sensor (Assemble > Joint Origin). Cameras use the
   ROS optical convention: +Z looks forward, +X points right, +Y points down.
2. Run **USD Sensor Properties**, select the joint origin, pick a type and fill in the fields (SI
   units unless noted, empty = default). Type *None* removes all `fusion2usd` attributes from the origin.

| Type | Fields | USD / Isaac Lab |
|---|---|---|
| Camera | width, height (px), fov (horizontal, deg), focal_length, horizontal_aperture, near, far (m), rate (Hz), data_types (e.g. `rgb,depth`) | USD `Camera` prim under the link, Isaac Lab `CameraCfg(spawn=None)` |
| IMU | rate | `ImuCfg` with the offset to its link |
| Contact | rate, history_length, track_air_time, force_threshold (N) | `ContactSensorCfg` on the link |
| Frame | none | `Xform` marker under the link, fixed link in the URDF |

The sensor belongs to the link that contains the joint origin's occurrence. Like other attributes,
tags are stored on the component, so every instance of a reused component gets the sensor (names are
made unique with a suffix). The generated Isaac Lab cfg exposes
`sensors(robot_prim_path="{ENV_REGEX_NS}/Robot")`, which returns the sensor configs for the robot.

## Post-export actions

Actions turn a finished export into more outputs. Built-ins: `isaaclab_cfg` and `urdf`. To add your
own:

```python
from fusion2usd.actions import PostExportAction, ExportArtifacts, register_action

@register_action("my_action")
class MyAction(PostExportAction):
    def run(self, artifacts: ExportArtifacts) -> list[str]:
        # artifacts.usd_path, artifacts.model (processed RobotModel), artifacts.options, artifacts.report
        out = artifacts.sibling("_summary.txt")
        ...
        return [out]
```

Enable an action by its registered name, or by `"package.module:ClassName"`, in the dialog's extra
actions field, in `ExportOptions.post_actions`, or with `--action` on the CLI. A failing action shows
up as a warning and never loses the USD.

## Verifying in Isaac Lab

`tools/isaaclab_smoke_test.py` loads an export in Isaac Lab 3 (PhysX backend) and checks:
- the spawn and reset pose;
- per-environment placement of cloned robots;
- that the simulation stays stable and loop closures stay closed;
- that mimic joints follow their leaders;
- that sensors work (cameras render, and their optical axis matches Isaac Lab's ROS convention).

`tools/isaaclab_smoke.sky.yaml` runs it on a cloud GPU with SkyPilot:

```bash
# workdir contains isaaclab_smoke_test.py and exports/<name>/<name>.usda + <name>_cfg.py
sky launch -c fusion2usd-smoke tools/isaaclab_smoke.sky.yaml --workdir <dir> -i 15 --down
```

## CLI (offline, no Fusion)

```bash
pip install -e ".[dev]"
fusion2usd robot.model.json -o out/robot.usda --zero-pose current --action urdf
```

## Architecture

```
Fusion2USD/
  Fusion2USD.py, .manifest     add-in entry point
  fusion2usd/
    core/          math (Vec3/Quat/Transform), inertia, naming, report
    model/         CAD-agnostic RobotModel IR (SI units, world-space joints)
    fusion/        Fusion API -> raw RobotModel (grouping, joints, probing, meshes, mass, motion links, attributes)
    kinematics/    passes: properties -> naming -> up-axis -> tree/loops -> decomposition -> zero pose -> mass -> validation
    usd/           dependency-free USDA writer + RobotModel -> USD physics builder
    isaaclab/      ArticulationCfg generator
    urdf/          URDF + STL writer
    actions/       post-export action framework (+ built-ins)
    ui/            Fusion commands (export dialog, joint/link properties)
    exporter.py    orchestration (ModelSource -> Pipeline -> UsdWriter -> actions)
    cli.py         offline re-export
tests/             pure-Python tests (USD validated with usd-core)
```

Each stage is a small class with one job. To add a CAD source, implement `ModelSource`. To change
processing, edit or insert a `ModelPass` in `kinematics.default_pipeline()`. New outputs are
post-export actions.

## Development

```bash
uv venv -p 3.12 .venv && uv pip install --python .venv/bin/python usd-core pytest
.venv/bin/python -m pytest
```

## Live verification

The unit tests cannot reach Fusion. To check the exporter against the real API, run the self-test
script:

1. In Fusion, open **Utilities > Scripts and Add-Ins**, add `Fusion2USD/selftest` (the `+` next to
   *My Scripts*) and run **SelfTest**.
2. It builds a test mechanism in a **new, unsaved design**: a grounded base, a four-bar loop, a
   slider, a gear linked 2:1 to the crank, and a ball joint. It then exports the design and checks
   links, the loop cut, the mimic gearing, slider limits, the probed axes and origins, mesh frames,
   joint attributes, and that all joint values are restored. Your open designs are not touched.
3. The pass/fail report appears in a message box and is saved as `selftest_report.txt` next to the
   exported files in a temporary `fusion2usd_selftest_*` directory. The test design stays open so
   you can inspect it.

## Limitations

- Fusion's API has no per-design "up" axis. `source_up_axis = "auto"` reads your *default modeling
  orientation* preference. If a design was made under a different setting, choose `Y` or `Z`
  explicitly in the export dialog.
- Joint decomposition approximates Fusion's multi-DOF semantics. The decomposed chain is exact at the
  exported pose; large ball-joint motions may differ from Fusion's own parameterization.
- PhysX mimic joints need both joints inside the articulation, so a motion link that involves a
  loop-closing joint is dropped (with a warning).
- URDF cannot represent closed loops, D6 joints or free bodies; those are omitted from the URDF.

- Motion links are written as `PhysxMimicJointAPI`. Two constraints were verified in Isaac Sim 6.2 /
  Isaac Lab 3.0, where breaking either one silently disables the coupling:
  - The instance and reference-axis tokens are always `rotX`, also for prismatic joints. A `transX`
    instance is ignored.
  - Every revolute joint in a coupling needs finite limits. Continuous joints get ±3600° and a warning.

  Units are USD units: `offset` is in degrees (or meters), and `gearing` is in m/deg for
  rotary-to-linear couplings. Recent Isaac Sim importers author `NewtonMimicAPI` instead; the Newton
  backend is not supported yet.

## License

Apache License 2.0, see [LICENSE](LICENSE).
