# Fusion2USD

A Fusion add-in that exports assemblies as USD Physics articulations, ready to simulate in
**PhysX** (Isaac Sim / Isaac Lab) or **Newton** (MuJoCo, Kamino, Featherstone and XPBD solvers).

It handles what robot CAD actually contains: closed kinematic loops, multi-DOF joints (cylindrical,
pin-slot, planar, ball), motion links (gears, rack and pinion, mirrored fingers), per-joint actuation
data, sensors and cameras. Each export also writes an Isaac Lab `ArticulationCfg`, and optionally a
URDF.

## Install

1. In Fusion, open **Utilities → Add-Ins → Scripts and Add-Ins**, click **+** next to *My Add-Ins* and
   select this repo's `Fusion2USD/` folder. Don't symlink the folder into Fusion's `API/AddIns`
   directory: Fusion would then register it twice, once by the link and once by its real path.
2. Select **Fusion2USD → Run**. From then on the add-in loads on startup, and its commands appear in
   the **FUSION2USD** panel on the **UTILITIES** tab of the Design workspace.

Fusion needs no third-party packages: the add-in writes `.usda` text itself. Binary `.usd`/`.usdc`
output needs `usd-core`, which Fusion does not ship: run **Install USD support** (in the FUSION2USD
panel's menu) once to download it from PyPI into the add-in's config folder. Without it the add-in
writes `.usda` and says so.

**USD format** in the export dialog picks the output:

| Format | Files |
|---|---|
| Text (.usda) | one `robot.usda` |
| Binary (.usdc) | one `robot.usdc` |
| Text, meshes in a binary layer | `robot.usda` with links, joints, physics and materials, sublayering `robot_meshes.usdc` with the meshes |

Collision prims reference the visual meshes by default (**Collision mesh quality → Same as visual**), so
the geometry is stored once and the simulator builds its hulls from it; pick a quality to export a
separate, coarser collision mesh instead. **Max triangles per body** loosens Fusion's tessellation
tolerances for bodies above the limit. The report lists the mesh settings and every coarsened body.

## Quick start

1. Optional: select joints and run **Joint Properties** to set effort, velocity, gains and the like
   (see [Joint and link properties](#joint-and-link-properties)).
2. Run **Export to USD**, choose the output file and the **Physics engine**, and export. You get:
   - `robot.usda`: the articulation (links, joints, drives, mimic joints, materials, collisions)
   - `robot_cfg.py`: an Isaac Lab `ArticulationCfg` with actuators and the initial joint positions
   - `robot.urdf` + `robot_meshes/`: optional
   - `robot.model.json`: optional raw extraction, for re-exporting offline without Fusion
3. Load it:

   ```python
   from robot_cfg import MY_ROBOT_CFG                    # Isaac Lab; the variable is named after the robot
   robot_cfg = MY_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
   ```

   ```python
   import newton                                         # plain Newton
   builder = newton.ModelBuilder()
   builder.add_usd("robot.usda")
   ```

## Choosing a physics engine

Both engines read the standard UsdPhysics schemas. They differ in which joints they solve natively
and in their extension schemas, so the export is structured for one of them. Each export only
contains the extension schemas of its engine.

| | PhysX | Newton |
|---|---|---|
| Cylindrical, pin-slot, planar joints | chain of single-DOF joints through small helper bodies | one D6 joint (`PhysicsJoint`) in the articulation |
| Ball joints | three revolute joints through two helper bodies | `PhysicsSphericalJoint` |
| Multi-DOF joint that is actuated, in a motion link, or a pin-slot/planar reversed into the tree | split, as above | split as well (no D6 drives are written, Newton's mimic only couples single-DOF joints, and a D6 cannot slide along an axis that turns with it) |
| Motion links | `PhysxMimicJointAPI` | `NewtonMimicAPI` |
| Continuous joints in a motion link | limited to ±3600° (PhysX disables the coupling otherwise) | left continuous |
| Self-collisions, joint dynamics | `PhysxArticulationAPI`, `PhysxJointAPI` | `NewtonArticulationRootAPI`, `NewtonJointAPI` |
| SDF collision | `PhysxSDFMeshCollisionAPI` | `NewtonSDFCollisionAPI` |
| Contact sensors | `PhysxContactReportAPI` on the link | set up by the simulation (Isaac Lab `ContactSensorCfg`) |

Choose **PhysX** for Isaac Lab on its PhysX backend. Choose **Newton** for Newton, including Isaac
Lab's Newton backend. For Newton, also set **Joint zero** to *Current pose* (see
[Joint zero position](#joint-zero-position)).

### Newton solvers

Verified with Newton 1.7 using `tools/newton_smoke_test.py` (joints only, no contacts):

| Solver | Closed loops | Motion links |
|---|---|---|
| MuJoCo | yes; needs Joint zero = *Current pose* | yes (soft constraint, settles over a few hundred steps) |
| Kamino | yes | **no**: Kamino ignores mimic constraints |
| Featherstone | **no**: loop-closure joints are ignored | yes, exact |
| XPBD | approximately (mm-level drift) | yes, looser |

The export warns when a design has motion links (Kamino) or closed loops with Fusion's joint zero
(MuJoCo).

## How the CAD maps to USD

| Fusion | USD |
|---|---|
| Occurrences that move together (rigid joints, rigid groups, ground-to-parent, unjointed children) | one rigid-body link (`PhysicsRigidBodyAPI`, `PhysicsMassAPI` from Fusion's physical properties) |
| Grounded occurrences / root bodies | base link; a fixed base gets a `root_joint` to the world |
| Revolute / Slider | `PhysicsRevoluteJoint` / `PhysicsPrismaticJoint` with limits, `DriveAPI`, `JointStateAPI` |
| Cylindrical, Pin-slot, Planar, Ball | depends on the engine, see above |
| Closed loops | spanning tree; one joint per loop gets `excludeFromArticulation = true` (a multi-DOF one becomes a D6 joint, a ball on Newton a spherical joint) |
| Motion links | mimic constraint on the follower joint |
| Appearances | `UsdPreviewSurface` materials |
| Collision | per-body meshes (coarser tessellation), `convexDecomposition` by default |

Joint axes and signs are **measured**: with *Probe joint axes* enabled, the exporter nudges every joint
in Fusion, records how the parts actually move, and restores the pose. Fusion's documented axis
vectors are the fallback. A D6 axis that is up to 2° off its joint frame axis is snapped onto it, with
a warning.

### Asset origin

The USD's origin is the base link, axis-aligned with Z up, as in the Isaac Sim URDF importer. The
design's original placement goes into the Isaac Lab cfg's `init_state.pos`. The base component's CAD
orientation is kept in the base link's `customData` (`cad_frame_rotation`, w x y z), for sensor offsets
defined in that frame.

### Joint zero position

- **Fusion joint zero** (default): USD joint position 0 is Fusion's joint value 0, and the limits are
  exactly as set in Fusion. The exported pose is written as `JointStateAPI` positions and as the Isaac
  Lab cfg's `init_state.joint_pos`.
- **Current pose**: the exported pose becomes position 0 everywhere, and the limits are shifted to
  match.

Use *Current pose* for Newton. Newton's `add_usd` ignores `JointStateAPI` (it only reads it through
`SchemaResolverPhysx`), so with Fusion's zero the robot loads at that zero rather than at the exported
pose. On top of that, Newton's MuJoCo solver anchors loop closures at joint zero, so loops drift open
unless joint zero is the exported pose. With PhysX either setting works, as long as the initial
joint positions come from the cfg.

## Joint and link properties

Properties come from three sources, in increasing priority:

1. `defaults` in an overrides file
2. Fusion attributes in the group `fusion2usd` on joints and occurrences, set with **Joint
   Properties** or through the API
3. Per-name entries in the overrides file (glob patterns allowed; exact names beat patterns)

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

**Joint keys:** `effort` (aliases `torque`, `force`, `max_torque`), `velocity`, `stiffness` (`kp`),
`damping` (`kd`), `armature`, `friction`, `actuated`, `drive_type`, `target`, `lower`, `upper`, `name`,
`loop_closure`, `enabled`.

**Link keys:** `name`, `mass`, `collision`, `base`. A `mass` on a body that Fusion gave no mass gets the
inertia of a solid box over its meshes. `collision` accepts the approximation names in any case and
with separators (`convex_hull`).

**Units:** values without a unit are SI (N·m, N, rad/s, m/s, per-rad gains). Units such as `deg`,
`deg/s`, `rpm`, `mm`, `Nm`, `Ncm`, `kgf*cm` and `g` are converted.

**Other data:** unrecognized keys are kept and written to the prim's `customData`, along with Fusion's
component, part number, description and material.

Two scoping rules apply to Fusion attributes:

- A link is a rigid group of occurrences. A link property set on *any* occurrence of the group applies
  to the whole link; if occurrences disagree, the link's anchor occurrence wins.
- Attributes live on the component, so a property set on one instance of a reused component applies to
  *every* instance. For per-instance values, use the overrides file, which matches by occurrence or joint
  name.

| Property | PhysX | Newton |
|---|---|---|
| effort | `drive:*:physics:maxForce` | same |
| stiffness / damping | `drive:*:physics:stiffness/damping`, per degree for revolute joints | same |
| velocity | `physxJoint:maxJointVelocity` | `newton:velocityLimit` |
| armature / friction | `physxJoint:armature` / `physxJoint:jointFriction` | `newton:armature` / `newton:friction` |

On Newton, a multi-DOF joint marked `actuated` is split, so its pieces get drives as on PhysX. Native
multi-DOF joints are passive, and the export warns when one carries dynamics values.

### Warnings

- **Continuous joints**: a rotational DOF without both limits produces a warning unless the joint
  matches `ALLOW_CONTINUOUS` (comma-separated names or globs, matched against the export name or the
  Fusion name). `DISABLE_CONTINUOUS_WARN=1` silences them. macOS GUI apps don't inherit shell
  variables, so for Fusion use `launchctl setenv ALLOW_CONTINUOUS "wheel_*"` and restart Fusion.
- **Default values**: every driven joint whose effort, velocity, stiffness or damping falls back to the
  export defaults is listed.
- Loop-closing joints, dropped motion links, disconnected parts, missing mass, probe deviations and
  engine-specific caveats are reported as well.

## Sensors and cameras

1. Add a Joint Origin on the part that carries the sensor (Assemble > Joint Origin). Cameras use the ROS
   optical convention: +Z looks forward, +X points right, +Y points down.
2. Run **USD Sensor Properties**, select the joint origin, pick a type and fill in the fields (SI units
   unless noted; empty means default). Type *None* removes all `fusion2usd` attributes from the origin.

| Type | Fields | USD / Isaac Lab |
|---|---|---|
| Camera | width, height (px), fov (horizontal, deg), focal_length, horizontal_aperture, near, far (m), rate (Hz), data_types (e.g. `rgb,depth`) | USD `Camera` prim under the link, Isaac Lab `CameraCfg(spawn=None)` |
| IMU | rate | `ImuCfg` with the offset to its link |
| Contact | rate, history_length, track_air_time, force_threshold (N) | `ContactSensorCfg` on the link |
| Frame | none | `Xform` marker under the link, fixed link in the URDF |

The sensor belongs to the link that contains the joint origin's occurrence. Like other attributes, tags
are stored on the component, so every instance of a reused component gets the sensor (names are made
unique with a suffix). The generated cfg exposes `sensors(robot_prim_path="{ENV_REGEX_NS}/Robot")`, which
returns the sensor configs for the robot.

## Post-export actions

Actions turn a finished export into more outputs. The built-ins are `isaaclab_cfg` and `urdf`. To add
your own:

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

Enable an action by its registered name or as `"package.module:ClassName"`: in the dialog's extra
actions field, in `ExportOptions.post_actions`, or with `--action` on the CLI. A failing action is
reported as an error (the CLI then exits with 1) and never loses the USD.

## CLI (offline, no Fusion)

Re-export a `robot.model.json` written by the add-in:

```bash
pip install -e ".[dev]"
fusion2usd robot.model.json -o out/robot.usda --engine newton --zero-pose current --action urdf
```

## Verifying an export

**Newton**: `tools/newton_smoke_test.py` runs on CPU. Per solver it checks that the model imports and
steps without NaNs, that loop closures stay closed, and that mimic followers obey their coupling after
the leader is displaced.

```bash
pip install -e ".[newton]"
python tools/newton_smoke_test.py out/robot.usda --solver mujoco kamino featherstone xpbd
```

**Isaac Lab**: `tools/isaaclab_smoke_test.py` loads an export in Isaac Lab 3 and checks the spawn and
reset pose, the placement of cloned robots, stability, loop closures, mimic joints (either schema) and
sensors, including that cameras render along the ROS optical axis. It has been run on the PhysX backend.
`tools/isaaclab_smoke.sky.yaml` runs it on a cloud GPU with SkyPilot:

```bash
# workdir contains isaaclab_smoke_test.py and exports/<name>/<name>.usda + <name>_cfg.py
sky launch -c fusion2usd-smoke tools/isaaclab_smoke.sky.yaml --workdir <dir> -i 15 --down
```

**Inside Fusion**: the unit tests can't reach Fusion, so a self-test script checks the exporter against
the real API.

1. Click **+** next to *My Scripts* in **Scripts and Add-Ins**, select `Fusion2USD/selftest/`, and run
   **SelfTest**.
2. It builds a test mechanism in a **new, unsaved design**: a grounded base, a four-bar loop, a slider,
   a gear linked 2:1 to the crank, a ball joint, a camera and an IMU. It exports the design and checks
   links, the loop cut, the mimic gearing, slider limits, probed axes and origins, mesh frames, joint
   attributes and sensors, and that all joint values are restored. Your open designs are not touched.
3. The report appears in a message box and is saved as `selftest_report.txt` next to the exported files
   in a temporary `fusion2usd_selftest_*` directory. The test design stays open for inspection.

## Development

```bash
uv venv -p 3.12 .venv && uv pip install --python .venv/bin/python usd-core pytest
.venv/bin/python -m pytest
```

`tests/test_newton_sim.py` simulates Newton exports end to end. It is skipped unless `newton` is
installed (`pip install -e ".[newton]"`).

## Architecture

```
Fusion2USD/
  Fusion2USD.py, .manifest     add-in entry point
  fusion2usd/
    core/          math (Vec3/Quat/Transform), inertia, naming, report
    model/         CAD-agnostic RobotModel IR (SI units, world-space joints)
    fusion/        Fusion API -> raw RobotModel (grouping, joints, probing, meshes, mass, motion links, attributes)
    kinematics/    passes: properties -> naming -> up axis -> tree/loops -> decomposition -> zero pose -> mass -> validation
    usd/           dependency-free USDA writer; RobotModel -> UsdPhysics builder; engines.py holds all PhysX/Newton schemas
    isaaclab/      ArticulationCfg generator
    urdf/          URDF + STL writer
    actions/       post-export action framework (+ built-ins)
    ui/            Fusion commands (export dialog, joint/link/sensor properties)
    exporter.py    orchestration (ModelSource -> Pipeline -> UsdWriter -> actions)
    cli.py         offline re-export
  selftest/        live self-test script for Fusion
tests/             pure-Python tests (USD validated with usd-core; Newton simulation when installed)
tools/             Newton and Isaac Lab smoke tests
```

Each stage is a small class with one job. To add a CAD source, implement `ModelSource`. To change
processing, edit or insert a `ModelPass` in `kinematics.default_pipeline()`. Engine-specific USD goes
into `usd/engines.py`. New outputs are post-export actions.

## Limitations

**Fusion**
- Fusion's API has no per-design up axis. `source_up_axis = "auto"` reads your *default modeling
  orientation* preference; for a design made under the other setting, choose `Y` or `Z` explicitly.

**PhysX**
- Splitting a multi-DOF joint approximates Fusion's semantics: the chain is exact at the exported pose,
  but large ball-joint motions may differ from Fusion's own parameterization.
- Mimic joints need both joints inside the articulation, so a motion link that involves a loop-closing
  joint is dropped, with a warning.
- D6 loop-closure joints get no rotation limits (Fusion's ranges are not verified against PhysX's D6
  twist limits).
- Two mimic constraints were verified in Isaac Sim 6.2 / Isaac Lab 3.0, and breaking either one silently
  disables the coupling. The instance and reference-axis tokens are always `rotX`, also for prismatic
  joints. Every revolute joint in a coupling needs finite limits. Units are USD units: `offset` in
  degrees (or meters), `gearing` in m/deg for rotary-to-linear couplings.

**Newton**
- Ball joints become spherical joints, which only support a cone limit, so Fusion's pitch/yaw/roll
  limits on ball joints are not exported (with a warning).
- Native multi-DOF joints are passive and carry no joint dynamics; mark a joint `actuated` to have it
  split and driven.
- Newton (1.7) only creates a D6 DOF for an axis that carries a limit, so free axes get an explicit
  infinite one.
- `newton:mimicCoef1` is in joint coordinates (m/rad for rotary-to-linear couplings), because that is
  what Newton's importer applies; NVIDIA's schema document specifies USD units instead.
- Isaac Lab's Newton backend has not been run with these exports yet; plain Newton has (see
  [Newton solvers](#newton-solvers)).

**Both**
- `friction` is written unconverted to every target, and the targets interpret it differently (a
  coefficient in the USD, a Coulomb force or torque in the URDF).

**URDF**
- URDF cannot represent closed loops, D6 or spherical joints, or free bodies; those, and the subtrees
  behind them, are left out of the URDF with a warning. Newton exports keep multi-DOF joints whole,
  so their URDF loses more than a PhysX export's does.

## License

Apache License 2.0, see [LICENSE](LICENSE).
