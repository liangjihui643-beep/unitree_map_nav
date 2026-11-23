# Architecture

## Public-map planned survey (2026-11-18)

The user selected a known-scene survey instead of frontier exploration and
requires avoiding rock piles without a validated rubble locomotion policy.
`--planned-survey` uses a separately declared, offline generated public-map
prior for A*/oriented-body DWA and scan-to-map localization. Online code still
uses only sensor Frame observations for the current pose. This is known-map
localization, not unknown-layout SLAM, and never uses simulator root state.

`Autonomy.grid` remains an independent measured occupancy/elevation/semantic
map. Its missing ground samples do not erase navigable prior floor. The
separate `PriorNavigationGrid` forbids rock footprints and walls, verifies
known roof-underpass ground/headroom offline, and incorporates online hazards
and independently repeated visual risk candidates. It is never exported as
the measured map. Each run freezes the prior asset and its SHA256.

The route contains survey viewpoints, with explicit heading gates at narrow
passages. Existing full-body margin escape and SE(2) maneuvers recover a safe
alignment when a pure turn is blocked; all motion edges retain body and
hazard-envelope checks. Measured slope limits remain active across route
stage transitions. The original observed-floor/frontier mode and native
backend remain available. Known-route truth control is a distinct diagnostic
flag that is excluded from autonomous acceptance.

The default `run_full_coverage.py` entry now selects planned survey;
`--legacy-exploration` selects the previous strategy. Actual performance is
documented in `预规划巡查实施与验证-2026-11-18.md`; offline connectivity is not
physical completion or competition acceptance.

## Data flow

```text
Lidar provider + SDK state/camera / localhost bridge / recording
MuJoCo raycast + camera + joint/IMU kinematic estimator
    -> Frame (timestamp, body XYZ, IMU wxyz, measured body twist, RGB)
    -> Localization (quality-gated ICP or KISS-ICP)
    -> ground/support/clearance GridMap
    -> ordered fixed-map stages OR frontiers + A* + DWA -> body Velocity
RGB -> undistortion -> football arcs/texture -> calibrated position
    -> multi-frame Tracker -> target events
RGB -> local ArUco finish -> Mission controller
Velocity -> DryRunMotion (default) or opt-in UnitreeMotion -> factory locomotion
Velocity -> pinned Go2 convex MPC -> 12 named actuator torques -> MuJoCo mj_step
```

## Interfaces

`Frame` contains observations only; synthetic truth remains in the software fixture.
`Pose` is start-relative map SE(2) with quality and an uncertainty indicator.
`BallObservation` includes optional estimated map XY; tracking produces de-duplicated target events.
`Step` combines the outgoing velocity, current pose, mission state, reason and task events.
`go2-rescue-frame/1` defines recording/TCP packet metadata; live point-frame and clock semantics are validated.

## Subsystems

- Planar ICP uses trimmed nearest-neighbour fits to recent keyframes with a measured-twist prior. Low-confidence scans do not advance the accepted pose. Conservative loop candidates add SE(2) graph constraints; accepted corrections rebuild the map and transform retained tracks. This is a prototype, not a validated large-scale SLAM product.
- Optional KISS-ICP 1.3.0 supplies 3-D odometry. Current quality gating uses planar correspondence checks. Per-point motion compensation is disabled; no fake point timestamps are used, and graph corrections are not applied to its internal map.
- Grid cells require observed floor support; free rays alone cannot make a drop-off traversable. A lower-envelope graph connects gently varying ground returns; bounded small-triangle interpolation fills sampling gaps up to ground_max_gap (0.45m), without extrapolating broad surfaces. Overlapping ground observations align scan elevations. Local tangent gradients/steps and ceiling clearance are relative to the measured terrain, and DWA uses that same ground classification. This 2.5D approximation is not an arbitrary 3D multi-level map.
- Unknown hardware retains the conservative circular footprint/10-degree slope defaults. MuJoCo uses its Go2 oriented envelope (half-length .35m, half-width .23m, margin .05m), inscribed-disk global inflation and a sampled full oriented footprint in DWA. Simulator slope limit is 14 degrees, backed by a focused official-ramp ascent test; this is not real-hardware calibration.
- The user-authorized local entry-ramp scene selects a separate map_start route with a bounded26-degree classification region around the measured18.7-degree ramp and its transition, not a global capability increase. There, established lower floor support cannot be replaced by a roof return; absolute ceiling height determines updated headroom. Torque control queries bounded piecewise lidar support under each foot, uses a1.5Hz/75% stance gait and surface-normal friction constraints(mu=.4), and accepts quality-gated lidar longitudinal motion to reduce stance-slip bias. Outside this region the original2.5Hz gait and upstream MPC solver remain. Flat SE(2) recovery is disabled in the steep region. East22cm descent stays explicitly guarded before leaving the pipe; local ramp entry has stage evidence, not full-course or hardware validation. See docs/ENTRY_RAMP_NAVIGATION.md.
- A* forbids diagonal corner cutting. Vectorized DWA samples bounded velocities and includes a braking tail, even when measured speeds exceed configured bounds. Explorer defers recently visited/failed frontiers. Recovery searches observed SE(2) space with simultaneous lateral translation and rotation, checks intermediate poses, and falls back to observed turning bays/breadcrumbs. Expired recovery does not immediately retry its failed goal.
- Optional RGB water cues require ground-consistent lidar hits. Projection is bounded to measured neighborhoods; confirmed polygons form a separate hazard grid. Walls use the full body envelope, water uses a configured foot-contact envelope (unknown hardware falls back conservatively). Camera-only color extrapolation cannot paint a distant entire lane as water.
- Ball detection combines Hough/contour candidates, RANSAC arcs and black/white texture, excluding recognized fiducials. Position uses calibrated ground geometry with optional local lidar sphere fit. It never ranges from an occluded bounding-box width. Weak arcs remain inspection candidates; repeated weak observations cannot produce a report without two recent strong observations.
- Mission stages are initialization, exploration, viewpoint verification, finish approach, stopped confirmation, finish/timeout. It emits start once, suppresses duplicate target reports, and does not declare finish just because a time budget expired.
- UnitreeMotion lazily imports SDK2, initializes DDS once and calls only Move/StopMove. A watchdog thread stops stale commands. Sensor/processing deadlines and hardware configuration are checked before real output.

## Assumptions

FLU body coordinates, metres/radians/seconds; image optical coordinates are right/down/forward. Transforms are proper 4x4 rigid matrices.
Near-floor mapping is locally planar with configurable conservative relief limits. Camera geometry and frame rate are configuration values, not fixed by the simulation screenshots.
The unknown-device config contains software-fixture intrinsics/extrinsics and cannot enable hardware motion.
ArUco 42 is a lab-only completion proxy. Competition world alignment and terminal signal wire formats are not implemented. map_start is anchored at the first accepted scan after warmup, not the simulation reset pose. A separate read-only SimulationEvaluator records that true transform at start for diagnostics; neither Autonomy nor MPC receives it.

## Entry points

`src/go2_rescue/cli.py`: init-config, finish-marker, demo, replay, live, native, mujoco.
See README.md for run commands and docs/HARDWARE_ADAPTER.md for hardware provider requirements.
Simulation design/provenance and run results are in docs/MUJOCO.md. Its native model adapter reads plant joints/IMU, not root pose/twist; the renderer and diagnostics can inspect truth, but Frame and Autonomy do not receive it.
Simulation MPC additionally estimates a near-foot support plane from body-frame lidar; upstream swing XY/timing are retained while touchdown Z and COM height/pitch/roll follow that plane. Joint/contact-scheduled vertical velocity is integrated rather than forcing the estimated support to a constant flat-world height. No scene geometry/true height is passed to the controller.

## Mandatory fixed-map course

Accepted V1 scope (2026-11-16) ends at safe parking inside pipe2; exit descent and subsequent competition completion are excluded by the user. The extended ordered route is retained behind its drop-capability guard. A successful V1 entry test does not change mission_complete or official full_course_pass.

OrderedCourse consumes an explicit manually authored map_start route plus estimated poses, gravity-levelled lidar and IMU pitch. Its seven stages cannot be bypassed by target quota, finish-marker visibility or a time reserve. Ascent/descent require measured elevation and pitch; dead end requires frontal returns and a stopped inspection followed by reverse exit; pipe requires observed overhead returns and distance. Ball reporting is enabled only at finish_search in this declared single-finish-ball local fixture. No simulator imports, geometry IDs or target coordinates enter this module.

Ordered local paths use clearance-weighted A*. Strict headings are maintained during approach to narrow portals. Point-to-line ICP retains measured motion in unobservable wall directions; an explicitly enabled stable IMU reference constrains yaw and pose-graph headings, avoiding repeated scan-fit angular drift. Unknown hardware leaves this option disabled. A separate read-only evaluator verifies actual terrain contacts, dead-end entry/exit, rubble-region travel and tube passage before full_course_pass.

Safety-buffer recovery is enabled only for the configured simulation envelope. If the physical body still fits but its extra margin does not, search a short physical-body-safe exit ending at restored full clearance; sample intermediate poses, preserve foot-water checks and cap translation at0.05m/s. No map cells or physical collision constraints are removed. This is recovery from tracking/grid error, not permission to force through an obstacle.

The ordered simulation uses0.10m terrain cells through the ramp/S/dead end, then rebuilds0.05m cells from stored sensor keyframes and estimated poses at the rubble-stage transition (course_map_refined event); its U-exit portal waypoints use a separate DWA. Historical settings were0.02m extra margin and0.08m/s forward; current forward limit comes from passage_max_vx(defaultsim0.5), and the portal extra margin is again min(safety_margin,.02); only the explicitly marked points use it. Lateral/turn caps remain0.05m/s and0.15rad/s. The physical rectangle and water envelope are unchanged. Full0.05m margin resumes after those points; the portal controller cannot accept a wall-intersecting body. Ordinary hardware does not infer portal settings from scene geometry.

A margin escape targets its checked short exit endpoint and terminates as soon as the normal buffered footprint fits again. It must not spend its entire timeout chasing millimetre interpolation subgoals. The retained intermediate route is safety evidence, not a sequence of mandatory stops.

Horizontal overhead planes are removed only from point-to-line planar localization scans using local3D plane normals. The full gravity-levelled cloud still reaches mapping/ceiling checks. ICP quality measures residual agreement on usable line correspondences, with an explicit coverage penalty and minimum12 inliers; a tiny matching subset of an unrelated cloud remains low-confidence. This avoids counting intentionally excluded ceiling samples as failed wall matches.

Pose-alignment recovery searches the waypoint’s declared goal region and heading, rather than forcing its possibly unsafe nominal center. Broad ramp transit gates are0.60m half-width; two S-corner and mirrored retreat regions are0.35m. Body collision constraints are unchanged; narrow portals and stopped-inspection points retain small tolerances.


## Required circular tube2 and experimental curb control

The fixed-map route now specifically requires tunnel2 west-to-east; required step height is0.22m, height-change evidence0.15m, stage travel2.4m. Autonomous navigation retains its8cm configured terrain capability and blocks this stage until a validated step-navigation adapter is integrated; do not increase global max_step_height or clear occupancy to bypass it.

CurbObservation consumes only joint/IMU-frame lidar. It requires repeated upper-plateau and vertical-front-face observations, then provides a discrete foot support height and gradual body reference. Opt-in ConvexMPC uses1Hz diagonal timing, piecewise minimum-jerk lift/translate/lower trajectories, distinct front/rear entry timing and torque clipping. After estimated entry completion it restores standard gait and measured support-plane adaptation; no scene lookup/true body position is read. scripts/test_pipe2_controller.py is a scripted partial-stage diagnostic, not the shared Autonomy implementation.

SimulationEvaluator independently checks circular2 floor contact, actual internal progress and exit. It distinguishes foot-sphere support on the floor/low curved bottom/threshold from body/leg/shell collisions. The stage harness evaluates contacts every2ms physics tick and cannot pass full_course_pass by construction. See docs/PIPE2.md for failures and remaining integration work.


Simulation speed priority: --max-speed overrides ordinary and passage forward limits; otherwise explicit config values are honored, otherwise simulation defaults both0.5m/s. Unknown hardware defaults remain0.3/0.08. DWA lookahead grows by measured speed and braking distance(up to1.2m); collision and stopping-tail checks remain hard constraints. No faster gait or changed actuator limits are required for the verified short clear-strip/U-exit controller tests. See docs/SPEED.md for physical evidence and remaining full-route/curb limitations.

Runtime rendering/output: camera observations stay10Hz; passive viewer sync is independently wall-clock capped at60Hz. Video encoding, raw sensor recording, physics archives and map/image snapshots are separate opt-ins. No unbounded Frame/RGB history is retained. Simulation wraps small-matrix BLAS in a single-thread scope, restoring the caller limit afterward. Terrain lower-envelope medians batch equal-size columns with unchanged median semantics.

## Local entry-ramp variant (later user authorization)

Official source assets remain immutable. --task-fixture now defaults to pipe_ramp=entry; --pipe-ramp official selects the former unmodified-static task. task_scene transforms only tunnel2_to3_sloped_continuous_threshold by180degrees about tube2's world centre before compiling, so both rendering and collision BVHs use the relocated mesh. Provenance lists the one intended change and verifies every imported mesh against its expected transform. The inset shows the actual navigation RGB stream at10Hz, up to480x360 display pixels, without another renderer. Modified geometry is always excluded from official full_course_pass. Autonomous pipe/exit-drop capability remains unintegrated; no terrain guard/map-cell deletion is used to claim traversal.
