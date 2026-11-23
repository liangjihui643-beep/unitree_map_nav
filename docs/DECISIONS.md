# Architecture and Research Decisions

## 2026-11-18: user-selected public-map survey, avoid rubble

Use an explicit offline public-scene map and preplanned survey viewpoints,
sensor-estimated localization and online body/obstacle checks. Do not require
per-cell measured floor support to traverse the known continuous floor.
Retain measured mapping separately and label the prior visibly; no root-state
navigation or claims of unknown-layout SLAM. Avoid rubble as a region until a
locomotion strategy exists; unknown height does not establish traversability.
Repair under-roof topology with declared ground and clearance instead of
treating every top-down roof shadow as a solid wall. Preserve the pipe2 exit
drop guard and the hardware calibration/command opt-in boundaries.

## 2026-11-15: add official MuJoCo + open-source MPC, preserve real backend

Explicitly requested by the user after the real-robot V1. Use the same Autonomy and sensor Frame interfaces. Pin MIT-licensed go2-convex-mpc rather than introduce RL. Adapt it to the supplied model using MuJoCo-native mass/inertia/Jacobians, named joints and IMU/leg-kinematic odometry; no second URDF/Pinocchio model is required.

Default simulation loads the original binary. A separately labelled local task imports its static collision geometry, validates world vertices, and adds a dynamic football/experimental finish. This is necessary because the MJB has no football/camera or usable local finish marker. Preserve originals and record provenance; do not pass its object coordinates to navigation.

## 2026-11-15: real-robot V1, factory locomotion, no training

Accepted by the user: invoke Go2's built-in walking interface; leave model and sensor details unknown. ROS 2 is not mandatory. The initial training/simulation/ONNX plan is superseded for this version.

Implementation: a ROS-independent Python package, native SDK2 motion transport and swappable sensor providers. Do not replace the vendor gait with RL or a home-grown joint controller. Vendor internal locomotion implementation is outside this project's control.

## 2026-11-15: hardware uncertainty is an interface boundary

No guessed lidar DDS schema, fake calibration or automatic network discovery. A concrete lidar driver and actual calibration are required for hardware operation. The unknown template remains uncalibrated; native/live default to no motion output.

Use the existing Python 3.10 interpreter in a new .venv. Competition Python/MuJoCo/ONNX version pinning is not a V1 physical-robot dependency.

## 2026-11-15: classical recognition and staged evidence

Use geometric/texture football detection without training. Preserve perception/localization/planning interfaces so learned replacements can be added later.
Synthetic regression missions and fake SDK tests establish software behaviour only; hardware performance and 80% completion remain unmeasured. ArUco is only a local finish proxy.

## 2026-11-15: enforce the user’s complete ordered course

The user requires original-start ascent/descent, S curve, a dead-end visit and retreat, rubble region, pipe, then far-finish ball/report. Free exploration and a short route directly to the ball do not satisfy this task. Use an explicit manually authored fixed-public-map prior in configs/official-course-route.json, with online sensor localization, supported-ground A* and footprint-checked DWA. Preserve all official static geometry. Do not claim unknown-layout generalization or use runtime plant truth. Ordered stages require sensor evidence and cannot be skipped by quotas/time/finish visibility; full-course evaluation additionally audits actual physical traversal.

Default full-course time limit is600sim seconds (not wall time), instead of the previous20s smoke default. Local custom-spawn diagnostics must select explore and remain labelled stage tests. The user subsequently explicitly required circular tunnel2. The route now requires west-to-east tunnel2 traversal, its22cm curb and descending exit. Historical tunnel3 traversal is insufficient. A sensor-only experimental MPC curb adapter is opt-in to isolated scripted stage tests; it is not integrated into Autonomy. Ordered mode stops at the unmet declared step capability rather than silently using tunnel3 or globally relaxing terrain limits. See docs/PIPE2.md.

## 2026-11-15: user-authorized local entry ramp and camera inset

User explicitly requested moving the pipe2 ramp to the current west entrance, superseding the earlier no-map-change constraint for a labelled local variant. Preserve all original SHA256-covered files. Recompile a derived task scene, rotate only the existing threshold180degrees around tube2 centre, validate every other geometry and the intended transformed mesh, and record provenance. --task-fixture defaults to entry; --pipe-ramp official preserves old geometry. Modified static geometry can never produce full_course_pass=true for the official course.

Display the existing10Hz sensor image through pinned MuJoCo3.17 Handle.set_images in the main viewer. No extra camera rendering, video, snapshot or raw archive is enabled. Resize overlay only for display, label its source, and support --hide-camera. Handle.close only signals a daemon render thread; retain model/image storage and wait for the pinned Handle weakref to expire before process/GLFW shutdown. Actual Ctrl+C frozen-view and automatic-close tests passed after fixing that lifecycle race.

## 2026-11-16: sensor-based local entry terrain and bounded corridor planning

The user reported a stopped approach after moving the entry ramp. Use a separate local-map route and a bounded26degree classification region for its18.7degree measured ramp, retaining14degree global/8cm step limits. Plan declared straight pipe segments with the full oriented body/support/water envelope rather than disk inflation; a route prior is a direction with a safe arrival region, not permission to cross an observed ramp edge. Near-goal fallback may use a shorter safe advance than25cm. Accepted scan displacements filter slow-trot stance alias in the pipe planner.

Original-start regression exposed a vertical-reference difference from ascent/descent that isolated fresh-map tests lacked. Start a sensor-only terrain submap at pipe entry, retain the original global map and water observations, replay only nearby/entry keyframes on corrections, and leave unknown support blocked. This is not physical scene editing, teleportation, or invented traversable terrain. Retain the east22cm descent capability guard; entry verification does not establish full pipe exit, finish reporting, official scoring, or hardware performance.

In this local terrain region only, reuse the upstream1.5Hz/75% stance gait, bounded per-foot lidar surfaces and surface-normal friction constraints(mu.4), with longitudinal accepted-scan motion feedback. Other segments retain the original2.5Hz MPC solver/gait; the real backend remains factory locomotion. No ROS2, RL or submission workflow introduced. Detailed evidence and failed trials are in docs/ENTRY_RAMP_NAVIGATION.md.

## 2026-11-16: accept pipe-interior V1 endpoint and publish by merge

The user explicitly cancels further descent implementation and accepts current pipe-interior stop as V1 scope. Keep the existing physical drop guard and truthful time_limit/mission_complete=false/full_course_pass=false diagnostics. Do not change geometry, fabricate finish events or label this official competition completion. Preserve later components as unvalidated work outside this milestone.

Replace the user's old unitree_map_nav repository tree with the current runnable project on a separate branch, then use a history-preserving merge. Preserve official resource hashes/licenses and pinned MPC source. Exclude environment, output recordings, caches and credentials; preserve local source workspace and all remote history.

## 2026-11-16: one concise README and optional pipe bypass

The user explicitly requests merging the official resource overview, README_V1 and current progress/issues/work allocation into root README.md. This supersedes the old root-README immutability rule for documentation only. Preserve original official README and SHA256SUMS bytes in source-metadata/training-package-original/; redirect only the README path in the root checksum list, retaining every resource hash and all other paths. Remove the redundant README_V1 and update references. No runtime code or physical/visual asset changes.

The user reports organizers allow bypassing the pipe. Prioritize validating an original-layout detour to finish in a future implementation; current mandatory pipe route/drop guard remain unchanged. No exit descent work. Python3.10.22 is the tested baseline;3.12 compatibility and autonomous coverage remain future validation tasks.
