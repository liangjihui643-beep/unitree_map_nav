"""MPC walking fixture for a raised ball. No simulator positions enter perception."""
import argparse
from dataclasses import replace
import sys
from pathlib import Path

import mujoco
import numpy as np
from go2_rescue import mujoco_scene, mujoco_sim
from go2_rescue.cli import main
from go2_rescue.perception_demo import PerceptionDemo

parser = argparse.ArgumentParser(description="Diagnostic MPC lateral motion for an elevated ball")
parser.add_argument("output", type=Path)
parser.add_argument("--raise-m", type=float, default=.31, choices=(.10, .20, .31))
parser.add_argument("--unlimited-reports", action="store_true",
                    help="Do not let the fixture target quota hide a false or duplicate report")
mode = parser.add_mutually_exclusive_group()
mode.add_argument("--outline-ball", action="store_true")
mode.add_argument("--hybrid-ball", action="store_true")
options = parser.parse_args()
out, use_outline, use_hybrid = options.output, options.outline_ball, options.hybrid_ball
original_scene=mujoco_scene.task_scene
original_motion=mujoco_sim.smoke_command
original_init=PerceptionDemo.__init__

def fixture(*args, **kwargs):
    import inspect
    import textwrap
    code=textwrap.dedent(inspect.getsource(original_scene))
    old='    model = spec.compile()'
    new=f"""    ball.pos=ball_position=np.array([2.,.8,{.11+options.raise_m}])
    spec.worldbody.add_geom(name="test_ball_pedestal",type=mujoco.mjtGeom.mjGEOM_BOX,
                           pos=[2.,.8,{options.raise_m/2}],size=[.19,.19,{options.raise_m/2}],rgba=[.40,.42,.47,1.])
    model = spec.compile()"""
    if code.count(old)!=1:
        raise RuntimeError("Fixture model builder changed")
    namespace=dict(vars(mujoco_scene))
    exec(code.replace(old,new),namespace)
    model=namespace["task_scene"](*args,**kwargs)
    import json
    path=out/"scene_provenance.json"
    provenance=json.loads(path.read_text())
    provenance["test_only"]="MPC lateral raised-ball fixture; not official competition run"
    provenance["pedestal_height_m"]=options.raise_m
    provenance["unlimited_reports"]=options.unlimited_reports
    provenance["experimental_ball_estimator"]="hybrid" if use_hybrid else "outline" if use_outline else "multi_view"
    provenance["additions"].append("local raised ball pedestal")
    path.write_text(json.dumps(provenance,indent=2))
    return model

def commands(t, profile, max_speed):
    if profile!="perception-walk":
        return original_motion(t,profile,max_speed)
    gain=max(0.,min(1.,(t-1.)/1.4,(9.-t)/1.4))
    return mujoco_sim.Velocity(vy=.11*gain)

def init(self, *args, **kwargs):
    original_init(self,*args,**kwargs)
    if options.unlimited_reports:
        self.tracker.c = replace(self.tracker.c, target_quota=0)
    # Enable slope perception after the simulator's normal default assignment
    # via observe; map height uses measured stance on this verified level fixture.
    original=self.observe
    def observe(frame, metadata, estimated=None):
        self.slope_perception=True
        self.outline_ball=use_outline
        self.hybrid_ball=use_hybrid
        return original(frame,metadata,estimated)
    self.observe=observe
    def stance_z():
        plane=self.stance_surface.plane
        return -float(plane[2]) if plane is not None else np.nan
    self.estimated_base_z=stance_z

mujoco_scene.task_scene=fixture
mujoco_sim.smoke_command=commands
PerceptionDemo.__init__=init
sys.argv=[sys.argv[0],"mujoco","--seconds","10","--output",str(out),
          "--control","perception-walk","--task-fixture","--task-layout","perception",
          "--pipe-ramp","official","--perception-demo","--camera-every","2","--record"]
raise SystemExit(main())
