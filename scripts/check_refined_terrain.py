"""Independently score inferred and directly observed terrain cells."""
import argparse
import json
from pathlib import Path
import numpy as np
from evaluate_perception_demo import rubble_map_precision


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    parser.add_argument("layer",type=Path)
    args=parser.parse_args()
    data=json.loads(args.layer.read_text())
    points=[x["xy_m"] for x in data["cells"]]
    inferred=[x["xy_m"] for x in data["cells"] if x.get("inferred_footprint")]
    anchor=np.array([.8,.8])
    rotation=np.eye(2)
    results={name:rubble_map_precision(args.run,anchor,rotation,mapped_xy=xy)
             for name,xy in (("combined",points),("inferred_only",inferred))}
    (args.layer.parent/"refined_terrain_precision.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    print(json.dumps(results,indent=2))


if __name__=="__main__":
    main()
