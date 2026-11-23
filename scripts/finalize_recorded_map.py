"""Build a delivery bundle from a continuous run without changing its raw evidence."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    scripts = Path(__file__).resolve().parent
    bundle = run / "delivery"
    bundle.mkdir(exist_ok=True)

    def invoke(name, *arguments):
        subprocess.run([sys.executable, str(scripts/name), *map(str,arguments)], check=True)

    invoke("probe_structural_map.py", run, bundle, "--evidence-only")
    invoke("refine_terrain_footprint.py", run, bundle)
    invoke("evaluate_scene_coverage.py", run, "--structural-mask", bundle/"structural_mask.npz",
           "--terrain-layer", bundle/"terrain_layer.json", "--report-dir", bundle)
    invoke("check_refined_terrain.py", run, bundle/"terrain_layer.json")
    if not (run/"terrain_acceptance.json").exists():
        invoke("evaluate_terrain.py", run)
    for name in (
        "summary.json", "config.json", "run_manifest.json", "scene_provenance.json",
        "perception_evaluation.json", "terrain_acceptance.json", "showcase.mp4", "showcase-h264.mp4",
        "showcase_live.png", "perception_overlay.png", "ball-evidence.png", "terrain-evidence.png",
    ):
        if (run/name).exists():
            shutil.copyfile(run/name,bundle/name)
    manifest = {
        "schema":"go2-fused-delivery-manifest/1",
        "physical_run":run.name,
        "physical_evidence":"../summary.json",
        "mapping":"same recorded LiDAR and estimated poses; repeated structural returns and bounded terrain interpolation",
        "height_of_inferred_cells":None,
        "reference_assets_used_in_postprocessing":False,
    }
    (bundle/"delivery_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    invoke("evaluate_perception_delivery.py",bundle)


if __name__ == "__main__":
    main()
