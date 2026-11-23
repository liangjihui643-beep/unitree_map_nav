"""Combine independently computed route, mapping and perception evidence."""
import argparse
from html import escape
import json
from pathlib import Path
import subprocess
import sys


def write_report(run, result, coverage, perception, summary):
    names = {
        "coverage_95pct": "可观测地面覆盖",
        "trajectory_p90_20cm": "定位精度",
        "lidar_occupied_surface_p90_15cm": "激光占据位置误差",
        "safe": "行走安全",
        "lidar_wall_boundary_recall_90pct": "激光墙体边界",
        "fused_boundary_recall_90pct": "融合边界完整度",
        "complete_original_start_route": "原起点走完整条路线",
        "rubble_map_minimum_support": "低矮障碍地图落点",
        "refined_terrain_precision": "补全碎石区域的落点核验",
        "rubble_height_evidence": "低矮障碍激光高度",
        "target_detection_range_map": "球检测、测距和落图",
        "terrain_handoff": "运动控制可读接口",
        "demonstration_files": "演示文件齐全",
        "autonomous_without_truth_navigation": "全程自主定位与行走",
        "sensor_contract": "正式激光回波契约",
    }
    rows = "".join(
        f"<tr><td>{escape(label)}</td><td class={'ok' if result['gates'][key] else 'bad'}>"
        f"{'通过' if result['gates'][key] else '未通过'}</td></tr>"
        for key, label in names.items() if key in result["gates"]
    )
    items = [
        ("fused_map.png", "全局融合图。蓝线是估计轨迹，橙色是视觉碎石，浅橙是推断轮廓且高度未知，绿色是激光实测高度，紫色是实验视觉估高，蓝星是确认球点。"),
        ("coverage_acceptance.png", "参考边界、实测地图与连续轨迹"),
        ("showcase_live.png", "终帧相机与语义地图"),
        ("terrain-evidence.png", "低矮碎石的激光高度证据"),
        ("ball-evidence.png", "足球检测、测距和地图标记"),
    ]
    figures = "".join(f'<figure><img src="{name}" loading="lazy" alt="{label}">'
                      f"<figcaption>{label}</figcaption></figure>"
                      for name,label in items if (run/name).exists())
    video = ("showcase-h264.mp4" if (run/"showcase-h264.mp4").exists() else "showcase.mp4")
    video_html = (f'<a href="{video}">打开实际运行录像</a>'
                  if (run/video).exists() else "")
    video_player = (f'<video controls preload="metadata" src="{video}"></video>'
                    if video.endswith("-h264.mp4") else "")
    route = summary.get("survey_route") or {}
    advanced = route.get("count", 0) if route.get("route_complete") else route.get("index", 0)
    skipped = sum(item["index"] < advanced for item in summary.get("survey_unvisited_viewpoints", []))
    visited = advanced-skipped
    target_err = perception.get("football_confirmed_marker_errors_m", [])
    target_text = f"{max(target_err)*100:.1f} cm" if target_err else "无确认标记"
    coverage_fraction = coverage["coverage_fraction"]
    wall_fused = coverage.get("fused_reference_boundary_within_20cm",0.)
    title = "通过整场验收" if result["accepted"] else "整场验收未通过"
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Go2 感知验收</title>
<style>
body{{background:#111922;color:#eaf1f6;font:16px system-ui;margin:auto;max-width:1260px;padding:24px}}
a{{color:#81d6ff}} nav{{display:flex;gap:1.2rem;flex-wrap:wrap;margin:1.4rem 0}}
.facts{{display:flex;gap:16px;flex-wrap:wrap}} .facts span{{background:#23303d;padding:12px 16px;border-radius:8px}}
table{{border-collapse:collapse;width:100%;max-width:740px}} td{{padding:.5rem .8rem;border-bottom:1px solid #405264}}
.ok{{color:#9ce4ab}} .bad{{color:#ffc6a0}} figure{{margin:26px 0}}
img,video{{display:block;width:100%;border-radius:8px}} figcaption{{margin:.5rem 0;color:#bdd2de}}
</style>
<h1>{title}</h1><p>{escape(result["run"])} · MuJoCo 原场景几何、公开导航先验与本地足球任务物件</p>
<div class="facts"><span>路线 {visited}/{route.get("count",0)}</span>
<span>实测地面覆盖 {coverage_fraction:.1%}</span><span>融合边界 {wall_fused:.1%}</span>
<span>球点误差 {target_text}</span></div>
<nav>{video_html} <a href="terrain_layer.json">运动控制接口</a>
<a href="perception_delivery_acceptance.json">交付门禁</a>
<a href="coverage_acceptance.json">地图误差</a>
<a href="perception_evaluation.json">感知误差</a></nav>
<table>{rows}</table>
{video_player}
{figures}
"""
    (run / "perception_delivery_report.html").write_text(page, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run
    coverage = json.loads((run / "coverage_acceptance.json").read_text())
    perception = json.loads((run / "perception_evaluation.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    terrain = json.loads((run / "terrain_layer.json").read_text()) if (run / "terrain_layer.json").exists() else {}
    height_report = run / "terrain_acceptance.json"
    if not height_report.exists() and (run / "recording").is_dir() and (run/"terrain_heights.json").exists():
        subprocess.run([sys.executable, str(Path(__file__).with_name("evaluate_terrain.py")), str(run)],
                       check=True, stdout=subprocess.DEVNULL)
    height_quality = json.loads(height_report.read_text()) if height_report.exists() else {}
    manifest_path = run/"delivery_manifest.json"
    delivery_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    gates = dict(coverage["gates"])
    raw_boundary_gate = gates.pop("boundary_recall_90pct")
    restricted_boundary_distance_gate = gates.pop("boundary_p90_15cm")
    surface_distance = coverage.get("occupied_to_any_source_structure") or {}
    gates["lidar_occupied_surface_p90_15cm"] = surface_distance.get("p90_m", float("inf")) <= .15+1e-8
    groups = coverage.get("boundary_by_structure", {})
    wall = [groups[key] for key in ("course_walls", "outer_boundary") if key in groups]
    wall_total = sum(group["reference_cells"] for group in wall)
    gates["lidar_wall_boundary_recall_90pct"] = bool(
        wall_total and sum(group["within_20cm"] for group in wall)/wall_total >= .90)
    gates["fused_boundary_recall_90pct"] = coverage.get("fused_reference_boundary_within_20cm",0.) >= .90
    route = summary.get("survey_route") or {}
    gates["complete_original_start_route"] = bool(
        coverage["original_start"] and summary.get("survey_segment") is None
        and summary.get("mission_complete") and route.get("route_complete")
        and route.get("all_requested_viewpoints_reached"))
    rubble = perception["rubble_map_agreement"]
    n = rubble["evaluated_cells"]
    height = perception["low_height_map_agreement"]
    evidence = perception.get("acceptance", {})
    gates.update(
        sensor_contract=summary["perception_demo"]["lidar_contract_violations"] == 0,
        rubble_map_minimum_support=n >= 15 and rubble["within_0_20m_cells"] / max(1,n) >= .85,
        rubble_height_evidence=height["in_mesh_footprint_cells"] >= 3 and height_quality.get("pass") is True,
        target_detection_range_map=evidence.get("ball_image_range_map") is True
        and len(perception.get("ball_confirmed_range_errors_m", [])) == 1
        and max(perception["ball_confirmed_range_errors_m"]) <= .20,
        terrain_handoff=terrain.get("schema") == "go2-terrain-layer/1"
        and bool(terrain.get("cells"))
        and bool((terrain.get("navigation_visual_avoidance") or {}).get("visual_avoid_indices")),
    )
    inferred_count = (terrain.get("footprint_refinement") or {}).get("inferred_cells",0)
    if inferred_count:
        audit_path = run/"refined_terrain_precision.json"
        audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
        inferred = audit.get("inferred_only",{})
        combined = audit.get("combined",{})
        gates["refined_terrain_precision"] = (
            inferred.get("evaluated_cells") == inferred_count
            and inferred.get("within_0_20m_cells",0)/inferred_count >= .85
            and combined.get("within_0_20m_cells",0)/max(1,combined.get("evaluated_cells",0)) >= .85)
    required = ("showcase.mp4", "perception_overlay.png", "fused_map.png", "ball-evidence.png", "terrain-evidence.png")
    gates["demonstration_files"] = all((run / f).is_file() and (run / f).stat().st_size for f in required)
    result = {
        "schema": "go2-perception-delivery/1",
        "run": delivery_manifest.get("physical_run",run.name),
        "scope": "public-map simulation survey and local football task fixture",
        "accepted": all(gates.values()),
        "gates": gates,
        "pending": [key for key, ok in gates.items() if not ok],
        "raw_lidar_all_boundaries_recall_gate": raw_boundary_gate,
        "raw_lidar_distance_to_accessible_boundary_gate": restricted_boundary_distance_gate,
        "artifacts": list(required),
    }
    (run / "perception_delivery_acceptance.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(run, result, coverage, perception, summary)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
