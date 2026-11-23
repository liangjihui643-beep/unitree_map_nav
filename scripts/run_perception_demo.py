"""Launch a reproducible moving demo, with a live browser monitor."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


PAGE = """<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><title>Go2 感知演示</title>
<style>
body{background:#191c21;color:#f4f4f4;font:16px system-ui;padding:1.5rem;max-width:1120px;margin:auto}
img,video{width:100%;background:#222;border-radius:8px} a{color:#8ccaeb}
nav a{margin-right:1.2em} h1{margin-bottom:.5rem}
</style>
<h1>Go2 移动感知</h1>
<p>左上看 Go2 实际行走，左下看机器狗相机。蓝框是这一帧用于落图的足球观测，黄区是识别出的碎石。右侧地图蓝线是估计轨迹，蓝星是确认位置，橙点是多帧碎石位置。深灰表示未知。</p>
<p>可打开足球原图与确认帧，图中同一个编号连接原图、测距和地图。</p>
<nav><a href="RUN/map.png">激光地图</a><a href="RUN/perception_overlay.png">语义叠图</a>
<a href="RUN/perception_evaluation.json">评估数据</a><a href="RUN/showcase-h264.mp4">三视图录像</a> · <a href="RUN/terrain-evidence.png">低矮激光高度证据</a> · <a href="RUN/scene-coverage.png">全场范围覆盖检查</a> <a href="RUN/ball-evidence.png">足球原图与地图证据</a></nav>
<p id="status">等待第一帧，仿真正在初始化…</p>
<img id="view" alt="相机、激光回波与地图预览" hidden>
<script>
const view=document.getElementById('view'), status=document.getElementById('status');
function refresh() {
  const image=new Image();
  image.onload=()=>{view.src=image.src;view.hidden=false;status.textContent='预览每 2 秒读取一次 · 图中 t 为仿真时间'};
  image.src='RUN/showcase_live.png?t='+Date.now();
}
refresh();
setInterval(refresh,2000);
</script>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--survey", action="store_true", help="Extend the physical route and scan around the robot")
    parser.add_argument("--variant", choices=("a", "b"), default="a")
    parser.add_argument("--tag", default=datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--original", action="store_true", help="Use the original MJB without adding the football fixture")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.tag):
        parser.error("--tag must contain letters, digits, hyphens or underscores")
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    name = f"perception-{args.variant}-{args.tag}"
    destination = Path("output") / name
    destination.parent.mkdir(exist_ok=True)
    if destination.exists():
        parser.error(f"Output already exists: {destination}")
    monitor = destination.with_suffix(".html")
    monitor.write_text(PAGE.replace("RUN", name), encoding="utf-8")
    print("Monitor:", monitor.resolve(), flush=True)
    if not args.no_browser and shutil.which("explorer.exe") and shutil.which("wslpath"):
        try:
            win_path = subprocess.check_output(["wslpath", "-w", str(monitor.resolve())], text=True).strip()
            subprocess.Popen(["explorer.exe", win_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.CalledProcessError) as error:
            print(f"Open the monitor file manually ({error})", file=sys.stderr, flush=True)
    spawn = ["5.15", "2.56", "0"] if args.variant == "a" else ["5.28", "2.59", "0.005"]
    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", "egl")
    env["PYTHONPATH"] = str(root / "src")
    command = [sys.executable, "-m", "go2_rescue.cli", "mujoco", "--perception-demo",
               "--control", "perception-survey" if args.survey else "perception-walk",
               "--max-speed", ".4", "--seconds", "32" if args.survey else "11",
               "--spawn", *spawn[:2], "--spawn-yaw", spawn[2], "--course-policy", "explore",
               "--pipe-ramp", "official", "--camera-every", "3", "--seed",
               str(int(args.variant == "b")), "--record", "--output", str(destination)]
    if not args.original:
        command.extend(["--task-fixture", "--task-layout", "perception"])
    subprocess.run(command, env=env, check=True)
    subprocess.run([sys.executable, "scripts/finish_fused_acceptance.py", str(destination)], env=env, check=True)
    encoder = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if encoder:
        source = str((destination/"showcase.mp4").resolve())
        target = str((destination/"showcase-h264.mp4").resolve())
        if encoder.lower().endswith(".exe"):
            source = subprocess.check_output(["wslpath","-w",source],text=True).strip()
            target = subprocess.check_output(["wslpath","-w",target],text=True).strip()
        subprocess.run([encoder,"-v","error","-n","-i",source,"-c:v","libx264",
                        "-pix_fmt","yuv420p","-movflags","+faststart",target],check=True)
    print("Results:", destination.resolve(), flush=True)


if __name__ == "__main__":
    main()
