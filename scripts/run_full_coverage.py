"""One entry for original-start public-scene autonomous coverage, with a monitor."""
import argparse,json,os,re,shutil,subprocess,sys
from datetime import datetime
from pathlib import Path

PAGE='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Go2 整场自主覆盖</title><style>body{margin:auto;max-width:1400px;padding:30px;background:#121a22;color:#edf5fa;font:17px system-ui}a{color:#81d6ff}h1{font-size:30px}.cards{display:flex;gap:15px;flex-wrap:wrap}.card{padding:18px;background:#23303e;border-radius:12px;min-width:180px}.card b{display:block;font-size:25px}img,video{width:100%;border-radius:12px}nav{padding:20px 0;display:flex;gap:25px;flex-wrap:wrap}.notice{padding:16px;background:#293426;border-left:5px solid #8fce82}footer{color:#aec1cd}</style><h1>Go2 · 整场自主覆盖监控</h1><p>原起点连续自主行走。公开场景巡查点引导，激光、视觉、关节与 IMU 决定实际动作；管道外绕行不等于比赛全流程成绩。</p><div class="cards"><div class="card">仿真时间<b id="time">初始化</b></div><div class="card">当前区域<b id="stage">加载场景</b></div><div class="card">已观测地图面积<b id="area">等待</b></div><div class="card">确认目标<b id="targets">等待</b></div></div><p class="notice" id="state">首次初始化可能较慢。面积表示当前已看见的栅格，不是整场完成率。结束后以独立验收结果为准。</p><img id="view" src="RUN/showcase_live.png" alt="等待仿真画面"><nav><a href="RUN/map.png">激光地图</a><a href="RUN/perception_overlay.png">语义地图</a><a href="RUN/coverage_acceptance.png">整场覆盖与地图核对</a><a href="RUN/coverage_acceptance.json">验收指标</a><a href="RUN/showcase-h264.mp4">实际运行视频</a><a href="RUN/ball-evidence.png">目标识别和落图证据</a><a href="RUN/terrain-evidence.png">低矮高度证据</a></nav><footer>蓝线：估计轨迹；蓝星：已确认目标；橙色：视觉地表候选；绿色：激光高度支持。灰暗区：未知。位置精度由实际运行后的独立统计给出。</footer><script>window.coverageStatus=d=>{document.getElementById('time').textContent=d.t.toFixed(1)+' 秒';document.getElementById('stage').textContent=d.stage;document.getElementById('area').textContent=typeof d.known_area_m2==='number'&&d.known_area_m2>0?d.known_area_m2.toFixed(1)+' m²':'图中查看';document.getElementById('targets').textContent=d.targets;document.getElementById('state').textContent=d.message};function tick(){document.getElementById('view').src='RUN/showcase_live.png?t='+Date.now();const old=document.getElementById('data');if(old)old.remove();const s=document.createElement('script');s.id='data';s.src='RUN/coverage_status.js?t='+Date.now();document.body.appendChild(s)}setInterval(tick,2500);tick();</script></html>'''

def main():
 p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=900.);p.add_argument('--tag',default=datetime.now().strftime('%Y%m%d-%H%M%S'));p.add_argument('--no-browser',action='store_true');p.add_argument('--legacy-exploration',action='store_true');p.add_argument('--overview-video',action='store_true',help='also render the separate simulation.mp4');a=p.parse_args()
 if not re.fullmatch('[a-zA-Z0-9_-]+',a.tag):p.error('tag must use letters, digits, hyphen or underscore')
 root=Path(__file__).resolve().parents[1];os.chdir(root);run=Path('output')/('coverage-'+a.tag)
 if run.exists():p.error('Output already exists; choose another tag')
 run.parent.mkdir(exist_ok=True);monitor=run.with_suffix('.html')
 page=PAGE.replace('RUN',run.name)
 page=page.replace('</nav>',f'<a href="{run.name}/terrain_layer.json">碎石栅格与避让数据</a>'
                   f'<a href="{run.name}/perception_delivery_acceptance.json">整场感知验收</a></nav>')
 if not a.legacy_exploration:
  page=page.replace('公开场景巡查点引导，激光、视觉、关节与 IMU 决定实际动作','公开地图预规划；激光/IMU/里程计估计自身位置，实测地图独立累计')
  page=page.replace('<nav>','<nav><a href="../configs/public-navigation-prior.png">导航先验（非实测）</a>')
 monitor.write_text(page,encoding='utf-8');print('监控页面:',monitor.resolve(),flush=True)
 if not a.no_browser and shutil.which('explorer.exe'):
  path=subprocess.check_output(['wslpath','-w',str(monitor.resolve())],text=True).strip();subprocess.Popen(['explorer.exe',path],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 env=dict(os.environ,MUJOCO_GL=os.environ.get('MUJOCO_GL','egl'),PYTHONPATH=str(root/'src'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
 command=[sys.executable,'-m','go2_rescue.cli','mujoco','--coverage','--task-fixture','--pipe-ramp','official','--perception-demo','--camera-every','5','--record','--seconds',str(a.seconds),'--output',str(run)]
 if not a.legacy_exploration:command.append('--planned-survey')
 if a.overview_video:command.extend(['--video','--video-fps','2'])
 subprocess.run(command,env=env,check=True)
 for script,extra in [('evaluate_scene_coverage.py',[]),('evaluate_perception_demo.py',['--no-localization-ab'])]:subprocess.run([sys.executable,'scripts/'+script,str(run),*extra],env=env,check=True)
 encoder=shutil.which('ffmpeg') or shutil.which('ffmpeg.exe')
 if encoder:
  source=str((run/'showcase.mp4').resolve());target=str((run/'showcase-h264.mp4').resolve())
  if encoder.lower().endswith('.exe'):source=subprocess.check_output(['wslpath','-w',source],text=True).strip();target=subprocess.check_output(['wslpath','-w',target],text=True).strip()
  subprocess.run([encoder,'-v','error','-n','-i',source,'-c:v','libx264','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',target],check=True)
 subprocess.run([sys.executable,'scripts/evaluate_perception_delivery.py',str(run)],env=env,check=True)
 if not a.legacy_exploration:
  subprocess.run([sys.executable,'scripts/finalize_recorded_map.py',str(run)],env=env,check=True)
 acceptance=run if a.legacy_exploration else run/'delivery'
 result=json.loads((acceptance/'perception_delivery_acceptance.json').read_text());print('整场感知验收:', '通过' if result['accepted'] else '未通过；查看 perception_delivery_acceptance.json 各项门禁',flush=True)
 print('结果目录:',run.resolve(),flush=True)
if __name__=='__main__':main()
