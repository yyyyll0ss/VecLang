"""Package already-built artifacts without including original datasets or model files."""
import json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.9'}},'cells':[
 {'cell_type':'markdown','id':'intro','metadata':{},'source':['# VecLang Interactive Demo\n','真实离线预测的 Image ↔ SVL ↔ Vector 展示。将本 notebook 与 `VecLang-Standalone.html` 放在同一目录，信任 notebook 后运行下一格。\n','\n无需 GPU、模型服务或联网；如 notebook 前端限制交互 HTML，请直接在浏览器中打开该 HTML。']},
 {'cell_type':'code','id':'viewer','metadata':{},'execution_count':None,'outputs':[],'source':['from pathlib import Path\n','from html import escape\n','from IPython.display import HTML, display\n','\n','page = Path("VecLang-Standalone.html")\n','assert page.exists(), "请将 VecLang-Standalone.html 放到 notebook 当前工作目录"\n','html = page.read_text(encoding="utf-8")\n','display(HTML(\'<iframe title="VecLang Interactive Demo" style="width:100%;height:1150px;border:0;border-radius:12px" srcdoc="\' + escape(html, quote=True) + \'"></iframe>\'))\n']}
]}
(ROOT/'VecLang_Demo.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=2))
art=ROOT/'artifacts';art.mkdir(exist_ok=True)
with zipfile.ZipFile(art/'VecLang-demo-site.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted((ROOT/'dist').rglob('*')):
  if p.is_file():z.write(p,p.relative_to(ROOT/'dist'))
with zipfile.ZipFile(art/'VecLang-demo-source.zip','w',zipfile.ZIP_DEFLATED) as z:
 for name in ['src','scripts','tests','public','.gitignore','artifacts/desktop-preview.png','artifacts/road-topology-preview.png','artifacts/single-case-preservation.json','artifacts/road_stitching_junction','artifacts/case-selection.json','artifacts/wide-preview.png','artifacts/data-validation.json','artifacts/browser-validation.json','artifacts/browser-offline-validation.json','index.html','package.json','package-lock.json','README.md','VecLang_Demo.ipynb','VecLang-Standalone.html']:
  p=ROOT/name
  for f in ([p] if p.is_file() else sorted(p.rglob('*'))):
   if f.is_file() and '__pycache__' not in f.parts:z.write(f,Path('VecLang-demo')/f.relative_to(ROOT))
for p in art.glob('*.zip'):print(p.name,round(p.stat().st_size/1048576,2),'MB')
