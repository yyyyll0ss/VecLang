"""Chrome DevTools integration checks. Start Chrome with --remote-debugging-port=9227."""
import base64,json,time,sys,os
os.environ['NO_PROXY']='127.0.0.1,localhost'
from pathlib import Path
import requests,websocket
ROOT=Path(__file__).resolve().parents[1]
class Browser:
 def __init__(self):
  target=requests.get('http://127.0.0.1:9227/json').json()[0]
  self.ws=websocket.create_connection(target['webSocketDebuggerUrl'],timeout=15);self.i=0;self.errors=[]
  self.call('Runtime.enable');self.call('Page.enable')
 def call(self,method,params=None):
  self.i+=1;i=self.i;self.ws.send(json.dumps({'id':i,'method':method,'params':params or {}}))
  while True:
   r=json.loads(self.ws.recv())
   if r.get('method')=='Runtime.exceptionThrown':self.errors.append(r['params'])
   if r.get('id')==i:
    if 'error' in r:raise RuntimeError(r['error'])
    return r.get('result',{})
 def js(self,expression):
  r=self.call('Runtime.evaluate',{'expression':expression,'returnByValue':True,'awaitPromise':True})
  if 'exceptionDetails' in r:raise AssertionError(r['exceptionDetails'])
  return r.get('result',{}).get('value')
 def wait(self,expression):
  for _ in range(100):
   if self.js(expression):return
   time.sleep(.1)
  raise AssertionError('Timed out: '+expression)
 def screenshot(self,name):
  r=self.call('Page.captureScreenshot',{'format':'png','captureBeyondViewport':True});(ROOT/'artifacts'/name).write_bytes(base64.b64decode(r['data']))
 def click(self,selector):
  box=self.js(f"(()=>{{const el=document.querySelector({json.dumps(selector)});el.scrollIntoView({{block:'nearest'}});const r=el.getBoundingClientRect();return {{x:r.x+r.width/2,y:r.y+r.height/2}}}})()")
  self.call('Input.dispatchMouseEvent',{'type':'mousePressed','button':'left','clickCount':1,**box});self.call('Input.dispatchMouseEvent',{'type':'mouseReleased','button':'left','clickCount':1,**box})

b=Browser();b.call('Emulation.setDeviceMetricsOverride',{'width':1512,'height':1100,'deviceScaleFactor':1,'mobile':False})
url=sys.argv[1] if len(sys.argv)>1 else 'http://127.0.0.1:4173/'
b.call('Network.enable');b.call('Network.emulateNetworkConditions',{'offline':url.startswith('file:'),'latency':0,'downloadThroughput':-1,'uploadThroughput':-1})
downloads=Path('/tmp')/('veclang-downloads-'+str(time.time_ns()));downloads.mkdir()
b.call('Browser.setDownloadBehavior',{'behavior':'allow','downloadPath':str(downloads)})
b.call('Page.navigate',{'url':url});b.wait("document.querySelectorAll('#vector-map .feature').length>20")
assert b.js("document.querySelector('#case-title').textContent.includes('Unified')")
assert b.js("document.querySelectorAll('#vector-map [data-id=road_network]').length===1")
b.screenshot('desktop-preview.png')
# Actual browser pointer event selects an SVL card; both canvases share the same ID.
b.click('#svl [data-feature]')
assert b.js("document.querySelectorAll('#image-map .feature.selected').length===1 && document.querySelectorAll('#vector-map .feature.selected').length===1")
b.js("document.querySelector('#vertices').click();document.querySelector('#topology').click()")
assert b.js("document.querySelectorAll('#vector-map .junction').length>0 && document.querySelectorAll('#vector-map .vertex').length>0")
# Junction click is dispatched to its actual SVG coordinates.
b.click('#vector-map .junction')
assert b.js("document.querySelector('#feature-info').textContent.includes('Degree') && document.querySelector('#feature-info').textContent.includes('stitched graph')")
assert b.js("document.querySelector('#svl .selected')!==null && document.querySelector('#svl .junction-reference')!==null")
b.js("document.querySelector('#view').value='both';document.querySelector('#view').dispatchEvent(new Event('change'))")
assert b.js("document.querySelectorAll('#vector-map [data-source=gt]').length>0")
b.js("document.querySelector('#view').value='gt';document.querySelector('#view').dispatchEvent(new Event('change'))")
b.click('#svl [data-feature]')
assert b.js("document.querySelector('#svl .selected').dataset.source==='gt' && document.querySelector('#image-map .feature.selected').dataset.source==='gt'")
# Switch to one polygon and verify valid edit changes actual rendered path, invalid JSON leaves it intact.
b.js("document.querySelector('[data-case=building_01]').click()")
b.wait("document.querySelector('#case-title').textContent.includes('Building')")
b.js("document.querySelector('#view').value='prediction';document.querySelector('#view').dispatchEvent(new Event('change'))")
b.click('#svl [data-feature]');before=b.js("document.querySelector('#vector-map .feature.selected').getAttribute('d')")
b.js("document.querySelector('#edit').click();const a=document.querySelector('.editor');const f=JSON.parse(a.value);f.geometry.coordinates[0][1][0]+=8;a.value=JSON.stringify(f,null,2);a.dispatchEvent(new Event('input'))")
after=b.js("document.querySelector('#vector-map .feature.selected').getAttribute('d')");assert after!=before
b.js("document.querySelector('#export').click()")
for _ in range(100):
 files=list(downloads.glob('*.geojson'))
 if files:break
 time.sleep(.1)
assert files,'Export did not download'
exported=json.loads(files[0].read_text());assert exported['provenance']['locallyEdited'] is True
assert exported['coordinateSystem']['georeferenced'] is False
assert exported['features'][0]['geometry']['coordinates'][0][1][0]==b.js("JSON.parse(document.querySelector('.editor').value).geometry.coordinates[0][1][0]")
b.js("document.querySelector('.editor').value='{';document.querySelector('.editor').dispatchEvent(new Event('input'))")
assert b.js("document.querySelector('#parse-status').classList.contains('error')")
assert b.js("document.querySelector('#vector-map .feature.selected').getAttribute('d')")==after
b.js("document.querySelector('#reset').click()")
assert b.js("document.querySelector('#vector-map .feature').getAttribute('d')")==before
# Layers, bbox drawing, raw output, all 20 cases, camera sync.
b.js("document.querySelector('[data-layer=building]').click()")
assert b.js("document.querySelectorAll('#vector-map .feature').length===0")
b.js("document.querySelector('[data-layer=building]').click();document.querySelector('#boxes').click()")
assert b.js("document.querySelectorAll('#image-map rect.bound').length>0")
b.js("document.querySelector('#raw').click()")
assert b.js("document.querySelector('#svl').textContent.includes('Polygon')")
b.js("document.querySelector('#image-map [data-zoom=in]').click()")
assert b.js("document.querySelector('#image-map svg').getAttribute('viewBox')===document.querySelector('#vector-map svg').getAttribute('viewBox')")
case_ids=[m['id'] for m in json.loads((ROOT/'public/cases/index.json').read_text())]
assert len(case_ids)==20
for cid in case_ids:
 b.js(f"document.querySelector('[data-case={cid}]').click()")
 b.wait(f"location.hash==='#{cid}'")
 assert b.js("document.querySelectorAll('#vector-map .feature').length>0")
 assert b.js("Array.from(document.images).filter(i=>i.complete&&i.naturalWidth===0).length===0")
b.screenshot('road-topology-preview.png')
b.js("document.querySelector('[data-case=multi_01]').click()")
b.wait("location.hash==='#multi_01'")
# Navigation exposes every case, and desktop margins stay compact on wide monitors.
for filt,number in [('multi-category',5),('building',5),('water',5),('road',5),('all',20)]:
 b.js(f"document.querySelector('[data-filter={filt}]').click()")
 assert b.js("document.querySelectorAll('#gallery [data-case]').length")==number
 assert b.js("document.querySelectorAll('#case-jump option').length")==number+1
b.js("document.querySelector('#case-jump').value='multi_05';document.querySelector('#case-jump').dispatchEvent(new Event('change'))")
b.wait("location.hash==='#multi_05'")
for width in [1920,2560]:
 b.call('Emulation.setDeviceMetricsOverride',{'width':width,'height':1200,'deviceScaleFactor':1,'mobile':False})
 assert b.js("document.querySelector('.workspace').getBoundingClientRect().left")<=20
 assert b.js("document.documentElement.scrollWidth<=window.innerWidth+1")
b.screenshot('wide-preview.png')
b.js("document.querySelector('[data-case=multi_01]').click()")
b.wait("location.hash==='#multi_01'")
b.call('Emulation.setDeviceMetricsOverride',{'width':390,'height':844,'deviceScaleFactor':1,'mobile':True})
assert b.js("document.documentElement.scrollWidth<=window.innerWidth+1")
b.screenshot('mobile-preview.png')
assert not b.errors,b.errors
report={'result':'PASS','url':url,'offline':url.startswith('file:'),'cases':len(case_ids),'javascriptExceptions':0,'checks':['three-column selection','junction SVL localization','GT','vertices','layers','bboxes','raw SVL','edit validation','edited GeoJSON download','reset','zoom sync','responsive layout']}
(ROOT/'artifacts'/('browser-offline-validation.json' if url.startswith('file:') else 'browser-validation.json')).write_text(json.dumps(report,indent=2))
print('PASS: 20 cases; actual pointer selection; three-column sync; GT; junctions; vertices; layers; bbox; raw SVL; editing and recovery; zoom; responsive layout; no JS exceptions.')
