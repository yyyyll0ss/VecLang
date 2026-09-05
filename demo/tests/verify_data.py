"""Check provenance, inverse transforms and source-image alignment against local inputs."""
import sys,ast,json,collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_cases import HERE,ROOT,DATA,VEC,read,transform
import cv2,numpy as np
script=ROOT/'data_process/1_multi_class_data_conversion/building_process/1_instance_cut_building.py'
fn=next(n for n in ast.parse(script.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='crop_instance');fn.returns=None
for a in fn.args.args:a.annotation=None
ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(script),'exec'),ns)
records=collections.defaultdict(dict);checks=[];nfeatures=0
for meta in read(HERE/'public/cases/index.json'):
 base=HERE/'public/cases'/meta['id'];raw=read(base/'raw-svl.json');pred=read(base/'prediction.geojson')['features'];gt=read(base/'gt.geojson')['features']
 for r in raw:
  records[r['resultFile']][r['line']]=r
  if r.get('stage')=='before-stitching':
   nfeatures+=1
   continue
  for source,fc in [('predict',pred),('label',gt)]:
   for k,feature in enumerate(json.loads(r[source])):
    fid=r['featureId'] if not k else r['featureId']+f'_{k}';actual=next(f for f in fc if f['id']==fid)
    assert transform(feature,fid,actual['properties']['class'],r['cropTransform'])['geometry']==actual['geometry'], (meta['id'],fid,source,transform(feature,fid,actual['properties']['class'],r['cropTransform'])['geometry'],actual['geometry'])
  nfeatures+=1
  if meta['dataset'] in ['WHU','WB']:
   scene=cv2.imread(str(DATA/(meta['dataset']+'_test_patches_512')/meta['source']['image']))
   x,y,w,h=r['cropTransform'];bbox=[x+.15*w/1.3,y+.15*h/1.3,w/1.3,h/1.3]
   cropped,*_=ns['crop_instance'](scene,bbox,scene.shape[0],scene.shape[1],1.3)
   reconstructed=cv2.resize(cropped,(256,256));original=cv2.imread(str(DATA/r['image']))
   error=float(np.abs(reconstructed.astype(float)-original.astype(float)).mean())
   assert error<3,(r['image'],error);checks.append({'image':r['image'],'mae':round(error,6)})
 if meta.get('roadStitching'):
  from stitch_demo_roads import WORK,graph_edges
  import pickle
  for source,fc in [('prediction',pred),('gt',gt)]:
   graph=pickle.load(open(WORK/source/'graph'/(meta['roadStitching']['region']+'.p'),'rb'))
   road=next(f for f in fc if f['properties']['class']=='road')
   actual={tuple(sorted((tuple(reversed(a)),tuple(reversed(b))))) for line in road['geometry']['coordinates'] for a,b in zip(line,line[1:])}
   assert actual==set(graph_edges(graph)),(meta['id'],source,'graph edges changed')
   adj=collections.defaultdict(set)
   for a,b in actual:adj[a].add(b);adj[b].add(a)
   tp=read(base/('topology.json' if source=='prediction' else 'gt-topology.json'))
   assert tp['mode']=='stitched-graph'
   for node in tp['nodes']:assert node['degree']==len(adj[tuple(reversed(node['coordinates']))])
 checks.extend({'image':r['image'],'mae':r['meanAbsolutePixelError']} for r in meta.get('registrationChecks',[]))
for name,lines in records.items():
 with (VEC/name).open() as stream:
  for i,line in enumerate(stream,1):
   if i in lines:
    source=json.loads(line);r=lines[i];assert source['predict']==r['predict'] and source['label']==r['label']
report={'cases':len(read(HERE/'public/cases/index.json')),'sourceRecordsVerified':nfeatures,'pixelRegistrationChecks':len(checks),'maxMeanAbsolutePixelError':max(c['mae'] for c in checks),'checks':checks,'result':'PASS'}
(HERE/'artifacts/data-validation.json').write_text(json.dumps(report,indent=2));print({k:v for k,v in report.items() if k!='checks'})
