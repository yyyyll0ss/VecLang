"""Reproduce the curated demo from existing predictions; never runs a model."""
import argparse, collections, importlib.util, json, math, re
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
VEC = ROOT/'new_model/Qwen3-VL-SFT-0502/eval_2026-05-02-22-05-43-instance'
DET = ROOT/'new_model/Qwen3-VL-SFT-0502/eval_2026-05-03-22-20-43-object-detection'
DATA = ROOT/'data'
OUT = HERE/'public/cases'
NAMES = {'building':'WHU_building_instance_test_0129','water':'WB_WaterBody_instance_test_0323','road':'Cityscales_road_instance_test_0301'}

def read(p): return json.loads(Path(p).read_text())
def write(p, d): Path(p).write_text(json.dumps(d,ensure_ascii=False,separators=(',',':')))
def pairs(c):
    if not c: return []
    if isinstance(c[0],list): return c
    if len(c)%2: raise ValueError('Odd coordinate count')
    return [c[i:i+2] for i in range(0,len(c),2)]
def parsed(s):
    return json.loads(s.strip().removeprefix('```json').removeprefix('```').removesuffix('```'))
def rows(name, directory=VEC):
    manifest=read(DATA/(name+'.json'))
    with (directory/(name+'.jsonl')).open() as f:
        count=0
        for i,line in enumerate(f):
            row=json.loads(line); m=manifest[i]
            expected=m['messages'][-1]['content'].strip(); actual=row['label'].strip()
            if actual!=expected:
                assert expected.startswith(actual), (name,i,'label alignment')
                count+=1
                continue  # Truncated labels cannot support a complete GT comparison.
            yield i,m['images'][0],row
            count+=1
    assert count==len(manifest), (name,'count mismatch')
def transform(raw, fid, cls, box):
    x,y,w,h=map(float,box)
    def pts(c):return [[round(x+a*w/1000,3),round(y+b*h/1000,3)] for a,b in pairs(c)]
    g=raw['geometry']; prop=raw.get('properties',{})
    if g['type']=='Polygon':
        rings=[pts(g['coordinates'])]+[pts(r) for r in prop.get('holes',[])]
        for r in rings:
            if len(r)<3:raise ValueError('Short ring')
            if r[-1]!=r[0]:r.append(r[0].copy())
        geometry={'type':'Polygon','coordinates':rings}
    elif g['type']=='MultiLineString':
        geometry={'type':'MultiLineString','coordinates':[pts(r) for r in g['coordinates']]}
        if any(len(r)<2 for r in geometry['coordinates']):raise ValueError('Short line')
    else:raise ValueError('Unsupported geometry')
    return {'type':'Feature','id':fid,'properties':{'class':cls},'geometry':geometry,'junctions':pts(prop.get('junction',[]))}
def topology(features):
    nodes=[]; segments=[]
    for f in features:
        if f['geometry']['type']!='MultiLineString':continue
        lines=f['geometry']['coordinates']; ids=[]
        for i,line in enumerate(lines):
            sid=f'{f["id"]}:line_{i+1:03}';ids.append(sid)
            segments.append({'id':sid,'featureId':f['id'],'coordinates':line})
        for j,p in enumerate(f.get('junctions',[])):
            connected=[]; degree=0
            for sid,line in zip(ids,lines):
                if f['properties'].get('topologyMode')=='stitched-graph':
                    incident=sum(endpoint==p for endpoint in [line[0],line[-1]])
                    if incident:connected.append(sid);degree+=incident
                    continue
                hits=[]
                for k,(a,b) in enumerate(zip(line,line[1:])):
                    dx,dy=b[0]-a[0],b[1]-a[1]; d=dx*dx+dy*dy
                    t=max(0,min(1,((p[0]-a[0])*dx+(p[1]-a[1])*dy)/d)) if d else 0
                    if math.hypot(p[0]-a[0]-t*dx,p[1]-a[1]-t*dy)<=2: hits.append(k)
                if hits:
                    connected.append(sid)
                    degree+=1 if min(math.dist(p,line[0]),math.dist(p,line[-1]))<=2 else 2
            nodes.append({'id':f'{f["id"]}:junction_{j+1:03}','featureId':f['id'],'coordinates':p,'degree':degree,'connectedPolylines':connected})
    if any(f['properties'].get('topologyMode')=='stitched-graph' for f in features):
        return {'mode':'stitched-graph','provenance':'Region graph produced by the repository extraction and stitching scripts. Junctions are graph nodes of degree >= 3; incidence uses exact shared polyline endpoints. Degree-2 chains are compressed without changing graph edges.','tolerancePixels':0,'nodes':nodes,'segments':segments}
    return {'provenance':'Junction coordinates from model/label; incidence derived within each crop at 2 image pixels. No global graph stitching.','tolerancePixels':2,'nodes':nodes,'segments':segments}
def crop_box(b):
    x,y,w,h=b;return [x-.15*w,y-.15*h,1.3*w,1.3*h]
def emit(cid,name,dataset,image,pred,gt,raw,detections,source,extra=None):
    p=OUT/cid;p.mkdir(parents=True,exist_ok=True)
    image.convert('RGB').save(p/'image.jpg',quality=94)
    thumb=image.copy();thumb.thumbnail((280,180));thumb.convert('RGB').save(p/'thumbnail.jpg',quality=85)
    classes=[c for c in ['building','road','water'] if any(f['properties']['class']==c for f in pred)]
    counts={c:sum(f['properties']['class']==c for f in pred) for c in ['building','road','water']}
    counts['roadPolylines']=sum(len(f['geometry']['coordinates']) for f in pred if f['properties']['class']=='road')
    counts['junctions']=sum(len(f.get('junctions',[])) for f in pred)
    meta={'counts':counts,'id':cid,'name':name,'dataset':dataset,'mode':'multi-category' if len(classes)>1 else 'single-category','classes':classes,'width':image.width,'height':image.height,'description':{'IRSAMap':'Polygonal objects and road networks in one scene.','WHU':'Instance-level building footprints.','WB':'Water boundaries and polygon rings.','Cityscale':'Road centerlines and predicted junctions.'}[dataset],'coordinateSystem':{'type':'image-pixel','origin':'top-left','x':'right','y':'down','georeferenced':False},'source':source,**(extra or {})}
    for key,features in [('prediction',pred),('gt',gt)]:write(p/(key+'.geojson'),{'type':'FeatureCollection','coordinateSystem':meta['coordinateSystem'],'features':features})
    topo=topology(pred);write(p/'topology.json',topo);write(p/'gt-topology.json',topology(gt))
    svl={'version':'veclang-demo/1','coordinateSystem':meta['coordinateSystem'],'features':[{'id':f['id'],'class':f['properties']['class'],'geometry':f['geometry'],'topology':{**({'mode':'stitched-graph'} if f['properties'].get('topologyMode')=='stitched-graph' else {}),'junctions':[n for n in topo['nodes'] if n['featureId']==f['id']]}} for f in pred]}
    write(p/'svl.json',svl);write(p/'raw-svl.json',raw);write(p/'detections.json',detections);write(p/'metadata.json',meta)
    print(cid,len(pred),'features',len(topo['nodes']),'junctions',flush=True);return meta

def single_polygons(cls):
    name=NAMES[cls]; dataset='WHU' if cls=='building' else 'WB'
    coco=read(DATA/'coco_label_patches_512'/('WHU_0129_test_coco.json' if cls=='building' else 'WB_0323_test_coco.json'))
    ann={a['id']:a for a in coco['annotations']}; ims={i['id']:i for i in coco['images']}
    grouped=collections.defaultdict(list)
    for i,path,row in rows(name):
        aid=int(Path(path).stem.split('_inst_')[-1]);a=ann.get(aid)
        if a:grouped[Path(ims[a['image_id']]['file_name']).name].append((i,path,row,a))
    pinned=['3001007_2.png','3001020_0.png'] if cls=='building' else ['11_396.png','12_288.png']
    candidates=pinned+sorted((k for k in grouped if k not in pinned),key=lambda k:(-len(grouped[k]),k))
    detection={Path(path).name:(i,row) for i,path,row in rows(dataset+'_test_patches_512',DET)}
    result=[]
    for key in candidates:
        pred=[];gt=[];raw=[]
        try:
            for i,path,row,a in grouped[key]:
                fid=f'{cls}_{a["id"]:06}';box=crop_box(a['bbox'])
                for field,dest in [('predict',pred),('label',gt)]:
                    for n,r in enumerate(parsed(row[field])):dest.append(transform(r,fid if n==0 else fid+f'_{n}',cls,box))
                raw.append({'featureId':fid,'resultFile':name+'.jsonl','line':i+1,'image':path,'cropTransform':box,'predict':row['predict'],'label':row['label']})
            if not pred:continue
            image=Image.open(DATA/(dataset+'_test_patches_512')/key)
            i,dr=detection[key];dets={'prediction':parsed(dr['predict']),'gt':parsed(dr['label']),'coordinateScale':1000,'sourceLine':i+1,'sourceFile':dataset+'_test_patches_512.jsonl'}
            cid=f'{cls}_{len(result)+1:02}'
            result.append(emit(cid,f'{"Building" if cls=="building" else "Water"} scene {len(result)+1:02}',dataset,image,pred,gt,raw,dets,{'image':key,'vectorResults':VEC.name,'detectionResults':DET.name,'transform':'Crop bounds reconstructed from original COCO bbox using repository crop_instance scale_factor=1.3.'}))
            if len(result)==5:break
        except (ValueError,KeyError,json.JSONDecodeError) as e:print('Skip',key,str(e)[:80])
    return result

def roads():
    name=NAMES['road'];candidates=[]
    for i,path,row in rows(name):
        try:
            p=parsed(row['predict']);g=parsed(row['label'])
            score=min(len(p[0]['properties'].get('junction',[])),len(g[0]['properties'].get('junction',[])))
            if score>=3:candidates.append((score,i,path,row))
        except (ValueError,IndexError,KeyError):continue
    out=[]
    for score,i,path,row in sorted(candidates,key=lambda c:(-c[0],c[1])):
        if any(m['source']['image'].split('_patch')[0]==Path(path).name.split('_patch')[0] for m in out):continue
        image=Image.open(DATA/path);box=[0,0,*image.size];fid='road_001'
        pred=[transform(r,fid if n==0 else f'road_{n+1:03}','road',box) for n,r in enumerate(parsed(row['predict']))]
        gt=[transform(r,fid if n==0 else f'road_{n+1:03}','road',box) for n,r in enumerate(parsed(row['label']))]
        out.append(emit(f'road_{len(out)+1:02}',f'Road network {len(out)+1:02}','Cityscale',image,pred,gt,[{'featureId':fid,'resultFile':name+'.jsonl','line':i+1,'image':path,'cropTransform':box,'predict':row['predict'],'label':row['label']}],{'prediction':[],'gt':[],'unavailableReason':'The supplied detection run contains no Cityscale road detection task.'},{'image':Path(path).name,'vectorResults':VEC.name,'transform':'Normalized coordinates multiplied by source patch dimensions.'}))
        if len(out)==5:break
    return out

def multicategory():
    import ast
    script=ROOT/'data_process/1_multi_class_data_conversion/building_process/1_instance_cut_building.py'
    node=next(n for n in ast.parse(script.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='crop_instance')
    node.returns=None
    for arg in node.args.args:arg.annotation=None
    ns={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(script),'exec'),ns)
    crop=ns['crop_instance']
    coco=read(DATA/'coco_label_patches_512/IRSAMap/test_patches_512_building_water_road_coco.json')
    ims={i['id']:i for i in coco['images']}; regions={Path(i['source_image']).stem for i in ims.values()}
    entries={}
    for cls in ['Building','WaterBody','Road']:
        name=f'IRSAMap_{cls}_instance_test_sft';group=collections.defaultdict(list)
        for i,path,row in rows(name):
            region=Path(path).name.split('_')[0]
            if region in regions:group[region].append((i,path,row))
        entries[cls]=group
    dets=collections.defaultdict(list)
    for i,path,row in rows('IRSAMap_test_patches_512',DET):
        region=Path(path).name.split('_')[0]
        if region in regions:dets[region].append((i,path,row))
    def build_region(region,index,focus):
        source_images=[im for im in ims.values() if im['source_image']==region+'.png']
        width=source_images[0]['source_width'];height=source_images[0]['source_height']
        canvas=Image.new('RGB',(width,height))
        for im in source_images:canvas.paste(Image.open(DATA/'IRSAMap_test_patches_512'/im['file_name']),(im['patch_x'],im['patch_y']))
        arr=cv2.cvtColor(np.array(canvas),cv2.COLOR_RGB2BGR)
        bounds=collections.defaultdict(list)
        for a in coco['annotations']:
            if a.get('source_image')==region+'.png' and a['category_id'] in [1,2]:
                im=ims[a['image_id']];x,y,w,h=a['bbox'];bounds[(a['category_id'],a['source_feature_index'])].extend([[x+im['patch_x'],y+im['patch_y']],[x+w+im['patch_x'],y+h+im['patch_y']]])
        pred=[];gt=[];raw=[];checks=[];skips=[]
        for cls,cat in [('Building',1),('WaterBody',2),('Road',3)]:
            norm={'Building':'building','WaterBody':'water','Road':'road'}[cls]
            for i,path,row in entries[cls][region]:
                stem=Path(path).stem
                if cls=='Road':
                    _,x,y=map(int,stem.split('_patch_')[1].split('_'))
                    if x%128 or y%128:continue # Non-overlapping tiling, no invented network merging.
                    box=[x,y,128,128];fid=f'road_{x:04}_{y:04}'
                    reconstructed=cv2.resize(arr[y:y+128,x:x+128],(256,256))
                else:
                    inst=int(stem.split('_inst_')[-1]);points=np.array(bounds[(cat,inst)])
                    if not len(points):raise ValueError(f'Missing original COCO feature {stem}')
                    lo=points.min(0);hi=points.max(0)
                    cropped,*box=crop(arr,[*lo,*(hi-lo)],height,width,1.3)
                    reconstructed=cv2.resize(cropped,(256,256));fid=f'{norm}_{inst:03}'
                original=cv2.imread(str(DATA/path));error=float(np.abs(reconstructed.astype(float)-original.astype(float)).mean())
                if error>3:raise ValueError(f'Crop mismatch: {path}, MAE={error}')
                checks.append({'image':path,'meanAbsolutePixelError':round(error,6)})
                try:
                    pf=[transform(r,fid if k==0 else fid+f'_{k}',norm,box) for k,r in enumerate(parsed(row['predict']))]
                    gf=[transform(r,fid if k==0 else fid+f'_{k}',norm,box) for k,r in enumerate(parsed(row['label']))]
                except (ValueError,KeyError) as e:skips.append({'image':path,'reason':str(e)});continue
                pred.extend(pf);gt.extend(gf)
                raw.append({'featureId':fid,'resultFile':f'IRSAMap_{cls}_instance_test_sft.jsonl','line':i+1,'image':path,'cropTransform':box,'predict':row['predict'],'label':row['label']})
        detections={'prediction':[],'gt':[],'coordinateScale':width,'sources':[]}
        assert width==height
        for i,path,row in dets[region]:
            m=re.search(r'_x(\d+)_y(\d+)',path);x,y=map(int,m.groups())
            for field,key in [('predict','prediction'),('label','gt')]:
                for d in parsed(row[field]):
                    b=d['bbox_2d'];detections[key].append({'label':d['label'],'bbox_2d':[round(x+b[0]*.512,3),round(y+b[1]*.512,3),round(x+b[2]*.512,3),round(y+b[3]*.512,3)]})
            detections['sources'].append({'image':path,'line':i+1,'file':'IRSAMap_test_patches_512.jsonl'})
        assert {f['properties']['class'] for f in pred}=={'road','building','water'}
        return emit(f'multi_{index:02}',f'Unified scene {index:02}','IRSAMap',canvas,pred,gt,raw,detections,{'image':region+'.png','vectorResults':VEC.name,'detectionResults':DET.name,'transform':'Polygon crop bounds reconstructed from COCO source_feature_index and bbox using repository crop_instance; verified against actual input images. Road patches restored to original 128 pixels with filename offsets.','composition':'Class-specific offline predictions are assembled in one common scene; not claimed to be a single joint decoding.','roadTiling':'Only non-overlapping x%128=0,y%128=0 patches; no global junction merging.'},{'registrationChecks':checks,'excludedMalformedOutputs':skips,'selectionFocus':focus})
    # Rank actual parseable predictions, counting road polylines rather than network containers.
    density={}
    for region in sorted(regions):
        counts={}
        for cls in ['Building','WaterBody','Road']:
            count=0
            for _,path,row in entries[cls].get(region,[]):
                if cls=='Road':
                    _,x,y=map(int,Path(path).stem.split('_patch_')[1].split('_'))
                    if x%128 or y%128:continue
                try:
                    features=parsed(row['predict'])
                    count+=sum(len(f['geometry']['coordinates']) if cls=='Road' else 1 for f in features)
                except (ValueError,KeyError,TypeError):continue
            counts[cls]=count
        if counts['Building']>=10 and counts['WaterBody']>=1 and counts['Road']>=4 and len(dets.get(region,[]))==4:
            density[region]=counts
    out=[];selected=set();rejected=[]
    for focus in ['original','original','Building','WaterBody','Road','Building','WaterBody','Road','Building','WaterBody']:
        candidates=[['102','117'][len(out)]] if focus=='original' else sorted(density,key=lambda r:(-density[r][focus],-sum(density[r].values()),int(r)))
        for region in candidates:
            if region in selected:continue
            try:
                meta=build_region(region,len(out)+1,focus)
            except (ValueError,KeyError,AssertionError) as e:
                selected.add(region);rejected.append({'region':region,'reason':str(e)});print('Skip multi',region,str(e)[:100],flush=True);continue
            out.append(meta);selected.add(region);break
        else:raise RuntimeError(f'Could not select enough validated {focus} scenes')
    write(HERE/'artifacts/case-selection.json',{'selection':'Preserve original scenes; add three building-rich, three water-rich and two road-rich regions, ranked by real predicted object/polyline counts. All multi scenes contain all three classes.','selected':[{'id':m['id'],'region':m['source']['image'],'focus':m['selectionFocus'],'counts':m['counts']} for m in out],'rejected':rejected})
    return out

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    meta=multicategory()+single_polygons('building')+single_polygons('water')+roads()
    assert len(meta)==25
    write(OUT/'index.json',meta)
    if '--prepare-only' not in __import__('sys').argv:
        from stitch_demo_roads import main as stitch_roads
        from apply_stitched_cases import main as publish_cases
        stitch_roads()
        publish_cases()
