"""Run the requested multiclass extraction and junction-stitch scripts on all overlapping patches."""
import hashlib,json,pickle,subprocess,sys,collections
from pathlib import Path
import cv2,numpy as np
from prepare_cases import ROOT,HERE,DATA,VEC,read,write
WORK=HERE/'artifacts/road_stitching_junction'
EXTRACT=ROOT/'VecLang/eval_tools/multiclass/road/1.extract_road_instance_patch.py'
STITCH=ROOT/'VecLang/eval_tools/multiclass/road/2.stitch_coco_polylines_junction.py'
NAME='IRSAMap_Road_instance_test_sft'

def graph_edges(graph):
    # The script serializes integer (row, column) adjacency. Canonicalize its undirected edges.
    return sorted({tuple(sorted((tuple(a),tuple(b)))) for a,ns in graph.items() for b in ns if tuple(a)!=tuple(b)})

def graph_feature(path):
    with Path(path).open('rb') as f:graph=pickle.load(f) # Locally produced by the trusted repository script.
    edges=graph_edges(graph);adj=collections.defaultdict(set)
    for a,b in edges:adj[a].add(b);adj[b].add(a)
    used=set();chains=[]
    def trace(a,b):
        line=[a,b];used.add(tuple(sorted((a,b))))
        while len(adj[b])==2:
            n=next(n for n in sorted(adj[b]) if n!=a);key=tuple(sorted((b,n)))
            if key in used:break
            used.add(key);line.append(n);a,b=b,n
        chains.append([[c,r] for r,c in line])
    for a in sorted(adj):
        if len(adj[a])!=2:
            for b in sorted(adj[a]):
                if tuple(sorted((a,b))) not in used:trace(a,b)
    for a,b in edges:
        if (a,b) not in used:trace(a,b)
    assert len(used)==len(edges)
    junctions=[[c,r] for (r,c) in sorted(adj) if len(adj[(r,c)])>=3]
    return {'type':'Feature','id':'road_network','properties':{'class':'road','topologyMode':'stitched-graph'},'geometry':{'type':'MultiLineString','coordinates':chains},'junctions':junctions},adj

def run(command,log):
    print('Running',Path(command[1]).name,log.name,flush=True)
    with log.open('w') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True,cwd=ROOT)

def raster(fc,cls,size=1024):
    mask=np.zeros((size,size),np.uint8)
    for f in fc:
        if f['properties']['class']!=cls:continue
        coords=f['geometry']['coordinates']
        if cls=='road':
            for line in coords:cv2.polylines(mask,[np.rint(line).astype(np.int32)],False,1,1)
        else:
            one=np.zeros_like(mask);cv2.fillPoly(one,[np.rint(coords[0]).astype(np.int32)],1)
            for hole in coords[1:]:cv2.fillPoly(one,[np.rint(hole).astype(np.int32)],0)
            mask|=one
    return mask

def main():
    WORK.mkdir(parents=True,exist_ok=True)
    import shutil
    candidates=[m for m in read(HERE/'public/cases/index.json') if m['dataset']=='IRSAMap']
    expected=[item['region'] for item in read(HERE/'scripts/selected_scenes.json')]
    assert [Path(m['source']['image']).stem for m in candidates] == expected
    write(WORK/'candidates.json',candidates)
    for m in candidates:
        shutil.copytree(HERE/'public/cases'/m['id'],WORK/'candidate_cases'/m['id'],dirs_exist_ok=True)
    regions=set(expected)
    manifest=read(DATA/(NAME+'.json'));images=[];sources=[];prediction=[];truth=[]
    with (VEC/(NAME+'.jsonl')).open() as stream:
        for i,line in enumerate(stream):
            m=manifest[i];path=m['images'][0];region=Path(path).name.split('_')[0]
            if region not in regions:continue
            row=json.loads(line);expected=m['messages'][-1]['content'].strip();assert expected.startswith(row['label'].strip())
            fid='region_'+Path(path).name
            images.append({'id':i+1,'file_name':fid,'width':256,'height':256})
            prediction.append({'predict':row['predict']});truth.append({'predict':expected})
            sources.append({'featureId':'road_network','region':region,'resultFile':NAME+'.jsonl','line':i+1,'image':path,'predict':row['predict'],'label':row['label'],'fullGroundTruth':expected,'stage':'before-stitching','displayMapping':'Many overlapping source patches map to the stitched road_network feature.'})
    assert len(manifest)==i+1
    write(WORK/'patch_index.json',{'images':images,'categories':[{'id':1,'name':'road'}],'annotations':[],'purpose':'Ordered image index for extraction; GT geometries come from the source manifest assistant labels.'})
    write(WORK/'sources.json',sources)
    commands=[]
    for source,rows in [('prediction',prediction),('gt',truth)]:
        inp=WORK/(source+'.jsonl');inp.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        cmd=[sys.executable,str(EXTRACT),'--inference-file',str(inp),'--gt-file',str(WORK/'patch_index.json'),'--output-file',str(WORK/(source+'.json'))]
        run(cmd,WORK/(source+'-extract.log'));commands.append(cmd)
        cmd=[sys.executable,str(STITCH),'--input_json',str(WORK/(source+'_full.json')),'--output_dir',str(WORK/source),'--crop_size_orig','128','--patch_size_model','256','--stride','64','--junction_json',str(WORK/(source+'_junctions.json')),'--preset','balanced','--enable_viz','0']
        run(cmd,WORK/(source+'-stitch.log'));commands.append(cmd)
        for region in regions:assert (WORK/source/'graph'/(region+'.p')).exists()
    ranked=[]
    for meta in candidates:
        region=Path(meta['source']['image']).stem
        pred,adj=graph_feature(WORK/'prediction/graph'/(region+'.p'));gt,_=graph_feature(WORK/'gt/graph'/(region+'.p'))
        pm=raster([pred],'road');gm=raster([gt],'road');kernel=np.ones((7,7),np.uint8)
        precision=float((pm*cv2.dilate(gm,kernel)).sum()/max(1,pm.sum()));recall=float((gm*cv2.dilate(pm,kernel)).sum()/max(1,gm.sum()));f1=2*precision*recall/max(1e-9,precision+recall)
        base=WORK/'candidate_cases'/meta['id'];pf=read(base/'prediction.geojson')['features'];gf=read(base/'gt.geojson')['features'];ious={}
        for cls in ['building','water']:
            p=raster(pf,cls);g=raster(gf,cls);ious[cls]=float((p&g).sum()/max(1,(p|g).sum()))
        score=.6*f1+.25*ious['building']+.15*ious['water']
        ranked.append({'id':meta['id'],'region':region,'roadPrecision3px':precision,'roadRecall3px':recall,'roadF1_3px':f1,'buildingIoU':ious['building'],'waterIoU':ious['water'],'selectionScore':score,'graphNodes':len(adj),'graphEdges':len(graph_edges(pickle.load(open(WORK/'prediction/graph'/(region+'.p'),'rb')))),'stitchedPolylines':len(pred['geometry']['coordinates']),'junctions':len(pred['junctions'])})
    ranked.sort(key=lambda r:-r['selectionScore'])
    write(WORK/'ranking.json',{'note':'Internal case selection, not a benchmark. Roads compared to the independently stitched full manifest labels with a 3-pixel raster tolerance; polygon IoU compares existing evaluation labels. Visual review is also required.','candidates':ranked})
    write(WORK/'provenance.json',{'sourceFile':NAME+'.jsonl','overlappingPatches':len(images),'regionIds':sorted(regions),'scripts':[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [EXTRACT,STITCH]],'commands':[[str(Path(v).relative_to(ROOT)) if str(v).startswith(str(ROOT)+'/') else v for v in cmd] for cmd in commands],'parameters':{'crop_size_orig':128,'patch_size_model':256,'stride':64,'preset':'balanced','junction_json':'Corresponding extracted prediction/GT junctions','remaining':'Unmodified script CLI defaults'},'gt':'Full source manifest labels independently passed through the same extraction and stitching pipeline; not original full-region GIS ground truth.'})
    print(json.dumps(ranked,indent=2),flush=True)
if __name__=='__main__':main()
