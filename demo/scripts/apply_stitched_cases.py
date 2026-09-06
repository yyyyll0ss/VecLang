"""Publish the five user-specified IRSAMap scenes using serialized junction-stitched graphs."""
import hashlib,json,shutil
from pathlib import Path
from prepare_cases import HERE,OUT,read,write,topology
from stitch_demo_roads import WORK,graph_feature
SELECTED=[item['region'] for item in read(HERE/'scripts/selected_scenes.json')]

def main():
    inventory=read(WORK/'candidates.json');snapshot=WORK/'candidate_cases';snapshot.mkdir(exist_ok=True)
    # Preserve the existing ten candidate inputs for audit and reproducible re-selection.
    catalog=read(OUT/'index.json')
    if len([m for m in catalog if m['dataset']=='IRSAMap'])==10:
        for m in inventory:shutil.copytree(OUT/m['id'],snapshot/m['id'],dirs_exist_ok=True)
    assert all((snapshot/m['id']/'metadata.json').exists() for m in inventory)
    source_records=read(WORK/'sources.json');pipeline=read(WORK/'provenance.json');ranking=read(WORK/'ranking.json')['candidates']
    published=[]
    for index,region in enumerate(SELECTED,1):
        old=next(m for m in inventory if m['source']['image']==region+'.png');base=snapshot/old['id'];cid=f'multi_{index:02}';dest=OUT/cid
        shutil.copytree(base,dest,dirs_exist_ok=True)
        road,adj=graph_feature(WORK/'prediction/graph'/(region+'.p'));gtroad,gtadj=graph_feature(WORK/'gt/graph'/(region+'.p'))
        prediction=read(base/'prediction.geojson');truth=read(base/'gt.geojson')
        prediction['features']=[f for f in prediction['features'] if f['properties']['class']!='road']+[road]
        truth['features']=[f for f in truth['features'] if f['properties']['class']!='road']+[gtroad]
        pt=topology(prediction['features']);gt=topology(truth['features'])
        for node in pt['nodes']:assert node['degree']==len(adj[tuple(reversed(node['coordinates']))])
        for node in gt['nodes']:assert node['degree']==len(gtadj[tuple(reversed(node['coordinates']))])
        svl=read(base/'svl.json');svl['features']=[f for f in svl['features'] if f['class']!='road']+[{'id':road['id'],'class':'road','geometry':road['geometry'],'topology':{'mode':'stitched-graph','junctions':pt['nodes']}}]
        raw=[r for r in read(base/'raw-svl.json') if not r['featureId'].startswith('road_')]+[r for r in source_records if r['region']==region]
        meta=read(base/'metadata.json');meta.update({'id':cid,'name':f'Unified scene {index:02}','selectionFocus':'stitched','description':f'IRSAMap region {region}: buildings, water and a junction-assisted stitched road network.'})
        meta['counts']['road']=1;meta['counts']['roadPolylines']=len(road['geometry']['coordinates']);meta['counts']['junctions']=len(pt['nodes'])
        meta['source']['roadTiling']='All overlapping 128-pixel patches at stride 64; multiclass extraction followed by junction-assisted stitching with balanced preset and extracted junctions.'
        meta['source']['transform']='Polygon crops retain their verified original inverse transforms. Road results are the serialized region graphs from the extraction + stitching pipeline; (row, col) converted to (x, y), and degree-2 chains compressed without changing any undirected graph edge.'
        meta['roadStitching']={'region':region,'pipeline':pipeline,'graphFile':f'artifacts/road_stitching_junction/prediction/graph/{region}.p','graphSha256':hashlib.sha256((WORK/'prediction/graph'/(region+'.p')).read_bytes()).hexdigest(),'sourcePatchCount':sum(r.get('region')==region for r in source_records),'graphNodes':len(adj),'junctionSource':'Graph adjacency (degree >= 3), not unstitched patch junction coordinates.','rawMapping':'All original overlapping road patch outputs map many-to-one to road_network. Map SVL is the postprocessed graph; Raw output remains pre-stitch model text.'}
        meta['gtDescription']=pipeline['gt'];meta['selectionReview']={'criterion':'Exact five user-specified regions, in requested order; no score-based substitution.','originalCandidateId':old['id']}
        for file,value in [('prediction.geojson',prediction),('gt.geojson',truth),('svl.json',svl),('raw-svl.json',raw),('topology.json',pt),('gt-topology.json',gt),('metadata.json',meta)]:write(dest/file,value)
        published.append(meta)
    # Remove only generated retired multi-case folders, including stale copies from prior builds.
    for p in OUT.glob('multi_*'):
        if p.name not in {m['id'] for m in published}:shutil.rmtree(p)
    singles=[m for m in catalog if m['mode']=='single-category'];assert len(singles)==15
    write(OUT/'index.json',published+singles)
    write(WORK/'selection.json',{'selectedRegions':SELECTED,'selected':[{'id':m['id'],'region':m['source']['image'],'counts':m['counts']} for m in published],'review':'User-specified regions 50, 457, 41, 259, 726 in requested order.','ranking':ranking})
    print('Published 5 stitched multicategory + 15 unchanged single-category scenes.')
if __name__=='__main__':main()
