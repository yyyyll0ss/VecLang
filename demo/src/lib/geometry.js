export const classes = ['building','road','water'];
export function featurePoints(f) { return f.geometry.coordinates.flat(); }
export function featureBounds(f) { const p=featurePoints(f); return [Math.min(...p.map(p=>p[0])),Math.min(...p.map(p=>p[1])),Math.max(...p.map(p=>p[0])),Math.max(...p.map(p=>p[1]))]; }
export function pathGeometry(g) { return g.coordinates.map(r=>r.map((p,i)=>`${i?'L':'M'}${p[0]},${p[1]}`).join(' ')+(g.type==='Polygon'?'Z':'')).join(' '); }
export function distanceToLine(p,a,b) { const dx=b[0]-a[0],dy=b[1]-a[1],d=dx*dx+dy*dy;const t=d?Math.max(0,Math.min(1,((p[0]-a[0])*dx+(p[1]-a[1])*dy)/d)):0;return Math.hypot(p[0]-a[0]-t*dx,p[1]-a[1]-t*dy); }
export function deriveTopology(features,tolerance=2) {
 const nodes=[],segments=[];
 for(const f of features){if(f.geometry.type!=='MultiLineString')continue;
  const lines=f.geometry.coordinates; const ids=lines.map((coordinates,i)=>{const id=`${f.id}:line_${String(i+1).padStart(3,'0')}`;segments.push({id,featureId:f.id,coordinates});return id;});
  for(const j of f.topology.junctions){let degree=0;const connectedPolylines=[];
   lines.forEach((line,i)=>{if(f.topology.mode==='stitched-graph'){const incident=[line[0],line.at(-1)].filter(p=>p[0]===j.coordinates[0]&&p[1]===j.coordinates[1]).length;if(incident){connectedPolylines.push(ids[i]);degree+=incident;}return;}if(line.slice(1).some((b,k)=>distanceToLine(j.coordinates,line[k],b)<=tolerance)){connectedPolylines.push(ids[i]);degree+=Math.min(Math.hypot(j.coordinates[0]-line[0][0],j.coordinates[1]-line[0][1]),Math.hypot(j.coordinates[0]-line.at(-1)[0],j.coordinates[1]-line.at(-1)[1]))<=tolerance?1:2;}});
   nodes.push({...j,featureId:f.id,degree,connectedPolylines});
  }
 }
 if(features.some(f=>f.topology.mode==='stitched-graph'))return {nodes,segments,mode:'stitched-graph',tolerancePixels:0,provenance:'Stitched region graph: incidence uses exact shared polyline endpoints; no proximity-based connections.'};
 return {nodes,segments,tolerancePixels:tolerance,provenance:'Junction coordinates from SVL; incidence derived within each source feature at 2 image pixels.'};
}
export function parseSVL(text) {
 const doc=JSON.parse(text);if(doc.version!=='veclang-demo/1'||!Array.isArray(doc.features)||doc.features.length>5000)throw Error('Expected veclang-demo/1 with at most 5000 features.');
 const ids=new Set();const point=p=>Array.isArray(p)&&p.length===2&&p.every(v=>typeof v==='number'&&Number.isFinite(v)&&Math.abs(v)<=1e7);
 for(const f of doc.features){
  if(typeof f.id!=='string'||!/^[a-zA-Z0-9_:-]+$/.test(f.id)||ids.has(f.id))throw Error('Feature IDs must be unique letters, numbers, underscores, colons or hyphens.');ids.add(f.id);
  if(!classes.includes(f.class))throw Error(`${f.id}: class must be building, road or water.`);
  const g=f.geometry;if(!g||!['Polygon','MultiLineString'].includes(g.type)||!Array.isArray(g.coordinates)||!g.coordinates.length)throw Error(`${f.id}: expected Polygon or MultiLineString.`);
  if((f.class==='road')!==(g.type==='MultiLineString'))throw Error(`${f.id}: road requires MultiLineString; building and water require Polygon.`);
  for(const r of g.coordinates){if(!Array.isArray(r)||r.length<(g.type==='Polygon'?4:2)||!r.every(point))throw Error(`${f.id}: invalid coordinate pairs or too few vertices.`);if(g.type==='Polygon'&&(r[0][0]!==r.at(-1)[0]||r[0][1]!==r.at(-1)[1]))throw Error(`${f.id}: polygon rings must close (last vertex = first).`);}
  if(!f.topology||!Array.isArray(f.topology.junctions))throw Error(`${f.id}: topology.junctions must be an array.`);
  if(f.topology.mode!==undefined&&f.topology.mode!=='stitched-graph')throw Error(`${f.id}: unknown topology mode.`);
  if(g.type==='Polygon'&&f.topology.mode)throw Error(`${f.id}: graph mode requires road geometry.`);
  if(g.type==='Polygon'&&f.topology.junctions.length)throw Error(`${f.id}: polygon cannot contain road junctions.`);
  for(const j of f.topology.junctions){if(typeof j.id!=='string'||!/^[a-zA-Z0-9_:-]+$/.test(j.id)||ids.has(j.id)||!point(j.coordinates))throw Error(`${f.id}: invalid or duplicate junction.`);ids.add(j.id);}
 }
 const topology=deriveTopology(doc.features);
 doc.features.forEach(f=>f.topology.junctions=topology.nodes.filter(n=>n.featureId===f.id));
 return {svl:doc,prediction:{type:'FeatureCollection',coordinateSystem:doc.coordinateSystem,features:doc.features.map(f=>({type:'Feature',id:f.id,properties:{class:f.class,...(f.topology.mode?{topologyMode:f.topology.mode}:{})},geometry:f.geometry}))},topology};
}
export function gtToSVL(fc,topology,coordinateSystem) {return {version:'veclang-demo/1',coordinateSystem,features:fc.features.map(f=>({id:f.id,class:f.properties.class,geometry:f.geometry,topology:{...(f.properties.topologyMode?{mode:f.properties.topologyMode}:{}),junctions:topology.nodes.filter(n=>n.featureId===f.id)}}))};}
