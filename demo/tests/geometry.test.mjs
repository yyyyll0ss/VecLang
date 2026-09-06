import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile,readdir} from 'node:fs/promises';
import {parseSVL,deriveTopology,pathGeometry,gtToSVL} from '../src/lib/geometry.js';
const read=async p=>JSON.parse(await readFile(new URL(p,import.meta.url),'utf8'));
test('every real case round trips through the editable parser and keeps feature / junction IDs',async()=>{
 const catalog=await read('../public/cases/index.json');assert.equal(catalog.length,20);
 assert.equal(catalog.filter(m=>m.mode==='multi-category').length,5);
 for(const c of ['building','water','road'])assert.equal(catalog.filter(m=>m.mode==='single-category'&&m.classes.includes(c)).length,5);
 for(const m of catalog){const prefix=`../public/cases/${m.id}/`;const svl=await read(prefix+'svl.json'),prediction=await read(prefix+'prediction.geojson'),topology=await read(prefix+'topology.json');const parsed=parseSVL(JSON.stringify(svl));
  for(const c of ['building','road','water'])assert.equal(m.counts[c],prediction.features.filter(f=>f.properties.class===c).length);
  const truth=await read(prefix+'gt.geojson'),gtTopology=await read(prefix+'gt-topology.json');assert.deepEqual(parseSVL(JSON.stringify(gtToSVL(truth,gtTopology,m.coordinateSystem))).topology.nodes,gtTopology.nodes);
  if(m.mode==='multi-category'){assert.ok(m.classes.includes('building'));assert.ok(m.classes.includes('road'));assert.equal(topology.mode,'stitched-graph');assert.equal(svl.features.filter(f=>f.class==='road').length,1);assert.equal(parsed.prediction.features.find(f=>f.properties.class==='road').properties.topologyMode,'stitched-graph');}assert.equal(parsed.prediction.features.length,prediction.features.length);assert.deepEqual(parsed.topology.nodes,topology.nodes);
  parsed.prediction.features.forEach((f,i)=>{assert.deepEqual(f.geometry,prediction.features[i].geometry);assert.equal(f.id,prediction.features[i].id);assert.ok(pathGeometry(f.geometry).startsWith('M'));});
 }
});
test('a T junction has degree three but two polylines, and incidence changes after a coordinate edit',()=>{
 const f={id:'road_001',class:'road',geometry:{type:'MultiLineString',coordinates:[[[0,0],[10,0],[20,0]],[[10,0],[10,10]]]},topology:{junctions:[{id:'road_001:junction_001',coordinates:[10,0]}]}};
 let t=deriveTopology([f]);assert.equal(t.nodes[0].degree,3);assert.equal(t.nodes[0].connectedPolylines.length,2);f.topology.junctions[0].coordinates=[500,500];t=deriveTopology([f]);assert.equal(t.nodes[0].degree,0);
});
test('polygon holes survive conversion; invalid coordinates, open rings and duplicate IDs fail',()=>{
 const f={id:'water_001',class:'water',geometry:{type:'Polygon',coordinates:[[[0,0],[20,0],[20,20],[0,0]],[[5,5],[6,5],[6,6],[5,5]]]},topology:{junctions:[]}};
 const doc={version:'veclang-demo/1',features:[f]};assert.equal(parseSVL(JSON.stringify(doc)).prediction.features[0].geometry.coordinates.length,2);
 const invalid=structuredClone(doc);invalid.features[0].geometry.coordinates[0][3]=[1,1];assert.throws(()=>parseSVL(JSON.stringify(invalid)),/close/);
 assert.throws(()=>parseSVL(JSON.stringify({...doc,features:[f,f]})),/unique/);
 const bad=structuredClone(doc);bad.features[0].geometry.coordinates[0][0][0]='NaN';assert.throws(()=>parseSVL(JSON.stringify(bad)),/coordinate/);
});
test('changing a junction cannot reuse another feature ID',()=>{const f={id:'road_001',class:'road',geometry:{type:'MultiLineString',coordinates:[[[0,0],[10,0]]]},topology:{junctions:[{id:'water_001',coordinates:[0,0]}]}};const water={id:'water_001',class:'water',geometry:{type:'Polygon',coordinates:[[[0,0],[1,0],[1,1],[0,0]]]},topology:{junctions:[]}};assert.throws(()=>parseSVL(JSON.stringify({version:'veclang-demo/1',features:[f,water]})),/unique/);});

test('stitched junction incidence uses exact endpoints, preserving loops and nearby disconnected lines',()=>{
 const f={id:'road_network',class:'road',geometry:{type:'MultiLineString',coordinates:[[[0,0],[5,0],[5,5],[0,0]],[[0,0],[-5,0]],[[1,0],[1,5]]]},topology:{mode:'stitched-graph',junctions:[{id:'road_network:junction_001',coordinates:[0,0]}]}};
 const t=deriveTopology([f]);assert.equal(t.mode,'stitched-graph');assert.equal(t.nodes[0].degree,3);assert.equal(t.nodes[0].connectedPolylines.length,2);assert.equal(t.tolerancePixels,0);
});
