/** Future providers can implement predict(image: File): Promise<VecLangResult>.
 * The UI consumes the same result shape as StaticCaseProvider.load(id). */
export class StaticCaseProvider {
 async catalog(){if(window.__CASES__)return Object.values(window.__CASES__).map(c=>c.metadata);return this.json('public/cases/index.json');}
 async json(url){const r=await fetch(url);if(!r.ok)throw Error(`${r.status}: ${url}`);return r.json();}
 async load(id){if(window.__CASES__)return structuredClone(window.__CASES__[id]);const base=`public/cases/${id}/`;const keys={'metadata':'metadata.json','prediction':'prediction.geojson','gt':'gt.geojson','svl':'svl.json','raw':'raw-svl.json','topology':'topology.json','gtTopology':'gt-topology.json','detections':'detections.json'};const entries=await Promise.all(Object.entries(keys).map(async([k,f])=>[k,await this.json(base+f)]));return {...Object.fromEntries(entries),image:base+'image.jpg',thumbnail:base+'thumbnail.jpg'};}
 thumbnail(id){return window.__CASES__?.[id]?.thumbnail||`public/cases/${id}/thumbnail.jpg`;}
}
export function escapeHTML(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
export function downloadFile(name,data,type='application/json'){const url=URL.createObjectURL(new Blob([typeof data==='string'?data:JSON.stringify(data,null,2)],{type}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
