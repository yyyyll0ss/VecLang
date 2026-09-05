import {pathGeometry,featurePoints,featureBounds} from '../lib/geometry.js';
export class VectorViewer {
 constructor(container,onSelect,onHover,onCamera){this.container=container;this.onSelect=onSelect;this.onHover=onHover;this.onCamera=onCamera;this.ns='http://www.w3.org/2000/svg';container.innerHTML='<svg class="map-svg" role="img" aria-label="Interactive vector canvas"></svg><div class="map-tag"></div><div class="map-controls"><button data-zoom="in" title="Zoom in">+</button><button data-zoom="out" title="Zoom out">−</button><button data-zoom="fit" title="Fit to screen">⌖</button></div>';this.svg=container.querySelector('svg');
 container.querySelectorAll('[data-zoom]').forEach(b=>b.onclick=()=>{if(b.dataset.zoom==='fit')this.fit();else this.zoom(b.dataset.zoom==='in'?.8:1.25);});
 this.svg.addEventListener('wheel',e=>{e.preventDefault();this.zoom(e.deltaY>0?1.15:.87,this.position(e));},{passive:false});
 this.svg.addEventListener('pointerdown',e=>{if(e.button!==0)return;this.drag={x:e.clientX,y:e.clientY,box:[...this.box],moved:false,target:e.target};this.svg.setPointerCapture(e.pointerId);});
 this.svg.addEventListener('pointermove',e=>{if(this.drag){const d=this.drag,r=this.svg.getBoundingClientRect(),scale=Math.max(this.box[2]/r.width,this.box[3]/r.height);const dx=e.clientX-d.x,dy=e.clientY-d.y;if(Math.abs(dx)+Math.abs(dy)>4)d.moved=true;if(d.moved){this.box=[d.box[0]-dx*scale,d.box[1]-dy*scale,...d.box.slice(2)];this.camera();}}else{const t=e.target.closest('[data-id]');this.onHover(t?.dataset.id||null,t?.dataset.source||'prediction');}});
 this.svg.addEventListener('pointerup',e=>{if(this.drag&&!this.drag.moved){const t=this.drag.target.closest('[data-id]');if(t)this.onSelect(t.dataset.id,t.dataset.source||'prediction',t.dataset.junction||null);}this.drag=null;this.svg.releasePointerCapture(e.pointerId);});
 this.svg.addEventListener('pointercancel',()=>this.drag=null);this.svg.addEventListener('pointerleave',()=>{if(!this.drag)this.onHover(null);});
 this.svg.addEventListener('keydown',e=>{if(['Enter',' '].includes(e.key)&&e.target.dataset.id){e.preventDefault();this.onSelect(e.target.dataset.id,e.target.dataset.source,e.target.dataset.junction||null);}});
 }
 position(e){const p=this.svg.createSVGPoint();p.x=e.clientX;p.y=e.clientY;const v=p.matrixTransform(this.svg.getScreenCTM().inverse());return [v.x,v.y];}
 fit(){if(!this.size)return;this.box=[-this.size[0]*.025,-this.size[1]*.025,this.size[0]*1.05,this.size[1]*1.05];this.camera();}
 zoom(scale,point){if(!this.box)return;const [x,y,w,h]=this.box;const nw=w*scale;if(nw<this.size[0]/30||nw>this.size[0]*6)return;const p=point||[x+w/2,y+h/2];this.box=[p[0]-(p[0]-x)*scale,p[1]-(p[1]-y)*scale,nw,h*scale];this.camera();}
 camera(notify=true){this.svg.setAttribute('viewBox',this.box.join(' '));if(notify)this.onCamera?.(this.box);}
 setCamera(box){this.box=[...box];this.camera(false);}
 element(type,attrs,parent=this.svg){const e=document.createElementNS(this.ns,type);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);parent.append(e);return e;}
 render(data,state,isImage){const m=data.metadata;this.size=[m.width,m.height];if(!this.box)this.fit();this.svg.replaceChildren();const showImage=isImage||state.vectorBackground;
 if(showImage)this.element('image',{href:data.image,x:0,y:0,width:m.width,height:m.height});
 const mode=isImage?state.imageMode:state.view;
 const sources=mode==='image'?[]:mode==='both'?['gt','prediction']:mode==='gt'?['gt']:['prediction'];
 const order={water:0,building:1,road:2};
 for(const source of sources){const features=[...data[source].features].sort((a,b)=>order[a.properties.class]-order[b.properties.class]);
  for(const f of features){if(!state.layers.has(f.properties.class))continue;const gt=source==='gt';const color=gt?'var(--gt)':`var(--${f.properties.class})`;
   const attrs={d:pathGeometry(f.geometry),fill:f.geometry.type==='Polygon'?color:'none','fill-rule':'evenodd','fill-opacity':gt?.04:showImage?.16:.19,stroke:color,'stroke-width':gt?1.6:f.properties.class==='road'?2.4:1.5,'stroke-dasharray':gt?'5 4':'none','data-id':f.id,'data-source':source,class:'feature',tabindex:0,role:'button','aria-label':`${source} ${f.id}`};const path=this.element('path',attrs);this.element('title',{},path).textContent=`${f.id} · ${f.properties.class} · ${source}`;
   if(state.vertices)for(const p of featurePoints(f))this.element('circle',{cx:p[0],cy:p[1],r:m.width/270,fill:color,stroke:showImage?'#fff':'white','stroke-width':.6,class:'vertex'});
  }
  if(state.topology&&state.layers.has('road'))for(const n of data[source==='gt'?'gtTopology':'topology'].nodes){const circle=this.element('circle',{cx:n.coordinates[0],cy:n.coordinates[1],r:m.width/100,fill:source==='gt'?'var(--gt)':'var(--junction)',stroke:'white','stroke-width':1.5,class:'junction','data-id':n.featureId,'data-source':source,'data-junction':n.id,tabindex:0,role:'button','aria-label':n.id});this.element('title',{},circle).textContent=`${n.id} · degree ${n.degree}`;}
 }
 if(isImage&&state.boxes){const d=data.detections;for(const src of state.view==='both'?['gt','prediction']:[state.view])for(const b of d[src]||[]){const cls=b.label.toLowerCase().replace('waterbody','water');if(!state.layers.has(cls))continue;const [x,y,x2,y2]=b.bbox_2d,s=d.coordinateScale||1000;this.element('rect',{x:x*m.width/s,y:y*m.height/s,width:(x2-x)*m.width/s,height:(y2-y)*m.height/s,fill:'none',stroke:src==='gt'?'var(--gt)':`var(--${cls})`,'stroke-width':1,'stroke-dasharray':'4 3',class:'bound'});}}
 this.container.querySelector('.map-tag').textContent=isImage?`${m.dataset} · ${mode==='image'?'RGB imagery':mode==='both'?'GT + prediction':mode==='gt'?'Ground truth':'Prediction overlay'}`:state.topology?(data.topology.mode==='stitched-graph'?'Stitched road topology':'Local road topology'):'Executable vector geometry';
 this.highlight(state.hover||state.selected,state.hover?state.hoverSource:state.selectedSource,data,state);
 }
 highlight(id,source,data,state){this.svg.querySelectorAll('.selected,.selection-box').forEach(e=>e.classList.contains('selection-box')?e.remove():e.classList.remove('selected'));if(!id)return;this.svg.querySelectorAll('[data-id]').forEach(e=>{if(e.dataset.id===id&&e.dataset.source===source)e.classList.add('selected');});const f=data[source]?.features.find(f=>f.id===id);if(!f||!state.layers.has(f.properties.class))return;const [x,y,x2,y2]=featureBounds(f);this.element('rect',{x:x-3,y:y-3,width:x2-x+6,height:y2-y+6,fill:'none',stroke:'var(--highlight)','stroke-width':1,'stroke-dasharray':'4 3',class:'bound selection-box'});
 // Keep a selected shape visible even when the image panel is in RGB-only mode.
 const node=!state.hover&&state.junction?data[source==='gt'?'gtTopology':'topology'].nodes.find(n=>n.id===state.junction):null;
 if(node){this.svg.querySelectorAll('.feature.selected').forEach(e=>e.classList.remove('selected'));const topology=data[source==='gt'?'gtTopology':'topology'];for(const segment of topology.segments.filter(s=>node.connectedPolylines.includes(s.id)))this.element('path',{d:pathGeometry({type:'MultiLineString',coordinates:[segment.coordinates]}),fill:'none',stroke:'var(--highlight)','stroke-width':4,class:'bound selection-box'});this.element('circle',{cx:node.coordinates[0],cy:node.coordinates[1],r:data.metadata.width/80,fill:'none',stroke:'var(--highlight)','stroke-width':3,class:'bound selection-box'});}
 else this.element('path',{d:pathGeometry(f.geometry),fill:'none',stroke:'var(--highlight)','stroke-width':2,class:'bound selection-box'});
 }
}
