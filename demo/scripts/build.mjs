import {readFile,writeFile,mkdir,readdir,cp,rm} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {resolve,dirname,join} from 'node:path';
const root=resolve(dirname(fileURLToPath(import.meta.url)),'..'),out=join(root,'dist');
const catalog=JSON.parse(await readFile(join(root,'public/cases/index.json'),'utf8'));
const cases={};
const files={metadata:'metadata.json',prediction:'prediction.geojson',gt:'gt.geojson',svl:'svl.json',raw:'raw-svl.json',topology:'topology.json',gtTopology:'gt-topology.json',detections:'detections.json'};
for(const m of catalog){const base=join(root,'public/cases',m.id),c={};for(const [key,file]of Object.entries(files))c[key]=JSON.parse(await readFile(join(base,file),'utf8'));for(const file of ['image','thumbnail'])c[file]='data:image/jpeg;base64,'+(await readFile(join(base,file+'.jpg'))).toString('base64');cases[m.id]=c;}
let js='';for(const file of ['lib/geometry.js','lib/caseLoader.js','components/VectorViewer.js','components/SVLViewer.js','app.js'])js+=(await readFile(join(root,'src',file),'utf8')).replace(/^import .*;\s*$/gm,'').replace(/^export /gm,'')+'\n';
// Parse the generated bundle before publishing it. No bundler or CDN dependency.
new Function(js);
let html=await readFile(join(root,'index.html'),'utf8');const css=await readFile(join(root,'src/theme.css'),'utf8');
html=html.replace('<link rel="stylesheet" href="src/theme.css">',()=>`<style>${css}</style>`).replace('<script type="module" src="src/app.js"></script>',()=>`<script>window.__CASES__=${JSON.stringify(cases).replace(/</g,'\\u003c')};</script><script>(()=>{${js.replace(/<\/script/gi,'<\\/script')}})();</script>`);
await mkdir(out,{recursive:true});await writeFile(join(out,'index.html'),html);await writeFile(join(root,'VecLang-Standalone.html'),html);await rm(join(out,'cases'),{recursive:true,force:true});await cp(join(root,'public/cases'),join(out,'cases'),{recursive:true});await writeFile(join(out,'.nojekyll'),'');
console.log(`Built ${catalog.length} real-data cases → dist/index.html (${(Buffer.byteLength(html)/1048576).toFixed(2)} MB). Standalone HTML also ready.`);
