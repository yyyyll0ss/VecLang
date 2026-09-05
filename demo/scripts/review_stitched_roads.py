"""Render selection-only contact sheets; the actual demo remains interactive SVG."""
import sys
from pathlib import Path
from PIL import Image,ImageDraw
import cv2,numpy as np
from stitch_demo_roads import WORK,graph_feature
from prepare_cases import HERE,read
ranking=read(WORK/'ranking.json')['candidates'];sheet=Image.new('RGB',(1500,5*410),'#f5f6f8');draw=ImageDraw.Draw(sheet)
for i,r in enumerate(ranking):
 base=(WORK/'candidate_cases' if (WORK/'candidate_cases').exists() else HERE/'public/cases')/r['id'];im=cv2.imread(str(base/'image.jpg'))
 pred,_=graph_feature(WORK/'prediction/graph'/(r['region']+'.p'));gt,_=graph_feature(WORK/'gt/graph'/(r['region']+'.p'))
 for pts in gt['geometry']['coordinates']:cv2.polylines(im,[np.array(pts,np.int32)],False,(190,90,160),2)
 for pts in pred['geometry']['coordinates']:cv2.polylines(im,[np.array(pts,np.int32)],False,(30,220,255),2)
 for f in read(base/'prediction.geojson')['features']:
  if f['properties']['class']=='road':continue
  for pts in f['geometry']['coordinates']:cv2.polylines(im,[np.array(pts,np.int32)],True,(220,140,40) if f['properties']['class']=='water' else (100,120,225),2)
 thumb=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)).resize((370,370))
 x=(i%2)*750;y=(i//2)*410
 sheet.paste(thumb,(x,y+30))
 draw.text((x+5,y+6),f"{r['id']} / {r['region']} score {r['selectionScore']:.3f}",fill='black')
 clean=Image.new('RGB',(1024,1024),'white');cd=ImageDraw.Draw(clean)
 for pts in gt['geometry']['coordinates']:cd.line([tuple(p) for p in pts],fill='#b277c4',width=2)
 for pts in pred['geometry']['coordinates']:cd.line([tuple(p) for p in pts],fill='#dfaa1c',width=2)
 sheet.paste(clean.resize((370,370)),(x+375,y+30))
sheet.save(WORK/'contact-sheet.jpg',quality=95)
