"""Small data-contract and regression tests; no datasets or model inference required."""
import contextlib
import importlib.util
import io
import json
import pickle
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

def module(relative):
    spec = importlib.util.spec_from_file_location('eval_module', ROOT / relative)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding='utf-8')

class PipelineTest(unittest.TestCase):
    def test_polygon_conversion_and_empty_metrics(self):
        for name, category in [('building', 'building'), ('waterbody', 'water_body')]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
                d = Path(tmp)
                gt = {'images': [{'id': i+1, 'file_name': f'100_x{x:04d}_y0000.png', 'width': 64, 'height': 64,
                    'source_image': '100.png', 'patch_x': x, 'patch_y': 0} for i, x in enumerate([0,64])],
                    'categories': [{'id': 7, 'name': category}], 'annotations': []}
                save(d/'gt.json',gt)
                save(d/'annotations/crop.json',{'original_image':'100.png','crop_info': {'crop_x':48,'crop_y':8,'crop_w':32,'crop_h':32,'output_size':256}})
                obj={'type':'Feature','geometry':{'type':'Polygon','coordinates':[0,0,1000,0,1000,1000,0,1000]},'properties':{}}
                save(d/'pred.jsonl', {'predict':json.dumps(obj)})
                conv=module(f'{name}/1.convert_to_coco_format_instance_patch.py')
                conv.convert_instance_predictions_to_coco(str(d/'pred.jsonl'), str(d/'annotations'), str(d),str(d/'dt.json'), gt_file=str(d/'gt.json'))
                anns=json.loads((d/'dt.json').read_text())
                self.assertEqual(len(anns),2)
                self.assertEqual({a['category_id'] for a in anns},{7})
                self.assertEqual({a['image_id'] for a in anns},{1,2})
                self.assertAlmostEqual(sum(a['area'] for a in anns),1024)
                for a in anns: a['iscrowd']=0;a['id']=a['image_id']
                gt['annotations']=anns;save(d/'gt.json',gt)
                ev=module(f'{name}/2.eval_{name}.py')
                self.assertEqual(ev.resolve_category(str(d/'gt.json'),category,None)[0],7)
                perfect=ev.compute_map(str(d/'gt.json'),anns,iou_type='segm',category_id=7)
                self.assertAlmostEqual(perfect['metrics']['segm']['AP_50'],1)
                empty=ev.compute_map(str(d/'gt.json'),[],iou_type='segm',category_id=7)
                self.assertTrue(empty['available']);self.assertEqual(empty['metrics']['segm']['AP_50'],0)
                self.assertEqual(ev.compute_polygon_metrics(str(d/'gt.json'),[],category_id=7)['metrics']['IoU'],0)
                (d/'pred.jsonl').write_text('')
                with self.assertRaises(ValueError):
                    conv.convert_instance_predictions_to_coco(str(d/'pred.jsonl'),str(d/'annotations'),str(d),str(d/'bad.json'),gt_file=str(d/'gt.json'))

    def test_scene_ids_and_empty_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            gt = {"images": [{"id": i, "file_name": f"{region}_x0000_y0000.png", "width": 64, "height": 64}
                              for i, region in enumerate([100, 102], 1)],
                  "categories": [{"id": 1, "name": "building"}],
                  "annotations": [{"id": i, "image_id": i, "category_id": 1, "iscrowd": 0,
                                   "bbox": [10, 10, 20, 20], "area": 400,
                                   "segmentation": [[10, 10, 30, 10, 30, 30, 10, 30]]} for i in [1, 2]]}
            save(d / "gt.json", gt)
            save(d / "dt.json", [])
            self.cli("building/5.eval_by_scene.py", "--gt", d / "gt.json", "--dt", d / "dt.json", "--output", d / "scenes.json")
            scenes = json.loads((d / "scenes.json").read_text())["scenes"]
            self.assertEqual(set(scenes), {"100", "102"})
            for scene in scenes.values():
                self.assertEqual(scene["coco_metrics"]["metrics"]["segm"]["AP_50"], 0)

    def test_road_extract_stitch_and_display(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)
            save(d/'patch.json',{'images':[{'id':1,'file_name':'100_patch_0_0_0.png','width':256,'height':256}], 'categories':[{'id':1,'name':'road'}],'annotations':[]})
            save(d/'pred.jsonl',{'predict':json.dumps({'type':'Feature','geometry':{'type':'LineString','coordinates':[100,500,900,500]},'properties':{}})})
            self.cli('road/1.extract_road_instance_patch.py','--inference-file',d/'pred.jsonl','--gt-file',d/'patch.json','--output-file',d/'road.json')
            for stitch in ['2.stitch_coco_polylines.py','2.stitch_coco_polylines_junction.py']:
                out=d/stitch
                self.cli('road/'+stitch,'--input_json',d/'road_full.json','--output_dir',out,'--enable_viz','0')
                self.assertTrue((out/'graph/100.p').is_file())
                with (out/'graph/100.p').open('rb') as f: graph=pickle.load(f)
                self.assertGreater(len(graph),0)
                import argparse
                vis=module('visualize_multiclass_inference_results.py')
                graphs=vis.build_road_graphs(argparse.Namespace(road_graph_dir=str(out/'graph')))
                self.assertEqual(set(map(tuple,graphs[100][0])),set(graph))
            empty = json.loads((d/'road_full.json').read_text())
            empty['annotations'] = []
            save(d/'empty.json', empty)
            for stitch in ['2.stitch_coco_polylines.py','2.stitch_coco_polylines_junction.py']:
                out = d / ('empty-' + stitch)
                self.cli('road/'+stitch, '--input_json', d/'empty.json', '--output_dir', out, '--enable_viz', '0')
                with (out/'graph/100.p').open('rb') as f:
                    self.assertEqual(pickle.load(f), {})
            (d/'pred.jsonl').write_text('')
            result=subprocess.run([sys.executable,str(ROOT/'road/1.extract_road_instance_patch.py'),'--inference-file',str(d/'pred.jsonl'),'--gt-file',str(d/'patch.json'),'--output-file',str(d/'bad.json')],capture_output=True)
            self.assertNotEqual(result.returncode,0)

    def test_explicit_empty_road_prediction(self):
        extractor = module('road/1.extract_road_instance_patch.py')
        self.assertTrue(extractor.is_explicit_empty_prediction('```json\n[]\n```'))
        self.assertFalse(extractor.is_explicit_empty_prediction('broken'))

    def test_gt_conversion_and_region_discovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)
            save(d/'geo/100.geojson',{'type':'FeatureCollection','features':[{'type':'Feature','geometry':{'type':'LineString','coordinates':[[10,20],[80,20],[80,90]]}}]})
            self.cli('road/0.convert_irsamap_road_geojson_to_graph_gt.py','--geojson-dir',d/'geo','--output-root',d/'gt')
            with (d/'gt/region_100_graph_gt.pickle').open('rb') as f: graph=pickle.load(f)
            self.assertIn((20,10),graph)
            runner=module('road/3.run_irsamap_metrics.py')
            self.assertEqual(runner.parse_tiles_arg('',d/'gt'),[100])
            with self.assertRaises(ValueError): runner.parse_tiles_arg('',d/'none')
            result=subprocess.run([sys.executable,str(ROOT/'road/3.run_irsamap_metrics.py'),'--gt-root',str(d/'gt'),'--savedir',str(d/'missing'),'--only','topo'],capture_output=True)
            self.assertNotEqual(result.returncode,0)

    def cli(self, script, *args):
        result=subprocess.run([sys.executable,str(ROOT/script),*map(str,args)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)

if __name__ == '__main__':
    unittest.main()
