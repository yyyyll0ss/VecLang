export type VecClass = 'building' | 'road' | 'water';
export type Point = [number, number];
export interface Geometry { type: 'Polygon' | 'MultiLineString'; coordinates: Point[][]; }
export interface Junction { id: string; featureId: string; coordinates: Point; degree: number; connectedPolylines: string[]; }
export interface SVLFeature { id: string; class: VecClass; geometry: Geometry; topology: { mode?: 'stitched-graph'; junctions: Junction[] }; }
export interface VecLangResult {
  metadata: { id: string; name: string; width: number; height: number; classes: VecClass[]; coordinateSystem: Record<string, unknown> };
  image: string;
  prediction: { type: 'FeatureCollection'; features: { type: 'Feature'; id: string; properties: {class: VecClass; topologyMode?: 'stitched-graph'}; geometry: Geometry }[] };
  svl: { version: 'veclang-demo/1'; features: SVLFeature[] };
  topology: { nodes: Junction[]; segments: {id: string; featureId: string; coordinates: Point[]}[] };
}
/** Future online adapter; not implemented or called by the offline demo. */
export interface InferenceProvider { predict(image: File): Promise<VecLangResult>; }
/** The current StaticCaseProvider selects a known precomputed result. */
export interface CaseProvider { load(id: string): Promise<VecLangResult>; }
