package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"os"
	//"strconv"
	//"sort"
	"container/heap"
	"github.com/dhconnelly/rtreego"
)

// for 2k by 2k tiles

// The unit of interval is in 'hop' - each hop is 2 meters.
var interval_1 int = 37       // 50 meters for control points (25*1.5=37)
var interval_2 float64 = 25.0 // 50 meters for control points  (25*2=50)

var min_distance_filter float64 = 100.0 // don't consider shortest path shorter than 100.0
var prop_step int = 4                   // one-to-one matching, if two control points mapped to A,B in the inferred graph, distance between A and B on the graph should greater than '4' hops, each hop is 2 meters.

var region_size float64 = 2048.0 // unit is meter
var margin_size float64 = 100.0  // unit is meter

type locKey struct {
	Lat int64
	Lon int64
}

type gridKey struct {
	X int
	Y int
}

type graph struct {
	// base info
	Nodes [][2]float64
	Edges [][2]int

	// aux info
	loc2index map[locKey]int
	neighbors map[int]map[int]int
}

func (g *graph) propagate(nid int, step int, action func(nid int)) {
	if nid < 0 || nid >= len(g.Nodes) {
		return
	}

	visited := make([]int, len(g.Nodes))
	for i := range visited {
		visited[i] = -1
	}

	queue := make([]int, 1, 64)
	queue[0] = nid
	visited[nid] = 0

	for head := 0; head < len(queue); head++ {
		current_nid := queue[head]

		if visited[current_nid] > step {
			continue
		}

		action(current_nid)

		for next_nid := range g.neighbors[current_nid] {
			if visited[next_nid] >= 0 {
				continue
			}
			queue = append(queue, next_nid)
			visited[next_nid] = visited[current_nid] + 1
		}
	}
}

func GPSDistance(p1 [2]float64, p2 [2]float64) float64 {
	a := (p1[0] - p2[0]) * 111111.0
	b := (p1[1] - p2[1]) * 111111.0 * math.Cos(p1[0]/360.0*2.0*math.Pi)

	return math.Sqrt(a*a + b*b)
}

func GPSInBound(p1 [2]float64) bool {
	lat_top_left := 41.0
	lon_top_left := -71.0

	lat2 := lat_top_left - region_size/111111.0
	lon2 := lon_top_left + region_size/111111.0/math.Cos(lat_top_left/180.0*3.1415926)

	if p1[0] > lat2+margin_size/111111.0 && p1[0] < lat_top_left-margin_size/111111.0 && p1[1] > lon_top_left+margin_size/111111.0/math.Cos(lat_top_left/180.0*3.1415926) && p1[1] < lon2-margin_size/111111.0/math.Cos(lat_top_left/180.0*3.1415926) {
		return true
	} else {
		return false
	}

	return false
}

type gpsnode struct {
	nid int
	loc [2]float64
}

var tol = 0.000001

func (s *gpsnode) Bounds() rtreego.Rect {
	return rtreego.Point{s.loc[0], s.loc[1]}.ToRect(tol)
}

func gpsDistanceInt(p1 [2]float64, p2 [2]float64) int {
	return int(GPSDistance(p1, p2) * 100.0)
}

func loc2key(loc [2]float64) locKey {
	return locKey{
		Lat: int64(math.Round(loc[0] * 10000000.0)),
		Lon: int64(math.Round(loc[1] * 10000000.0)),
	}
}

func LoadGraphFromJson(filename string) *graph {
	// fmt.Println("loading graphs", filename)
	g := new(graph)

	g.loc2index = make(map[locKey]int)
	g.neighbors = make(map[int]map[int]int)

	byteValue, err := ioutil.ReadFile(filename)
	if err != nil {
		panic(err)
	}

	var rawresult [2]json.RawMessage
	if err := json.Unmarshal(byteValue, &rawresult); err != nil {
		panic(err)
	}
	if err := json.Unmarshal(rawresult[0], &g.Nodes); err != nil {
		panic(err)
	}
	if err := json.Unmarshal(rawresult[1], &g.Edges); err != nil {
		panic(err)
	}

	// fmt.Println("loading nodes", len(nodes))
	for ind, loc := range g.Nodes {
		key := loc2key(loc)
		if _, ok := g.loc2index[key]; ok {
			// fmt.Println("Warning: duplicated location", sk)
		} else {
			g.loc2index[key] = ind
		}
	}

	return g
}

func (g *graph) addEdge(loc1 [2]float64, loc2 [2]float64) {
	sk1 := loc2key(loc1)
	sk2 := loc2key(loc2)

	var nid1, nid2 int

	if _, ok := g.loc2index[sk1]; ok {
		nid1 = g.loc2index[sk1]
		//fmt.Println("never hit?")
	} else {
		nid1 = len(g.Nodes)
		g.Nodes = append(g.Nodes, loc1)
		g.loc2index[sk1] = nid1
	}

	if _, ok := g.loc2index[sk2]; ok {
		nid2 = g.loc2index[sk2]
	} else {
		nid2 = len(g.Nodes)
		g.Nodes = append(g.Nodes, loc2)
		g.loc2index[sk2] = nid2
	}

	g.Edges = append(g.Edges, [2]int{nid1, nid2})
	weight := gpsDistanceInt(loc1, loc2)

	if _, ok := g.neighbors[nid1]; ok {
		g.neighbors[nid1][nid2] = weight
	} else {
		g.neighbors[nid1] = make(map[int]int)
		g.neighbors[nid1][nid2] = weight
	}

	if _, ok := g.neighbors[nid2]; ok {
		g.neighbors[nid2][nid1] = weight
	} else {
		g.neighbors[nid2] = make(map[int]int)
		g.neighbors[nid2][nid1] = weight
	}
}

func GraphDensify(g *graph) *graph {
	ng := new(graph)
	ng.loc2index = make(map[locKey]int)
	ng.neighbors = make(map[int]map[int]int)

	for _, edge := range g.Edges {
		n1 := edge[0]
		n2 := edge[1]

		d := GPSDistance(g.Nodes[n1], g.Nodes[n2])

		if d > 3.0 {
			n := int(d/2.0) + 1
			for i := 0; i < n; i++ {
				alpha1 := float64(i) / float64(n)
				alpha2 := float64(i+1) / float64(n)

				if i == 0 {
					loc1 := g.Nodes[n1]
					loc2 := [2]float64{g.Nodes[n1][0]*(1-alpha2) + g.Nodes[n2][0]*alpha2, g.Nodes[n1][1]*(1-alpha2) + g.Nodes[n2][1]*alpha2}

					ng.addEdge(loc1, loc2)
				} else if i == n-1 {
					loc1 := [2]float64{g.Nodes[n1][0]*(1-alpha1) + g.Nodes[n2][0]*alpha1, g.Nodes[n1][1]*(1-alpha1) + g.Nodes[n2][1]*alpha1}
					loc2 := g.Nodes[n2]

					ng.addEdge(loc1, loc2)
				} else {
					loc1 := [2]float64{g.Nodes[n1][0]*(1-alpha1) + g.Nodes[n2][0]*alpha1, g.Nodes[n1][1]*(1-alpha1) + g.Nodes[n2][1]*alpha1}
					loc2 := [2]float64{g.Nodes[n1][0]*(1-alpha2) + g.Nodes[n2][0]*alpha2, g.Nodes[n1][1]*(1-alpha2) + g.Nodes[n2][1]*alpha2}

					ng.addEdge(loc1, loc2)
				}
			}
		} else {
			loc1 := g.Nodes[n1]
			loc2 := g.Nodes[n2]
			ng.addEdge(loc1, loc2)
		}
	}

	// fmt.Println("densify graph ", len(g.Nodes), len(ng.Nodes))
	return ng
}

func lockey(loc [2]float64, dist float64) gridKey {
	return gridKey{
		X: int(loc[0] * 111111.0 / dist),
		Y: int(loc[1] * 111111.0 / dist),
	}
}

func apls_one_way(graph_gt *graph, graph_prop *graph, ret chan float64) {

	// sample control point on graph
	visited := make([]bool, len(graph_gt.Nodes))
	lockeys := make(map[gridKey]bool)

	control_point_gt := make(map[int]int)

	node_cover_map_gt := make([]bool, len(graph_gt.Nodes))

	for nid, _ := range graph_gt.Nodes {
		if len(graph_gt.neighbors[nid]) != 2 {
			//fmt.Println("len(graph_gt.neighbors[nid])", nid, graph_gt.neighbors[nid])
			for next_nid, _ := range graph_gt.neighbors[nid] {
				if visited[next_nid] {
					continue
				}
				var chain []int

				chain = append(chain, nid)
				chain = append(chain, next_nid)

				last_nid := nid
				current_nid := next_nid

				//fmt.Println("inloop")
				for len(graph_gt.neighbors[current_nid]) == 2 {
					var s int = 0
					for k, _ := range graph_gt.neighbors[current_nid] {
						s = s + k
					}

					current_nid, last_nid = s-last_nid, current_nid

					chain = append(chain, current_nid)
				}
				//fmt.Println("outloop")

				// city wide parameter: 37 50 meters 25.0
				// spacenet : 15 20 meters 10.0

				if len(chain) > interval_1 { // 50 meters
					n := int(float64(len(chain))/interval_2) + 1

					for i := 1; i < n; i++ {
						idx := int(float64(len(chain)) * float64(i) / float64(n))

						if GPSInBound(graph_gt.Nodes[chain[idx]]) && node_cover_map_gt[chain[idx]] == false {

							lk := lockey(graph_gt.Nodes[chain[idx]], 2.0)

							if _, ok := lockeys[lk]; !ok {
								lockeys[lk] = true
								control_point_gt[chain[idx]] = -1

								graph_gt.propagate(chain[idx], 4, func(nid int) {
									node_cover_map_gt[nid] = true
								})

							}
						}
					}
				}

				for _, cnid := range chain {
					visited[cnid] = true
				}
			}

			if GPSInBound(graph_gt.Nodes[nid]) && (node_cover_map_gt[nid] == false || len(graph_gt.neighbors[nid]) == 1) {
				lk := lockey(graph_gt.Nodes[nid], 2.0)
				if _, ok := lockeys[lk]; !ok {
					lockeys[lk] = true
					control_point_gt[nid] = -1

					graph_gt.propagate(nid, 4, func(nid int) {
						node_cover_map_gt[nid] = true
					})
				}
			}

			//control_point_gt[nid] = -1
		} else {

		}
	}

	// fmt.Println("ground truth map control points: ", len(control_point_gt))

	// snap to proposal map
	// - create index
	rt := rtreego.NewTree(2, 25, 50)

	node_cover_map := make([]bool, len(graph_prop.Nodes))

	for nid, loc := range graph_prop.Nodes {
		var gNode gpsnode
		gNode.nid = nid
		gNode.loc = loc
		rt.Insert(&gNode)
	}

	var matched_point int = 0

	// change this to one-to-one matching
	// propagate for 4 steps

	for nid1, _ := range control_point_gt {
		q := rtreego.Point{graph_gt.Nodes[nid1][0], graph_gt.Nodes[nid1][1]}

		results := rt.NearestNeighbors(10, q)

		for _, result := range results {

			if node_cover_map[result.(*gpsnode).nid] == true {
				continue
			}

			if GPSDistance(result.(*gpsnode).loc, graph_gt.Nodes[nid1]) < 10.0 {
				control_point_gt[nid1] = result.(*gpsnode).nid
				matched_point += 1

				graph_prop.propagate(result.(*gpsnode).nid, prop_step, func(nid int) {
					node_cover_map[nid] = true
				})

				break
			}
		}

	}

	// fmt.Println("snapped to proposal graph, matched nodes:", matched_point)

	// fmt.Println("Finding shortest paths")

	var cc float64 = 0.0
	var sum float64 = 0.0
	var pair_num int = 0

	var control_point_prop_list []int
	control_point_prop_map := make(map[int]bool)
	var control_point_gt_list []int

	for cp1_gt, cp1_prop := range control_point_gt {

		if cp1_prop < 0 {
			continue
		}
		control_point_gt_list = append(control_point_gt_list, cp1_gt)
		if _, ok := control_point_prop_map[cp1_prop]; ok {

		} else {
			control_point_prop_map[cp1_prop] = true
			control_point_prop_list = append(control_point_prop_list, cp1_prop)
		}
	}

	shortest_paths_gt := make(map[int]map[int]float64)
	shortest_paths_prop := make(map[int]map[int]float64)

	var counter = 0
	//var debugind = 0
	//var debug map[int]int

	for _, cp_prop := range control_point_prop_list {
		shortest_paths_prop[cp_prop], _ = graph_prop.ShortestPaths(cp_prop, control_point_prop_list)
		counter += 1

		if counter%100 == 0 {
			// fmt.Println("computing prop graph shortest paths ", counter, len(control_point_prop_list))

			// if counter == 100 {
			// 	fmt.Println("dump debug paths")
			// 	for _, cp_prop2 := range control_point_prop_list {
			// 		var trace [][2]float64
			// 		current := cp_prop2

			// 		for {
			// 			if debug[current] < 0 {
			// 				break
			// 			}

			// 			if debug[current] == current {
			// 				break
			// 			}

			// 			trace = append(trace,graph_prop.Nodes[current])
			// 			current = debug[current]
			// 		}

			// 		if len(trace)>0 {
			// 			dat, _ := json.MarshalIndent(trace, "  ", "  ")
			// 			_=ioutil.WriteFile(fmt.Sprintf("debug/trace%d.json", debugind), dat, 0644)
			// 			debugind += 1
			// 		}
			// 	}
			// }

		}
	}

	counter = 0

	for _, cp_gt := range control_point_gt_list {
		shortest_paths_gt[cp_gt], _ = graph_gt.ShortestPaths(cp_gt, control_point_gt_list)
		counter += 1

		if counter%100 == 0 {
			// fmt.Println("computing gt graph shortest paths ", counter, len(control_point_gt_list))
		}
	}

	for cp1_gt, cp1_prop := range control_point_gt {
		for cp2_gt, cp2_prop := range control_point_gt {
			if cp2_gt <= cp1_gt {
				continue
			}

			pair_num += 1

			d1 := shortest_paths_gt[cp1_gt][cp2_gt]

			//if (cp1_prop == -1 || cp2_prop == -1) && (d1 > 0.0) {
			if cp1_prop == -1 || cp2_prop == -1 {
				cc += 1.0
				sum += 1.0
				continue
			}

			//d1 := shortest_paths_gt[cp1_gt][cp2_gt]
			if d1 > min_distance_filter {
				d2 := shortest_paths_prop[cp1_prop][cp2_prop]

				if d2 < 0 {
					d2 = 0
				}

				s := math.Abs(d1-d2) / d1
				if s > 1.0 {
					s = 1.0
				}

				cc += 1.0
				sum += s
			}

			// if d1 < 0.0 {
			// 	d2 := shortest_paths_prop[cp1_prop][cp2_prop]
			// 	if d2 > 0 {
			// 		cc += 1.0
			// 		sum += 1.0
			// 	}
			// }

			// if int(cc) % 1000 == 0 {
			// 	fmt.Println(int(cc), "current apls:", 1.0 - sum/cc, "progress:", float64(pair_num) / float64(len(control_point_gt) * len(control_point_gt)/2) * 100.0)
			// }

		}
	}

	ret <- 1.0 - sum/cc
	//return
}

type NodeItem struct {
	nid      int
	distance int
}

type PriorityQueue []*NodeItem

func (pq PriorityQueue) Len() int { return len(pq) }

func (pq PriorityQueue) Less(i, j int) bool {

	return pq[i].distance < pq[j].distance
}

func (pq PriorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
}

func (pq *PriorityQueue) Push(x interface{}) {
	*pq = append(*pq, x.(*NodeItem))
}

func (pq *PriorityQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	item := old[n-1]
	*pq = old[0 : n-1]
	return item
}

func (g *graph) ShortestPath(nid1 int, nid2 int) float64 {
	result, _ := g.ShortestPaths(nid1, []int{nid2})
	return result[nid2]
}

func (g *graph) ShortestPaths(nid1 int, nid2 []int) (map[int]float64, map[int]int) {

	result := make(map[int]float64, len(nid2))
	previous := make(map[int]int, len(nid2)+1)
	targets := make(map[int]struct{}, len(nid2))

	for _, v := range nid2 {
		if _, ok := targets[v]; ok {
			continue
		}
		targets[v] = struct{}{}
		result[v] = -1.0
		previous[v] = -1
	}

	if nid1 < 0 || nid1 >= len(g.Nodes) {
		return result, previous
	}
	previous[nid1] = nid1

	maxDistance := int(^uint(0) >> 1)
	mindistance := make([]int, len(g.Nodes))
	previousAll := make([]int, len(g.Nodes))
	visited := make([]bool, len(g.Nodes))

	for nid := range g.Nodes {
		mindistance[nid] = maxDistance
		previousAll[nid] = -1
	}

	mindistance[nid1] = 0
	previousAll[nid1] = nid1
	pq := make(PriorityQueue, 0, 64)

	heap.Init(&pq)
	heap.Push(&pq, &NodeItem{nid: nid1, distance: 0})

	remainingTargets := len(targets)

	for pq.Len() > 0 && remainingTargets > 0 {
		cur_node_item := heap.Pop(&pq).(*NodeItem)
		if visited[cur_node_item.nid] {
			continue
		}
		if cur_node_item.distance != mindistance[cur_node_item.nid] {
			continue
		}
		visited[cur_node_item.nid] = true

		if _, ok := targets[cur_node_item.nid]; ok {
			result[cur_node_item.nid] = float64(cur_node_item.distance) / 100.0
			previous[cur_node_item.nid] = previousAll[cur_node_item.nid]
			remainingTargets -= 1
		}

		for next_nid, weight := range g.neighbors[cur_node_item.nid] {
			if visited[next_nid] {
				continue
			}

			d := cur_node_item.distance + weight
			if d < mindistance[next_nid] {
				mindistance[next_nid] = d
				previousAll[next_nid] = cur_node_item.nid
				heap.Push(&pq, &NodeItem{nid: next_nid, distance: d})
			}
		}
	}

	return result, previous
}

func apls(graph_gt *graph, graph_prop *graph) {
	c1 := make(chan float64, 1)
	c2 := make(chan float64, 1)

	go apls_one_way(graph_gt, graph_prop, c1)
	go apls_one_way(graph_prop, graph_gt, c2)

	apls_gt := <-c1
	apls_prop := <-c2
	// fmt.Println(apls_gt, apls_prop, "apls:",(apls_gt+apls_prop)/2.0)

	d1 := []byte(fmt.Sprintf("%f %f %f\n", apls_gt, apls_prop, (apls_gt+apls_prop)/2.0))
	_ = ioutil.WriteFile(os.Args[3], d1, 0644)

}

func main() {
	graph_gt := LoadGraphFromJson(os.Args[1])
	graph_prop := LoadGraphFromJson(os.Args[2])
	// os.Args[3] is the output file
	if len(os.Args) > 4 {
		// See the header of this file for a detailed description of these parameters
		// fmt.Println("Use parameters for small tiles (region size=352)")
		interval_2 = 15.0 // 30 meters (5 * 2)
		interval_1 = int(interval_2 * 1.5)
		min_distance_filter = 30.0 // 30 meters
		prop_step = 3
		margin_size = 30.0
		region_size = 352.0
	}

	graph_gt_dense := GraphDensify(graph_gt)
	graph_prop_dense := GraphDensify(graph_prop)

	apls(graph_gt_dense, graph_prop_dense)

}
