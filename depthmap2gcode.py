#!/usr/bin/env python

import sys
import argparse
from decimal import Decimal
from PIL import Image, ImageOps

# TODO: Run final simulation pass and allow for variable movement rate trying to create
#       constant material volume / second.

NEIGHBOURS = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
NEIGHBOURS_AND_SELF = NEIGHBOURS + [(0, 0)]
NEIGHBOURS2 = [(x, y) for x in range(-2, 3) for y in range(-2, 3) if (x, y) not in NEIGHBOURS_AND_SELF]

class PythonImage(object):
    def __init__(self, img):
        self.img = img
        self.size = img.size
        self.width = self.size[0]
        self.height = self.size[1]
        self.data = list(img.getdata())

    def getpixel(self, p):
        return self.data[p[0] + self.width * p[1]]

    def putpixel(self, p, v):
        self.data[p[0] + self.width * p[1]] = v

    def copy(self):
        self.img.putdata(self.data)
        return PythonImage(self.img.copy())

    def clone(self):
        clone = PythonImage(self.img.copy())
        clone.data = list(self.data)
        return clone

    def show(self):
        self.img.putdata(self.data)
        return self.img.show()


class DistanceImage(PythonImage):
    def show(self, heights):
        img = Image.new('RGB', self.size)
        for p in [(x, y) for x in range(0, self.img.size[0]) for y in range(0, self.img.size[1])]:
            dist = self.getpixel(p)
            marked = [h for h in heights if dist > h and dist < h + 2]
            col = (
                int(dist / 20),
                255 if dist > 14 and dist < 16 else 0,
                255 if marked else 0,
            )
            img.putpixel(p, col)
        img.show()


class BooleanImage(PythonImage):
    def __init__(self, img):
        self.size = img.size
        self.width = self.size[0]
        self.height = self.size[1]
        self.data = list([False for x in range(0, self.width) for y in range(0, self.height)])


def formatFloat(args, f):
    d = (Decimal(f) / Decimal(args.str_precision)).quantize(1) * Decimal(args.str_precision)
    return d.quantize(Decimal(1)) if d == d.to_integral() else d.normalize()


def filterTrace(trace, distance):
    i = 1
    while i < len(trace) - 1:
        dx = trace[i - 1]['x'] - trace[i + 1]['x']
        dy = trace[i - 1]['y'] - trace[i + 1]['y']
        if dx * dx <= 1 and dy * dy <= 1:
            ds = distance.getpixel((trace[i - 1]['x'], trace[i - 1]['y']))
            dm = distance.getpixel((trace[i    ]['x'], trace[i    ]['y']))
            de = distance.getpixel((trace[i + 1]['x'], trace[i + 1]['y']))
            if ds < dm and de < dm:
                trace = trace[:i] + trace[i + 1:]
                continue
        i = i + 1

    return trace


def optimizeTrace(trace):
    while trace and not trace[0]['useful']:
        trace = trace[1:]
    while trace and not trace[-1]['useful']:
        trace = trace[:-1]

    while True:
        i = 0
        while i < len(trace) and trace[i]['useful']:
            i = i + 1
        if i >= len(trace):
            break

        visited = {}
        shortened = False
        while i < len(trace) and not trace[i]['useful']:
            position = (trace[i]['x'], trace[i]['y'])
            if position in visited:
                start = visited[(trace[i]['x'], trace[i]['y'])]
                trace = trace[0:start] + trace[i:]
                shortened = True
                break
            visited[position] = i
            i = i + 1

        if not shortened:
            break

    return trace


def emitTrace(args, z, trace, out):
    if not trace:
        return

    print("G90", file=out)
    print("G0 Z%s" % formatFloat(args, args.zspace), file=out)
    for i, step in enumerate(trace):
        if i == 0:
            print("G0 X%s Y%s" % (
                formatFloat(args, step['x'] * args.precision),
                formatFloat(args, step['y'] * args.precision),
            ), file=out)
            print("G1 Z%s" % formatFloat(args, -z), file=out)
        else:
            print("G1 X%s Y%s" % (
                formatFloat(args, step['x'] * args.precision),
                formatFloat(args, step['y'] * args.precision),
            ), file=out)


def toolPixels(args, diameter):
    pixel_diameter = int(diameter / args.precision)
    edge = pixel_diameter * pixel_diameter / 4

    pixel_diameter += 2
    center = int(pixel_diameter / 2)

    hit = []
    for p in [(x, y) for x in range(0, pixel_diameter) for y in range(0, pixel_diameter)]:
        dx = p[0] - center
        dy = p[1] - center
        if dx * dx + dy * dy < edge:
            hit.append((p[0] - center, p[1] - center))

    return hit


def toolEdge(shape):
    tool = set(shape)
    edge = set()

    for p in shape:
        for n in NEIGHBOURS:
            e = (p[0] + n[0], p[1] + n[1])
            if e not in tool:
                edge.add(e)

    return list(edge)


def applyTool(state, distance, z, shape, pos, distance_map):
    useful = False

    state_data = state.data
    state_width = state.width
    state_height = state.height

    px = pos[0]
    py = pos[1]
    for t in shape:
        x = px + t[0]
        y = py + t[1]
        if(x < 0 or x >= state_width or
                y < 0 or y >= state_height):
            continue

        idx = x + state_width * y
        state_old = state_data[idx]
        if z > state_old:
            state_data[idx] = z
            useful = True

    distance_data = distance.data
    distance_width = distance.width
    distance_map_data = distance_map.data

    surface = distance_map_data[pos[0] + distance_width * pos[1]][1]
    sdir_x = surface[0] - px
    sdir_y = surface[1] - py
    sdir_len = (sdir_x * sdir_x + sdir_y * sdir_y) ** 0.5

    cutoff_distance = distance.getpixel(pos) + 2

    queue = set()
    queue.add(pos)
    while queue:
        p = queue.pop()
        distance_data[p[0] + distance_width * p[1]] = 0

        for n in NEIGHBOURS:
            pn = (p[0] + n[0], p[1] + n[1])
            idx = pn[0] + distance_width * pn[1]
            d = distance_data[idx]
            if d == 0 or d > cutoff_distance:
                continue
            if distance_map_data[idx][1] != surface:
                continue

            pdir_x = pn[0] - px
            pdir_y = pn[1] - py
            pdir_len = (pdir_x * pdir_x + pdir_y * pdir_y) ** 0.5
            if pdir_x * sdir_x + pdir_y * sdir_y >= -0.7 * pdir_len * sdir_len:
                continue

            queue.add(pn)

    return useful


def buildDistanceMap(distance, all_coords):
    # So performance, much wow...
    distance_data = distance.data
    distance_width = distance.width
    distance_height = distance.height

    checked = set()
    next_stratum = 0
    max_stratum = 0
    strata = {}
    for p in all_coords:
        dist = distance_data[p[0] + distance_width * p[1]][0]
        if dist >= 0 and dist < 99999999:
            disti = int(dist)
            if disti not in strata:
                strata[disti] = []
            strata[disti].append(p)
            max_stratum = max(max_stratum, disti)

    running = True
    while True:
        while not strata.get(next_stratum):
            next_stratum = next_stratum + 1
            print("\x1B[1G...", next_stratum, "  \x1B[1F")
            if next_stratum > max_stratum:
                running = False
                break
        if not running:
            break

        p = strata[next_stratum].pop()
        if p in checked:
            continue

        checked.add(p)
        nearest = distance_data[p[0] + distance_width * p[1]][1]
        if not nearest:
            continue

        for d in NEIGHBOURS:
            x = p[0] + d[0]
            y = p[1] + d[1]
            if(x < 0 or x >= distance_width or
                y < 0 or y >= distance_height):
                continue

            dx = (nearest[0] - x)
            dy = (nearest[1] - y)
            distP = dx * dx + dy * dy
            idx = x + distance_width * y

            if distP < distance_data[idx][0]:
                distance_data[idx] = (distP, nearest)
                np = (x, y)
                if np in checked:
                    checked.remove(np)
                distPi = int(distP)
                if distPi not in strata:
                    strata[distPi] = []
                    max_stratum = max(max_stratum, distPi)
                strata[distPi].append(np)
                if distPi < next_stratum:
                    next_stratum = distPi


def initDistanceMap(target, state, distance, image_cutoff, all_coords):
    # So performance, much wow...
    distance_data = distance.data
    distance_width = distance.width

    for p in all_coords:
        t = target.getpixel(p)

        if(t >= image_cutoff or
            p[0] <= 0 or p[0] >= state.size[0] - 1 or
            p[1] <= 0 or p[1] >= state.size[1] - 1):
            distance_data[p[0] + distance_width * p[1]] = (0, p)
        else:
            distance_data[p[0] + distance_width * p[1]] = (999999999, None)


def sortTraces(args, traces):
    traces = list(filter(bool, traces))
    result = []

    pos = (0, 0)
    while traces:
        minimum = 99999999999
        best = None
        reverse = False
        for trace in traces:
            dx = trace[0]['x'] - pos[0]
            dy = trace[0]['y'] - pos[1]
            dist = dx * dx + dy * dy
            if dist < minimum:
                best = trace
                reverse = False
                minimum = dist

            dx = trace[-1]['x'] - pos[0]
            dy = trace[-1]['y'] - pos[1]
            dist = dx * dx + dy * dy
            if dist < minimum:
                best = trace
                reverse = True
                minimum = dist

        if reverse:
            result.append(list(reversed(best)))
            pos = (best[0]['x'], best[0]['y'])
        else:
            result.append(best)
            pos = (best[-1]['x'], best[-1]['y'])
        traces.remove(best)

    return result


def findFreeConnection(args, z, start, end, may_cut_map):
    may_cut_data = may_cut_map.data
    may_cut_width = may_cut_map.width

    checked = set()
    strata = {}
    next_stratum = 0
    max_stratum = 0

    steps = {}

    s = (start['x'], start['y'])
    e = (end['x'], end['y'])

    planar_dist = ((s[0] - e[0]) ** 2 + (s[1] - e[1]) ** 2) ** 0.5
    if planar_dist > 5 / args.precision:
        return None

    dist_cutoff = (
        2 * args.zspace + 2 * z +
        args.precision * planar_dist
    ) / args.precision

    # 5mm maximum reconnection length
    dist_cutoff = min(dist_cutoff, 5 / args.precision)

    steps[(s[0], s[1])] = (0, None)
    dist = int(((s[0] - e[0]) ** 2 + (s[1] - e[1]) ** 2) ** 0.5)
    strata[0] = [(s[0], s[1])]

    running = True
    found = None
    while True:
        while not strata.get(next_stratum):
            next_stratum = next_stratum + 1
            if next_stratum > max_stratum:
                running = False
                break
        if not running:
            break

        p = strata[next_stratum].pop()
        if p in checked:
            continue
        checked.add(p)

        if p == e:
            found = True
            break

        if steps[p][0] > dist_cutoff:
            continue

        for d in NEIGHBOURS:
            n = (p[0] + d[0], p[1] + d[1])
            if not may_cut_data[n[0] + may_cut_width * n[1]]:
                continue

            s_dist = steps[p][0] + (1.414213562373 if d[0] and d[1] else 1)
            if n not in steps or s_dist < steps[n][0]:
                steps[n] = (s_dist, p)
                if n in checked:
                    checked.remove(n)

                dx = n[0] - e[0]
                dy = n[1] - e[1]
                e_dist = dx * dx + dy * dy
                if e_dist not in strata:
                    strata[e_dist] = []
                    max_stratum = max(max_stratum, e_dist)
                strata[e_dist].append(n)
                if e_dist < next_stratum:
                    next_stratum = e_dist

    if found:
        i = e
        connection = []
        while i != s:
            connection.append(i)
            i = steps[i][1]
        connection.append(s)
        return list(map(lambda xy: {
            'x': xy[0],
            'y': xy[1],
            'useful': False,
        }, reversed(connection)))

    return None


def connectTraces(args, z, traces, may_cut_map, connection_cache):
    result = []
    if not traces:
        return result

    last = traces[0]
    traces = traces[1:]
    progress = 0
    for trace in traces:
        progress = progress + 1
        print("\x1B[1G...", progress, "  \x1B[1F")

        connection_identifier = (last[-1]['x'], last[-1]['y'], trace[0]['x'], trace[0]['y'])
        if connection_identifier not in connection_cache:
            connection = findFreeConnection(args, z, last[-1], trace[0], may_cut_map)
            connection_cache[connection_identifier] = connection
        else:
            connection = connection_cache[connection_identifier]

        if connection:
            last = last + connection[1:-2] + trace
        else:
            result.append(last)
            last = trace

    result.append(last)
    return result
    

def linearizeTrace(args, trace):
    if len(trace) <= 2:
        return trace

    result = [trace[0]]

    i = 1
    while i < len(trace) - 1:
        print("\x1B[1G...", i, "/", len(trace), "  \x1B[1F")

        dx = trace[i]['x'] - result[-1]['y']
        dy = trace[i]['y'] - result[-1]['y']
        len_ = (dx * dx + dy * dy) ** 0.5

        if not len_:
            i += 1
            continue

        dx /= len_
        dy /= len_

        while i < len(trace) - 1:
            n = i + 1
            ndx = trace[n]['x'] - result[-1]['x']
            ndy = trace[n]['y'] - result[-1]['y']
            nlen = (ndx * ndx + ndy * ndy) ** 0.5
            if not nlen:
                i += 1
                continue

            ndx /= nlen
            ndy /= nlen

            if abs(dx - ndx) < 1e-6 and abs(dy - ndy) < 1e-6:
                i = n
            else:
                break

        result.append(trace[i])
        i += 1

    if i < len(trace):
        result.append(trace[i])
    return trace


def buildMayCutMap(distance, distance_to_cut, all_coords):
    may_cut = BooleanImage(distance)
    may_cut.data = list(map(lambda v: distance_to_cut <= v, distance.data))
    return may_cut


def generateSweep(target, state, args, diameter, diameter_before, padding, out, image_cutoff, z, all_coords, all_idx, tool_shape):
    distance = DistanceImage(Image.new('I', target.size))
    # So performance, much wow...
    distance_data = distance.data
    distance_width = distance.width

    initDistanceMap(target, state, distance, image_cutoff, all_coords)
    buildDistanceMap(distance, all_coords)

    distance_map = distance.clone()

    for q in all_idx:
        distance_data[q] = distance_data[q][0] ** 0.5

    distance_to_cut = (diameter / 2 + padding) / args.precision
    distance_stop = None
    if diameter_before:
        distance_stop = diameter_before / args.precision
    may_cut_map = buildMayCutMap(distance, distance_to_cut, all_coords)
    original_distance = distance.clone()

    any_at_distance = False

    distance_strata = list(map(lambda i: [], range(0, distance_width + distance.height + 3)))
    for q in all_idx:
        d = int(distance_data[q])
        distance_strata[d].append(q)

    plane_traces = []

    pos = (0, 0)
    while True:
        minimum = 99999999999
        start = None

        for i in [0, 1, 2]:
            for q in distance_strata[int(distance_to_cut) + i]:
                pdist = distance_data[q]
                if pdist >= distance_to_cut and pdist < distance_to_cut + 2:
                    start_x = q % distance_width
                    start_y = q // distance_width
                    dist = start_x * start_x + start_y + start_y
                    if dist < minimum:
                        minimum = dist
                        start = (start_x, start_y)
            if start:
                break
        
        if not start:
            if not any_at_distance:
                break
            else:
                any_at_distance = False
                distance_to_cut += (diameter / 2 - args.overlap) / args.precision
                if distance_stop and distance_stop < distance_to_cut:
                    break
                continue

        any_at_distance = True

        pos = start
        useful = applyTool(state, distance, z, tool_shape, pos, distance_map)
        trace_steps = []
        trace_steps.append({
            'x': start[0],
            'y': start[1],
            'useful': useful,
        })
        while True:
            step = None
            minimum = 999999999
            for offset in NEIGHBOURS:
                p = (pos[0] + offset[0], pos[1] + offset[1])
                pdist = distance_data[p[0] + distance_width * p[1]]
                if pdist >= distance_to_cut and pdist < distance_to_cut + 2 and pdist < minimum:
                    step = p
                    minimum = pdist
                    break

            if not step:
                break

            pos = step
            useful = applyTool(state, distance, z, tool_shape, pos, distance_map)
            trace_steps.append({
                'x': pos[0],
                'y': pos[1],
                'useful': useful,
            })

        trace_steps = filterTrace(trace_steps, original_distance)
        trace_steps = optimizeTrace(trace_steps)
        plane_traces.append(trace_steps)

    print("Traces considered: ", len(plane_traces))
    plane_traces = sortTraces(args, plane_traces)
    last_relevant = -1
    connection_cache = {}
    while last_relevant != len(plane_traces):
        print("Traces relevant: ", len(plane_traces))
        last_relevant = len(plane_traces)
        plane_traces = connectTraces(args, z, plane_traces, may_cut_map, connection_cache)
        plane_traces = sortTraces(args, plane_traces)

    print("Traces to emit: ", len(plane_traces))
    for trace in plane_traces:
        trace = linearizeTrace(args, trace)
        emitTrace(args, z=z, trace=trace, out=out)


def generateCommands(target, state, padding, args, diameter, diameter_before, out):
    planes = list(range(0, args.planes))
    cut_early = []
    cut_late = []

    last_cut = 0
    next_cut = 0
    for plane in planes:
        z = (plane + 1) * (args.depth / args.planes)
        if z - last_cut < args.cutdepth:
            pass
        elif cut_late:
            cut_early.append(cut_late.pop())
            last_cut = next_cut
        else:
            print("Not enough cut planes to satisfy --cutdepth constraint", file=sys.stderr)
            sys.exit(1)

        cut_late.append(plane)
        next_cut = z

    tool_shape = sorted(toolPixels(args, diameter), key=lambda p: p[0] * p[0] + p[1] * p[1])
    all_coords = [(x, y) for x in range(0, state.size[0]) for y in range(0, state.size[1])]
    state_width = state.size[0]
    all_idx = list(map(lambda c: c[0] + state_width * c[1], all_coords))
    for plane in cut_early + list(reversed(cut_late)):
        image_cutoff = 255.0 - (plane + 1) * (255.0 / (args.planes + 1)) + padding * 255.0 / args.depth
        z = (plane + 1) * (args.depth / args.planes)
        print("plane %d: img %03.3f z %03.3f" % (plane, image_cutoff, z))

        generateSweep(target=target, state=state, args=args, diameter=diameter,
                diameter_before=diameter_before, padding=padding,
                out=out, image_cutoff=image_cutoff, z=z, all_coords=all_coords, all_idx=all_idx,
                tool_shape=tool_shape)

    print("G0 Z%s" % formatFloat(args, args.zspace), file=out)


def main():
    parser = argparse.ArgumentParser(description="""
    Convert depthmap (black = deep, white = high) to G-code.
    """)

    parser.add_argument('input', nargs='?', help='Input image file')
    required = parser.add_argument_group('required arguments')
    required.add_argument('--depth', dest='depth', required=True, type=float,
                        help='Total depth difference between black + white, in mm')
    required.add_argument('--width', dest='width', required=True, type=float,
                        help='Total X difference between left and right image border, in mm')
    required.add_argument('--height', dest='height', required=True, type=float,
                        help='Total Y difference between top and bottom image border, in mm')
    parser.add_argument('--planes', dest='planes', default='256', type=int, help="""
    Number of planes to sweep during cut.
    """)
    parser.add_argument('--zspace', dest='zspace', default='10', type=float, help="""
    Z distance to hover above origin when moving to disconnected region, in mm
    """)
    parser.add_argument('--precision', dest='str_precision', default='0.1', type=str, help="""
    Pixel quantization size, this is the smallest surface size you don't care about, in mm.
    Smaller values will increase runtime.
    """)
    parser.add_argument('--overlap', dest='overlap', default='0.1', type=float, help="""
    How much parallel sweeps should overlap to make sure material is cleared, in mm.
    """)
    parser.add_argument('--cutdepth', dest='cutdepth', default='5', type=float, help="""
    Maximum depth to cut in one pass, in mm.
    """)
    parser.add_argument('--result', dest='result', type=str, help="""
    Image file to store result state in.
    """)
    parser.add_argument('--inverse', dest='inverse', action='store_true', help="""
    Invert image before cutting, i.e. now assume white as deep into material.
    """)
    required.add_argument('--tool', metavar='<diameter>:[padding:]outputfile', action='append',
    required=True, help="""
    Specify a G-code output file for a tool of given diameter in mm. Can be specified
    multiple times to generate a set of files for multi-tool cuts. If a padding is specified,
    the target geometry is padded (in all 3 dimensions) by this amount, in mm, for this tool.
    """)

    args = parser.parse_args()
    if not args.input:
        parser.print_help()
        print("\nNo input file given.", file=sys.stderr)
        sys.exit(1)

    args.precision = float(args.str_precision)

    input = Image.open(args.input).convert('L', dither=None).transpose(Image.FLIP_LEFT_RIGHT)
    if args.inverse:
        input = ImageOps.invert(input)
    target = input.resize((int(args.width / args.precision), int(args.height / args.precision)),
            resample=Image.LANCZOS)

    target = PythonImage(target)
    state = PythonImage(Image.new('L', target.size, 255))
    state.data = list(map(lambda v: 0.0, state.data))
    diameter_before = None

    for i, tool in enumerate(args.tool):
        parts = tool.split(':')
        if len(parts) == 2:
            (diameter, outfile) = tool.split(':')
            if i == len(args.tool) - 1:
                padding = 0;
            else:
                padding = 0.3;
        elif len(parts) == 3:
            (diameter, padding, outfile) = tool.split(':')
            padding = float(padding)
        else:
            parser.print_help()
            print("\nCould not parse output file from --tool %s" % tool, file=sys.stderr)
            sys.exit(1)
        with open(outfile, 'w') as out:
            generateCommands(target=target, state=state, padding=padding,
                    args=args, diameter=float(diameter), diameter_before=diameter_before, out=out)
            diameter_before = float(diameter)

    depth_map = {
        0: 255,
    }
    for plane in range(0, args.planes):
        z = (plane + 1) * (args.depth / args.planes)
        depth_map[z] = int(255 - (z / args.depth * 255))

    result = PythonImage(Image.new('RGB', state.size, (255, 255, 255)))
    for p in [(x, y) for x in range(0, state.size[0]) for y in range(0, state.size[1])]:
        depth = state.getpixel(p)
        v = depth_map[depth]
        result.putpixel(p, (v, v, v))

    result = result.copy().img.transpose(Image.FLIP_LEFT_RIGHT)
    result.show()

    if args.result:
        result.save(args.result)


if __name__ == '__main__':
    main()
