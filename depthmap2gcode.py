#!/usr/bin/env python

import sys
import argparse
from decimal import Decimal
from PIL import Image

NEIGHBOURS = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]

def formatFloat(args, f):
    d = (Decimal(f) / Decimal(args.str_precision)).quantize(1) * Decimal(args.str_precision)
    return d.quantize(Decimal(1)) if d == d.to_integral() else d.normalize()

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


def toolFits(target, image_cutoff, tool_shape, position):
    for t in tool_shape:
        tp = (position[0] + t[0], position[1] + t[1])
        if(tp[0] < 0 or tp[0] >= target.size[0] or
            tp[1] < 0 or tp[1] >= target.size[1]):
            return False
        if target.getpixel(tp) >= image_cutoff:
            return False

    return True


def applyTool(state, distance, cut_depth, shape, inner, pos):
    depth = int(cut_depth)

    for t in shape:
        p = (pos[0] + t[0], pos[1] + t[1])
        state.putpixel(p, min(depth, state.getpixel(p)))

    for i in inner:
        p = (pos[0] + t[0], pos[1] + t[1])
        distance.putpixel(p, 0)


def generateSweep(target, state, args, diameter, out, image_cutoff, z):
    distance = Image.new('I', target.size, 999999999)
    for p in [(x, y) for x in range(0, state.size[0]) for y in range(0, state.size[1])]:
        t = target.getpixel(p)
        s = state.getpixel(p)
        if t >= image_cutoff or t > s:
            distance.putpixel(p, 0)

    to_check = set((x, y) for x in range(0, state.size[0]) for y in range(0, state.size[1]))
    while to_check:
        if len(to_check) % 1000 == 0:
            print("\x1B[1G...", len(to_check), "  \x1B[1F")
        p = to_check.pop()
        for d in NEIGHBOURS:
            pd = (p[0] + d[0], p[1] + d[1])
            if(pd[0] < 0 or pd[0] >= distance.size[0] or
                pd[1] < 0 or pd[1] >= distance.size[1]):
                continue

            if distance.getpixel(pd) > distance.getpixel(p) + 1:
                distance.putpixel(pd, distance.getpixel(p) + 1)
                to_check.add(pd)

    tool_shape = sorted(toolPixels(args, diameter), key=lambda p: p[0] * p[0] + p[1] * p[1])
    tool_inner = sorted(toolPixels(args, diameter - args.overlap), key=lambda p: p[0] * p[0] + p[1] * p[1])
    tool_edge = sorted(toolEdge(tool_shape), key=lambda p: p[0] * p[0] + p[1] * p[1])
    tool_region = sorted(tool_shape + tool_edge, key=lambda p: p[0] * p[0] + p[1] * p[1])
    # print("tool inner:", tool_inner)
    # print("tool shape:", tool_shape)
    # print("tool edge:", tool_edge)

    trace = 0
    while True:
        trace = trace + 1
        print("Emitting trace", trace)

        maximum = 0
        start = None
        for p in [(x, y) for x in range(0, state.size[0]) for y in range(0, state.size[1])]:
            if distance.getpixel(p) > maximum:
                for t in tool_inner:
                    if toolFits(target, image_cutoff, reversed(tool_shape), (p[0] - t[0], p[1] - t[1])):
                        maximum = distance.getpixel(p)
                        start = (p[0] - t[0], p[1] - t[1])
                        break
                else:
                    distance.putpixel(p, 0)
        
        if not start:
            break

        print("G90", file=out)
        print("G0 Z%s" % formatFloat(args, args.zspace), file=out)
        print("G0 X%s Y%s" % (
            formatFloat(args, start[0] * args.precision),
            formatFloat(args, start[1] * args.precision)), file=out)
        print("G1 Z%s" % formatFloat(args, -z), file=out)

        unreachable = set()

        pos = start
        applyTool(state, distance, image_cutoff - 1e-6, tool_shape, tool_inner, pos)
        while True:
            minimum = 999999999
            step = None
            for offset in tool_region:
                p = (pos[0] + offset[0], pos[1] + offset[1])
                if p in unreachable:
                    continue
                if(p[0] < 0 or p[0] >= target.size[0] or
                        p[1] < 0 or p[1] >= target.size[1]):
                    unreachable.add(p)
                    continue

                dist = distance.getpixel(p)
                if dist < minimum and dist > 0:
                    minimum = dist
                    step = p

            if not step:
                break

            minimum = (step[0] - pos[0]) ** 2 + (step[1] - pos[1]) ** 2
            next_pos = None
            for d in NEIGHBOURS:
                p = (pos[0] + d[0], pos[1] + d[1])
                dist = (step[0] - p[0]) ** 2 + (step[1] - p[1]) ** 2
                if dist < minimum and toolFits(target, image_cutoff, reversed(tool_shape), p):
                    minimum = dist
                    next_pos = p

            if not next_pos:
                unreachable.add(step)
                distance.putpixel(step, 0);
                continue

            print("G1 X%s Y%s" % (
                formatFloat(args, next_pos[0] * args.precision),
                formatFloat(args, next_pos[1] * args.precision)), file=out)
            pos = next_pos
            applyTool(state, distance, image_cutoff - 1e-6, tool_shape, tool_inner, pos)

    print("G0 Z%s" % formatFloat(args, args.zspace), file=out)

def generateCommands(target, state, args, diameter, out):
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

    for plane in cut_early + list(reversed(cut_late)):
        image_cutoff = 255.0 - (plane + 1) * (255.0 / (args.planes + 1))
        z = (plane + 1) * (args.depth / args.planes)
        print("plane %d: img %03.3f z %03.3f" % (plane, image_cutoff, z))

        generateSweep(target=target, state=state, args=args, diameter=diameter,
                out=out, image_cutoff=image_cutoff, z=z)


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
    required.add_argument('--tool', metavar='<diameter>:outputfile', action='append',
    required=True, help="""
    Specify a G-code output file for a tool of given diameter in mm. Can be specified
    multiple times to generate a set of files for multi-tool cuts.
    """)

    args = parser.parse_args()
    if not args.input:
        parser.print_help()
        print("\nNo input file given.", file=sys.stderr)
        sys.exit(1)

    args.precision = float(args.str_precision)

    input = Image.open(args.input).convert('L', dither=None)
    target = input.resize((int(args.width / args.precision), int(args.height / args.precision)),
            resample=Image.LANCZOS)
    state = Image.new('L', target.size, 255)

    for i, tool in enumerate(args.tool):
        if len(tool.split(':')) != 2:
            parser.print_help()
            print("\nCould not parse output file from --tool %s" % tool, file=sys.stderr)
            sys.exit(1)

        tool_target = target
        if i != len(args.tool) - 1:
            print("Padding target geometry for coarse tool run")
            tool_target = target.copy()
            for j in [0, 1]:
                new_target = tool_target.copy()
                for p in [(x, y) for x in range(0, state.size[0]) for y in range(0, state.size[1])]:
                    maximum = tool_target.getpixel(p) + 1
                    for n in NEIGHBOURS:
                        pn = (p[0] + n[0], p[1] + n[1])
                        if(pn[0] < 0 or pn[0] >= target.size[0] or
                            pn[1] < 0 or pn[1] >= target.size[1]):
                            maximum = 256
                        else:
                            maximum = max(maximum, tool_target.getpixel(pn))
                    new_target.putpixel(p, maximum)
                tool_target = new_target

        (diameter, outfile) = tool.split(':')
        with open(outfile, 'w') as out:
            generateCommands(target=tool_target, state=state, args=args, diameter=float(diameter), out=out)

        state.show()


if __name__ == '__main__':
    main()
