#!/usr/bin/env python

import sys
import argparse
from PIL import Image

NEIGHBOURS = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]

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
        state.putpixel(p, depth)

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
            print("...", len(to_check))
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
    print("tool inner:", tool_inner)
    print("tool shape:", tool_shape)
    print("tool edge:", tool_edge)

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
        print("G0 Z%f" % args.zspace, file=out)
        print("G0 X%f Y%f" % (start[0] * args.precision, start[1] * args.precision), file=out)
        print("G1 Z%f" % -z, file=out)

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

            print("G1 X%f Y%f" % (next_pos[0] * args.precision, next_pos[1] * args.precision), file=out)
            pos = next_pos
            applyTool(state, distance, image_cutoff - 1e-6, tool_shape, tool_inner, pos)

    print("G0 Z%f" % args.zspace, file=out)

def generateCommands(target, state, args, diameter, out):
    planes = args.planes
    depth = args.depth

    for plane in range(0, planes):
        image_cutoff = 255.0 - (plane + 1) * (255.0 / (planes + 1))
        z = (plane + 1) * (depth / planes)
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
    parser.add_argument('--precision', dest='precision', default='0.1', type=float, help="""
    Pixel quantization size, this is the smallest surface size you don't care about, in mm.
    Smaller values will increase runtime.
    """)
    parser.add_argument('--overlap', dest='overlap', default='0.1', type=float, help="""
    How much parallel sweeps should overlap to make sure material is cleared, in mm.
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

    input = Image.open(args.input).convert('L', dither=None)
    target = input.resize((int(args.width / args.precision), int(args.height / args.precision)),
            resample=Image.LANCZOS)
    state = Image.new('L', target.size, 255)

    for tool in args.tool:
        if len(tool.split(':')) != 2:
            parser.print_help()
            print("\nCould not parse output file from --tool %s" % tool, file=sys.stderr)
            sys.exit(1)
        (diameter, outfile) = tool.split(':')
        with open(outfile, 'w') as out:
            generateCommands(target=target, state=state, args=args, diameter=float(diameter), out=out)

    state.show()


if __name__ == '__main__':
    main()
