#!/usr/bin/env python

import sys
import argparse
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description="""
    Convert depthmap (black = deep, white = high) to G-code.
    """)

    parser.add_argument('input', nargs='?', help='Input image file')
    required = parser.add_argument_group('required arguments')
    required.add_argument('--depth', dest='depth', required=True,
                        help='Total depth difference between black + white, in mm')
    required.add_argument('--width', dest='width', required=True,
                        help='Total X difference between left and right image border, in mm')
    required.add_argument('--height', dest='height', required=True,
                        help='Total Y difference between top and bottom image border, in mm')
    parser.add_argument('--planes', dest='planes', default='256', help="""
    Number of planes to sweep during cut.
    """)
    parser.add_argument('--zspace', dest='zspace', default='10', help="""
    Z distance to hover above origin when moving to disconnected region, in mm
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

    target = Image.open(args.input).convert('L', dither=None)
    state = Image.new('L', target.size, 255)

    for tool in args.tool:
        if len(tool.split(':')) != 2:
            parser.print_help()
            print("\nCould not parse output file from --tool %s" % tool, file=sys.stderr)
            sys.exit(1)
        (diameter, outfile) = tool.split(':')
        outfile = sys.stdout # debugging for now

        generateCommands(args=args, diameter=diameter, out=outfile)


if __name__ == '__main__':
    main()
