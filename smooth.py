#!/usr/bin/env python

import sys
import argparse
from decimal import Decimal

def formatFloat(precision, f):
    d = (Decimal(f) / Decimal(precision)).quantize(1) * Decimal(precision)
    return d.quantize(Decimal(1)) if d == d.to_integral() else d.normalize()


def main():
    parser = argparse.ArgumentParser(description="""
    Creates G-code to clear large blocks of crap out of your work area.
    """)

    required = parser.add_argument_group('required arguments')
    required.add_argument('--zstart', dest='zstart', required=True, type=float,
                        help='Start z plane, in mm')
    required.add_argument('--zstop', dest='zstop', required=True, type=float,
                        help='Stop z plane, in mm')
    required.add_argument('--width', dest='width', required=True, type=float,
                        help='Area width, in mm')
    required.add_argument('--height', dest='height', required=True, type=float,
                        help='Area height, in mm')
    required.add_argument('--ystep', dest='ystep', required=True, type=float,
                        help='Raster distance in y, in mm')
    parser.add_argument('--feedrate', dest='feedrate', default=400,
                        help='Feedrate in mm/min.')
    required.add_argument('--output', required=True, dest='output', type=str, help="""
    Output file name.
    """)
    parser.add_argument('--cutdepth', dest='cutdepth', default='2', type=float, help="""
    Maximum depth to cut per pass, in mm.
    """)

    args = parser.parse_args()

    precision = Decimal(1)
    while precision > args.ystep / 20:
        precision = precision / Decimal(10)

    with open(args.output, "w") as out:
        print("G90", file=out)
        print("G0 X0 Y0 Z%s F%s" % (
            formatFloat(precision, args.zstart), formatFloat(precision, args.feedrate)),
            file=out)
        xpos = 0
        layers = []
        z = args.zstart
        while z > args.zstop:
            layers.append(z)
            z -= args.cutdepth
        if layers[-1] != args.zstop:
            layers.append(args.zstop)

        rows = []
        y = 0
        while y < args.height:
            rows.append(y)
            y += args.ystep
        if rows[-1] != args.height:
            rows.append(args.height)

        for z in layers:
            print("G1 Z%s F%s" % (
                formatFloat(precision, z), formatFloat(precision, args.feedrate)),
                file=out)
            print("G1 X%s" % formatFloat(precision, args.width), file=out)
            print("G1 Y%s" % formatFloat(precision, args.height), file=out)
            print("G1 X%s" % formatFloat(precision, 0), file=out)
            print("G1 Y%s" % formatFloat(precision, 0), file=out)

            for y in rows:
                print("G1 Y%s" % formatFloat(precision, y), file=out)
                if xpos == 0:
                    print("G1 X%s" % formatFloat(precision, args.width), file=out)
                    xpos = args.width
                else:
                    print("G1 X0", file=out)
                    xpos = 0

            print("G0 Z0", file=out)
            print("G0 X0 Y0", file=out)

        print("G0 Z%s" % formatFloat(precision, args.zstart), file=out)

if __name__ == '__main__':
    main()
