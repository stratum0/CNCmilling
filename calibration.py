#!/usr/bin/env python

import sys
import argparse
from decimal import Decimal

def formatFloat(precision, f):
    d = (Decimal(f) / Decimal(precision)).quantize(1) * Decimal(precision)
    return d.quantize(Decimal(1)) if d == d.to_integral() else d.normalize()


def main():
    parser = argparse.ArgumentParser(description="""
    Creates G-code to help in calibrating effective tool width.
    """)

    required = parser.add_argument_group('required arguments')
    required.add_argument('--max', dest='max', required=True, type=float,
                        help='Maximum estimated tool width, in mm')
    required.add_argument('--min', dest='min', required=True, type=float,
                        help='Minimum estimated tool width, in mm')
    required.add_argument('--step', required=True, type=float,
                        help='Step size to probe tool width in, in mm')
    required.add_argument('--output', required=True, dest='output', type=str, help="""
    Output file name.
    """)
    parser.add_argument('--cutdepth', dest='cutdepth', default='2', type=float, help="""
    Depth to cut, in mm.
    """)
    parser.add_argument('--zspace', dest='zspace', default='10', type=float, help="""
    Z distance to hover above origin when moving to disconnected region, in mm
    """)

    args = parser.parse_args()

    precision = Decimal(1)
    while precision > args.min / 20:
        precision = precision / Decimal(10)

    width = args.max
    x = 0
    with open(args.output, "w") as out:
        print("G90")

        while width > args.min - 1e-6:
            print("G0 Z%s" % formatFloat(precision, args.zspace), file=out)
            print("G0 X%s Y0" % formatFloat(precision, x), file=out)
            print("G1 Z%s" % formatFloat(precision, -args.cutdepth), file=out)
            print("G0 Z%s" % formatFloat(precision, args.zspace), file=out)
            print("G0 Y%s" % formatFloat(precision, width), file=out)
            print("G1 Z%s" % formatFloat(precision, -args.cutdepth), file=out)
            print("G0 Z%s" % formatFloat(precision, args.zspace), file=out)
            print("G0 Y%s" % formatFloat(precision, -2 * width), file=out)
            print("G1 Z%s" % formatFloat(precision, -args.cutdepth), file=out)
            print("G1 Y%s" % formatFloat(precision, -width), file=out)

            x += width * 3
            width -= args.step

        print("G0 Z%s" % formatFloat(precision, args.zspace), file=out)

if __name__ == '__main__':
    main()
