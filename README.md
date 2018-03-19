As always, @Drahflow has severe not-invented-here-syndrome.

```
mkvirtualenv -p $(which python3) -r requirements.pip cnc
workon cnc
./depthmap2gcode.py --help
```

This code converts grayscale depth maps to G-code for
CNC machines.

Simplest case: You have a b/w image and wish to engrave
some object with it. The code assumes **dark = deep into material, white = top**.

```
./depthmap2gcode.py --depth 10 --width 50 --height 50 --planes 4 --tool 4:my-image.g my-image.png
```

Creates G-code for a 4mm tool to cut (an approximation of) the image quantized into
4 z-planes into the material.

The generated G-gode uses coordinates: (0, 0, 0) -> (width, height, -depth).


## Tool calibration

Unfortunately, the effective cut width of the end mill is not equal to what the end mill
producer claims as diameter. To find the actual width of cuts made in the concrete material
you want to use, you can generate a test pattern like this (assuming you have a tool which
probably cuts between 1.6mm and 1.2mm wide):

```
./calibration.py --max 1.6 --min 1.2 --step 0.05 --cutdepth 2 --zspace 2 --output calibration.g
```

It will generate a series of pairs of holes with decreasing (by `--step`) separation.
The first pair which connects tells you the effective tool width.


## Intarsia

The thing I made all this for. Take a lovely black (form you want to inlay) and white (background)
example.png file and
```
./intarsia.sh example
```

Check the rendered `example-collisions.png` and ensure there are no white areas (except in the
outermost corners). White areas should have been cut from the inlay wood but were not reachable
due to tool diameter. Wood would collide during assembly there, so if you have white areas you
*must* choose a different, rounder input image.

Put one of the "1.6mm" Dremel HSS tool into the spindle, set speed somewhere between 3-4, position
the mill on top right corner of target area exactly on top of the material, then manually
`G92 X0 Y0 Z0` to set the coordinate system. Play `example-background.g`.

Change material to inlay wood, reposition coordinate system (as before) and play `example-foreground.g`.
Detach milled foreground area from rest of inlay wood such that a matching piece for the background
comes free. Put some wood glue in the background wood align everything well and press together
hard, i.e. use a bench vice.

Wait until wood glue is set (30min did it for me) and somehow remove the now attached block of inlay wood
and minimally mill into the background where the inlay was positioned. If you are too lazy to setup the
router (**but not too lazy to vacuum a lot**), you can delegate this to the CNC mill by using a
huge cutter end, generating
```
./smooth.g --zstart <height of inlay block + 1mm> --zstop -0.1 --width 60 --height 60 --ystep 5 --cutdepth 1 --output smooth.g
```
then playing `smooth.g`, and then vacuum.


### Troubleshooting

* Need different size: Change `SIZE` in `intarsia.sh`, its in mm.
* Milled pieces don't fit:
  1. Press harder.
  2. Align better.
  3. Change `intarsia.sh` specifically increase the `-blur` radius, it's given as AxB where
     `A` is just image kernel size and `B` the actual blurring. Increase B and use A = 2*B.
     Units are `--precision`-sized pixels.
