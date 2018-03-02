As always, @Drahflow has severe not-invented-here-syndrome.

```
mkvirtualenv -p $(which python3) -r requirements.pip cnc
workon cnc
./depthmap2gcode.py --help
```

This code converts grayscale depth maps to G-code for
CNC machines.

Simplest case: You have a b/w image and which to engrave
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
