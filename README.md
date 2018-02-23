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
