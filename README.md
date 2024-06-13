# GCS Box Unloading

Using Graph of Convex Sets (GCS) algorithm to plan paths for a robot to unload boxes from a truck trailer.

## Installation
To run this code, you must obtain the directory `data/unload-gen0` which contains robot assets (that are not public).

Our recommended setup requires using a Linux machine, or using WSL2 on Windows, with the following requirements:
- `python` 3.8 or higher
- `pip` 23.3.1 or higher

Necessary installs:
- `pip install manipulation`
- `pip install drake`
- `pip install ipython`
- `pip install pyvirtualdisplay`

## Running

```
cd src
python main.py
```