# GCS Box Unloading

Using Graph of Convex Sets (GCS) algorithm to plan paths for a robot to unload boxes from a truck trailer.

## Installation
You must obtain the directory `data/unload-gen0` which contains robot assets.

Our recommended setup requires using a Linux machine, or using WSL2 on Windows, with the following requirements:
- `python` 3.8 or higher
- `pip` 23.3.1 or higher

Necessary installs:
- `pip install manipulation`
- `pip install --extra-index-url https://drake-packages.csail.mit.edu/whl/nightly/ 'drake==0.0.20231210'` (or any newer version of drake)
- `pip install ipython`
- `pip install pyvirtualdisplay`

If you are running WSL on Windows, ensure you install the following to enable the graphics libraries to work:
 - Update to the latest version of WSL
 - `sudo apt install mesa-utils`
 - Install XcXsrv software on your Windows Machine: https://sourceforge.net/projects/vcxsrv/files/latest/download
 - Before running this code, start an instance of XcXsrv (by starting the XLaunch application). Leave all settings in XLaunch at the default, except, *disable Access Control*. You should only need to do this once (unless you kill XcXsrv or restart your machine).
 - Test that the display forwarding is working by runing `glxgears` in your WSL terminal. You should see a new window appear with an animation of spinning gears.
 - If you ever run src/main.py, but see nothing happen in the meshcat Window (but also receive no error message), you likely do not have an instance of XcXsrv running.