# pyFastcamTools

Welcome to the Stan's pyFastcamTools.<br/>
Here are some basic info about how to use the tools to read the fast-camera data.<br/>

These tools are used to read and analyse fast-camera data from MAST.
These are loosely based on Nick Walkden tools from:
https://github.com/nick-walkden/pyFastcamTools

This repository is meant to clean up those tools, develop a user-friendly wiki documentation, and make it ready for insertion into the ukaea github space.<br/>
The camera movies can be found in /projects/SOL/Data/Cameras/ at CCFE for authorized personel.<br/>


# Python packages
Before running, you'll need to make sure you have a few packages installed, particularly open-cv, the main reading package.<br/>
On Ubuntu, these can be installed with

```
sudo apt-get install python3-opencv python3-matplotlib
```
but of course you can do that with pip.

# Playing a movie
This will simply play a movie on your screen.<br/>
In this case, we play a movie stored in the file <code>29786/C001H001S0001/C001H001S0001-04.mraw</code>

```
python3
>>> from pyFastcamTools.utilities.play_movie import play_movie
>>> play_movie('29786/C001H001S0001/C001H001S0001-04.mraw')
```

# Reading a movie
This will read the entire movie frames, which you can then process.

```
python3
>>> from pyFastcamTools.utilities.read_movie import read_movie
>>> read_movie('29786/C001H001S0001/C001H001S0001-04.mraw')
```

# Saving a movie
You can also use that command to save frames into .png or .jpg files.<br/>
In that example, we only save the frames 4100->4200 into the folder <code>movie_frames</code>

```
python3
>>> from pyFastcamTools.utilities.read_movie import read_movie
>>> read_movie('29786/C001H001S0001/C001H001S0001-04.mraw',save_frames=True,frames_directory='movie_frames',startframe=4100,endframe=4200)
```

# Subtracting background
A useful tool subtracts the background average of the image to visualise the filaments at the plasma edge.<br/>
Sorry, still need to import this from Nick's space...




