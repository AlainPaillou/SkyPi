# SkyPi

This program is mainly dedicated to astronomy live video with live treatments.

This software runs with a raspberry pi (at least 3B+ or better 4)

This program require a picamera V2.

You will also need :

    Python 3
    openCV
    numpy
    pillow
    picamera

## Python installation 
If you don't have Python3 on your package manager, add it in sources.list:
`sudo add-apt-repository ppa:deadsnakes/ppa`

``` 
# Python 3
sudo apt install python3.8

# Pip
sudo apt install python3-pip
```
You can check version with `python --version`

## Packages installation
``` 
# openCV
sudo apt install python3-opencv

# numpy
python3 -m pip install --upgrade numpy

# Pillow
python3 -m pip install --upgrade Pillow

# Picamera
sudo pip install picamera
```
[Picamera installation documentation](https://picamera.readthedocs.io/en/release-1.13/install.html)

## Run program
``` 
python3
```

You will also have to create 2 directories in you installation directory :

    one for the images you will acquire
    one for the video you will acquire

This is an old version of SkyPi and i guess it must get bugs. This software won't get any update (no further developments).

IMPORTANT license information : This softawre and any part of this software are free of use for any kind of use.
