#!/bin/bash
cd ~
sudo insmod v4l2loopback/v4l2loopback.ko video_nr=0,1,2,3
cd flirone-v4l2
sudo ./flirone palettes/Grey.raw