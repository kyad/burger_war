#!/bin/sh -x

rm -f ~/.ros/result.csv
rosclean purge -y
rm -f burger_war/scripts/weight.hdf5.*
