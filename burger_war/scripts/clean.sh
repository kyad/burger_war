#!/bin/sh -x

rm -f ~/.ros/results.csv
rosclean purge -y
rm -f burger_war/scripts/weight.hdf5.*
