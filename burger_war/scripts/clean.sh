#!/bin/sh -x

rm -f ~/.ros/result.csv
rosclean purge -y
git checkout burger_war/scripts/weight.hdf5
rm -f burger_war/scripts/weight.hdf5.*
git status
