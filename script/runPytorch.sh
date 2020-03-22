#!/bin/sh
#add classes in varyLr.py file
python varyLr.py
cd models
for i in *.py 
do
	python $i
done