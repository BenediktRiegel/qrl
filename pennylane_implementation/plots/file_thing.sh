#!/bin/sh
x="2/value_qnn_not_learning.html"
cat ./$x >> temp
rm $x
mv temp $x
