# Project1: Dynamic Programming

This project develop a method using forward dynamic programming to solve the door & key problem.



## Running

All modified code are in doorkey.py and utils.py files. Run doorkey.py and it will run partA, generate_policy and partB function consecutively. partA function generate control sequence for the door & key problem in known map and draw a gif for each map. generate_policy function iterate 36 random maps , generate and save policy in .npy file. partB function load policy, choose a random map and extract control sequence for current map, then draw a gif.



