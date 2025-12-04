#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is mean to test one architecture for a the JointOracle functionality.
This functionality is meant to learn a joint probability distribution from some
inhomogeneous data and should be capable of creating samples from various conditional
distributions dericted from that joint distribution.
We will test this concept on data of Temperature T, Pressure P that sometimes
is measured incompletely, i.e. a sample can be (T,P) or (T,None), (None, P), or
even (None, None). Training creates a representation of the joint P(T,P) and
the maps to conditionals P(T,P | (T,P)), P(T,P | (T, None)), P(T,P | (None, P)),
and P(T,P | (None,None)). Then when passing input tuples, the output is a sample
from the posterior. For this, do the following:
    1. Definitions and imports
    2. Simulate some data
    3. Build Model
    4. Build guide
    5. Assemble to JointOracle 
    6. Perform inference
    7. Plots and illustrations
"""

"""
    1. Definitions and imports
"""




"""
    2. Simulate some data
"""




"""
    3. Build Model
"""




"""
    4. Build guide
"""




"""
    5. Assemble to JointOracle 
"""




"""
    6. Perform inference
"""




"""
    7. Plots and illustrations
"""