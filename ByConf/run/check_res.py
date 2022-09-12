#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 20/10/08 21:32:54

@author: Ziyin Huang
"""
import json
import sys
from collections import defaultdict

a = json.load(open(sys.argv[1], 'r'))
for key in a :
    print(len(a[key]))
print(a["5"])
print(a["6"])
print(a["7"])
