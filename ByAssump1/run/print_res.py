#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 21/04/07 14:14:10

@author: Ziyin Huang
"""
import json,sys
res = json.load(open(sys.argv[1], 'r'))
for key in res :
    print(key)
    print(len(res[key]))

