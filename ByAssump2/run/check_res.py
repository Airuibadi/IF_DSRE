#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 20/10/08 21:32:54

@author: Ziyin Huang
"""
import json
import sys
from collections import defaultdict
res = json.load(open(sys.argv[1], 'r'))
ori_data = []
with open(sys.argv[2], 'r') as f :
    for line in f :
        ori_data.append(eval(line))
for key in res :
    with open("bpc_data"+str(key), 'w') as f :
        for line in ori_data :
            if line['id'] in res[key] :
                f.writelines(str(line)+'\n')

