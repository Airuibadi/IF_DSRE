#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 20/07/14 02:11:58

@author: Ziyin Huang
"""
import sys
from pathlib import Path
import os

file_list = []
for f in Path(sys.argv[1]+'/').glob("*outdir"):
    file_list.append(str(f))
file_list.sort()
split = int(sys.argv[2])
for idx, f in enumerate(file_list) :
    last_infix = -1
    for r in sorted(Path(f).glob("*.s_test"), key=lambda x: (len(str(x)), str(x))) :
        ori_name = str(r)
        prefix_id = ori_name.rfind('/')
        prefix = ori_name[:prefix_id+1]
        sufix = ori_name[prefix_id+1:]
        sufix_id = sufix.find('_') + prefix_id + 1
        sufix = ori_name[sufix_id:]
        infix = int(ori_name[prefix_id+1:sufix_id])
        assert infix > last_infix
        last_infix = infix
        infix = infix + idx*split
        new_name = prefix+str(infix)+sufix
        os.rename(ori_name, new_name)




