#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 20/07/14 02:11:58

@author: Ziyin Huang
"""
import sys
from pathlib import Path
import os
f = sys.argv[1]
last_infix = -1
for idy, r in enumerate(sorted(Path(f).glob("*.s_test"), key=lambda x: (len(str(x)), str(x)))) :
    ori_name = str(r)
    prefix_id = ori_name.rfind('/')
    prefix = ori_name[:prefix_id+1]
    sufix = ori_name[prefix_id+1:]
    sufix_id = sufix.find('_') + prefix_id + 1
    sufix = ori_name[sufix_id:]
    infix = int(ori_name[prefix_id+1:sufix_id])
    if last_infix != -1:
        assert infix-last_infix == 1
    last_infix = infix
    infix = idy
    new_name = prefix+str(infix)+sufix
    os.rename(ori_name, new_name)




