#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 20/07/16 01:43:04

@author: Ziyin Huang
"""
import numpy as np
import sys
from pathlib import Path
import json
import time
file_list = []

for f in Path(sys.argv[1]).glob("*outdir") :
    file_list.append(str(f))
file_list.sort(key=lambda x: (len(str(x)), str(x)))
res_list = [np.load(f+'/IF_value_tmp.npy') for f in file_list]
influences = np.concatenate(res_list, axis=1)
np.save(sys.argv[2]+"/IF_value_result.npy", influences)
test_num = influences.shape[1]
influences = np.sum(influences, axis=1)/test_num
influences = influences.tolist()
print(len(influences))
harmful = np.argsort(influences)
helpful = harmful[::-1]
influence_results = {}
influence_results['influences'] = influences
influence_results['harmful'] = harmful.tolist()
influence_results['helpful'] = helpful.tolist()
print(influences[:11])
print(harmful[:10])
print(helpful[:10])
json.dump(influence_results, open(sys.argv[2]+'/IFresult.json', 'w'))
