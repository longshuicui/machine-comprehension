# -*- coding: utf-8 -*-
"""
@author: longshuicui
@date  :   2020/7/3
@function:
"""
import json

with open("./dev-v2.0.json",encoding="utf8") as file:
    sentences=file.readlines()
for sent in sentences:
    content=json.loads(sent.strip())

json.dump(content,open("dev.json","w",encoding="utf8"),ensure_ascii=False,indent=4)
