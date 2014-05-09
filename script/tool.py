#!/usr/bin/env python
#encoding=utf8

#Copyright [2014] [Wei Zhang]

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#http://www.apache.org/licenses/LICENSE-2.0
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

###################################################################
# Date: 2014/5/9                                                  #
# Providing all tool functions for each algorithms                #
###################################################################

import sys, csv, json, math, random
from collections import defaultdict

with open("../SETTINGS.json") as fp:
    settings = json.loads(fp.read())


def checkRating(rating):
    if rating < settings["MIN_RATING_VAL"]:
        rating = settings["MIN_RATING_VAL"]
    elif rating > settings["MAX_RATING_VAL"]:
        rating = settings["MAX_RATING_VAL"]
    return rating
