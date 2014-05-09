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
# Segment the whole dataset into training and test dataset.       #
###################################################################

import sys, csv, json, argparse, random, math

with open("../SETTINGS.json") as fp:
    settings = json.loads(fp.read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=int, action='store',
            dest='data_num', help='choose which data set to use')
    if len(sys.argv) != 3:
        print 'Command e.g.: python segDataset.py -d 0(1)'
        sys.exit(1)

    para = parser.parse_args()
    if para.data_num == 0:
        data_infile = settings["ROOT_PATH"] + settings["SRC_DATA_FILE1_1"]
        train_outfile = settings["ROOT_PATH"] + settings["TRAIN_DATA_FILE1"]
        vali_outfile = settings["ROOT_PATH"] + settings["VALI_DATA_FILE1"]
        test_outfile = settings["ROOT_PATH"] + settings["TEST_DATA_FILE1"]
    elif para.data_num == 1:
        data_infile = settings["ROOT_PATH"] + settings["SRC_DATA_FILE2_1"]
        train_outfile = settings["ROOT_PATH"] + settings["TRAIN_DATA_FILE2"]
        vali_outfile = settings["ROOT_PATH"] + settings["VALI_DATA_FILE2"]
        test_outfile = settings["ROOT_PATH"] + settings["TEST_DATA_FILE2"]
    else:
        print 'Invalid choice of dataset'
        sys.exit(1)

    tr_wfd = open(train_outfile, "w")
    va_wfd = open(vali_outfile, "w")
    te_wfd = open(test_outfile, "w")
    tr_uid = set([])
    tr_pid = set([])
    cache_vadata = []
    cache_tedata = []
    for line in open(data_infile):
        s_ratio = random.random()
        uid, pid = line.strip("\r\t\n").split(" ")[:2]
        if s_ratio < settings["TRAIN_RATIO"]:
            tr_wfd.write(line)
            tr_uid.add(uid)
            tr_pid.add(pid)
        elif s_ratio < settings["TRAIN_RATIO"] + settings["VALI_RATIO"]:
            cache_vadata.append(line)
        else:
            cache_tedata.append(line)
    for line in cache_vadata:
        uid, pid = line.strip("\r\t\n").split(" ")[:2]
        if uid in tr_uid and pid in tr_pid:
            va_wfd.write(line)
        else:
            tr_wfd.write(line)
    for line in cache_tedata:
        uid, pid = line.strip("\r\t\n").split(" ")[:2]
        if uid in tr_uid and pid in tr_pid:
            te_wfd.write(line)
        else:
            tr_wfd.write(line)


if __name__ == "__main__":
    main()

