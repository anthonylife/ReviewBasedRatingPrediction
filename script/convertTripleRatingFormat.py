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
# Convert raw format of data into triple format for LIBMF:        #
#   Two formats: (1). triple format; (2). sparse matrix.          #
###################################################################

import sys, csv, json, argparse

with open("../SETTINGS.json") as fp:
    settings = json.loads(fp.read())


def convert(infile, outfile):
    ''' for libmf '''
    wfd = open(outfile, "w")
    for line in open(infile):
        uid, pid, rating = line.strip("\r\t\n").split(" ")[:3]
        wfd.write("%s %s %s\n" % (uid, pid, rating))
    wfd.close()

def convert1(infile, outfile):
    ''' for max-margin matrix factorization '''
    max_uid = 0
    for line in open(infile):
        uid, pid, rating = line.strip("\r\t\n").split(" ")[:3]
        if int(uid) > max_uid:
            max_uid = int(uid)

    output_result = [[] for i in xrange(max_uid+1)]
    for line in open(infile):
        uid, pid, rating = line.strip("\r\t\n").split(" ")[:3]
        uid = int(uid)
        pid = int(pid)+1
        rating = float(rating)
        output_result[uid].append([pid, rating])

    wfd = open(outfile, "w")
    for mul_entry in output_result:
        for entry in mul_entry:
            wfd.write("%d:%.1f " % (entry[0], entry[1]))
        wfd.write("\n")
    wfd.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=int, action='store',
            dest='data_num', help='choose which data set to use')

    if len(sys.argv) != 3:
        print 'Command e.g.: python evaluation.py -d 0(1)'
        sys.exit(1)

    para = parser.parse_args()
    if para.data_num == 0:
        train_infile = settings["ROOT_PATH"] + settings["TRAIN_DATA_FILE1"]
        vali_infile = settings["ROOT_PATH"] + settings["VALI_DATA_FILE1"]
        test_infile = settings["ROOT_PATH"] + settings["TEST_DATA_FILE1"]
        train_outfile = settings["ROOT_PATH"] + settings["TRAIN_TRIPLE_FILE1"]
        vali_outfile = settings["ROOT_PATH"] + settings["VALI_TRIPLE_FILE1"]
        test_outfile = settings["ROOT_PATH"] + settings["TEST_TRIPLE_FILE1"]
        train_outfile1 = settings["ROOT_PATH"] + settings["TRAIN_SPMAT_FILE1"]
        vali_outfile1 = settings["ROOT_PATH"] + settings["VALI_SPMAT_FILE1"]
        test_outfile1 = settings["ROOT_PATH"] + settings["TEST_SPMAT_FILE1"]
    elif para.data_num == 1:
        train_infile = settings["ROOT_PATH"] + settings["TRAIN_DATA_FILE2"]
        vali_infile = settings["ROOT_PATH"] + settings["VALI_DATA_FILE2"]
        test_infile = settings["ROOT_PATH"] + settings["TEST_DATA_FILE2"]
        train_outfile = settings["ROOT_PATH"] + settings["TRAIN_TRIPLE_FILE2"]
        vali_outfile = settings["ROOT_PATH"] + settings["VALI_TRIPLE_FILE2"]
        test_outfile = settings["ROOT_PATH"] + settings["TEST_TRIPLE_FILE2"]
        train_outfile1 = settings["ROOT_PATH"] + settings["TRAIN_SPMAT_FILE2"]
        vali_outfile1 = settings["ROOT_PATH"] + settings["VALI_SPMAT_FILE2"]
        test_outfile1 = settings["ROOT_PATH"] + settings["TEST_SPMAT_FILE2"]
    else:
        print 'Invalid choice of dataset'
        sys.exit(1)

    #convert(train_infile, train_outfile)
    #convert(vali_infile, vali_outfile)
    #convert(test_infile, test_outfile)
    convert1(train_infile, train_outfile1)
    convert1(vali_infile, vali_outfile1)
    convert1(test_infile, test_outfile1)

if __name__ == "__main__":
    main()
