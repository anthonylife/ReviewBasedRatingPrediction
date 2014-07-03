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
# Date: 2014/7/2                                                  #
# Converting our raw specified data format to satisfy the input   #
#   requirements of LDA model.                                    #
###################################################################

import sys, csv, json, argparse, random
sys.path.append("../")
from collections import defaultdict

settings = json.loads(open("../../SETTINGS.json").read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=int, action='store',
            dest='data_num', help='choose which data set to use')

    if len(sys.argv) != 3:
        print 'Command e.g.: python convertDataFormat.py -d 0(1)'
        sys.exit(1)

    para = parser.parse_args()
    if para.data_num == 0:
        tr_data_path = settings["ROOT_PATH"]+settings["TRAIN_DATA_FILE1"]
        va_data_path = settings["ROOT_PATH"]+settings["VALI_DATA_FILE1"]
        te_data_path = settings["ROOT_PATH"]+settings["TEST_DATA_FILE1"]
        tr_review_path = settings["ROOT_PATH"]+settings["TRAIN_REVIEW_FILE1"]
        va_review_path = settings["ROOT_PATH"]+settings["VALI_REVIEW_FILE1"]
        te_review_path = settings["ROOT_PATH"]+settings["TEST_REVIEW_FILE1"]
    elif para.data_num == 1:
        tr_data_path = settings["ROOT_PATH"]+settings["TRAIN_DATA_FILE2"]
        va_data_path = settings["ROOT_PATH"]+settings["VALI_DATA_FILE2"]
        te_data_path = settings["ROOT_PATH"]+settings["TEST_DATA_FILE2"]
        tr_review_path = settings["ROOT_PATH"]+settings["TRAIN_REVIEW_FILE2"]
        va_review_path = settings["ROOT_PATH"]+settings["VALI_REVIEW_FILE2"]
        te_review_path = settings["ROOT_PATH"]+settings["TEST_REVIEW_FILE2"]
    elif para.data_num == 2:
        tr_data_path = settings["ROOT_PATH"]+settings["TRAIN_DATA_FILE3"]
        va_data_path = settings["ROOT_PATH"]+settings["VALI_DATA_FILE3"]
        te_data_path = settings["ROOT_PATH"]+settings["TEST_DATA_FILE3"]
        tr_review_path = settings["ROOT_PATH"]+settings["TRAIN_REVIEW_FILE3"]
        va_review_path = settings["ROOT_PATH"]+settings["VALI_REVIEW_FILE3"]
        te_review_path = settings["ROOT_PATH"]+settings["TEST_REVIEW_FILE3"]
    elif para.data_num == 3:
        tr_data_path = settings["ROOT_PATH"]+settings["TRAIN_DATA_FILE4"]
        va_data_path = settings["ROOT_PATH"]+settings["VALI_DATA_FILE4"]
        te_data_path = settings["ROOT_PATH"]+settings["TEST_DATA_FILE4"]
        tr_review_path = settings["ROOT_PATH"]+settings["TRAIN_REVIEW_FILE4"]
        va_review_path = settings["ROOT_PATH"]+settings["VALI_REVIEW_FILE4"]
        te_review_path = settings["ROOT_PATH"]+settings["TEST_REVIEW_FILE4"]
    else:
        print 'Invalid choice of dataset'
        sys.exit(1)

    words_cnt = defaultdict(int)
    for line in open(tr_data_path):
        parts = line.strip("\r\t\n").split(" ")
        for word in parts[4:]:
            words_cnt[word] += 1

    saved_words = set([pair[0] for pair in sorted(words_cnt.items(), key=lambda x:x[1], reverse=True)][:settings["MAX_WORDS"]])

    doc_num = 0
    for line in open(tr_data_path):
        doc_num += 1
    wfd = open(tr_review_path, "w")
    wfd.write("%d\n" % doc_num)
    for line in open(tr_data_path):
        parts = line.strip("\r\t\n").split(" ")
        for word in parts[4:]:
            if word in saved_words:
                wfd.write("%s " % word)
        wfd.write("\n")
    wfd.close()

    doc_num = 0
    for line in open(va_data_path):
        doc_num += 1
    wfd = open(va_review_path, "w")
    wfd.write("%d\n" % doc_num)
    for line in open(va_data_path):
        parts = line.strip("\r\t\n").split(" ")
        for word in parts[4:]:
            if word in saved_words:
                wfd.write("%s " % word)
        wfd.write("\n")
    wfd.close()

    doc_num = 0
    for line in open(te_data_path):
        doc_num += 1
    wfd = open(te_review_path, "w")
    wfd.write("%d\n" % doc_num)
    for line in open(te_data_path):
        parts = line.strip("\r\t\n").split(" ")
        for word in parts[4:]:
            if word in saved_words:
                wfd.write("%s " % word)
        wfd.write("\n")
    wfd.close()


if __name__ == "__main__":
    main()

