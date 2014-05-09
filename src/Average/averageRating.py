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
# User and Item based Average Rating                              #
###################################################################

import sys, csv, json, argparse
sys.path.append("../")
from collections import defaultdict
from data_io import write_submission

with open("../../SETTINGS.json") as fp:
    settings = json.loads(fp.read())


class AverageRating():
    def __init__(self, trdata_path, vadata_path, tedata_path, m_choice):
        self.trdata_path = trdata_path
        self.vadata_path = vadata_path
        self.tedata_path = tedata_path
        self.m_choice = m_choice

    def train(self):
        self.ave_rating = {}
        if self.m_choice == 0:
            uid_srate = defaultdict(int)
            uid_rnum = defaultdict(int)
            for line in open(self.trdata_path):
                uid, pid, rating = line.strip("\r\t\n").split(" ")[:3]
                uid_srate[uid] += float(rating)
                uid_rnum[uid] += 1
            for uid in uid_srate:
                self.ave_rating[uid] = uid_srate[uid]/uid_rnum[uid]
        elif self.m_choice == 1:
            pid_srate = defaultdict(int)
            pid_rnum = defaultdict(int)
            for line in open(self.trdata_path):
                uid, pid, rating = line.strip("\r\t\n").split(" ")[:3]
                pid_srate[pid] += float(rating)
                pid_rnum[pid] += 1
            for pid in pid_srate:
                self.ave_rating[pid] = pid_srate[pid]/pid_rnum[pid]
        else:
            print 'Invalid choice of average rating method!'
            sys.exit(1)

    def predict(self, submission_path):
        prediction_result = []
        for line in open(self.tedata_path):
            uid, pid = line.strip("\r\t\n").split(" ")[:2]
            if self.m_choice == 0:
                prediction_result.append([uid, pid, self.ave_rating[uid]])
            elif self.m_choice == 1:
                prediction_result.append([uid, pid, self.ave_rating[pid]])
            else:
                print 'Invalid choice of average rating method!'
                sys.exit(1)
        write_submission(prediction_result, submission_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=int, action='store',
            dest='data_num', help='choose which data set to use')
    parser.add_argument('-m', type=int, action='store',
            dest='m_choice', help='choose which method to predict')
    if len(sys.argv) != 5:
        print 'Command e.g.: python averageRating.py -d 0(1) -m 0(1) '
        sys.exit(1)

    para = parser.parse_args()
    if para.data_num == 0:
        trdata_path = settings["ROOT_PATH"] + settings["TRAIN_DATA_FILE1"]
        vadata_path = settings["ROOT_PATH"] + settings["VALI_DATA_FILE1"]
        tedata_path = settings["ROOT_PATH"] + settings["TEST_DATA_FILE1"]
    elif para.data_num == 1:
        trdata_path = settings["ROOT_PATH"] + settings["TRAIN_DATA_FILE2"]
        vadata_path = settings["ROOT_PATH"] + settings["VALI_DATA_FILE2"]
        tedata_path = settings["ROOT_PATH"] + settings["TEST_DATA_FILE2"]
    else:
        print 'Invalid choice of dataset'
        sys.exit(1)

    if para.m_choice == 0:
        submission_path = settings["ROOT_PATH"] + settings["USER_AVERAGE_RATING_RESULT_FILE"]
    elif para.m_choice == 1:
        submission_path = settings["ROOT_PATH"] + settings["ITEM_AVERAGE_RATING_RESULT_FILE"]
    else:
        print 'Invalid choice of rating method'
        sys.exit(1)

    averageRating =AverageRating(trdata_path,
                                 vadata_path,
                                 tedata_path,
                                 para.m_choice)
    averageRating.train()
    averageRating.predict(submission_path)


if __name__ == "__main__":
    main()
