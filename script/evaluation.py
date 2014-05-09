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
# Date: 2014/5/1                                                  #
# Evaluation on file results                                      #
# Note:                                                           #
#   1. evaluation metrics including Precision, Recall, F-score.   #
###################################################################

import sys, csv, json, argparse
from collections import defaultdict
from tool import checkRating

with open("../SETTINGS.json") as fp:
    settings = json.loads(fp.read())


class Evaluation():
    def __init__(self, standard_result_file, prediction_result_file):
        self.standard_result_file = standard_result_file
        self.prediction_result_file = prediction_result_file

    def evaluate(self):
        rmse = 0.0
        mae = 0.0
        rating_num = 0
        for line1, line2 in zip(open(self.standard_result_file), open(self.prediction_result_file)):
            uid1, pid1, rating1 = line1.strip("\r\t\n").split(" ")[:3]
            uid2, pid2, rating2 = line2.strip("\r\t\n").split(" ")[:3]
            if uid1 != uid2 or pid1 != pid2:
                print uid1, pid1, rating1
                print uid2, pid2, rating2
                sys.exit(1)
            rmse += (checkRating(float(rating1))-checkRating(float(rating2)))**2
            mae += abs(checkRating(float(rating1))-checkRating(float(rating2)))
            rating_num += 1
        rmse = rmse/rating_num
        mae = mae/rating_num
        return rmse, mae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=int, action='store',
            dest='data_num', help='choose which data set to use')
    parser.add_argument('-a', type=int, action='store',
            dest='algorithm_num', help='specify the algorithm which genenrate the recommendation results')
    parser.add_argument('-m', type=int, action='store',
            dest='eval_method', help='choose which method to evaluate the results')

    if len(sys.argv) != 7:
        print 'Command e.g.: python evaluation.py -d 0(1) -a 0(0,1,...) -m 0(...)'
        sys.exit(1)

    para = parser.parse_args()
    if para.data_num == 0:
        standard_result_file = settings["ROOT_PATH"] + settings["TEST_DATA_FILE1"]
    elif para.data_num == 1:
        standard_result_file = settings["ROOT_PATH"] + settings["TEST_DATA_FILE2"]
    else:
        print 'Invalid choice of data set!'
        sys.exit(1)

    if para.algorithm_num == 0:
        prediction_result_file = settings["ROOT_PATH"] + settings["USER_AVERAGE_RATING_RESULT_FILE"]
        tips = "User based average rating prediction algorithm"
    elif para.algorithm_num == 1:
        prediction_result_file = settings["ROOT_PATH"] + settings["ITEM_AVERAGE_RATING_RESULT_FILE"]
        tips = "Item based average rating prediction algorithm"
    else:
        print 'Invalid choice of algorithm!'
        sys.exit(1)

    evaluation = Evaluation(standard_result_file,
                            prediction_result_file)
    rmse, mae = evaluation.evaluate()
    print "%s: RMSE=%.3f, MAE=%.3f" %(tips, rmse, mae)

if __name__ == "__main__":
    main()

