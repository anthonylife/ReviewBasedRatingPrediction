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
# Date: 2014/5/8                                                  #
# Count the basic statistics of the specified dataset             #
#  e.g. 1.User number; 2.Item number; 3. Review Number;           #
#       4.User average review num; 5. Item average review num;    #
#       6.Average length of review; 7. Rating distribution.       #
###################################################################

import sys, csv, json, argparse, pylab
import numpy as np
from collections import defaultdict

with open("../SETTINGS.json") as fp:
    settings = json.loads(fp.read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=int, action='store',
            dest='data_num', help='choose which data set to use')

    if len(sys.argv) != 3:
        print 'Command e.g.: python cntBasicStatics.py -d 0(1)'
        sys.exit(1)

    para = parser.parse_args()
    if para.data_num == 0:
        review_file = settings["ROOT_PATH"] + settings["SRC_DATA_FILE1_1"]
    elif para.data_num == 1:
        review_file = settings["ROOT_PATH"] + settings["SRC_DATA_FILE2_1"]
    else:
        print 'Invalid choice of dataset'
        sys.exit(1)

    uid_review_cnt = defaultdict(int)
    pid_review_cnt = defaultdict(int)
    review_num = 0
    ave_ur_num = 0
    ave_pr_num = 0
    ave_lenr_num = 0
    rating_ratio = defaultdict(int)
    for line in open(review_file):
        uid, pid, rating, date, wcnt = line.strip("\r\t\n").split(" ")[:5]
        uid_review_cnt[uid] += 1
        pid_review_cnt[pid] += 1
        review_num += 1
        ave_lenr_num += int(wcnt)
        rating_ratio[float(rating)] += 1
    cnt_num = [[entry[0], float(entry[1])/review_num] for entry in rating_ratio.items()]
    #keys = [entry[0] for entry in cnt_num]
    #vals = [entry[1] for entry in cnt_num]
    #width = 0.2
    #pylab.xticks(np.array(keys[:50])+width/2.0, keys, rotation=45)
    #pylab.bar(keys[:50], vals[:50], width, color='r')
    #pylab.show()

    print '1.User number:\t\t%d' % len(uid_review_cnt)
    print '2.Item number:\t\t%d' % len(pid_review_cnt)
    print '3.Review number:\t\t%d' % review_num
    print '4.User average review num:\t\t%.2f' % (float(review_num)/len(uid_review_cnt))
    print '5.Item average review num:\t\t%.2f' % (float(review_num)/len(pid_review_cnt))
    print '6.Average length of review:\t\t%.2f' % (float(ave_lenr_num)/review_num)
    print '7.Rating distribution:'
    print cnt_num

if __name__ == "__main__":
    main()

