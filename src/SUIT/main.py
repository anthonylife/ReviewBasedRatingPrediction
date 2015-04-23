#!/usr/bin/env python
#encoding=utf8

#Copyright [2015] [Wei Zhang]

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#http://www.apache.org/licenses/LICENSE-2.0
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.


import argparse, json, sys
from suit import SUIT

with open("../../SETTINGS.json") as fp:
    settings = json.loads(fp.read())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=int, action='store',
            dest='data_num', help='choose which data set to use')

    para = parser.parse_args()
    if para.data_num == 1:
        trdata_path = settings["ROOT_PATH"]+settings["TRAIN_DATA_FILE2_AWARE"]
        vadata_path = settings["ROOT_PATH"]+settings["VALI_DATA_FILE2_AWARE"]
        tedata_path = settings["ROOT_PATH"]+settings["TEST_DATA_FILE2_AWARE"]
        trtopic_path = "../GibbsLDA++/food_tr_topic.dat"
        vatopic_path = "../GibbsLDA++/food_va_topic.dat"
        tetopic_path = "../GibbsLDA++/food_te_topic.dat"
        beta_path = "../GibbsLDA++/food_word_dis.dat"
        word_map_path = "../GibbsLDA++/food_wordmap.dat"
        submit_path = settings["ROOT_PATH"]+settings["SUIT_RESULT_FILE2"]
    elif para.data_num == 6:
        trdata_path = settings["ROOT_PATH"]+settings["TRAIN_DATA_FILE7_AWARE"]
        vadata_path = settings["ROOT_PATH"]+settings["VALI_DATA_FILE7_AWARE"]
        tedata_path = settings["ROOT_PATH"]+settings["TEST_DATA_FILE7_AWARE"]
        trtopic_path = "../GibbsLDA++/video_tr_topic.dat"
        vatopic_path = "../GibbsLDA++/video_va_topic.dat"
        tetopic_path = "../GibbsLDA++/video_te_topic.dat"
        beta_path = "../GibbsLDA++/video_word_dis.dat"
        word_map_path = "../GibbsLDA++/video_wordmap.dat"
        submit_path = settings["ROOT_PATH"]+settings["SUIT_RESULT_FILE5"]
    elif para.data_num == 7:
        trdata_path = settings["ROOT_PATH"]+settings["TRAIN_DATA_FILE8_AWARE"]
        vadata_path = settings["ROOT_PATH"]+settings["VALI_DATA_FILE8_AWARE"]
        tedata_path = settings["ROOT_PATH"]+settings["TEST_DATA_FILE8_AWARE"]
        trtopic_path = "../GibbsLDA++/sport_tr_topic.dat"
        vatopic_path = "../GibbsLDA++/sport_va_topic.dat"
        tetopic_path = "../GibbsLDA++/sport_te_topic.dat"
        beta_path = "../GibbsLDA++/sport_word_dis.dat"
        word_map_path = "../GibbsLDA++/video_wordmap.dat"
        submit_path = settings["ROOT_PATH"]+settings["SUIT_RESULT_FILE6"]
    else:
        print 'Invalid choice of data set!'
        sys.exit(1)

    suit = SUIT(trdata_path, vadata_path, tedata_path, trtopic_path, vatopic_path, tetopic_path, beta_path, word_map_path)
    suit.train()
    suit.inference()
    suit.submitPrediction(submit_path)

