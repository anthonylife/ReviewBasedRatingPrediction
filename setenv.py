#!/usr/bin/env python
#encoding=utf8

import os, json

if __name__ == "__main__":
    settings = {}
    setting_file = "SETTINGS.json"
    wfp = open(setting_file, "w")
    settings["ROOT_PATH"] = os.getcwd() + "/"

    # Configuration
    settings["SRC_DATA_FILE1_1"] = "data/yelp_review.dat"
    settings["SRC_DATA_FILE1_2"] = "data/yelp_friendship.dat"
    settings["SRC_DATA_FILE2_1"] = "data/amazonfood_review.dat"
    settings["TRAIN_DATA_FILE1"] = "data/yelp_train.dat"
    settings["VALI_DATA_FILE1"] = "data/yelp_vali.dat"
    settings["TEST_DATA_FILE1"] = "data/yelp_test.dat"
    settings["TRAIN_DATA_FILE2"] = "data/amazonfood_train.dat"
    settings["VALI_DATA_FILE2"] = "data/amazonfood_vali.dat"
    settings["TEST_DATA_FILE2"] = "data/amazonfood_test.dat"
    settings["TRAIN_RATIO"] = 0.7
    settings["TEST_RATIO"] = 0.2
    settings["VALI_RATIO"] = 0.1

    # Write result
    json.dump(settings, wfp, sort_keys=True, indent=4)
