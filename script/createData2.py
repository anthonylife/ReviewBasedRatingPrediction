#!/usr/bin/env python
#encoding=utf8

import csv, json, sys, re
from collections import defaultdict


reload(sys)
sys.setdefaultencoding('utf8')


def createReviewDataset(infile, outfile):
    wfd = open(outfile, "w")
    for line in open(infile):
        try:
            parse_data = json.loads(line.strip("\r\t\n").replace(r"\\", "").replace(r"\n", " ").replace(r"\"", "").replace("  ", " "))
            #parse_data = json.loads(line.strip("\r\t\n").replace("\\\"", "").replace("\\n", "").replace("\\\\", "\\"))
        except:
            print line.strip("\r\t\n").replace("\\\"", "").replace("\\n", "")
            wfd1 = open("error.txt", "w")
            wfd1.write(line)
            wfd1.close()
            sys.exit(1)
        uid = parse_data["user_id"]
        date = parse_data["date"]
        rating = float(parse_data["stars"])
        if rating > 5:
            rating = 5
        elif rating < 0:
            rating = 0
        text = parse_data["text"]
        wd_cnt = len(text.split(" "))
        bid = parse_data["business_id"]
        #print uid, bid, rating, date, wd_cnt, text
        try:
            wfd.write("%s %s %.1f %s %d %s\n" % (uid, bid, rating, date, wd_cnt, text))
        except:
            print text.encode()
            sys.exit(1)
    wfd.close()


def createFriendshipDataset(infile, outfile):
    wfd = open(outfile, "w")
    for line in open(infile):
        parse_data = json.loads(line.strip("\r\t\n"))
        uid = parse_data["user_id"]
        friends = parse_data["friends"]
        if len(friends) > 0:
            wfd.write("%s %s\n" % (uid, " ".join(friends)))
    wfd.close()

if __name__ == "__main__":
    infile = "./yelp_academic_dataset_review.json"
    outfile = "/home/anthonylife/Doctor/Code/MyPaperCode/ReviewBasedRatingPrediction/data/yelp_review.dat"
    createReviewDataset(infile, outfile)

    infile = "./yelp_academic_dataset_user.json"
    outfile = "/home/anthonylife/Doctor/Code/MyPaperCode/ReviewBasedRatingPrediction/data/yelp_friendship.dat"
    createFriendshipDataset(infile, outfile)

