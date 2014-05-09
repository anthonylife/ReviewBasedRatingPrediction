#!/usr/bin/env python
#encoding=utf8

import csv, json, sys, re


def createReviewDataset(infile, outfile):
    wfd = open(outfile, "w")
    one_review = []
    for line in open(infile):
        cstr = line.strip("\r\t\n").replace(r"\\", "").replace(r"\n", " ").replace(r"\"", "").replace("  ", " ")
        if len(cstr) == 0:
            uid, pid, rating, date, text = parseData(one_review)
            if uid != "" and pid != "" and rating != -1 and date != "" and text != "":
                wd_cnt = len(text.split(" "))
                if rating > 5 or rating < 0:
                    print 'haha'
                wfd.write("%s %s %.1f %s %d %s\n" % (uid, pid, rating, date, wd_cnt, text))
            one_review = []
        else:
            one_review.append(cstr)
    wfd.close()

def parseData(one_review):
    uid = ""
    pid = ""
    rating = -1
    date = ""
    text = ""
    for entry in one_review:
        if entry.startswith("product/productId"):
            pair = entry.split(" ")
            if len(pair) < 2:
                return uid, pid, rating, date, text
            pid = pair[1]
        elif entry.startswith("review/userId"):
            pair = entry.split(" ")
            if len(pair) < 2:
                return uid, pid, rating, date, text
            uid = pair[1]
        elif entry.startswith("review/score"):
            pair = entry.split(" ")
            if len(pair) < 2:
                return uid, pid, rating, date, text
            rating = float(pair[1])
        elif entry.startswith("review/time"):
            pair = entry.split(" ")
            if len(pair) < 2:
                return uid, pid, rating, date, text
            date = pair[1]
        elif entry.startswith("review/text"):
            text = entry.replace("review/text: ", "")
    return uid, pid, rating, date, text

if __name__ == "__main__":
    infile = "./foods.txt"
    outfile = "/home/anthonylife/Doctor/Code/MyPaperCode/ReviewBasedRatingPrediction/data/amazonfood_review.dat"
    createReviewDataset(infile, outfile)

