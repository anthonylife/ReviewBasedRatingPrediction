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

import sys
sys.path.append("../")
from collections import defaultdict
from utils import loadReviewData


class Corpus():
    def __init__(self, trdata_path, vadata_path, tedata_path, max_words):
        self.trdata_path = trdata_path
        self.vadata_path = vadata_path
        self.tedata_path = tedata_path
        self.max_words = max_words
        self.word_ids = {}
        self.rword_ids = {}
        self.user_ids = {}
        self.ruser_ids = {}
        self.item_ids = {}
        self.ritem_ids = {}

        word_count = defaultdict(int)
        for line in open(self.trdata_path):
            parts = line.strip("\r\t\n").split(" ")
            for word in parts[5:]:
                if len(word) > 0:
                    word_count[word] += 1

        word_count = sorted(word_count.items(), key=lambda x:x[1], reverse=True)
        self.n_words = min(len(word_count), self.max_words)
        wfd = open("word_id.txt", "w")
        for i in xrange(self.n_words):
            self.word_ids[word_count[i][0]] = i
            self.rword_ids[i] = word_count[i][0]
            wfd.write("%s %d\n" % (word_count[i][0], i))
        wfd.close()
        self.train_votes = loadReviewData(self.trdata_path, self.word_ids,
                self.user_ids, self.ruser_ids, self.item_ids, self.ritem_ids)
        self.vali_votes = loadReviewData(self.vadata_path, self.word_ids,
                self.user_ids, self.ruser_ids, self.item_ids, self.ritem_ids)
        self.test_votes = loadReviewData(self.tedata_path, self.word_ids,
                self.user_ids, self.ruser_ids, self.item_ids, self.ritem_ids)
        self.n_users = len(self.user_ids)
        self.n_items = len(self.item_ids)

