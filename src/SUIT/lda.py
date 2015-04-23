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


import numpy as np
import sys


class Lda():
    def __init__(self, K):
        self.K = K
        self.alpha = 0.5
        self.beta = 0.1
        self.tr_iters = 1000
        self.inf_iters = 100
        self.trained = False

    def train(self, votes, n_words):
        self.n_words = n_words
        print len(votes)
        review_topic_dis = np.zeros((self.K, len(votes)))
        topic_word_dis = np.zeros((self.n_words, self.K))
        self.S_iz = []
        self.S_vz = [[0 for i in xrange(self.K)] for i in xrange(len(votes))]
        self.S_zw = [[0 for i in xrange(self.n_words)] for i in xrange(self.K)]
        self.S_z = [0 for i in xrange(self.K)]
        # Random initialization
        for vidx, vote in enumerate(votes):
            v_zi = [-1 for i in xrange(len(vote.words))]
            for i, widx in enumerate(vote.words):
                k = int(np.random.rand()*self.K)
                v_zi[i] = k
                self.S_vz[vidx][k] += 1
                self.S_zw[k][widx] += 1
                self.S_z[k] += 1
            self.S_iz.append(v_zi)
        # Gibbs training
        Wbeta = self.n_words*self.beta
        for it in xrange(self.tr_iters):
            for vidx, vote in enumerate(votes):
                P = np.zeros(self.K+1)
                for i, widx in enumerate(vote.words):
                    k = self.S_iz[vidx][i]
                    self.S_vz[vidx][k] -= 1
                    self.S_zw[k][widx] -= 1
                    self.S_z[k] -= 1
                    for k in xrange(self.K):
                        P[k+1] = P[k] + (self.S_vz[vidx][k]+self.alpha)*(self.S_zw[k][widx]+self.beta)/(self.S_z[k]+Wbeta)
                    sval = np.random.rand()*P[self.K]
                    sk = -1
                    for k in xrange(self.K):
                        if P[k+1]>sval:
                            sk = k
                            break
                    self.S_iz[vidx][i] = sk
                    self.S_vz[vidx][sk] += 1
                    self.S_zw[sk][widx] += 1
                    self.S_z[sk] += 1
            sys.stdout.write("\rFinished iteration=%d in training process" % (it+1))
            sys.stdout.flush()
        print ''
        # Compute model para
        Kalpha = self.K*self.alpha
        for vidx, vote in enumerate(votes):
            review_topic_dis[:,vidx] = (np.array(self.S_vz[vidx])+self.alpha)/(len(vote.words)+Kalpha)
        for k in xrange(self.K):
            topic_word_dis[:,k] = (np.array(self.S_zw[k])+self.beta)/(self.S_z[k]+Wbeta)
        self.trained = True
        return (review_topic_dis, topic_word_dis)


    def inference(self, votes):
        if self.trained:
            print 'No trained model. Invalid use of inference function.'
        review_topic_dis = np.zeros(self.K, len(votes))
        nS_iz = []
        nS_vz = [[0 for i in xrange(self.K)] for i in xrange(len(votes))]
        nS_zw = [[0 for i in xrange(self.n_words)] for i in xrange(self.K)]
        nS_z = [0 for i in xrange(self.K)]
        # Random initialization
        for vidx, vote in enumerate(votes):
            v_zi = [-1 for i in xrange(len(vote.words))]
            for i, widx in enumerate(vote.words):
                k = int(np.random.rand()*self.K)
                v_zi[i] = k
                nS_vz[vidx][k] += 1
                nS_zw[k][widx] += 1
                nS_z[k] += 1
            nS_iz.append(v_zi)
        # Gibbs training
        Wbeta = self.n_words*self.beta
        for it in xrange(self.inf_iters):
            for vidx, vote in enumerate(votes):
                P = np.zeros(self.K+1)
                for i, widx in enumerate(vote.words):
                    k = nS_iz[vidx][i]
                    nS_vz[vidx][k] -= 1
                    nS_zw[k][widx] -= 1
                    nS_z[k] -= 1
                    for k in xrange(self.K):
                        P[k+1] = P[k] + (nS_vz[vidx][k]+self.alpha)*(self.S_zw[k][widx]+nS_zw[k][widx]+self.beta)/(self.S_z[k]+nS_z[k]+Wbeta)
                    sval = np.random.rand()*P[self.K]
                    sk = -1
                    for k in xrange(self.K):
                        if P[k+1]>sval:
                            sk = k
                            break
                    nS_iz[vidx][i] = sk
                    nS_vz[vidx][sk] += 1
                    nS_zw[sk][widx] += 1
                    nS_z[sk] += 1
            sys.stdout.write("\rFinished iteration=%d in training process" % (it+1))
            sys.stdout.flush()
        print ''
        Kalpha = self.K*self.alpha
        for vidx, vote in enumerate(votes):
            review_topic_dis[:,vidx] = (np.array(self.S_vz[vidx])+self.alpha)/(len(vote.words)+Kalpha)
        return review_topic_dis

