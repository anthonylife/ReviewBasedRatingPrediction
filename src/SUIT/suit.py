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


import time, sys
sys.path.append("../")
import numpy as np
from corpus import Corpus
from utils import zeroCheck
from lda import Lda

class SUIT():
    def __init__(self, trdata_path, vadata_path, tedata_path, trtopic_path, vatopic_path, tetopic_path, beta_path, word_map_path):
        ## Parameters settings
        self.train_iters = 100
        self.infer_iters = 10
        self.grad_iters = 100
        self.K = 40
        self.max_words = 10000
        self.l_u = 0.1
        self.l_i = 0.1
        self.init_topic = 0        # 0=init from files, 1=init from implemented by us

        ## Reading review data
        self.trdata_path = trdata_path
        self.vadata_path = vadata_path
        self.tedata_path = tedata_path
        self.corp = Corpus(trdata_path, vadata_path, tedata_path, self.max_words)
        self.n_users = self.corp.n_users
        self.n_items = self.corp.n_items
        self.n_words = self.corp.n_words
        print 'N_users=%d, N_items=%d, N_words=%d, N_trreviews=%d' % (self.n_users, self.n_items, self.n_words, len(self.corp.train_votes))

        ## Model parameter allocation
        self.theta_user = np.random.random((self.K, self.n_users))
        self.theta_item = np.random.random((self.K, self.n_items))
        self.phi1_review = np.zeros((self.K, len(self.corp.train_votes)))
        self.phi2_review = np.zeros((self.K, len(self.corp.vali_votes)))
        self.phi3_review = np.zeros((self.K, len(self.corp.test_votes)))
        self.beta_kw = np.zeros((self.n_words, self.K))
        self.log_beta_kw = np.zeros((self.n_words, self.K))
        if self.init_topic == 0:
            for i, line in enumerate(open(trtopic_path)):
                parts = line.strip("\r\t\n")[:-1].split(" ")
                parts = map(float, parts)
                self.phi1_review[:, i] = parts
                self.phi1_review[:, i] /= np.sum(self.phi1_review[:, i])
            for i, line in enumerate(open(vatopic_path)):
                parts = line.strip("\r\t\n")[:-1].split(" ")
                parts = map(float, parts)
                self.phi2_review[:, i] = parts
                self.phi2_review[:, i] /= np.sum(self.phi2_review[:, i])
            for i, line in enumerate(open(tetopic_path)):
                parts = line.strip("\r\t\n")[:-1].split(" ")
                parts = map(float, parts)
                self.phi3_review[:, i] = parts
                self.phi3_review[:, i] /= np.sum(self.phi3_review[:, i])
            id_map_id = [-1 for i in xrange(self.n_words)]
            for i, line in enumerate(open(word_map_path)):
                if i == 0:
                    continue
                parts = line.strip("\r\t\n").split(" ")
                word = parts[0]
                r_widx = int(parts[1])
                if word not in self.corp.word_ids:
                    print 'Word mismatch'
                    sys.exit(1)
                widx = self.corp.word_ids[word]
                id_map_id[r_widx] = widx
            for i, line in enumerate(open(beta_path)):
                parts = line.strip("\r\t\n")[:-1].split(" ")
                parts = np.array(map(float, parts))
                self.beta_kw[:,i] = parts[id_map_id]
                self.beta_kw[:,i] /= np.sum(self.beta_kw[:,i])
            self.log_beta_kw = np.log(self.beta_kw)
        elif self.init_topic == 1:
            lda = Lda(self.K)
            (self.phi1_review, self.beta_kw) = lda.train(self.corp.train_votes, self.n_words)
            self.log_beta_kw = np.log(self.beta_kw)
            self.phi2_review = lda.inference(self.corp.vali_votes)
            self.phi3_review = lda.inference(self.corp.test_votes)
        else:
            print 'Invalid choice topic init method'
            sys.exit(1)

        self.train_votes_puser = [[] for u in xrange(self.n_users)]
        self.train_votes_pitem = [[] for i in xrange(self.n_items)]
        for i, vote in enumerate(self.corp.train_votes):
            uidx = vote.user
            iidx = vote.item
            self.train_votes_puser[uidx].append(i)
            self.train_votes_pitem[iidx].append(i)
        self.probValCheck()
        print "Finished model preprocessing"


    def probValCheck(self):
        for k in xrange(self.K):
            for v in xrange(len(self.corp.train_votes)):
                if self.phi1_review[k,v] == 0:
                    print 'phi1_review[%d,%d]=0' % (k, v)
            for v in xrange(len(self.corp.vali_votes)):
                if self.phi2_review[k,v] == 0:
                    print 'phi2_review[%d,%d]=0' % (k, v)
            for v in xrange(len(self.corp.test_votes)):
                if self.phi3_review[k,v] == 0:
                    print 'phi3_review[%d,%d]=0' % (k, v)
            for w in xrange(self.n_words):
                if self.beta_kw[w,k] == 0:
                    print 'beta_kw[%d,%d]=0' % (w,k)

    def train(self):
        likelihood = -np.exp(50)
        IK = np.eye(self.K)
        for it in xrange(self.train_iters):
            start = time.clock()
            old_likelihood = likelihood
            likelihood = 0
            ## Coordinate learning
            # Learning user factor
            for u in xrange(self.n_users):
                R_u = np.zeros(self.n_items)
                C_u = np.zeros((self.n_items, self.n_items))
                ITP = np.full((self.K, self.n_items), 1.0/self.K)
                for vidx in self.train_votes_puser[u]:
                    iidx = self.corp.train_votes[vidx].item
                    R_u[iidx] = self.corp.train_votes[vidx].rating
                    ITP[:,iidx] = self.phi1_review[:,vidx]
                    C_u[iidx, iidx] = 1
                A1 = self.theta_item*ITP
                A2 = np.dot(A1, C_u)
                A3 = np.dot(A2, A1.T)
                A4 = np.linalg.inv(A3+self.l_u*IK)
                self.theta_user[:,u] = np.dot(np.dot(A4, A2), R_u)
                likelihood += -self.l_u*np.dot(self.theta_user[:,u], self.theta_user[:,u])
                if (u+1) % 10 == 0:
                    sys.stdout.write("\rFinishing scanned user=%d" % (u+1))
                    sys.stdout.flush()
            # Learning item factor
            for i in xrange(self.n_items):
                R_i = np.zeros(self.n_users)
                C_i = np.zeros((self.n_users, self.n_users))
                UTP = np.full((self.K, self.n_users), 1.0/self.K)
                for vidx in self.train_votes_pitem[i]:
                    uidx = self.corp.train_votes[vidx].user
                    R_i[uidx] = self.corp.train_votes[uidx].rating
                    UTP[:,uidx] = self.phi1_review[:,vidx]
                    C_i[uidx, uidx] = 1
                A1 = self.theta_user*UTP
                A2 = np.dot(A1, C_i)
                A3 = np.dot(A2, A1.T)
                A4 = np.linalg.inv(A3+self.l_i*IK)
                self.theta_item[:,i] = np.dot(np.dot(A4, A2), R_i)
                likelihood += -self.l_i*np.dot(self.theta_item[:,i], self.theta_item[:,i])
                if (i+1) % 10 == 0:
                    sys.stdout.write("\rFinishing scanned item=%d" % (i+1))
                    sys.stdout.flush()
            # Learning review distributions
            beta_cum = np.zeros((self.n_words, self.K))
            for ridx, vote in enumerate(self.corp.train_votes):
                # Gradient projection method (NMF projection + Fast L1 projection)
                uidx = vote.user
                iidx = vote.item
                likelihood += -(np.dot(self.theta_user[:,uidx]*self.theta_item[:,iidx], self.phi1_review[:,ridx])-vote.rating)**2
                new_phi_review = np.copy(self.phi1_review[:,ridx])
                old_phi_review = self.phi1_review[:,ridx]
                obj_old = self.objTopic(vote, uidx, iidx, ridx, new_phi_review)
                grad = self.gradTopic(vote, uidx, iidx, ridx, beta_cum, False)
                grad_sum = np.sum(np.abs(grad))
                if grad_sum > 1:
                    grad /= grad_sum
                new_phi_review -= grad
                self.simplex_projection(new_phi_review)
                diff = new_phi_review-old_phi_review
                beta = 0.5
                r = np.dot(grad,diff)*0.5
                t = beta
                for it1 in xrange(self.grad_iters):
                    new_phi_review = old_phi_review + t*diff
                    obj_new = self.objTopic(vote, uidx, iidx, ridx, new_phi_review)
                    if obj_new > obj_old+r*t:
                        t *= beta
                    else:
                        likelihood += -obj_new
                        break
                self.phi1_review[:,ridx] = new_phi_review
                zeroCheck(new_phi_review)
                if (ridx+1) % 100 == 0:
                    sys.stdout.write("\rFinishing scanned reviews=%d" % (ridx+1))
                    sys.stdout.flush()
            # Learning topic distributions
            for k in xrange(self.K):
                self.beta_kw[:,k] = beta_cum[:,k]/np.sum(beta_cum[:,k])
            self.log_beta_kw = np.log(self.beta_kw)

            end = time.clock()
            time_cost = 1.0*(start-end)
            [tr_rmse, va_rmse, te_rmse] = self.evalRating(True, False, False)
            if likelihood < old_likelihood:
                print "Likelihood is decreasing."
            print "iter=%d, time_cost=%f, likelihood=%f, rmse=%f" % (it, time_cost, likelihood, tr_rmse)


    def objTopic(self, vote, uidx, iidx, ridx, phi_review):
        pred = np.sum(self.theta_user[:,uidx]*self.theta_item[:,iidx]*phi_review)
        obj = 0.5*(vote.rating-pred)**2
        for pair in vote.wordcnt:
            widx = pair[0]
            wcnt = pair[1]
            omega = phi_review*self.beta_kw[widx,:]
            omega = omega/np.sum(omega)
            obj -= wcnt*np.dot(omega, np.log(phi_review)+self.log_beta_kw[widx,:]-np.log(omega))
            if np.isnan(obj):
                print 'Nan in objTopic'
                print phi_review
                print self.beta_kw[widx,:]
                raw_input()
        return obj


    def gradTopic(self, vote, uidx, iidx, ridx, beta_cum, infer_tag):
        pred = np.sum(self.theta_user[:,uidx]*self.theta_item[:,iidx]*self.phi1_review[:,ridx])
        if not infer_tag:
            g = (vote.rating-pred)*self.theta_user[:,uidx]*self.theta_item[:,iidx]
        else:
            g = 0
        for pair in vote.wordcnt:
            widx = pair[0]
            wcnt = pair[1]
            omega = self.phi1_review[:,ridx]*self.beta_kw[widx,:]
            omega = omega/np.sum(omega)
            if not infer_tag:
                beta_cum[widx,:] += wcnt*omega
            g -= wcnt*omega/self.phi1_review[:,ridx]
        return g


    def simplex_projection(self, phi_review):
        cache_phi = np.copy(phi_review)
        cache_phi = np.sort(cache_phi)[::-1]
        cumsum = -1
        j = 0
        for val in cache_phi:
            cumsum += val
            if val > cumsum/(j+1):
                j += 1
            else:
                break
        theta = cumsum/j
        phi_review[:] = np.maximum(phi_review-theta, 0)
        phi_review /= np.sum(phi_review)


    def inference(self):
        likelihood = -np.exp(50)
        beta_cum = np.zeros((self.n_words, self.K))
        for it in xrange(self.infer_iters):
            start = time.clock()
            old_likelihood = likelihood
            likelihood = 0
            for ridx, vote in enumerate(self.corp.test_votes):
                # Gradient projection method (NMF projection + Fast L1 projection)
                uidx = vote.user
                iidx = vote.item
                likelihood += -(np.dot(self.theta_user[:,uidx]*self.theta_item[:,iidx], self.phi3_review[:,ridx])-vote.rating)**2
                new_phi_review = np.copy(self.phi3_review[:,ridx])
                old_phi_review = self.phi3_review[:,ridx]
                obj_old = self.objTopic(vote, uidx, iidx, ridx, new_phi_review)
                grad = self.gradTopic(vote, uidx, iidx, ridx, beta_cum, True)
                grad_sum = np.sum(np.abs(grad))
                if grad_sum > 1:
                    grad /= grad_sum
                new_phi_review -= grad
                self.simplex_projection(new_phi_review)
                diff = new_phi_review-old_phi_review
                beta = 0.5
                r = grad*diff*beta
                t = beta
                for it1 in xrange(self.grad_iters):
                    new_phi_review = old_phi_review + t*diff
                    obj_new = self.objTopic(uidx, iidx, ridx, new_phi_review)
                    if obj_new > obj_old+r*t:
                        t *= beta
                    else:
                        likelihood += -obj_new
                        break
                self.phi3_review[:,ridx] = new_phi_review
            end = time.clock()
            time_cost = 1.0*(start-end)/10e3
            if likelihood < old_likelihood:
                print "Likelihood is decreasing."
            print "iter=%d, time_cost=%f, likelihood=%f" % (it+1, time_cost, likelihood)


    def evalRating(self, tr_tag, va_tag, te_tag):
        tr_rmse = 0.0
        va_rmse = 0.0
        te_rmse = 0.0
        if tr_tag:
            for ridx, vote in enumerate(self.corp.train_votes):
                uidx = vote.user
                iidx = vote.item
                tr_rmse += (vote.rating-np.sum(self.theta_user[:,uidx]*self.theta_item[:,iidx]*self.phi1_review[:,ridx]))**2
            tr_rmse = np.sqrt(tr_rmse/len(self.corp.train_votes))
        if va_tag:
            for ridx, vote in enumerate(self.corp.vali_votes):
                uidx = vote.user
                iidx = vote.item
                va_rmse += (vote.rating-np.sum(self.theta_user[:,uidx]*self.theta_item[:,iidx]*self.phi2_review[:,ridx]))**2
            va_rmse = np.sqrt(va_rmse/len(self.corp.vali_votes))
        if te_tag:
            for ridx, vote in enumerate(self.corp.test_votes):
                uidx = vote.user
                iidx = vote.item
                te_rmse += (vote.rating-np.sum(self.theta_user[:,uidx]*self.theta_item[:,iidx]*self.phi3_review[:,ridx]))**2
            te_rmse = np.sqrt(te_rmse/len(self.corp.test_votes))
        return [tr_rmse, va_rmse, te_rmse]


    def predRating(self):
        pass


    def submitPredictions(self, submit_path):
        wfd = open(submit_path, "w")
        for ridx, vote in enumerate(self.corp.test_votes):
            uidx = vote.user
            iidx = vote.item
            pred = np.sum(self.theta_user[:,uidx], self.theta_item[:,iidx], self.phi3_review[:,ridx])
            wfd.write("%s %s %.6f\n" % (self.ruser_ids[uidx], self.ritem_ids[iidx], pred))
        wfd.close()

