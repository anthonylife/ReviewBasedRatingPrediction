#pragma once
//Copyright [2014] [Wei Zhang]

//Licensed under the Apache License, Version 2.0 (the "License");
//you may not use this file except in compliance with the License.
//You may obtain a copy of the License at
//http://www.apache.org/licenses/LICENSE-2.0
//Unless required by applicable law or agreed to in writing, software
//distributed under the License is distributed on an "AS IS" BASIS,
//WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//See the License for the specific language governing permissions and
//limitations under the License.

///////////////////////////////////////////////////////////////////
// Date: 2015/1/9                                                //
// Model Implementation (CMR-CIKM14).                            //
///////////////////////////////////////////////////////////////////

#include "../utils.hpp"
#include "corpus.hpp"

using namespace std;

class CMR{
public:
    //******Model parameter needed to be specified by User*******//
    const static int niters_gibbs=1000;                          //
    const static int ncluster_u=5;                              //
    const static int ncluster_i=5;                              //
    const static int nratings = 5;                               // 
    const static int K=40;                                       //
    const static int max_words = 5000;                           //
                                                                 // 
    constexpr static double alpha=0.5/ncluster_u;                 //
    constexpr static double beta=0.5/ncluster_i;                  //
    constexpr static double eta=0.5/nratings;                    //
    constexpr static double kappa=0.5/K;                         //
    constexpr static double gamma=0.01;                          //
    ///////////////////////////////////////////////////////////////

    //*******Model Parameter needed to be learned by Model*******//
    double ** pai_uc;                                            //
    double ** pai_ic;                                            //
    double ** psai_ccr;                                           //
    double ** theta_cck;                                          //  
    double ** fai_kw;                                            //
    ///////////////////////////////////////////////////////////////
   
    int ** N_uc;
    int ** N_ic;
    int ** N_ccr;
    int ** N_cck;
    int * sumN_cc;
    int ** N_kw;
    int * sumN_k;
    int * UC_vote;
    int * IC_vote;
    int ** WK_vote;

    vector<Vote*> train_votes;
    vector<Vote*> vali_votes;
    vector<Vote*> test_votes;
    vector<Vote*>* train_votes_puser;
    vector<Vote*>* train_votes_pitem;
    map<Vote*, double> best_vali_predictions;
    hash_set<int>* adj_adv_verb_set;

    Corpus* corp;
    int n_users;
    int n_items;
    int n_words;
   
    bool restart_tag;
    char* trdata_path;
    char* vadata_path;
    char* tedata_path;
    char* model_path;
   

public:
    CMR(char* trdata_path, char* vadata_path, char* tedata_path,
            char* model_path, int tr_method, bool restart_tag) {
        this->trdata_path = trdata_path;
        this->vadata_path = vadata_path;
        this->tedata_path = tedata_path;
        this->model_path  = model_path;
        this->restart_tag = restart_tag;

        printf("CMR.\n");
        printf("Loading data.\n");
        corp = new Corpus(trdata_path, vadata_path, tedata_path, max_words);
        n_users = corp->n_users;
        n_items = corp->n_items;
        n_words = corp->n_words;
        
        train_votes_puser = new vector<Vote*>[n_users];
        train_votes_pitem = new vector<Vote*>[n_items];
        for (vector<Vote*>::iterator it = corp->TR_V->begin(); 
                it != corp->TR_V->end(); it++) {
            train_votes.push_back(*it);
            train_votes_puser[(*it)->user].push_back(*it);
            train_votes_pitem[(*it)->item].push_back(*it);
        }
        for (vector<Vote*>::iterator it = corp->TE_V->begin();
                it != corp->TE_V->end(); it++)
            test_votes.push_back(*it);
        for (vector<Vote*>::iterator it = corp->VA_V->begin();
                it != corp->VA_V->end(); it++)
            vali_votes.push_back(*it);
        printf("Number of training votes: %d, vali votes: %d, test votes: %d", train_votes.size(), vali_votes.size(), test_votes.size());

        srandom(time(0));
        initialize();
        printf("Finishing all initialization.\n");
    }

    ~CMR() {
        delete[] train_votes_puser;
        delete[] train_votes_pitem;
        
        train_votes.clear();
        vector<Vote*>(train_votes).swap(train_votes);
        vali_votes.clear();
        vector<Vote*>(vali_votes).swap(vali_votes);
        test_votes.clear();
        vector<Vote*>(test_votes).swap(test_votes);
        best_vali_predictions.clear();
        map<Vote*, double>(best_vali_predictions).swap(best_vali_predictions);
        
        delete corp;
        
        for (int u=0; u<n_users; u++)
            delete[] pai_uc[u]; 
        delete[] pai_uc;
        for (int i=0; i<n_items; i++)
            delete[] pai_ic[i];
        delete[] pai_ic;
        for (int cc=0; cc<ncluster_u*ncluster_i; cc++) {
            delete[] psai_ccr[cc];
            delete[] theta_cck[cc];
        }
        delete[] psai_ccr;
        delete[] theta_cck;
        for (int k=0; k<K; k++)
            delete[] fai_kw[k];
  
        for (int u=0; u<n_users; u++)
            delete[] N_uc[u];
        delete[] N_uc;
        for (int i=0; i<n_items; i++)
            delete[] N_ic[i];
        delete[] N_ic;
        for (int cc=0; cc<ncluster_u*ncluster_i; cc++) {
            delete[] N_ccr[cc];
            delete[] N_cck[cc];
        }
        delete[] N_ccr;
        delete[] N_cck;
        delete[] sumN_cc;
    
        for (int k=0; k<K; k++)
            delete[] N_kw[k];
        delete[] sumN_k;
        delete[] UC_vote;
        delete[] IC_vote;
        int ind = 0;
        for (vector<Vote*>::iterator it=train_votes.begin();
                it!=train_votes.end(); it++)
            delete[] WK_vote[ind++];
    }

    void initialize() {
        pai_uc = new double*[n_users];
        for (int u=0; u<n_users; u++) {
            pai_uc[u] = new double[ncluster_u];
            memset(pai_uc[u], 0, sizeof(double)*ncluster_u);
        }
        pai_ic = new double*[n_items];
        for (int i=0; i<n_items; i++) {
            pai_ic[i] = new double[ncluster_i];
            memset(pai_ic[i], 0, sizeof(double)*ncluster_i);
        }
        psai_ccr = new double*[ncluster_u*ncluster_i];
        theta_cck = new double*[ncluster_u*ncluster_i];
        for (int cc=0; cc<ncluster_u*ncluster_i; cc++) {
            psai_ccr[cc] = new double[nratings];
            memset(psai_ccr[cc], 0, sizeof(double)*nratings);
            theta_cck[cc] = new double[K];
            memset(theta_cck[cc], 0, sizeof(double)*K);
        }
        fai_kw = new double*[K];
        for (int k=0; k<K; k++) {
            fai_kw[k] = new double[n_words];
        }

        N_uc = new int*[n_users];
        for (int u=0; u<n_users; u++) {
            N_uc[u] = new int[ncluster_u];
            memset(N_uc[u], 0, sizeof(int)*ncluster_u);
        }
        N_ic = new int*[n_items];
        for (int i=0; i<n_items; i++) {
            N_ic[i] = new int[ncluster_i];
            memset(N_ic[i], 0, sizeof(int)*ncluster_i);
        }
        N_ccr = new int*[ncluster_u*ncluster_i];
        N_cck = new int*[ncluster_u*ncluster_i];
        for (int cc=0; cc<ncluster_u*ncluster_i; cc++) {
            N_ccr[cc] = new int[nratings];
            memset(N_ccr[cc], 0, sizeof(int)*nratings);
            N_cck[cc] = new int[K];
            memset(N_cck[cc], 0, sizeof(int)*K);
        }
        sumN_cc = new int[ncluster_u*ncluster_i];
        memset(sumN_cc, 0, sizeof(int)*ncluster_u*ncluster_i);
        N_kw = new int*[K];
        for (int k=0; k<K; k++) {
            N_kw[k] = new int[n_words];
            memset(N_kw[k], 0, sizeof(int)*n_words);
        }
        sumN_k = new int[K];
        memset(sumN_k, 0, sizeof(int)*K);
        UC_vote = new int[(int)train_votes.size()];
        memset(UC_vote, 0, sizeof(int)*(int)train_votes.size());
        IC_vote = new int[(int)train_votes.size()];
        memset(IC_vote, 0, sizeof(int)*(int)train_votes.size());
        WK_vote = new int*[(int)train_votes.size()];
        int ind = 0;
        for (vector<Vote*>::iterator it=train_votes.begin();
                it!=train_votes.end(); it++) {
            WK_vote[ind++] = new int[(int)(*it)->words.size()];
        }
        ind = -1;
        for (vector<Vote*>::iterator it=train_votes.begin();
                it!=train_votes.end(); it++) {
            ind++;
            if ((*it)->words.size() == 0)
                continue;
            int rating = (int)(*it)->value;
            int user = (*it)->user;
            int item = (*it)->item;
            int uc = (int)(((double)random()/RAND_MAX)*ncluster_u);
            int ic = (int)(((double)random()/RAND_MAX)*ncluster_i);
            int cc = uc*ncluster_i+ic;
            N_uc[user][uc]++;
            N_ic[item][ic]++;
            N_ccr[cc][rating-1]++;
            sumN_cc[cc]++;
            UC_vote[ind] = uc;
            IC_vote[ind] = ic;
            int ind1=0;
            for (vector<int>::iterator it1=(*it)->words.begin();
                    it1!=(*it)->words.end(); it1++) {
                int k= (int)(((double)random()/RAND_MAX)*K);
                WK_vote[ind][ind1] = k;
                N_cck[cc][k]++;
                N_kw[k][*it1]++;
                sumN_k[k]++;
                ind1++;
            } 
        }
    }
   
    void train() {
        int uc, ic, cc, k, user, item, rating, topic3dim;
        double sval, cur_train, cur_valid, cur_test, best_valid=1e5, best_rmse;
        double * P = new double[ncluster_u*ncluster_i*K];
        printf("Start training.\n");
        int ind, ind1;

        for (int iter=0; iter<niters_gibbs; iter++) {
        ind = -1;
        for (vector<Vote*>::iterator it=train_votes.begin();
                it!=train_votes.end(); it++) {
            ind++;
            rating = (int)(*it)->value;
            user = (*it)->user;
            item = (*it)->item;
            /*uc = UC_vote[ind];
            ic = IC_vote[ind];
            N_uc[user][uc]--;
            N_ic[item][ic]--;
            cc = uc*ncluster_i+ic;
            N_ccr[cc][rating-1]--;
            sumN_cc[cc]--;*/
            
            ind1=0;
            for (vector<int>::iterator it1=(*it)->words.begin();
                    it1!=(*it)->words.end(); it1++) {
                uc = UC_vote[ind];
                ic = IC_vote[ind];
                N_uc[user][uc]--;
                N_ic[item][ic]--;
                cc = uc*ncluster_i+ic;
                N_ccr[cc][rating-1]--;
                sumN_cc[cc]--;
                k = WK_vote[ind][ind1];
                N_cck[cc][k]--; 
                N_kw[k][*it1]--;
                sumN_k[k]--;
                for (uc=0; uc<ncluster_u; uc++)
                    for (ic=0; ic<ncluster_i; ic++)
                        for (k=0; k<K; k++) {
                            topic3dim = uc*ncluster_i*K+ic*K+k;
                            if (uc==0 && ic==0 && k==0)
                                P[topic3dim] = sampling(*it, uc, ic, k, *it1);
                            else
                                P[topic3dim] = P[topic3dim-1]+sampling(*it, uc, ic, k, *it1);
                            //printf("%lf ", P[topic3dim]);
                        }
                //printf("\n");
                //utils::pause();
                sval = ((double)random()/RAND_MAX)*P[ncluster_u*ncluster_i*K-1];
                //printf("sval: %lf, P[-1]: %lf\n", sval, P[ncluster_u*ncluster_i*K-1]);
                for (uc=0; uc<ncluster_u; uc++)
                    for (ic=0; ic<ncluster_i; ic++)
                        for (k=0; k<K; k++) {
                            topic3dim = uc*ncluster_i*K+ic*K+k;
                            //printf("uc: %d, ic: %d, k: %d, K: %d\n", uc, ic, k, K);
                            if (sval <= P[topic3dim])
                                goto Next;
                        }
                Next:
                //printf("ind: %d, ind1: %d\n", ind, ind1);
                //printf("uc: %d, ic: %d, k: %d\n", uc, ic, k);
                //utils::pause();
                UC_vote[ind] = uc;
                IC_vote[ind] = ic;
                N_uc[user][uc]++;
                N_ic[item][ic]++;
                cc = uc*ncluster_i+ic;
                N_ccr[cc][rating-1]++;
                sumN_cc[cc]++;
                WK_vote[ind][ind1]=k;
                N_cck[cc][k]++; 
                N_kw[k][*it1]++;
                sumN_k[k]++;
                
                ind1++;
            }
        }
        if ((iter+1) % 2 == 0) { 
            updateModelPara();
            evalRmseError(cur_train, cur_valid, cur_test);
            if (cur_valid < best_valid) {
                best_valid = cur_valid;
                best_rmse = cur_test;
                for (vector<Vote*>::iterator it=train_votes.begin();
                        it!=train_votes.end(); it++)
                    best_vali_predictions[*it] = predRating(*it);
                for (vector<Vote*>::iterator it=vali_votes.begin();
                        it!=vali_votes.end(); it++)
                    best_vali_predictions[*it] = predRating(*it);
                for (vector<Vote*>::iterator it=test_votes.begin();
                        it!=test_votes.end(); it++)
                    best_vali_predictions[*it] = predRating(*it);
            }
            printf("Current iteration: %d, Train RMSE=%.6f, ", iter+1, cur_train);
            printf("Valid RMSE=%.6f, Test RMSE=%.6f;", cur_valid, cur_test);
            printf("Best valid RMSE=%.6f, test RMSE=%.6f!\n", best_valid, best_rmse);
        }
        }
        printf("Finish training.\n");
        updateModelPara();
        evalRmseError(cur_train, cur_valid, cur_test);
        if (cur_valid < best_valid) {
            best_valid = cur_valid;
            best_rmse = cur_test;
            for (vector<Vote*>::iterator it=train_votes.begin();
                    it!=train_votes.end(); it++)
                best_vali_predictions[*it] = predRating(*it);
            for (vector<Vote*>::iterator it=vali_votes.begin();
                    it!=vali_votes.end(); it++)
                best_vali_predictions[*it] = predRating(*it);
            for (vector<Vote*>::iterator it=test_votes.begin();
                    it!=test_votes.end(); it++)
                best_vali_predictions[*it] = predRating(*it);
        }
        printf("Train RMSE=%.6f, Best valid RMSE=%.6f, test RMSE=%.6f!\n", cur_train,  best_valid, best_rmse);
        delete[] P;
    }

    double sampling(Vote * vote, int uc, int ic, int topic, int wid) {
        double sval;
        int user = vote->user;
        int item = vote->item;
        int rating = (int)vote->value;
        int cc = uc*ncluster_i+ic;
        sval = alpha+N_uc[user][uc];
        sval *= beta+N_ic[item][ic];
        sval *= (eta+N_ccr[cc][rating-1])/(nratings*eta+sumN_cc[cc]);
        sval *= (kappa+N_cck[cc][topic])/(K*kappa+sumN_cc[cc]);
        sval *= (gamma+N_kw[topic][wid])/(n_words*gamma+sumN_k[topic]);
        return sval;
    }

    void updateModelPara() {
        int cc;
        double Calpha=ncluster_u*alpha, Cbeta=ncluster_i*beta, Reta=nratings*eta, Kkappa=K*kappa, Wgamma=n_words*gamma;
        for (int u=0; u<n_users; u++)
            for (int uc=0; uc<ncluster_u; uc++)
                pai_uc[u][uc] = (alpha+N_uc[u][uc])/(Calpha+train_votes_puser[u].size());
        for (int i=0; i<n_items; i++)
            for (int ic=0; ic<ncluster_i; ic++)
                pai_ic[i][ic] = (beta+N_ic[i][ic])/(Cbeta+train_votes_pitem[i].size());
        for (int uc=0; uc<ncluster_u; uc++)
            for (int ic=0; ic<ncluster_i; ic++) {
                cc = uc*ncluster_i+ic;
                for (int r=0; r<nratings; r++)
                    psai_ccr[cc][r] = (eta+N_ccr[cc][r])/(Reta+sumN_cc[cc]);
                for (int k=0; k<K; k++)
                    theta_cck[cc][k] = (kappa+N_cck[cc][k])/(Kkappa+sumN_cc[cc]);
            }
        for (int k=0; k<K; k++)
            for (int w=0; w<n_words; w++)
                fai_kw[k][w] = (gamma+N_kw[k][w])/(Wgamma+sumN_k[k]);
    }

    double predRating(Vote * vote) {
        int cc;
        int user = vote->user;
        int item = vote->item;
        double rating = 0;
        for (int uc=0; uc<ncluster_u; uc++)
            for (int ic=0; ic<ncluster_i; ic++) {
                cc = uc*ncluster_i+ic;
                for (int r=0; r<nratings; r++)
                    rating+= (r+1)*pai_uc[user][uc]*pai_ic[item][ic]*psai_ccr[cc][r];
            }
        return rating;   
    }

    double predRating1(Vote * vote) {
        int user = vote->user;
        int item = vote->item;
        double * rating_prob = new double[nratings];
        memset(rating_prob, 0, sizeof(double)*nratings);

        for (int uc=0; uc<ncluster_u; uc++)
            for (int ic=0; ic<ncluster_i; ic++) {
                cc = uc*ncluster_i+ic;
                for (int r=0; r<nratings; r++)
                    rating_prob[r] += pai_uc[user][uc]*pai_ic[item][ic]*psai_ccr[cc][r];
            }
        double rating = 0.0;
        int max_ind=0;
        for (int r=1; r<nratings; r++)
            if (rating_prob[r]> rating_prob[max_ind]) {
                rating = r + 1.0;
                max_ind = r;
            }
        return rating;
    }

    void evalRmseError(double& train, double& valid, double& test) {
        train = 0.0, valid = 0.0, test = 0.0;
        for (vector<Vote*>::iterator it = train_votes.begin();
                it != train_votes.end(); it++)
            train += utils::square(predRating(*it) - (*it)->value);
        for (vector<Vote*>::iterator it = vali_votes.begin();
                it != vali_votes.end(); it++)
            valid += utils::square(predRating(*it) - (*it)->value);
        for (vector<Vote*>::iterator it = test_votes.begin();
                it != test_votes.end(); it++)
            test += utils::square(predRating(*it) - (*it)->value);
        //cout << "Train: " << train << ", Size: " << train_votes.size() << endl;
        train = sqrt(train/train_votes.size());
        //cout << "Valid: " << valid << ", Size: " << vali_votes.size() << endl;
        valid = sqrt(valid/vali_votes.size());
        //cout << "Test: " << test << ", Size: " << test_votes.size() << endl;
        test = sqrt(test/test_votes.size());
    } 
   
    void submitPredictions(char* submission_path) {
        FILE* f = utils::fopen_(submission_path, "w");
        for (vector<Vote*>::iterator it = corp->TE_V->begin();
                it != corp->TE_V->end(); it ++)
            fprintf(f, "%s %s %.6f\n", corp->ruser_ids[(*it)->user].c_str(),
                    corp->ritem_ids[(*it)->item].c_str(),
                    best_vali_predictions[*it]);
        fclose(f);
    }
};

