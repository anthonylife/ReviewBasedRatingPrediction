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
// Date: 2014/12/16                                              //
// Model Implementation (JMARS-KDD14).                           //
///////////////////////////////////////////////////////////////////

#include "../utils.hpp"
#include "../LBFGSCPP.h"
#include "corpus.hpp"

class JMARS{
public:
    //******Model parameter needed to be specified by User*******//
    const static int niters_gibbs_em=500;                         //
    const static int niters_gibbs_e = 1;                          //
    const static int niters_gibbs_m = 10;                        //
                                                                 //
    const static int ndim = 40;   // latant factor dimension     //
    const static int naspects = 5;                               // 
    constexpr static double lambda_u = 0.1;                          //
    constexpr static double lambda_i = 1;                            //
    constexpr static double psai_u = 0.1;                            //
    constexpr static double psai_i = 0.1;                            //
    constexpr static double sigma_u = 0.1;                           //
    constexpr static double sigma_i = 0.1;                           //
    constexpr static double psai_tm = 0.1;                           //
                                                                 // 
    constexpr static double alpha  = 5;                              // 
    const static int max_words = 8000;                           //
    constexpr static double gamma = 1.0;                             //
    constexpr static double eta_bw = 0.001;                          //
    constexpr static double eta_sw = 0.001;                          //
    constexpr static double eta_aw = 0.001;                          //
    constexpr static double eta_mw = 0.001;                          //
    constexpr static double logit_c = 1;                             //
    constexpr static double logit_b = 3;                             //
    const static int eta_scale = 10;                             //
    const static int switch_num = 5;                             //
    const static int sentiment_num = 2;                          //
    ///////////////////////////////////////////////////////////////

    //*******Model Parameter needed to be learned by Model*******//
    double ** theta_user;       // user latent factor            //
    double ** theta_item;       // item latent factor            //
    double ** gamma_user;       // user topic factor             //
    double ** gamma_item;       // item topic factor             //
    double * b_user;            // user rating bias              //
    double * b_item;            // item rating bias              //
    double * mu;                // total rating average          //
    double *** aspect_mat;      // aspect matrix                 //
    ///////////////////////////////////////////////////////////////
    
    int NW;         // total number of parameters
    double * W;     // continuous version of all leared para
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
    int n_pos;
   
    int * N_y;
    int * N_y0w;
    int ** VW_y;
    int ** N_y1sw;
    int * N_y1s;
    int ** VW_s;
    int ** VC_y1s;
    int ** N_y2aw;
    int * N_y2a;
    int *** VC_y2sa;
    int ** N_y3aw;
    int * N_y3a;
    int ** VC_y3a;
    int ** VW_a;
    int ** N_y4mw;
    int * N_y4m;
    double sum_eta_y2sw;
    double sum_eta_y3sw;
    double sum_eta_y4aw;

    int tr_method;
    bool restart_tag;
    char* trdata_path;
    char* vadata_path;
    char* tedata_path;
    char* model_path;
   
    double * g_W;
    double ** gtheta_user; 
    double ** gtheta_item;
    double ** ggamma_user;
    double ** ggamma_item; 
    double * gb_user;
    double * gb_item; 
    double * gmu;
    double *** gaspect_mat; 
    CLBFGSCPP *m_lbfgs;
    double * P;
    double ** exp_gamma_u;
    double ** exp_gamma_i;
    double * aspect_dis;
    double ** aggregate_mat;
    double *** aspect_agg_mat;
    double * sentiment_val;

    double * cache_topic_vec; 
    double * cache_user_vec; 
    double * cache_item_vec;
    double ** cache_ui_mat;

public:
    JMARS(char* trdata_path, char* vadata_path, char* tedata_path,
            char* model_path, int tr_method, bool restart_tag) {
        this->trdata_path = trdata_path;
        this->vadata_path = vadata_path;
        this->tedata_path = tedata_path;
        this->model_path  = model_path;
        this->tr_method = tr_method;
        this->restart_tag = restart_tag;

        printf("JMARS.\n");
        printf("Loading data.\n");
        corp = new Corpus(trdata_path, vadata_path, tedata_path, max_words, true);
        n_users = corp->n_users;
        n_items = corp->n_items;
        n_words = corp->n_words;
        n_pos = corp->n_pos;
        
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
       
        m_lbfgs = new CLBFGSCPP();
        if (restart_tag == true) {
            printf("Para initialization from restart.\n");
            modelParaInit();
        } else {
            printf("Para loading from trained model.\n");
            loadModelPara();
            gibbsInit();
        }
        loadAdjAdvVerbSet();
        printf("Finishing all initialization.\n");
    }

    ~JMARS() {
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
        
        delete[] theta_user;
        delete[] theta_item;
        delete[] gamma_user;
        delete[] gamma_item;
        for (int a=0; a<naspects; a++)
            delete[] aspect_mat[a];
        delete[] aspect_mat;
        delete[] W;
        delete[] gtheta_user;
        delete[] gtheta_item;
        delete[] ggamma_user;
        delete[] ggamma_item;
        for (int a=0; a<naspects; a++)
            delete[] gaspect_mat[a];
        delete[] gaspect_mat;
        delete[] g_W;
        delete corp;
        
        if (P)
            delete P;
        if (exp_gamma_u)
            utils::free_matrix(exp_gamma_u, n_users);
        if (exp_gamma_i)    
            utils::free_matrix(exp_gamma_i, n_items);
        if (aggregate_mat)
            utils::free_matrix(aggregate_mat, ndim);
        if (aspect_agg_mat) {
            for (int a=0; a<naspects; a++)
                utils::free_matrix(aspect_agg_mat[a], ndim);
            delete[] aspect_agg_mat;
        }
        delete[] aspect_dis;
        if (adj_adv_verb_set)
            delete adj_adv_verb_set;
        
        if (cache_topic_vec)
            delete[] cache_topic_vec;
        if (cache_user_vec)
            delete[] cache_user_vec;
        if (cache_item_vec)
            delete[] cache_item_vec;
        if (cache_ui_mat) {
            utils::free_matrix(cache_ui_mat, ndim);
        }
    }

    void loadAdjAdvVerbSet() {
        adj_adv_verb_set = new hash_set<int>();
        adj_adv_verb_set->insert(corp->pos_ids["JJ"]);
        adj_adv_verb_set->insert(corp->pos_ids["JJR"]);
        adj_adv_verb_set->insert(corp->pos_ids["JJS"]);
        adj_adv_verb_set->insert(corp->pos_ids["RB"]);
        adj_adv_verb_set->insert(corp->pos_ids["RBR"]);
        adj_adv_verb_set->insert(corp->pos_ids["RBS"]);
        adj_adv_verb_set->insert(corp->pos_ids["VB"]);
        adj_adv_verb_set->insert(corp->pos_ids["VBD"]);
        adj_adv_verb_set->insert(corp->pos_ids["VBG"]);
        adj_adv_verb_set->insert(corp->pos_ids["VBN"]);
        adj_adv_verb_set->insert(corp->pos_ids["VBP"]);
        adj_adv_verb_set->insert(corp->pos_ids["VBZ"]);
    }

    void modelParaInit() {
        int ind=0;
        NW = 1 + (n_users+n_items)*(ndim+1) + (n_users+n_items)*naspects + naspects*pow(ndim, 2);
        W = new double[NW];
        memset(W, 0, sizeof(double)*NW);

        mu = W + ind;
        ind += 1;
        theta_user = new double*[n_users];
        for (int u=0; u < n_users; u++) {
            theta_user[u] = W + ind;
            utils::muldimGaussrand(theta_user[u], ndim);
            ind += ndim;
        }
        theta_item = new double*[n_items];
        for (int i=0; i < n_items; i++) {
            theta_item[i] = W + ind;
            utils::muldimGaussrand(theta_item[i], ndim);
            ind += ndim;
        }
        b_user = W + ind;
        ind += n_users;
        b_item = W + ind;
        ind += n_items;

        gamma_user = new double*[n_users];
        for (int u = 0; u < n_users; u++) {
            gamma_user[u] = W + ind;
            utils::muldimGaussrand(gamma_user[u], naspects);
            ind += naspects;
        }
        gamma_item = new double*[n_items];
        for (int i=0; i < n_items; i++) {
            gamma_item[i] = W + ind;
            utils::muldimGaussrand(gamma_item[i], naspects);
            ind += naspects;
        }
        aspect_mat = new double**[naspects];
        for (int a=0; a<naspects; a++) {
            aspect_mat[a] = new double*[ndim];
            for (int j=0; j<ndim; j++) {
                aspect_mat[a][j] = W + ind;
                utils::muldimGaussrand(aspect_mat[a][j], ndim);
                ind += ndim;
            }
        }
    }

    void train() {
        sentiment_val = new double[sentiment_num];
        sentiment_val[0] = -1, sentiment_val[1] = 1;
        exp_gamma_u = utils::alloc_matrix(n_users, naspects, (double)0.0);
        exp_gamma_i = utils::alloc_matrix(n_items, naspects, (double)0.0);
        aspect_dis = new double[naspects];
        aggregate_mat = utils::alloc_matrix(ndim, ndim, (double)0.0);
        aspect_agg_mat = new double**[naspects];
        for (int a=0; a<naspects; a++)
            aspect_agg_mat[a] = utils::alloc_matrix(ndim, ndim, (double)0.0);
        cache_topic_vec = new double[naspects];
        cache_user_vec = new double[ndim];
        cache_item_vec = new double[ndim];
        cache_ui_mat = utils::alloc_matrix(ndim, ndim, (double)0.0);

        g_W = new double[NW];
        gradParaMap();
        if (tr_method == 0) {
            printf("Gibbs sampling initialization.\n");
            gibbsInit();
            gibbsEm();
        } else {
            printf("Invalid choice of learning method!\n");
            exit(1);
        }
        saveModelPara();
    }

    void gradParaMap() {
        int ind=0;

        gmu = g_W;
        ind += 1;
        gtheta_user = new double*[n_users];
        for (int u=0; u < n_users; u++) {
            gtheta_user[u] = g_W + ind;
            ind += ndim;
        }
        gtheta_item = new double*[n_items];
        for (int i=0; i < n_items; i++) {
            gtheta_item[i] = g_W + ind;
            ind += ndim;
        }
        gb_user = g_W + ind;
        ind += n_users;
        gb_item = g_W + ind;
        ind += n_items;

        ggamma_user = new double*[n_users];
        for (int u = 0; u < n_users; u++) {
            ggamma_user[u] = g_W + ind;
            ind += naspects;
        }
        ggamma_item = new double*[n_items];
        for (int i=0; i < n_items; i++) {
            ggamma_item[i] = g_W + ind;
            ind += naspects;
        }
        gaspect_mat = new double**[naspects];
        for (int a=0; a<naspects; a++) {
            gaspect_mat[a] = new double*[ndim];
            for (int j=0; j<ndim; j++) {
                gaspect_mat[a][j] = g_W + ind;
                ind += ndim;
            }
        }
    }

    void gibbsInit() {
        int user=-1, item, voteidx, switchid, idx;

        P = new double[switch_num*naspects*sentiment_num];
        N_y = new int[switch_num];
        memset(N_y, 0, sizeof(int)*switch_num);
        N_y0w = new int[n_words];
        memset(N_y0w, 0, sizeof(int)*n_words);
        VW_y = new int*[(int)train_votes.size()];
        for (int i=0; i<(int)train_votes.size(); i++) {
            VW_y[i] = new int[(int)train_votes[i]->words.size()];
            memset(VW_y[i], 0, sizeof(int)*(int)train_votes[i]->words.size());
        }
        N_y1sw = new int*[sentiment_num];
        for (int s=0; s<sentiment_num; s++) {
            N_y1sw[s] = new int[n_words];
            memset(N_y1sw[s], 0, sizeof(int)*n_words);
        }
        N_y1s = new int[sentiment_num];
        memset(N_y1s, 0, sizeof(int)*sentiment_num);
        VW_s = new int*[(int)train_votes.size()];
        for (int i=0; i<(int)train_votes.size(); i++) {
            VW_s[i] = new int[(int)train_votes[i]->words.size()];
            memset(VW_s[i], 0, sizeof(int)*(int)train_votes[i]->words.size());
        }
        VC_y1s = new int*[(int)train_votes.size()];
        for (int i=0; i<(int)train_votes.size(); i++) {
            VC_y1s[i] = new int[sentiment_num];
            memset(VC_y1s[i], 0, sizeof(int)*sentiment_num);
        }
        N_y2aw = new int*[naspects];
        for (int a=0; a<naspects; a++) {
            N_y2aw[a] = new int[n_words];
            memset(N_y2aw[a], 0, sizeof(int)*n_words);
        }
        N_y2a = new int[naspects];
        memset(N_y2a, 0, sizeof(int)*naspects);
        VC_y2sa = new int**[(int)train_votes.size()];
        for (int i=0; i<(int)train_votes.size(); i++) {
            VC_y2sa[i] = new int*[sentiment_num];
            for (int j=0; j<sentiment_num; j++) {
                VC_y2sa[i][j] = new int[naspects];
                memset(VC_y2sa[i][j], 0, sizeof(int)*naspects);
            }
        }
        N_y3aw = new int*[naspects];
        for (int a=0; a<naspects; a++) {
            N_y3aw[a] = new int[n_words];
            memset(N_y3aw[a], 0, sizeof(int)*n_words);
        }
        N_y3a = new int[naspects];
        memset(N_y3a, 0, sizeof(int)*naspects);
        VW_a = new int*[(int)train_votes.size()];
        for (int i=0; i<(int)train_votes.size(); i++) {
            VW_a[i] = new int[(int)train_votes[i]->words.size()];
            memset(VW_a[i], 0, sizeof(int)*(int)train_votes[i]->words.size());
        }
        VC_y3a = new int*[(int)train_votes.size()];
        for (int i=0; i<(int)train_votes.size(); i++) {
            VC_y3a[i] = new int[naspects];
            memset(VC_y3a[i], 0, sizeof(int)*naspects);
        }
        N_y4mw = new int*[n_items];
        for (int i=0; i<n_items; i++) {
            N_y4mw[i] = new int[n_words];
            memset(N_y4mw[i], 0, sizeof(int)*n_words);
        }
        N_y4m = new int[n_items];
        memset(N_y4m, 0, sizeof(int)*n_items);
        sum_eta_y2sw = 0.0;
        sum_eta_y3sw = 0.0;
        sum_eta_y4aw = 0.0;

        user=-1,item=-1;
        voteidx = 0;
        for (vector<Vote*>::iterator it=train_votes.begin();
                it!=train_votes.end(); it++) {
            user = (*it)->user;
            item = (*it)->item;
            idx = 0;
            for (vector<int>::iterator it1=(*it)->words.begin();
                    it1!=(*it)->words.end(); it1++) {
                switchid = (int)(((double)random()/RAND_MAX)*switch_num);
                N_y[switchid]++;
                VW_y[voteidx][idx] = switchid;
                VW_s[voteidx][idx] = -1;
                VW_a[voteidx][idx] = -1;
                if (switchid == 0)  {
                    N_y0w[*it1]++;                    
                } else if (switchid == 1) {
                    int s = (int)(((double)random()/RAND_MAX)*sentiment_num);
                    N_y1sw[s][*it1]++;
                    N_y1s[s]++;
                    VW_s[voteidx][idx] = s;
                    VC_y1s[s]++;
                } else if (switchid == 2) {
                    int a = (int)(((double)random()/RAND_MAX)*naspects);
                    int s = (int)(((double)random()/RAND_MAX)*sentiment_num);
                    N_y2aw[a][*it1]++;
                    N_y2a[a]++;
                    VW_a[voteidx][idx] = a;
                    if (adj_adv_verb_set->find((*it)->pos[idx]) != adj_adv_verb_set->end())
                        sum_eta_y2sw += eta_sw*eta_scale;
                    else
                        sum_eta_y2sw += eta_sw;
                    VW_s[voteidx][idx] = s;
                    VC_y2sa[voteidx][s][a]++;
                } else if (switchid == 3) {
                    int a = (int)(((double)random()/RAND_MAX)*naspects);
                    N_y3aw[a][*it1]++;
                    N_y3a[a]++;
                    VW_a[voteidx][idx] = a;
                    if (adj_adv_verb_set->find((*it)->pos[idx]) != adj_adv_verb_set->end())
                        sum_eta_y3sw += eta_sw*eta_scale;
                    else
                        sum_eta_y3sw += eta_sw;
                    VC_y3a[voteidx][a]++;
                } else if (switchid == 4) {
                    N_y4mw[item][*it1]++;
                    N_y4m[item]++;
                    if (adj_adv_verb_set->find((*it)->pos[idx]) != adj_adv_verb_set->end())
                        sum_eta_y4aw += eta_sw;
                    else
                        sum_eta_y4aw += eta_sw*eta_scale;
                } else {
                    printf("Invalid switchid.\n");
                    exit(1);
                }
                idx++;
            }
            voteidx++;
        }
    }

    void gibbsEm() {
        int iter;
        double train_rmse, valid_rmse, test_rmse;
        double cur_valid, best_valid, best_rmse;

        printf("Start training.\n");
        best_valid = 1e5;
        best_rmse = 1e5;
        timeval start_t, end_t;
        utils::tic(start_t);
        calcExpGamma();
        for (iter=0; iter<niters_gibbs_em; iter++) {
            for (int iter_e=0; iter_e<niters_gibbs_e; iter_e++) {
                gibbsEstep();
            }
            for (int iter_m=0; iter_m<niters_gibbs_m; iter_m++) {
                gibbsMstep();
                calcExpGamma();
            }
            evalRmseError(train_rmse, valid_rmse, test_rmse);
            printf("Current iteration: %d, Train RMSE=%.6f, ", iter+1, train_rmse);
            printf("Valid RMSE=%.6f, Test RMSE=%.6f;", valid_rmse, test_rmse);
            cur_valid = valid_rmse;
            if (cur_valid < best_valid) {
                best_valid = cur_valid;
                best_rmse = test_rmse;
                for (vector<Vote*>::iterator it=train_votes.begin();
                        it!=train_votes.end(); it++)
                    best_vali_predictions[*it] = predRating(*it, false, false);
                for (vector<Vote*>::iterator it=vali_votes.begin();
                        it!=vali_votes.end(); it++)
                    best_vali_predictions[*it] = predRating(*it, false, false);
                for (vector<Vote*>::iterator it=test_votes.begin();
                        it!=test_votes.end(); it++)
                    best_vali_predictions[*it] = predRating(*it, false, false);
            }
            
            printf("Best valid RMSE=%.6f, test RMSE=%.6f!\n", best_valid, best_rmse);
            utils::toc(start_t, end_t, false);
            utils::tic(start_t);
        }
    }

    void gibbsEstep() {
        double pred=0.0;
        int user, item, dim3todim1, voteidx;
        int yid, wid, sid, aid, posid, sampleid;
        int nsample_space = 1+sentiment_num+sentiment_num*naspects+naspects+1;

        srandom(time(0));
        voteidx = 0;
        for (vector<Vote*>::iterator it=train_votes.begin();
                it!=train_votes.end(); it++) {
            pred = (*it)->value;
            user = (*it)->user;
            item = (*it)->item;
            calcAspectDis(user, item);
            for (int i=0; i<(int)(*it)->words.size(); i++) {
                // Preprocessing
                wid = (*it)->words[i];
                posid = (*it)->pos[i];
                yid = VW_y[voteidx][i];
                sid = VW_s[voteidx][i];
                aid = VW_a[voteidx][i];
                if (yid==0) {
                    N_y0w[wid]--;
                    if (N_y0w[wid] < 0) {
                        printf("N_y0w[%d]<0\n", wid);
                        exit(1);
                    }
                } else if (yid==1) {
                    VC_y1s[voteidx][sid]--;
                    N_y1sw[sid][wid]--;
                    if (N_y1sw[sid][wid] < 0) {
                        printf("N_y1sw[%d][%d]<0\n", sid, wid);
                        exit(1);
                    }
                    N_y1s[sid]--;
                    if (N_y1s[sid] < 0) {
                        printf("N_y1s[%d]<0\n", sid);
                        exit(1);
                    }
                } else if (yid==2) {
                    N_y2aw[aid][wid]--;
                    if (N_y2aw[aid][wid] < 0) {
                        printf("N_y2aw[%d][%d]<0\n", aid, wid);
                        exit(1);
                    }
                    N_y2a[aid]--;
                    if (N_y2a[aid] < 0) {
                        printf("N_y2a[%d]<0\n", aid);
                        exit(1);
                    }
                    VC_y2sa[voteidx][sid][aid]--;
                } else if (yid==3) {
                    N_y3aw[aid][wid]--;
                    if (N_y3aw[aid][wid] < 0) {
                        printf("N_y3aw[%d][%d]<0\n", aid, wid);
                        exit(1);
                    }
                    N_y3a[aid]--;
                    if (N_y3a[aid] < 0) {
                        printf("N_y3a[%d]<0\n", aid);
                        exit(1);
                    }
                    VC_y3a[voteidx][aid]--;
                    if (VC_y3a[voteidx][aid] < 0) {
                        printf("VC_y3a[%d][%d]<0\n", voteidx, aid);
                        exit(1);
                    }
                } else if (yid==4) {
                    N_y4mw[item][wid]--;
                    if (N_y4mw[item][wid] < 0) {
                        printf("N_y4mw[%d][%d]<0\n", item, wid);
                        exit(1);
                    }
                    N_y4m[item]--;
                    if (N_y4m[item] < 0) {
                        printf("N_y4m[%d]<0\n", item);
                        exit(1);
                    }
                } else {
                    printf("Invalid yid\n");
                    exit(1);
                }

                //// Sampling
                // 1-->background word generation
                sampleid = 0;
                P[sampleid] = getSamplingVal(0, -1, -1, wid, posid, user, item);
                sampleid += 1;

                // 2--> sentiment word generation
                for (int s=0; s<sentiment_num; s++) {
                    P[sampleid]  = P[sampleid-1] + getSamplingVal(1, s, -1, wid, posid, user, item);
                    sampleid += 1;
                }

                // 3--> aspect sentiment word generation
                for (int s=0; s<sentiment_num; s++) {
                    for (int a=0; a<naspects; a++) {
                        P[sampleid]  = P[sampleid-1] + getSamplingVal(2, s, a, wid, posid, user, item);
                        sampleid += 1;
                    }
                }

                // 4--> aspect word generation
                for (int a=0; a<naspects; a++) {
                    P[sampleid]  = P[sampleid-1] + getSamplingVal(3, -1, a, wid, posid, user, item);
                    sampleid += 1;
                }

                // 5--> item-based word generation
                P[sampleid]  = P[sampleid-1] + getSamplingVal(4, -1, -1, wid, posid, user, item);

                double sval = ((double)random()/RAND_MAX*P[nsample_space-1]);
                for (sampleid=0; sampleid<nsample_space; sampleid++) {
                    if (sval < P[sampleid])
                        break;
                }
                mapDim1ToDim3(sampleid, yid, sid, aid);
                VW_y[voteidx][i]=yid;
                VW_s[voteidx][i]=sid;
                VW_a[voteidx][i]=aid;
                if (yid==0) {
                    N_y0w[wid]++;
                } else if (yid==1) {
                    VC_y1s[voteidx][sid]++;
                    N_y1sw[sid][wid]++;
                    N_y1s[sid]++;
                } else if (yid==2) {
                    VC_y2sa[voteidx][sid][aid]++;
                    N_y2aw[aid][wid]++;
                    N_y2a[aid]++;
                } else if (yid==3) {
                    VC_y3a[voteidx][aid]++;
                    N_y3aw[aid][wid]++;
                    N_y3a[aid]++;
                } else if (yid==4) {
                    N_y4mw[item][wid]++;
                    N_y4m[item]++;
                } else {
                    printf("Invalid yid\n");
                    exit(1);
                }
            }
            voteidx++;
            if (voteidx % 10000 == 0) {
                printf("\rEstep: finished scanning votes: %d", voteidx);
                fflush(stdout);
            }
        }
        printf("\n");
    }

    void gibbsMstep() {
        int user, item, voteidx;
        double rating, *aspect_rating, res_rating, prob, aggregate_setiment;
        double local_cache1, local_cache2, cache_res_rating;

        // LBFGS related variables
        int opt_size = NW;
        double * diag_ = new double[opt_size];
        double f=0, eps=1e-5, xtol=1e-16;
        int m=5, iprint[2], iflag[1];
        bool diagco = false;
        iprint[0] = -1; iprint[1] = 0;
        iflag[0] = 0;

        aspect_rating = new double[naspects];
        memset(g_W, 0, sizeof(double)*opt_size);

        // Adding regularization terms to objective value.
        for (int u=0; u<n_users; u++) {
            for (int k=0; k<ndim; k++) {
                f+=psai_u*pow(theta_user[u][k], 2);
                gtheta_user[u][k] = psai_u*theta_user[u][k];
            }
            for (int a=0; a<naspects; a++) {
                f+=lambda_u*pow(gamma_user[u][a], 2);
                ggamma_user[u][a] = lambda_u*gamma_user[u][a];
            }
            f+=sigma_u*pow(b_user[u], 2);
            gb_user[u] = sigma_u*b_user[u];
        }
        for (int i=0; i<n_items; i++) {
            for (int k=0; k<ndim; k++) {
                f+=psai_i*pow(theta_item[i][k], 2);
                gtheta_item[i][k] = psai_i*theta_item[i][k];
            }
            for (int a=0; a<naspects; a++) {
                f+=lambda_i*pow(gamma_item[i][a], 2);
                ggamma_item[i][a] = lambda_i*gamma_item[i][a];
            }
            f+=sigma_i*pow(b_item[i], 2);
            gb_item[i] = sigma_i*b_item[i];
        }
        for (int a=0; a<naspects; a++) {
            for (int k1=0; k1<ndim; k1++) {
                for (int k2=0; k2<ndim; k2++) {
                    f+=psai_tm*pow(aspect_mat[a][k1][k2], 2);
                    gaspect_mat[a][k1][k2] = psai_tm*aspect_mat[a][k1][k2];
                }
            }
        }
        f *= 0.5;

        // Review text
        voteidx = 0;
        for (vector<Vote*>::iterator it=train_votes.begin();
                it!=train_votes.end(); it++) {
            user = (*it)->user;
            item = (*it)->item;
            calcAspectDis(user, item);
            calcAggregateMatrix();
            calcAspectAggregateMatrix(user, item);
            calcCacheTopicVec(user, item);
            calcOuterProductOfUserItem(user, item);

            // Divide the objective function into five parts (Four parts
            //   shown in orginal paper and the left part is regularization terms)
            // --> 1 <-- : total rating error from user behaviors
            rating = predRating(*it, true, true);
            aggregate_setiment = predAggregateSentiment(user, item, true);
            res_rating = rating - (*it)->value;
            cache_res_rating = 2*alpha*res_rating;
            f += alpha*pow(res_rating, 2);
            for (int k1=0; k1<ndim; k1++) {
                cache_user_vec[k1] = 0;
                cache_item_vec[k1] = 0;
                for (int k2=0; k2<ndim; k2++) {
                    cache_item_vec[k1] += aggregate_mat[k1][k2]*theta_item[item][k2];
                    cache_user_vec[k1] += theta_user[user][k2]*aggregate_mat[k2][k1];
                }
                gtheta_user[user][k1] += cache_res_rating*cache_item_vec[k1];
                gtheta_item[item][k1] += cache_res_rating*cache_user_vec[k1];
            }
            gb_user[user] += cache_res_rating; 
            gb_item[item] += cache_res_rating;
            *gmu += cache_res_rating;
            for (int a=0; a<naspects; a++) {
                local_cache1 = cache_res_rating*cache_topic_vec[a];
                ggamma_user[user][a] += local_cache1;
                ggamma_item[item][a] += local_cache1;
                for (int k1=0; k1<ndim; k1++)
                    for (int k2=0; k2<ndim; k2++)
                        gaspect_mat[a][k1][k2] += cache_res_rating*aspect_dis[a]*cache_ui_mat[k1][k2];
            }

            // --> 2 <-- : total rating error from reviews
            local_cache1 = 0.0;
            for (int s=0; s<sentiment_num; s++) {
                prob = calcLogitProb(sentiment_val[s], aggregate_setiment);
                local_cache1 += VC_y1s[voteidx][s]*(1-prob)*sentiment_val[s]*logit_c;
                f-=VC_y1s[voteidx][s]*log(prob);
            }
            for (int k1=0; k1<ndim; k1++) {
                gtheta_user[user][k1] -= local_cache1*cache_item_vec[k1];
                gtheta_item[item][k1] -= local_cache1*cache_user_vec[k1];
            }
            gb_user[user] -= local_cache1; 
            gb_item[item] -= local_cache1;
            *gmu -= local_cache1;
            for (int a=0; a<naspects; a++) {
                local_cache2 = local_cache1*cache_topic_vec[a];
                ggamma_user[user][a] -= local_cache2;
                ggamma_item[item][a] -= local_cache2;
                for (int k1=0; k1<ndim; k1++)
                    for (int k2=0; k2<ndim; k2++)
                        gaspect_mat[a][k1][k2] -= local_cache1*aspect_dis[a]*cache_ui_mat[k1][k2];
            }
            
            // --> 3 <-- : aspect rating error from reviews
            for (int a=0; a<naspects; a++)
                aspect_rating[a] = predAspectSentiment(a, user, item);
            for (int a=0; a<naspects; a++) {
                local_cache1 = 0.0;
                for (int s=0; s<sentiment_num; s++) {
                    prob = calcLogitProb(sentiment_val[s], aspect_rating[a]);
                    local_cache1 += VC_y2sa[voteidx][s][a]*(1-prob)*sentiment_val[s]*logit_c;
                    f-=VC_y2sa[voteidx][s][a]*log(prob);
                }
                for (int k1=0; k1<ndim; k1++) {
                    cache_user_vec[k1] = 0;
                    cache_item_vec[k1] = 0;
                    for (int k2=0; k2<ndim; k2++) {
                        cache_item_vec[k1] += aspect_mat[a][k1][k2]*theta_item[item][k2];
                        cache_user_vec[k1] += theta_user[user][k2]*aspect_mat[a][k2][k1];
                    }
                    gtheta_user[user][k1] -= local_cache1*cache_item_vec[k1];
                    gtheta_item[item][k1] -= local_cache1*cache_user_vec[k1];
                }
                gb_user[user] -= local_cache1;
                gb_item[item] -= local_cache1;
                *gmu -= local_cache1;
                for (int k1=0; k1<ndim; k1++)
                    for (int k2=0; k2<ndim; k2++)
                        gaspect_mat[a][k1][k2] -= local_cache1*cache_ui_mat[k1][k2];
            }
            
            // --> 4 <-- : content generated by item soly
            local_cache1 = 0.0;
            for (int a=0; a<naspects; a++) {
                local_cache1 += VC_y3a[voteidx][a]*aspect_dis[a];
                f-=VC_y3a[voteidx][a]*log(aspect_dis[a]);
            }
            for (int a=0; a<naspects; a++) {
                ggamma_user[user][a] += local_cache1-VC_y3a[voteidx][a];
                ggamma_item[item][a] += local_cache1-VC_y3a[voteidx][a];
            }
            voteidx++;
        }
       
        // LBFGS 
        m_lbfgs->lbfgs(opt_size, m, W, f, g_W, diagco, diag_, iprint, eps, xtol, iflag);
        
        delete[] aspect_rating;
        delete[] diag_;
    }

    double getSamplingVal(int y, int s, int a, int wid, int pos, int user, int item) {
        double sval, rating;

        sval = N_y[y]+gamma;
        if (y==0) {
            sval = sval*(N_y0w[wid]+eta_bw)/(N_y[0]+n_words*eta_bw);
        } else if (y==1) {
            if (adj_adv_verb_set->find(pos) != adj_adv_verb_set->end())
                sval *= (N_y1sw[s][wid]+eta_sw*eta_scale)/(N_y1s[s]+n_words*eta_sw);
            else 
                sval *= (N_y1sw[s][wid]+eta_sw)/(N_y1s[s]+n_words*eta_sw);
            rating = predAggregateSentiment(user, item, true);
            if (s == 0)
                sval *= calcLogitProb(-1, rating);
            else if (s == 1)
                sval *= calcLogitProb(1, rating);
            else {
                printf("Invalid s: %d\n, s");
                exit(1);
            }
        } else if (y==2) {
            if (adj_adv_verb_set->find(pos) != adj_adv_verb_set->end())
                sval *= (N_y2aw[a][wid]+eta_sw*eta_scale)/(N_y2a[a]+n_words*eta_sw);
            else 
                sval *= (N_y2aw[a][wid]+eta_sw)/(N_y2a[a]+n_words*eta_sw);
            sval *= aspect_dis[a];
            rating = predAspectSentiment(a, user, item);
            if (s == 0)
                sval *= calcLogitProb(-1, rating);
            else if (s == 1)
                sval *= calcLogitProb(1, rating);
            else {
                printf("Invalid s\n");
                exit(1);
            }
        } else if (y==3) {
            if (adj_adv_verb_set->find(pos) != adj_adv_verb_set->end())
                sval *= (N_y3aw[a][wid]+eta_aw)/(N_y2a[a]+n_words*eta_aw);
            else
                sval *= (N_y3aw[a][wid]+eta_aw*eta_scale)/(N_y2a[a]+n_words*eta_aw);
            sval *= aspect_dis[a];
        } else if (y==4) {
            sval *= (N_y4mw[item][wid]+eta_mw)/(N_y4m[item]+n_words*eta_mw);
        }
        if (sval < 0)
            return 0;
        return sval;
    }

    inline void mapDim1ToDim3(int sampleid, int &y, int &s, int &a) {
        int ss = sampleid;
        if (sampleid==0) {
            y=0, s=-1, a=-1;
            return;
        }
        sampleid -= 1;
        if (sampleid<sentiment_num) {
            y=1, s=sampleid, a=-1;
            return;
        }
        sampleid -= sentiment_num;
        if (sampleid<naspects*sentiment_num) {
            y=2, s=sampleid/naspects, a=sampleid%naspects;
            return;
        }
        sampleid -= naspects*sentiment_num;
        if (sampleid<naspects) {
            y=3, s=-1, a=sampleid;
            return;
        }
        sampleid -= naspects;
        if (sampleid == 0) {
            y=4, s=-1, a=-1;
            return;
        } else {
            printf("Error in sampleid: %d, current sampleid: %d\n", ss, sampleid);
            exit(1);
        }
    }

    void calcExpGamma() {
        for (int u=0; u<n_users; u++)  
            for (int k=0; k<naspects; k++)
                exp_gamma_u[u][k] = exp(gamma_user[u][k]);
        for (int i=0; i<n_items; i++)
            for (int k=0; k<naspects; k++)
                exp_gamma_i[i][k] = exp(gamma_item[i][k]);
    }

    inline double calcLogitProb(int s, double r) {
        return 1.0/(1+exp(-s*(logit_c*r-logit_b)));
    }

    void calcAspectDis(int user, int item) {
        double constant=0.0;
        for (int a=0; a<naspects; a++) {
            aspect_dis[a] = exp_gamma_u[user][a]*exp_gamma_i[item][a];
            constant += aspect_dis[a];
        }
        for (int a=0; a<naspects; a++)
            aspect_dis[a] = aspect_dis[a]/constant;
    }

    void calcAggregateMatrix() {
        for (int k1=0; k1<ndim; k1++) {
            for (int k2=0; k2<ndim; k2++) {
                aggregate_mat[k1][k2] = aspect_dis[0]*aspect_mat[0][k1][k2];
                for (int a=1; a<naspects; a++)
                    aggregate_mat[k1][k2] += aspect_dis[a]*aspect_mat[a][k1][k2];
            }
        }
    }
   
    void calcAspectAggregateMatrix(int user, int item) {
        for (int a=0; a<naspects; a++) {
            utils::set_matrix(aspect_agg_mat[a], ndim, ndim, 0);
            for (int a1=0; a1<naspects; a1++) {
                if (a1==a)
                    break;
                for (int k1=0; k1<ndim; k1++)
                    for (int k2=0; k2<ndim; k2++)
                        aspect_agg_mat[a][k1][k2] += aspect_dis[a1]*aspect_mat[a][k1][k2]-aspect_mat[a1][k1][k2];
            }
            for (int k1=0; k1<ndim; k1++)
                for (int k2=0; k2<ndim; k2++)
                    aspect_agg_mat[a][k1][k2] *= aspect_dis[a];
        }
    }

    void calcCacheTopicVec(int user, int item) {
        memset(cache_topic_vec, 0, sizeof(double)*naspects);
        for (int a=0; a<naspects; a++)
            for (int k1=0; k1<ndim; k1++)
                for (int k2=0; k2<ndim; k2++)
                    cache_topic_vec[a] += theta_user[user][k1]*aspect_agg_mat[a][k1][k2]*theta_item[item][k2];
    } 

    void calcOuterProductOfUserItem(int user, int item) {
        for (int k1=0; k1<ndim; k1++)
            for (int k2=0; k2<ndim; k2++)
                cache_ui_mat[k1][k2] = theta_user[user][k1]*theta_user[user][k2];
    }

    double predAspectSentiment(int a, int user, int item) {
        double sentiment = utils::vecMatVecDot(theta_user[user], aspect_mat[a], theta_item[item], ndim, ndim)+b_user[user]+b_item[item]+*mu; 
        return sentiment;
    }
    
    double predAggregateSentiment(int user, int item, bool aspectdis_tag) {
        double sentiment = 0.0;
        if (!aspectdis_tag) {
            calcAspectDis(user, item);
        }
        for (int a=0; a<naspects; a++) {
            sentiment += aspect_dis[a]*predAspectSentiment(a, user, item);
        }
        return sentiment;
    }

    double predRating(Vote *v, bool aspectdis_tag, bool aggregatemat_tag) {
        double rating = 0.0;
        int user = v->user;
        int item = v->item;
        if (!aspectdis_tag)
            calcAspectDis(user, item);
        if (!aggregatemat_tag)
            calcAggregateMatrix();
        rating = utils::vecMatVecDot(theta_user[user], aggregate_mat, theta_item[item], ndim, ndim)+b_user[user]+b_item[item]+*mu;
        return rating; 
    }
    
    void evalRmseError(double& train, double& valid, double& test) {
        train = 0.0, valid = 0.0, test = 0.0;
        for (vector<Vote*>::iterator it = train_votes.begin();
                it != train_votes.end(); it++)
            train += utils::square(predRating(*it, false, false) - (*it)->value);
        for (vector<Vote*>::iterator it = vali_votes.begin();
                it != vali_votes.end(); it++)
            valid += utils::square(predRating(*it, false, false) - (*it)->value);
        for (vector<Vote*>::iterator it = test_votes.begin();
                it != test_votes.end(); it++)
            test += utils::square(predRating(*it, false, false) - (*it)->value);
        //cout << "Train: " << train << ", Size: " << train_votes.size() << endl;
        train = sqrt(train/train_votes.size());
        //cout << "Valid: " << valid << ", Size: " << vali_votes.size() << endl;
        valid = sqrt(valid/vali_votes.size());
        //cout << "Test: " << test << ", Size: " << test_votes.size() << endl;
        test = sqrt(test/test_votes.size());
    } 
   
    void outputSentimentWords(int topk) {
        int voteidx, yid, sid, wid;
        int * pos_words_cnt = new int[n_words];
        memset(pos_words_cnt, 0, sizeof(int)*n_words);
        int * neg_words_cnt = new int[n_words];
        memset(neg_words_cnt, 0, sizeof(int)*n_words);

        voteidx = 0;
        for (vector<Vote*>::iterator it=train_votes.begin();
                it!=train_votes.end(); it++) {
            for (int i=0; i<(int)(*it)->words.size(); i++) {
                wid = (*it)->words[i];
                yid = VW_y[voteidx][i];
                sid = VW_s[voteidx][i];
                if (yid==1) {
                    if (sid==0) {
                        neg_words_cnt[wid]++;
                        pos_words_cnt[wid]--;
                    } else if (sid==1) {
                        neg_words_cnt[wid]--;
                        pos_words_cnt[wid]++;
                    }
                }
            }
        }

        vector<Cword>* results = new vector<Cword>();
        for (int w=0; w<n_words; w++) {
            Cword cword;
            cword.wid = w;
            cword.wcnt = pos_words_cnt[w];
            results->push_back(cword);
        }
        sort(results->begin(), results->end(), utils::greaterCmp1);
        printf("Top %d positive words--> ", topk);
        int i=0;
        for (vector<Cword>::iterator it=results->begin();
                it!=results->end(); it++) {
            printf("%s:%d ", corp->rword_ids[it->wid].c_str(), it->wcnt);
            i++;
            if (i == topk)
                break;
        }
        printf("||||||\n");
        delete results;
        
        results = new vector<Cword>();
        for (int w=0; w<n_words; w++) {
            Cword cword;
            cword.wid = w;
            cword.wcnt = neg_words_cnt[w];
            results->push_back(cword);
        }
        sort(results->begin(), results->end(), utils::greaterCmp1);
        printf("Top %d negative words--> ", topk);
        for (vector<Cword>::iterator it=results->begin();
                it!=results->end(); it++) {
            printf("%s:%d ", corp->rword_ids[it->wid].c_str(), it->wcnt);
            i++;
            if (i == topk)
                break;
        }
        printf("||||||\n");
        delete results;
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
    
    void saveModelPara() {
        FILE* f = utils::fopen_(model_path, "w");
        fwrite(W, sizeof(double), NW, f);
        fclose(f);
    }
    
    void loadModelPara() {
        int ind = 0;
        W = new double[NW];
        FILE* f = utils::fopen_(model_path, "r");
        utils::fread_(W, sizeof(double), NW, f);
        fclose(f);

        mu = W+ind;
        ind += 1;
        theta_user = new double*[n_users];
        for (int u=0; u < n_users; u++) {
            theta_user[u] = W + ind;
            ind += ndim;
        }
        theta_item = new double*[n_items];
        for (int i=0; i < n_items; i++) {
            theta_item[i] = W + ind;
            ind += ndim;
        }
        b_user = W + ind;
        ind += n_users;
        b_item = W + ind;
        ind += n_items;

        gamma_user = new double*[n_users];
        for (int u = 0; u < n_users; u++) {
            gamma_user[u] = W + ind;
            ind += naspects;
        }
        gamma_item = new double*[n_items];
        for (int i=0; i < n_items; i++) {
            gamma_item[i] = W + ind;
            ind += naspects;
        }
        aspect_mat = new double**[naspects];
        for (int a=0; a<naspects; a++) {
            aspect_mat[a] = new double*[ndim];
            for (int j=0; j<ndim; j++) {
                aspect_mat[a][j] = W + ind;
                ind += ndim;
            }
        }
    }
};

