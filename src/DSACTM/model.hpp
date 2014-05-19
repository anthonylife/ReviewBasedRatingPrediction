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
// Date: 2014/5/14                                               //
// Model Implementation (DSACTM).                                //
///////////////////////////////////////////////////////////////////

#include "utils.hpp"
#include "corpus.hpp"

//#define OL_M_T
//#define OL_S_T
//#define OL_I_T
#define OL_DEBUG
#define CD_DEBUG

using namespace std;
using namespace __gnu_cxx;

class DSACTM{
    //******Model parameter needed to be specified by User*******//
    /// Method control variable                                  //
    int niters;                                                  //
    int inter_niters;       // number of inter iterations        //
    double delta;           // convergence value                 //
                                                                 // 
    /// Online learning                                          //
    double lr;              // learning rate for online learning //
    int minibatch;                                               //
    // truncated gradient                                        //
    int truncated_k;                                             //
    double truncated_theta;                                      //
                                                                 //
    /// Hyper-parameter setting                                  //
    int K;                  // latant factor dimension           //
    double lambda_u;        // laplace hyper for user topic      //
    double lambda_i;        // laplace hyper for item topic      //
    double lambda_b;        // laplace hyper for backgroud topic //
    double psai_u;          // gaussian hyper for user factor    //
    double psai_i;          // gaussian hyper for item factor    //
    double sigma_u;         // gaussian hyper for user bias      //
    double sigma_i;         // gaussian hyper for item bias      //
    double sigma_a;         // gaussian hyper for average        //
    int max_words;          // maximal number of words used      //
    double alpha;           // relative weight of two objs       //
    ///////////////////////////////////////////////////////////////

    //*******Model Parameter needed to be learned by Model*******//
    double ** theta_user;       // user latent factor            //
    double ** theta_item;       // item latent factor            //
    double ** gamma_user;       // user topic factor             //
    double ** gamma_item;       // item topic factor             //
    double * b_user;            // user rating bias              //
    double * b_item;            // item rating bias              //
    double * mu;                // total rating average          //
    double ** topic_words;      // word distribution of topics   //
    double * background_topic;  // backgroud topic distribution  //
    double ** vb_word_topics;   // topic distribution of word    //
    ///////////////////////////////////////////////////////////////

    int NW;         // total number of parameters
    double * W;     // continuous version of all leared para
    vector<Vote*> train_votes;
    vector<Vote*> vali_votes;
    vector<Vote*> test_votes;
    vector<Vote*>* train_votes_puser;
    vector<Vote*>* train_votes_pitem;
  
    map<int, int> ntraining_puser;
    map<int, int> ntraining_pitem;
    map<Vote*, double> best_vali_predictions;

    Corpus* corp;
    int n_users;
    int n_items;
    int n_words;

    int tr_method;
    bool restart_tag;
    char* trdata_path;
    char* vadata_path;
    char* tedata_path;
    char* model_path;

public:
    DSACTM(char* trdata_path, char* vadata_path, char* tedata_path,
            char* model_path, int niters, double delta, int minibatch, double lr,
            int K, double lambda_u, double lambda_i, double lambda_b,
            double psai_u, double psai_i, double sigma_u, double sigma_i,
            double sigma_a, int max_words, double alpha, int inter_niters, 
            int truncated_k, double truncated_theta, int tr_method, bool restart_tag){
        this->trdata_path = trdata_path;
        this->vadata_path = vadata_path;
        this->tedata_path = tedata_path;
        this->model_path  = model_path;
        this->niters      = niters;
        this->delta       = delta;
        this->minibatch   = minibatch;
        this->lr          = lr;
        this->K           = K;
        this->lambda_u    = lambda_u;
        this->lambda_i    = lambda_i;
        this->lambda_b    = lambda_b;
        this->psai_u      = psai_u;
        this->psai_i      = psai_i;
        this->sigma_u     = sigma_u;
        this->sigma_i     = sigma_i;
        this->sigma_a     = sigma_a;
        this->max_words   = max_words;
        this->alpha       = alpha;
        this->inter_niters= inter_niters;
        this->truncated_k = truncated_k;                 // [5, 30]
        this->truncated_theta = truncated_theta;
        this->tr_method   = tr_method;
        this->restart_tag = restart_tag;

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
            if (ntraining_puser.find((*it)->user) == ntraining_puser.end())
                ntraining_puser[(*it)->user] = 0;
            if (ntraining_pitem.find((*it)->item) == ntraining_pitem.end())
                ntraining_pitem[(*it)->item] = 0;
            ntraining_puser[(*it)->user] ++;
            ntraining_pitem[(*it)->item] ++;
        }
        for (vector<Vote*>::iterator it = corp->TE_V->begin();
                it != corp->TE_V->end(); it++)
            test_votes.push_back(*it);
        for (vector<Vote*>::iterator it = corp->VA_V->begin();
                it != corp->VA_V->end(); it++)
            vali_votes.push_back(*it);
        
        if (restart_tag == true) {
            printf("Para initialization from restart.\n");
            modelParaInit();
        } else {
            printf("Para loading from trained model.\n");
            loadModelPara();
        }
        printf("Finishing all initialization.\n");
    }
    
    ~DSACTM() {
        delete corp;
        delete W;
        delete[] train_votes_puser;
        delete[] train_votes_pitem;
        
        if (theta_user) {
            delete[] theta_user;
            delete[] theta_item;
            delete[] gamma_user;
            delete[] gamma_item;
            delete[] topic_words;
            delete[] vb_word_topics;
            delete mu;
            delete b_user;
            delete b_item;
            delete background_topic;
        }

        train_votes.clear();
        vector<Vote*>(train_votes).swap(train_votes);
        vali_votes.clear();
        vector<Vote*>(vali_votes).swap(vali_votes);
        test_votes.clear();
        vector<Vote*>(test_votes).swap(test_votes);
        ntraining_puser.clear();
        map<int, int>(ntraining_puser).swap(ntraining_puser);
        ntraining_pitem.clear();
        map<int, int>(ntraining_pitem).swap(ntraining_pitem);
        best_vali_predictions.clear();
        map<Vote*, double>(best_vali_predictions).swap(best_vali_predictions);
    }

    void modelParaInit() {
        // total number of paramters to be learned
        NW = 1 + (n_users+n_items)*(K+1) + (n_users+n_items+1)*K + 2*K*n_words;
        W = new double[NW];
        memset(W, 0, sizeof(double)*NW);

        allocMemForPara(W, &mu, &b_user, &b_item, &theta_user, &theta_item,
                &gamma_user, &gamma_item, &background_topic, &topic_words,
                &vb_word_topics);

        // set mu to the average
        for (vector<Vote*>::iterator it=train_votes.begin();
                it!=train_votes.end(); it++)
            *mu += (*it)->value;
        *mu /= train_votes.size();
        
        // set user and item rating bias
        for (vector<Vote*>::iterator it=train_votes.begin();
                it!=train_votes.end(); it++) {
            b_user[(*it)->user] += (*it)->value - *mu;
            b_item[(*it)->item] += (*it)->value - *mu;
        }
        for (int u=0; u<n_users; u++)
            b_user[u] /= train_votes_puser[u].size();
        for (int i=0; i<n_items; i++) 
            b_item[i] /= train_votes_pitem[i].size();
        
        // randomly init user bias, item bias and average
        /*utils::muldimPosUniform(mu, 1, 1.0);
        utils::muldimPosUniform(b_user, n_users, 1.0);
        utils::muldimPosUniform(b_item, n_items, 1.0);*/

        // randomly init user and item latent factor and topic factor
        for (int u=0; u<n_users; u++) {
            //utils::muldimGaussrand(theta_user[u], K);
            //utils::muldimUniform(theta_user[u], K);
            //uitls::muldimZero(theta_user[u], K);
            utils::muldimPosUniform(theta_user[u], K, 1.0);
            //utils::muldimGaussrand(gamma_user[u], K);
            //utils::muldimUniform(gamma_user[u], K);
            //uitls::muldimZero(gamma_user[u], K);
            utils::muldimPosUniform(gamma_user[u], K, 1.0);
        }
        for (int i=0; i<n_items; i++) {
            //utils::muldimGaussrand(theta_item[i], K);
            //utils::muldimUniform(theta_item[i], K);
            //uitls::muldimZero(theta_item[i], K);
            utils::muldimPosUniform(theta_item[i], K, 1.0);
            //utils::muldimGaussrand(gamma_item[i], K);
            //utils::muldimUniform(gamma_item[i], K);
            //uitls::muldimZero(gamma_item[i], K);
            utils::muldimPosUniform(gamma_item[i], K, 1.0);
        }
       
        // background topic factor initialization
        //for (int i=0; i<K; i++)
            //background_topic[i] = 1.0/K;
        //uitls::muldimZero(background_topic, K);

        // topic words dictionary with K bases
        for (int k=0; k<K; k++)
            for (int i=0; i<n_words; i++)
                topic_words[k][i] = 1.0/n_words;
    }

    void allocMemForPara(double* g,
                         double** gmu,
                         double** gb_user,
                         double** gb_item,
                         double*** gtheta_user,
                         double*** gtheta_item,
                         double*** ggamma_user,
                         double*** ggamma_item,
                         double** gbackground_topic,
                         double*** gtopic_words,
                         double*** gword_topics) {
        int ind = 0;
        
        *gmu = g + ind;
        ind++;

        *gb_user = g + ind;
        ind += n_users;
        *gb_item = g + ind;
        ind += n_items;
        
        *gtheta_user = new double*[n_users];
        for (int u = 0; u < n_users; u++) {
            (*gtheta_user)[u] = g + ind;
            ind += K;
        }
        *gtheta_item = new double*[n_items];
        for (int i=0; i < n_items; i++) {
            (*gtheta_item)[i] = g + ind;
            ind += K;
        }
        *ggamma_user = new double*[n_users];
        for (int u = 0; u < n_users; u++) {
            (*ggamma_user)[u] = g + ind;
            ind += K;
        }
        *ggamma_item = new double*[n_items];
        for (int i=0; i < n_items; i++) {
            (*ggamma_item)[i] = g + ind;
            ind += K;
        }

        *gbackground_topic = g + ind;
        ind += K;
        *gtopic_words = new double*[K];
        for (int k=0; k < K; k++) {
            (*gtopic_words)[k] = g + ind;
            ind += n_words;
        }
        *gword_topics = new double*[n_words];
        for (int i=0; i < n_words; i++) {
            (*gword_topics)[i] = g + ind;
            ind += K;
        }
    }

    void freeMemForPara(double** gmu,
                        double** gb_user,
                        double** gb_item,
                        double*** gtheta_user,
                        double*** gtheta_item,
                        double*** ggamma_user,
                        double*** ggamma_item,
                        double** gbackground_topic,
                        double*** gtopic_words,
                        double*** gword_topics) {
        delete *gmu;
        delete *gb_user;
        delete *gb_item;
        delete *gbackground_topic;
        delete[] (*gtheta_user);
        delete[] (*gtheta_item);
        delete[] (*ggamma_user);
        delete[] (*ggamma_item);
        delete[] (*gtopic_words);
        delete[] (*gword_topics);
    }

    void train() {
        if (tr_method == 0) {
            // SGD
            onlineLearning();
        } else if (tr_method == 1) {
            // minibatch SGD
            minibatchLearning();
        } else if (tr_method == 2) {
            // GD + VB
            gdBatchLearning();
        } else if (tr_method == 3) {
            // Coordinate descent (ALS+VB)
            coordinateBatchLearning();
        } else if (tr_method == 4) {
            // Gibbs stochastic EM algorithm (E-step:Gibbs sampling, M-step:SGD)
            gibbsStochasticEm();
        } else {
            printf("Invalid choice of learning method!\n");
            exit(1);
        }
        saveModelPara();
    }

    void onlineLearning() {
        double train_rmse, valid_rmse, test_rmse;
        double train_perp, valid_perp, test_perp;
        double obj_new, obj_old, best_valid, cur_valid;
        int * user_scan, * item_scan;
        double * doc_topic;
        double exp_sum;
        int user, item, rating;
        double *gbackground_topic, *ggamma_user, *ggamma_item;
        double *gtheta_user, *gtheta_item;
        map<int, double> ** gtopic_words;
        double *tmp_val1, *tmp_val2;
        double res;
        int cur_iter, complete;
        bool converged; 
       
        gbackground_topic = new double[K];
        ggamma_user = new double[K];
        ggamma_item = new double[K];
        gtheta_user = new double[K];
        gtheta_item = new double[K];
        gtopic_words = new map<int, double>*[K];
        for (int k=0; k<K; k++)
            gtopic_words[k] = new map<int, double>();

        user_scan = new int[n_users];
        memset(user_scan, 0, sizeof(int)*n_users);
        item_scan = new int[n_items];
        memset(item_scan, 0, sizeof(int)*n_items);

        printf("Start online learning...\n");
        tmp_val1 = new double[K];
        tmp_val2 = new double[K];
        cur_iter = 0;
        obj_old = 1e5;
        best_valid = 1e5;
        doc_topic = new double[K];
        converged = false;
        timeval start_t, end_t;
        while(!converged && cur_iter < niters) {
            random_shuffle(train_votes.begin(), train_votes.end());
            complete = 0;
#ifdef OL_I_T
            utils::tic(start_t);
#endif
            for (vector<Vote*>::iterator it=train_votes.begin();
                    it!=train_votes.end(); it++) {
#ifdef OL_S_T
                utils::tic(start_t);
#endif
                user = (*it)->user;
                item = (*it)->item;
                //rating = (*it)->rating;
#ifdef OL_M_T
                printf("Debug: step 1\n");
                utils::tic(start_t);
#endif
                res = prediction(*it) - (*it)->value;
#ifdef OL_DEBUG
                cout << "Word cnt = " << (*it)->words.size() << endl;
                cout << "res = " << res << endl;
                cout << "value = " << (*it)->value << endl;
#endif
                memset(gbackground_topic, 0, sizeof(double)*K);
                memset(ggamma_user, 0, sizeof(double)*K);
                memset(ggamma_item, 0, sizeof(double)*K);
                memset(gtheta_user, 0, sizeof(double)*K);
                memset(gtheta_item, 0, sizeof(double)*K);
                for (int k=0; k<K; k++){
                    gtopic_words[k]->clear();
                    map<int, double>(*gtopic_words[k]).swap(*gtopic_words[k]);
                }
#ifdef OL_M_T
                utils::toc(start_t, end_t);
                utils::pause();
#endif
                
                // Note: first should compute doc topic distribution 
#ifdef OL_M_T
                printf("Debug: step 2\n");
                utils::tic(start_t);
#endif
                calDocTopic(&doc_topic, exp_sum, *it);
#ifdef OL_DEBUG
                cout << "exp_sum=" << exp_sum <<endl;
                cout << "doc_topic=";
                for (int i=0; i<K; i++)
                    cout << doc_topic[i] << " ";
                cout << endl;
                utils::pause();
#endif

#ifdef OL_M_T
                utils::toc(start_t, end_t);
                utils::pause();
#endif
                
                // compute variational doc word topic distribution para first
#ifdef OL_M_T
                printf("Debug: step 3\n");
                utils::tic(start_t);
#endif
                calVbPara(&vb_word_topics, doc_topic, topic_words, (*it)->words);
#ifdef OL_DEBUG
                /*for (vector<int>::iterator it1=(*it)->words.begin();
                    it1!=(*it)->words.end(); it1++) {
                    cout << "VB word topic:";
                    for (int k=0; k<K; k++)
                        cout << vb_word_topics[*it1][k] << " ";
                    cout << endl;
                    utils::pause();
                }*/
#endif

#ifdef OL_M_T
                utils::toc(start_t, end_t);
                utils::pause();
#endif

                /// Compute gradients
#ifdef OL_M_T
                printf("Debug: step 4\n");
                utils::tic(start_t);
#endif
                #pragma omp parallel for
                for (int k=0; k<K; k++) {
                    double bt_tmpval = 1-doc_topic[k]/exp_sum;
                    double gu_tmpval = 1-doc_topic[k]/exp_sum;
                    double gi_tmpval = 1-doc_topic[k]/exp_sum;
                    for (vector<int>::iterator it1=(*it)->words.begin();
                            it1!=(*it)->words.end(); it1++) {
                        // compute gradient of background topic factor
                        gbackground_topic[k] += vb_word_topics[*it1][k]
                                              * bt_tmpval;
                        // compute gradient of user topic factor
                        ggamma_user[k] += vb_word_topics[*it1][k]
                                        * gu_tmpval;
                        // compute gradient of item topic factor
                        ggamma_item[k] += vb_word_topics[*it1][k]
                                        * gi_tmpval;
                        // compute gradient of dictionary bases
                        if(gtopic_words[k]->find(*it1)==gtopic_words[k]->end())
                            (*gtopic_words[k])[*it1] = vb_word_topics[*it1][k]
                                                     / topic_words[k][*it1];
                        else
                            (*gtopic_words[k])[*it1] += vb_word_topics[*it1][k]
                                                      / topic_words[k][*it1];
                    }
                    tmp_val1[k] = psai_u*(gamma_user[user][k]-theta_user[user][k]);
                    tmp_val2[k] = psai_i*(gamma_item[item][k]-theta_item[item][k]);
                    ggamma_user[k] -= tmp_val1[k];
                    ggamma_item[k] -= tmp_val2[k];
                    // compute gradient of user latent factor
                    gtheta_user[k] = -theta_item[item][k]*res + tmp_val1[k];
                    // compute gradient of item latent factor
                    gtheta_item[k] = -theta_user[user][k]*res + tmp_val2[k];
                }
#ifdef OL_DEBUG
                cout << "background_topic:";
                for (int k=0; k<K; k++)
                    cout << gbackground_topic[k] << " ";
                cout << endl << "user topic factor:";
                for (int k=0; k<K; k++)
                    cout << ggamma_user[k] << " ";
                cout << endl << "item topic factor:";
                for (int k=0; k<K; k++)
                    cout << ggamma_item[k] << " ";
                cout << endl;
                utils::pause();
#endif

#ifdef OL_M_T
                utils::toc(start_t, end_t);
                utils::pause();
#endif                
                /// Update parameters
#ifdef OL_M_T
                printf("Debug: step 5\n");
                utils::tic(start_t);
#endif                
                #pragma omp parallel for
                for (int k=0; k<K; k++) {
                    // update background topic factor
                    background_topic[k] += lr*alpha*gbackground_topic[k];
                    // update user topic factor
                    gamma_user[user][k] += lr*alpha*ggamma_user[k];
                    // update item topic factor
                    gamma_item[item][k] += lr*alpha*ggamma_item[k];
                    // update dictionary base
                    for (map<int, double>::iterator it1=gtopic_words[k]->begin();
                            it1!=gtopic_words[k]->end(); it1++)
                        topic_words[k][it1->first] += lr*alpha*it1->second;
                    //utils::project_beta(topic_words[k], n_words, 1, 1e-30);
                    //utils::project_beta1(topic_words[k], n_words);
                    //utils::project_beta2(topic_words[k], n_words);
                    
                    // update user latent factor
                    theta_user[user][k] += lr*gtheta_user[k];
                    // update item latent factor
                    theta_item[item][k] += lr*gtheta_item[k]; 
                }
                // compute user bias gradient and update 
                b_user[user] += lr*(-res-sigma_u);
                // compute item bias gradient and update 
                b_item[item] += lr*(-res-sigma_i);
                // compute gradient of average para and update
#ifdef OL_DEBUG
                cout << "before mu=" << *mu <<endl;
#endif
                *mu += lr*(-res-sigma_a);
#ifdef OL_DEBUG
                cout << "after mu=" << *mu <<endl;
                utils::pause();
#endif
                if (*mu > 10 || *mu < -10) {
                    cout << "mu=" << *mu;
                    cout << "res=" << res;
                    cout << "lr=" << lr << endl;
                    for (int i=0; i<K; i++) {
                        cout << theta_user[user][i] << " ";
                    }
                    cout << endl << "theta item: ";
                    for (int i=0; i<K; i++) {
                        cout << theta_item[item][i] << " ";
                    }
                    cout << endl;
                    utils::pause();
                    
                }

#ifdef OL_M_T
                utils::toc(start_t, end_t);
                utils::pause();
#endif                

#ifdef OL_M_T
                printf("Debug: step 6\n");
                utils::tic(start_t);
#endif                
                complete++;
                user_scan[user]++;
                item_scan[item]++;
                if (complete % truncated_k == 0)
                    truncatedGradient(background_topic, lambda_b);
                if (user_scan[user] % truncated_k == 0)
                    truncatedGradient(gamma_user[user], lambda_u);
                if (item_scan[item] % truncated_k == 0)
                    truncatedGradient(gamma_item[item], lambda_i);
#ifdef OL_M_T
                utils::toc(start_t, end_t);
                utils::pause();
#endif                
            
                //printf("\rFinished Scanning pairs: %d ......", complete);
                //fflush(stdout);
                if (complete % 1000 == 0) {
                    printf("\rFinished Scanning pairs: %d ......", complete);
                    fflush(stdout);
#ifdef OL_I_T
                    utils::toc(start_t, end_t);
                    utils::pause();
                    utils::tic(start_t);
#endif                
                }
#ifdef OL_S_T
                utils::toc(start_t, end_t);
                utils::pause();
#endif                
            }

            evalRmseError(train_rmse, valid_rmse, test_rmse);
            printf("Current iteration: %d, Train RMSE=%.6f, ", cur_iter+1, train_rmse);
            printf("Valid RMSE=%.6f, Test RMSE=%.6f!\n", valid_rmse, test_rmse);
            
            cur_valid = valid_rmse;
            if (cur_valid < best_valid) {
                best_valid = cur_valid;
                for (vector<Vote*>::iterator it=train_votes.begin();
                        it!=train_votes.end(); it++)
                    best_vali_predictions[*it] = prediction(*it);
                for (vector<Vote*>::iterator it=vali_votes.begin();
                        it!=vali_votes.end(); it++)
                    best_vali_predictions[*it] = prediction(*it);
                for (vector<Vote*>::iterator it=test_votes.begin();
                        it!=test_votes.end(); it++)
                    best_vali_predictions[*it] = prediction(*it);
            }
            
            if (cur_iter == 0)
                obj_old = train_rmse;
            else {
                obj_new = train_rmse;
                if (obj_new >= obj_old || fabs(obj_new-obj_old) < delta)
                    converged = true;
                obj_old = obj_new;
            }
            cur_iter += 1;
        }
        printf("Online learning finish...\n");
        saveModelPara();

        // release memory
        for (int k=0; k<K; k++) {
            gtopic_words[k]->clear();
            map<int, double>(*gtopic_words[k]).swap(*gtopic_words[k]);
        }
        delete[] gtopic_words;
        
        delete user_scan;
        delete item_scan;
        delete doc_topic;
        delete gbackground_topic; 
        delete ggamma_user;
        delete ggamma_item;
        delete gtheta_user;
        delete gtheta_item;
        delete tmp_val1;
        delete tmp_val2;
    }
    
    void minibatchLearning() {
        /*double train_rmse, valid_rmse, test_rmse;
        double train_perp, valid_perp, test_perp;
        double obj_new, obj_old, best_valid, cur_valid;
        int * user_scan, * item_scan;
        double * doc_topic;
        double exp_sum;
        int user, item, rating;
        double *gbackground_topic, *ggamma_user, *ggamma_item;
        double *gtheta_user, *gtheta_item;
        map<int, double> ** gtopic_words;
        double *tmp_val1, *tmp_val2;
        double res;
        int cur_iter, complete;
        bool converged; 
       
        gbackground_topic = new double[K];
        ggamma_user = new double[K];
        ggamma_item = new double[K];
        gtheta_user = new double[K];
        gtheta_item = new double[K];
        gtopic_words = new map<int, double>*[K];
        for (int k=0; k<K; k++)
            gtopic_words[k] = new map<int, double>();

        user_scan = new int[n_users];
        memset(user_scan, 0, sizeof(int)*n_users);
        item_scan = new int[n_items];
        memset(item_scan, 0, sizeof(int)*n_items);

        printf("Start online learning...\n");
        tmp_val1 = new double[K];
        tmp_val2 = new double[K];
        cur_iter = 0;
        obj_old = 1e5;
        best_valid = 1e5;
        doc_topic = new double[K];
        converged = false;
        timeval start_t, end_t;
        while(!converged && cur_iter < niters) {
            random_shuffle(train_votes.begin(), train_votes.end());
            complete = 0;
#ifdef OL_I_T
            utils::tic(start_t);
#endif
            for (vector<Vote*>::iterator it=train_votes.begin();
                    it!=train_votes.end(); it++) {
#ifdef OL_S_T
                utils::tic(start_t);
#endif
                user =  (*it)->user;
                item = (*it)->item;
                //rating = (*it)->rating;
#ifdef OL_M_T
                printf("Debug: step 1\n");
                utils::tic(start_t);
#endif
                res = prediction(*it) - (*it)->value;
                memset(gbackground_topic, 0, sizeof(double)*K);
                memset(ggamma_user, 0, sizeof(double)*K);
                memset(ggamma_item, 0, sizeof(double)*K);
                memset(gtheta_user, 0, sizeof(double)*K);
                memset(gtheta_item, 0, sizeof(double)*K);
#ifdef OL_M_T
                utils::toc(start_t, end_t);
                utils::pause();
#endif
                
                // Note: first should compute doc topic distribution 
#ifdef OL_M_T
                printf("Debug: step 2\n");
                utils::tic(start_t);
#endif
                calDocTopic(&doc_topic, exp_sum, *it);
#ifdef OL_M_T
                utils::toc(start_t, end_t);
                utils::pause();
#endif
                
                // compute variational doc word topic distribution para first
#ifdef OL_M_T
                printf("Debug: step 3\n");
                utils::tic(start_t);
#endif
                calVbPara(&vb_word_topics, doc_topic, topic_words, (*it)->words);
#ifdef OL_M_T
                utils::toc(start_t, end_t);
                utils::pause();
#endif

                /// Compute gradients
#ifdef OL_M_T
                printf("Debug: step 4\n");
                utils::tic(start_t);
#endif
                #pragma omp parallel for
                for (int k=0; k<K; k++) {
                    for (vector<int>::iterator it1=(*it)->words.begin();
                            it1!=(*it)->words.end(); it1++)  {
                        // compute gradient of background topic factor
                        gbackground_topic[k] += vb_word_topics[*it1][k]
                                         * (1-exp(background_topic[k])/exp_sum);
                        // compute gradient of user topic factor
                        ggamma_user[k] += vb_word_topics[*it1][k]
                                    * (1-exp(gamma_user[user][k])/exp_sum);
                        // compute gradient of item topic factor
                        ggamma_item[k] += vb_word_topics[*it1][k]
                                    * (1-exp(gamma_item[item][k])/exp_sum);
                        // compute gradient of dictionary bases
                        if(gtopic_words[k]->find(*it1)==gtopic_words[k]->end())
                            (*gtopic_words[k])[*it1] = vb_word_topics[*it1][k]
                                                     / topic_words[k][*it1];
                        else
                            (*gtopic_words[k])[*it1] += vb_word_topics[*it1][k]
                                                      / topic_words[k][*it1];
                    }
                    tmp_val1[k] = psai_u*(gamma_user[k]-theta_user[k]);
                    tmp_val2[k] = psai_i*(gamma_item[k]-theta_item[k]);
                    ggamma_user[k] -= tmp_val1[k];
                    ggamma_item[k] -= tmp_val2[k];
                    // compute gradient of user latent factor
                    gtheta_user[k] = -theta_item[item][k]*res + tmp_val1[k];
                    // compute gradient of item latent factor
                    gtheta_item[k] = -theta_user[user][k]*res + tmp_val2[k];
                }
#ifdef OL_M_T
                utils::toc(start_t, end_t);
                utils::pause();
#endif                
                /// Update parameters
#ifdef OL_M_T
                printf("Debug: step 5\n");
                utils::tic(start_t);
#endif                
                #pragma omp parallel for
                for (int k=0; k<K; k++) {
                    // update background topic factor
                    background_topic[k] += lr*gbackground_topic[k];
                    // update user topic factor
                    gamma_user[user][k] += lr*ggamma_user[k];
                    // update item topic factor
                    gamma_item[item][k] += lr*ggamma_item[k];
                    // update user latent factor
                    theta_user[user][k] += lr*gtheta_user[k];
                    // update item latent factor
                    theta_item[item][k] += lr*gtheta_item[k]; 
                }
                // compute user bias gradient and update 
                b_user[user] += lr*(-res-sigma_u);
                // compute item bias gradient and update 
                b_item[item] += lr*(-res-sigma_i);
                // compute gradient of average para and update
                *mu += lr*(-res-sigma_a);
#ifdef OL_M_T
                utils::toc(start_t, end_t);
                utils::pause();
#endif                

#ifdef OL_M_T
                printf("Debug: step 6\n");
                utils::tic(start_t);
#endif                
                complete++;
                user_scan[user]++;
                item_scan[item]++;
                if (complete % truncated_k == 0)
                    truncatedGradient(background_topic, lambda_b);
                if (user_scan[user] % truncated_k == 0)
                    truncatedGradient(gamma_user[user], lambda_u);
                if (item_scan[item] % truncated_k == 0)
                    truncatedGradient(gamma_item[item], lambda_i);
                // minibatch based updation for dictionary base
                if (complete % minibatch == 0 || complete == train_votes.size()) {
                    for (int k=0; k<K; k++) {
                        for (map<int, double>::iterator it1=gtopic_words[k]->begin();
                                it1!=gtopic_words[k]->end(); it1++)
                            topic_words[k][it1->first] += lr*it1->second/minibatch;
                        // project dictionary codes to probabilistic simplex
                        utils::project_beta1(topic_words[k], n_words);
                        gtopic_words[k]->clear();
                        map<int, double>(*gtopic_words[k]).swap(*gtopic_words[k]);
                    }
                }
#ifdef OL_M_T
                utils::toc(start_t, end_t);
                utils::pause();
#endif                
            
                //printf("\rFinished Scanning pairs: %d ......", complete);
                //fflush(stdout);
                if (complete % 1000 == 0) {
                    printf("\rFinished Scanning pairs: %d ......", complete);
                    fflush(stdout);
#ifdef OL_I_T
                    utils::toc(start_t, end_t);
                    utils::pause();
                    utils::tic(start_t);
#endif                
                }
#ifdef OL_S_T
                utils::toc(start_t, end_t);
                utils::pause();
#endif                
            }

            evalRmseError(train_rmse, valid_rmse, test_rmse);
            printf("Current iteration: %d, Train RMSE=%.6f, ", cur_iter+1, train_rmse);
            printf("Valid RMSE=%.6f, Test RMSE=%.6f!\n", valid_rmse, test_rmse);
            
            cur_valid = valid_rmse;
            if (cur_valid < best_valid) {
                best_valid = cur_valid;
                for (vector<Vote*>::iterator it=train_votes.begin();
                        it!=train_votes.end(); it++)
                    best_vali_predictions[*it] = prediction(*it);
                for (vector<Vote*>::iterator it=vali_votes.begin();
                        it!=vali_votes.end(); it++)
                    best_vali_predictions[*it] = prediction(*it);
                for (vector<Vote*>::iterator it=test_votes.begin();
                        it!=test_votes.end(); it++)
                    best_vali_predictions[*it] = prediction(*it);
            }
            
            if (cur_iter == 0)
                obj_old = train_rmse;
            else {
                obj_new = train_rmse;
                if (obj_new >= obj_old || fabs(obj_new-obj_old) < delta)
                    converged = true;
                obj_old = obj_new;
            }
            cur_iter += 1;
        }
        printf("Online learning finish...\n");
        saveModelPara();

        // release memory
        for (int k=0; k<K; k++) {
            gtopic_words[k]->clear();
            map<int, double>(*gtopic_words[k]).swap(*gtopic_words[k]);
        }
        delete[] gtopic_words;
        
        delete user_scan;
        delete item_scan;
        delete doc_topic;
        delete gbackground_topic; 
        delete ggamma_user;
        delete ggamma_item;
        delete gtheta_user;
        delete gtheta_item;
        delete tmp_val1;
        delete tmp_val2;*/
    }
    
    void gdBatchLearning() {
        double train_rmse, valid_rmse, test_rmse;
        double train_perp, valid_perp, test_perp;
    }
    
    void coordinateBatchLearning() {
        double train_rmse, valid_rmse, test_rmse;
        double train_perp, valid_perp, test_perp;
        double obj_new, obj_old, best_valid, cur_valid;
        double ** inter_tu1, ** inter_tu2, ** inter_ti1, **inter_ti2;
        double ** exp_gu, ** exp_gi, * exp_bt;
        double ** grad_gu, ** grad_gi, * grad_bt;
        double * exp_topic, * exp_doctopic_sum, * exp_wordtopic;
        double exp_topic_sum, exp_wordtopic_sum;
        double ** exp_topic_u, ** exp_doctopic_sum_u, ** exp_wordtopic_u;
        double *exp_topic_sum_u, * exp_wordtopic_sum_u;
        double ** exp_topic_i, ** exp_doctopic_sum_i, ** exp_wordtopic_i;
        double *exp_topic_sum_i, * exp_wordtopic_sum_i;
        double *t_k, *t_k1, **t_k_u, **t_k1_u, **t_k_i, **t_k1_i;
        double *xk_bt, *xk1_bt, **xk_gu, **xk1_gu, **xk_gi, **xk1_gi;
        double *topic_words_sum, **topic_words_bak, **exp_dict, *exp_dict_sum;
        int cur_iter;
        bool converged;

        /// Allocate memory for caching intermidiate results
        inter_tu1 = new double*[n_users];
        inter_tu2 = new double*[n_users];
        for (int u=0; u<n_users; u++) {
            inter_tu1[u] = new double[K];
            inter_tu2[u] = new double[K];
        }
        inter_ti1 = new double*[n_items];
        inter_ti2 = new double*[n_items];
        for (int i=0;i<n_items; i++) {
            inter_ti1[i] = new double[K];
            inter_ti2[i] = new double[K];
        }
        exp_gu = new double*[n_users];
        for (int u=0; u<n_users; u++) {
            exp_gu[u] = new double[K];
            for (int k=0; k<K; k++)
                exp_gu[u][k] = exp(gamma_user[u][k]);
        }
        exp_gi = new double*[n_items];
        for (int i=0; i<n_items; i++) {
            exp_gi[i] = new double[K];
            for (int k=0; k<K; k++)
                exp_gi[i][k] = exp(gamma_item[i][k]);
        }
        exp_bt = new double[K];
        for (int k=0; k<K; k++)
            exp_bt[k] = exp(background_topic[k]);
        exp_topic = new double[K];
        exp_doctopic_sum = new double[K];   
        exp_wordtopic = new double[K];
        exp_topic_sum_u = new double[n_users];
        exp_wordtopic_sum_u = new double[n_users];
        exp_topic_u = new double*[n_users];
        exp_doctopic_sum_u = new double*[n_users];
        exp_wordtopic_u = new double*[n_users];
        for (int u=0; u<n_users; u++) {
            exp_topic_u[u] = new double[K];
            exp_doctopic_sum_u[u] = new double[K];
            exp_wordtopic_u[u] = new double[K];
        }
        exp_topic_sum_i = new double[n_items];
        exp_wordtopic_sum_i = new double[n_items];
        exp_topic_i = new double*[n_items];
        exp_doctopic_sum_i = new double*[n_items];
        exp_wordtopic_i = new double*[n_items];
        for (int i=0; i<n_items; i++) {
            exp_topic_i[i] = new double[K];
            exp_doctopic_sum_i[i] = new double[K];
            exp_wordtopic_i[i] = new double[K];
        }

        /// Allocate memory for storing gradients
        grad_gu = new double*[n_users];
        for (int u=0; u<n_users; u++) {
            grad_gu[u] = new double[K];
            memset(grad_gu[u], 0.0, sizeof(double)*K);
        }
        grad_gi = new double*[n_items];
        for (int i=0; i<n_items; i++) {
            grad_gi[i] = new double[K];
            memset(grad_gi[i], 0.0, sizeof(double)*K);
        }
        grad_bt = new double[K];
        memset(grad_bt, 0.0, sizeof(double)*K);
       
        /// Allocate memory for caching intermediate results of FISTA algorithm
        t_k = new double[K];
        t_k1 = new double[K];
        t_k_u =  new double*[n_users];
        t_k1_u = new double*[n_users];
        for (int u=0; u<n_users; u++) {
            t_k_u[u] = new double[K];
            t_k1_u[u] = new double[K];
        }
        t_k_i =  new double*[n_items];
        t_k1_i = new double*[n_items];
        for (int i=0; i<n_items; i++) {
            t_k_i[i] = new double[K];
            t_k1_i[i] = new double[K];
        }
        xk_bt = new double[K];
        xk1_bt = new double[K];
        xk_gu = new double*[n_users];
        xk1_gu = new double*[n_users];
        for (int u=0; u<n_users; u++) {
            xk_gu[u] = new double[K];
            xk1_gu[u] = new double[K];
        }
        xk_gi = new double*[n_items];
        xk1_gi = new double*[n_items];
        for (int i=0; i<n_items; i++) {
            xk_gi[i] = new double[K];
            xk1_gi[i] = new double[K]; 
        }
        
        /// Allocate memory for dictionary learning
        topic_words_sum = new double[K];
        topic_words_bak = new double*[K];
        //exp_dict = new double*[K];
        exp_dict_sum = new double[K];
        for (int k=0; k<K; k++) {
            topic_words_bak[k]  = new double[n_words];
            //exp_dict[k] = new double[K];
        }

        timeval start_t, end_t;
        printf("Start coordinate learning...\n");
        while(!converged && cur_iter < niters) {
#ifdef CD_DEBUG
        cout<< "CD for learning user latent factor" <<endl;
        utils::tic(start_t);
#endif
            /// CD for learning user latent factor
            for (int u=0; u<n_users; u++) {
                memset(inter_tu1[u], 0.0, sizeof(double)*K);
                memset(inter_tu2[u], 0.0, sizeof(double)*K);
            }
            #pragma omp parallel for
            for (int u=0; u<n_users; u++) {
                for (vector<Vote*>::iterator it=train_votes_puser[u].begin();
                        it!=train_votes_puser[u].end(); it++) {
                    int item = (*it)->item;
                    double res = (*it)->value - predictionWithoutFactorProd(*it);
                    for (int k=0; k<K; k++) {
                        inter_tu1[u][k] += theta_item[item][k]
                                         * theta_item[item][k];
                        inter_tu2[u][k] += theta_item[item][k]*res;
                    }
                }
                for (int k=0; k<K; k++)
                    theta_user[u][k]=(psai_u*gamma_user[u][k]+inter_tu2[u][k])
                                    /(psai_u+inter_tu1[u][k]);
            }
#ifdef CD_DEBUG
            utils::toc(start_t, end_t);
            utils::pause();
#endif

            /// CD for learning user rating bias
#ifdef CD_DEBUG
            cout << "CD for learning user rating bias" << endl;
            utils::tic(start_t);
#endif
            #pragma omp parallel for
            for (int u=0; u<n_users; u++) {
                double inter_b = 0.0;
                for (vector<Vote*>::iterator it=train_votes_puser[u].begin();
                        it!=train_votes_puser[u].end(); it++)
                    inter_b += (*it)->value - predictionWithoutUserBias(*it);
                b_user[u] = inter_b/(train_votes_puser[u].size()+sigma_u);
            }
#ifdef CD_DEBUG
            utils::toc(start_t, end_t);
            utils::pause();
#endif
            /// CD for learning item latent factor
#ifdef CD_DEBUG
            cout << "CD for learning item latent factor" << endl;
            utils::tic(start_t);
#endif
            for (int i=0;i<n_items; i++) {
                memset(inter_ti1[i], 0.0, sizeof(double)*K);
                memset(inter_ti2[i], 0.0, sizeof(double)*K);
            }
            #pragma omp parallel for
            for (int i=0; i<n_items; i++) {
                for (vector<Vote*>::iterator it=train_votes_pitem[i].begin();
                        it!=train_votes_pitem[i].end(); it++) {
                    int user = (*it)->user;
                    double res = (*it)->value - predictionWithoutFactorProd(*it);
                    for (int k=0; k<K; k++) {
                        inter_ti1[i][k] += theta_user[user][k]
                                         * theta_user[user][k];
                        inter_ti2[i][k]+=theta_user[user][k]*res;
                    }
                }
                for (int k=0; k<K; k++)
                    theta_item[i][k]=(psai_i*gamma_item[i][k]+inter_ti2[i][k])
                                    /(psai_i+inter_ti1[i][k]);
            }
#ifdef CD_DEBUG
            utils::toc(start_t, end_t);
            utils::pause();
#endif

            /// CD for learning item rating bias
#ifdef CD_DEBUG
            cout << "CD for learning item rating bias" << endl;
            utils::tic(start_t);
#endif
            #pragma omp parallel for
            for (int i=0; i<n_items; i++) {
                double inter_b = 0.0;
                for (vector<Vote*>::iterator it=train_votes_pitem[i].begin();
                        it!=train_votes_pitem[i].end(); it++)
                    inter_b += (*it)->value - predictionWithoutItemBias(*it);
                b_item[i] = inter_b/(train_votes_pitem[i].size()+sigma_i);
            }
#ifdef CD_DEBUG
            utils::toc(start_t, end_t);
            utils::pause();
#endif
            
            /// CD for learning whole rating bias
#ifdef CD_DEBUG
            cout << "CD for learning whole rating bias" << endl;
            utils::tic(start_t);
#endif
            double inter_mu = 0.0;
            for (vector<Vote*>::iterator it=train_votes.begin();
                    it!=train_votes.end(); it++)
                inter_mu += (*it)->value - predictionWithoutAverage(*it);
            *mu = inter_mu/(train_votes.size()+sigma_a);
#ifdef CD_DEBUG
            utils::toc(start_t, end_t);
            utils::pause();
#endif

            /// CD for learning background topic factor (FISTA algorithm)
#ifdef CD_DEBUG
            cout << "CD for background topic factor" << endl;
            utils::tic(start_t);
#endif
            for (int k=0; k<K; k++)
                t_k[k] = 1;
            memcpy(xk_bt, background_topic, sizeof(double)*K);
            for (int t=0; t<inter_niters; t++) {
                memset(grad_bt, 0.0, sizeof(double)*K);
                for (vector<Vote*>::iterator it=train_votes.begin();
                            it!=train_votes.end(); it++) {
                    int user = (*it)->user;
                    int item = (*it)->item;
                    exp_topic_sum = 0.0;
                    for (int k=0; k<K; k++) {
                        exp_topic[k]=exp_gu[user][k]*exp_gi[item][k]*exp_bt[k];
                        exp_topic_sum += exp_topic[k];
                    }
                    exp_wordtopic_sum = 0.0;
                    memset(exp_doctopic_sum, 0.0, sizeof(double)*K);
                    for (vector<int>::iterator it1=(*it)->words.begin();
                            it1!=(*it)->words.end(); it1++) {
                        for (int k=0; k<K; k++) {
                            exp_wordtopic[k] = exp_topic[k]*topic_words[k][*it1];
                            exp_wordtopic_sum += exp_wordtopic[k];
                        }
                        #pragma omp parallel for
                        for (int k=0; k<K; k++) 
                            exp_doctopic_sum[k] += exp_wordtopic[k]/exp_wordtopic_sum;
                    }
                    #pragma omp parallel for
                    for (int k=0; k<K; k++)
                        grad_bt[k]+=exp_doctopic_sum[k]*(1-exp_topic[k]/exp_topic_sum)
                                   /(*it)->words.size();
                }
                #pragma omp parallel for
                for (int k=0; k<K; k++) {
                    background_topic[k] = background_topic[k]
                                        + lr*alpha*1.0/train_votes.size()*grad_bt[k];
                    xk1_bt[k] = utils::soft(background_topic[k], lr*lambda_b);
                    t_k1[k] = (1+sqrt(1+4*t_k[k]*t_k[k]))/2;
                    background_topic[k] = xk1_bt[k]+(t_k[k]-1)/t_k1[k]
                                        * (xk1_bt[k]-xk_bt[k]);
                    t_k[k] = t_k1[k];
                    xk_bt[k] = xk1_bt[k];
                    exp_bt[k] = exp(background_topic[k]);
                }
#ifdef CD_DEBUG
                printf("\rCurrent iteration: %d...", t+1);
                fflush(stdout);
#endif
            }
#ifdef CD_DEBUG
            utils::toc(start_t, end_t);
            utils::pause();
#endif
            
            /// CD for learning user topic factor (FISTA algorithm)
#ifdef CD_DEBUG
            cout << "CD for user topic factor" << endl;
            utils::tic(start_t);
#endif
            for (int u=0; u<n_users; u++) {
                for (int k=0; k<K; k++)
                    t_k_u[u][k] = 1;
                memcpy(xk_gu[u], gamma_user[u], sizeof(double)*K);
            }
            #pragma omp parallel for
            for (int u=0; u<n_users; u++) {
                for (int t=0; t<inter_niters; t++) {
                    memset(grad_gu[u], 0.0, sizeof(double)*K);
                    for (vector<Vote*>::iterator it=train_votes_puser[u].begin();
                            it!=train_votes_puser[u].end(); it++) {
                        int item = (*it)->item;
                        exp_topic_sum_u[u] = 0.0;
                        for (int k=0; k<K; k++) {
                            exp_topic_u[u][k]=exp_gu[u][k]*exp_gi[item][k]*exp_bt[k];
                            exp_topic_sum_u[u] += exp_topic[k];
                        }
                        exp_wordtopic_sum_u[u] = 0.0;
                        memset(exp_doctopic_sum_u[u], 0.0, sizeof(double)*K);
                        for (vector<int>::iterator it1=(*it)->words.begin();
                                it1!=(*it)->words.end(); it1++) {
                            for (int k=0; k<K; k++) {
                                exp_wordtopic_u[u][k] = exp_topic_u[u][k]
                                                      * topic_words[k][*it1];
                                exp_wordtopic_sum_u[u] += exp_wordtopic_u[u][k];
                            }
                            #pragma omp parallel for
                            for (int k=0; k<K; k++) 
                                exp_doctopic_sum_u[u][k] += exp_wordtopic_u[u][k]
                                                          / exp_wordtopic_sum_u[u];
                        }
                        #pragma omp parallel for
                        for (int k=0; k<K; k++)
                            grad_gu[u][k]+=exp_doctopic_sum_u[u][k]
                                          *(1-exp_topic_u[u][k]/exp_topic_sum_u[u])
                                       /(*it)->words.size();
                    }
                    #pragma omp parallel for
                    for (int k=0; k<K; k++) {
                        gamma_user[u][k] = gamma_user[u][k]
                                         + lr*alpha*1.0/train_votes_puser[u].size()
                                           *grad_gu[u][k];
                        xk1_gu[u][k] = utils::soft(gamma_user[u][k], lr*lambda_u);
                        t_k1_u[u][k] = (1+sqrt(1+4*t_k_u[u][k]*t_k_u[u][k]))/2;
                        gamma_user[u][k] = xk1_gu[u][k]+(t_k_u[u][k]-1)/t_k1_u[u][k]
                                        * (xk1_gu[u][k]-xk_gu[u][k]);
                        t_k_u[u][k] = t_k1_u[u][k];
                        xk_gu[u][k] = xk1_gu[u][k];
                        exp_gu[u][k] = exp(gamma_user[u][k]);
                    }
#ifdef CD_DEBUG
                    printf("\rCurrent iteration: %d...", t+1);
                    fflush(stdout);
#endif
                }
            }
#ifdef CD_DEBUG
            utils::toc(start_t, end_t);
            utils::pause();
#endif
            
            /// CD for learning item topic factor (FISTA algorithm)
#ifdef CD_DEBUG
            cout << "CD for item topic factor" << endl;
            utils::tic(start_t);
#endif
            for (int i=0; i<n_items; i++) {
                for (int k=0; k<K; k++)
                    t_k_i[i][k] = 1;
                memcpy(xk_gi[i], gamma_item[i], sizeof(double)*K);
            }
            #pragma omp parallel for
            for (int i=0; i<n_items; i++) {
                for (int t=0; t<inter_niters; t++) {
                    memset(grad_gi[i], 0.0, sizeof(double)*K);
                    for (vector<Vote*>::iterator it=train_votes_pitem[i].begin();
                            it!=train_votes_pitem[i].end(); it++) {
                        int user = (*it)->user;
                        exp_topic_sum_i[i] = 0.0;
                        for (int k=0; k<K; k++) {
                            exp_topic_i[i][k]=exp_gi[user][k]*exp_gi[i][k]*exp_bt[k];
                            exp_topic_sum_i[i] += exp_topic[k];
                        }
                        exp_wordtopic_sum_i[i] = 0.0;
                        memset(exp_doctopic_sum_i[i], 0.0, sizeof(double)*K);
                        for (vector<int>::iterator it1=(*it)->words.begin();
                                it1!=(*it)->words.end(); it1++) {
                            for (int k=0; k<K; k++) {
                                exp_wordtopic_i[i][k] = exp_topic_i[i][k]
                                                      * topic_words[k][*it1];
                                exp_wordtopic_sum_i[i] += exp_wordtopic_i[i][k];
                            }
                            #pragma omp parallel for
                            for (int k=0; k<K; k++) 
                                exp_doctopic_sum_i[i][k] += exp_wordtopic_i[i][k]
                                                          / exp_wordtopic_sum_i[i];
                        }
                        #pragma omp parallel for
                        for (int k=0; k<K; k++)
                            grad_gi[i][k]+=exp_doctopic_sum_i[i][k]
                                          *(1-exp_topic_i[i][k]/exp_topic_sum_i[i])
                                       /(*it)->words.size();
                    }
                    #pragma omp parallel for
                    for (int k=0; k<K; k++) {
                        gamma_item[i][k] = gamma_item[i][k]
                                         + lr*alpha*1.0/train_votes_pitem[i].size()
                                           *grad_gi[i][k];
                        xk1_gi[i][k] = utils::soft(gamma_item[i][k], lr*lambda_i);
                        t_k1_i[i][k] = (1+sqrt(1+4*t_k_i[i][k]*t_k_i[i][k]))/2;
                        gamma_item[i][k] = xk1_gi[i][k]+(t_k_i[i][k]-1)/t_k1_i[i][k]
                                        * (xk1_gi[i][k]-xk_gi[i][k]);
                        t_k_i[i][k] = t_k1_i[i][k];
                        xk_gi[i][k] = xk1_gi[i][k];
                        exp_gi[i][k] = exp(gamma_item[i][k]);
                    }
#ifdef CD_DEBUG
                    printf("\rCurrent iteration: %d...", t+1);
                    fflush(stdout);
#endif
                }
            }
#ifdef CD_DEBUG
            utils::toc(start_t, end_t);
            utils::pause();
#endif

            /// CD for learning dictionary base (Closed form solution)
#ifdef CD_DEBUG
            cout << "CD for learning dictionary base" << endl;
            utils::tic(start_t);
#endif
            memset(topic_words_sum, 0.0, sizeof(double)*K);
            #pragma omp parallel for
            for (int k=0; k<K; k++) {
                memset(topic_words_bak[k], 0.0, sizeof(double)*n_words);
                for (vector<Vote*>::iterator it=train_votes.begin();
                            it!=train_votes.end(); it++) {
                    int user = (*it)->user;
                    int item = (*it)->item;
                    exp_dict_sum[k] = exp_bt[k]*exp_gu[user][k]*exp_gi[item][k];
                    for (vector<int>::iterator it1=(*it)->words.begin();
                            it1!=(*it)->words.end(); it1++)
                        topic_words_bak[k][*it1] += exp_dict_sum[k]
                                                  * topic_words[k][*it1]; 
                }
                utils::normalize(topic_words[k], topic_words_bak[k], n_words);
            }
#ifdef CD_DEBUG
            utils::toc(start_t, end_t);
            utils::pause();
#endif

            evalRmseError(train_rmse, valid_rmse, test_rmse);
            printf("Current iteration: %d, Train RMSE=%.6f, ",
                    cur_iter+1, train_rmse);
            printf("Valid RMSE=%.6f, Test RMSE=%.6f!\n",
                    valid_rmse, test_rmse);
            cur_valid = valid_rmse;
            if (cur_valid < best_valid) {
                best_valid = cur_valid;
                for (vector<Vote*>::iterator it=train_votes.begin();
                        it!=train_votes.end(); it++)
                    best_vali_predictions[*it] = prediction(*it);
                for (vector<Vote*>::iterator it=vali_votes.begin();
                        it!=vali_votes.end(); it++)
                    best_vali_predictions[*it] = prediction(*it);
                for (vector<Vote*>::iterator it=test_votes.begin();
                        it!=test_votes.end(); it++)
                    best_vali_predictions[*it] = prediction(*it);
            }
            
            if (cur_iter == 0)
                obj_old = train_rmse;
            else {
                obj_new = train_rmse;
                if (obj_new >= obj_old || fabs(obj_new-obj_old) < delta)
                    converged = true;
                obj_old = obj_new;
            }
            cur_iter += 1;
        }

        /// Release memory
        for (int u=0; u<n_users; u++) {
            delete inter_tu1[u];
            delete inter_tu2[u]; 
            delete exp_gu[u];
            delete grad_gu[u];
            delete exp_topic_u[u];
            delete exp_doctopic_sum_u[u];
            delete exp_wordtopic_u[u];
            delete t_k_u;
            delete t_k1_u;
            delete xk_gu[u];
            delete xk1_gu[u];
        }
        delete exp_topic_sum_u;
        delete exp_wordtopic_sum_u;
        delete[] inter_tu1;
        delete[] inter_tu2; 
        delete[] exp_gu;
        delete[] grad_gu;
        delete[] exp_topic_u;
        delete[] exp_doctopic_sum_u;
        delete[] exp_wordtopic_u;
        delete[] t_k_u;
        delete[] t_k1_u;
        delete[] xk_gu;
        delete[] xk1_gu;
       
        for (int i=0; i<n_items; i++) {
            delete inter_ti1[i];
            delete inter_ti2[i];
            delete exp_gi[i];
            delete grad_gi[i];
            delete exp_topic_i[i];
            delete exp_doctopic_sum_i[i];
            delete exp_wordtopic_i[i];
            delete t_k_i[i];
            delete t_k1_i[i];
            delete xk_gi[i];
            delete xk1_gi[i];
        }
        delete exp_topic_sum_i;
        delete exp_wordtopic_sum_i;
        delete[] inter_ti1;
        delete[] inter_ti2;
        delete[] exp_gi;
        delete[] grad_gi;
        delete[] exp_topic_i;
        delete[] exp_doctopic_sum_i;
        delete[] exp_wordtopic_i;
        delete[] t_k_i;
        delete[] t_k1_i;
        delete[] xk_gi;
        delete[] xk1_gi;
        
        delete exp_bt;
        delete grad_bt;
        delete exp_topic;
        delete exp_doctopic_sum;
        delete exp_wordtopic;
        delete t_k;
        delete t_k1;
        delete xk_bt;
        delete xk1_bt;
        delete topic_words_sum;
        delete exp_dict_sum;
        for (int k=0; k<K; k++)
            delete topic_words_bak[k];
        delete[] topic_words_bak;
    }
    
    void gibbsStochasticEm() {
        double train_rmse, valid_rmse, test_rmse;
        double train_perp, valid_perp, test_perp;
    }

    void calDocTopic(double ** doc_topic, double& exp_sum, Vote * v) {
        exp_sum = 0.0;
        for (int k=0; k<K; k++) {
            (*doc_topic)[k] = exp(background_topic[k]+gamma_user[v->user][k]
                    +gamma_item[v->item][k]);
            exp_sum += (*doc_topic)[k];
        }
        //for (int i=0; i<K; i++)
        //    (*doc_topic)[i] /= exp_sum;
    }

    void calVbPara(double *** vb_word_topics, double * doc_topic, 
            double ** dtopic_words, vector<int> review) {
        double normalization = 0.0;
        for (vector<int>::iterator it=review.begin();
                it!=review.end(); it++) {
            normalization = 0.0;
            for (int k=0; k<K; k++) {
                (*vb_word_topics)[*it][k] = doc_topic[k]*dtopic_words[k][*it];
                normalization += (*vb_word_topics)[*it][k];
            }
            if (normalization == 0) {
                cout << "doc_topic:"; 
                for (int k=0; k<K; k++) {
                    cout << doc_topic[k] << " ";
                }
                cout << endl << "topic words:";
                for (int k=0; k<K; k++) {
                    cout << dtopic_words[k][*it] << " ";
                }
                cout << endl;
                printf("Normalization == 0\n");
                exit(1);
            }
            for (int k=0; k<K; k++) {
                (*vb_word_topics)[*it][k] /= normalization;
            }
        }
    }

    void truncatedGradient(double * weights, double g) {
        for (int i=0; i<K; i++) {
            if (weights[i] <= truncated_theta && weights[i] >= 0)
                weights[i] = utils::max(0, weights[i]-lr*truncated_k*g);
            if (weights[i] >= -truncated_theta && weights[i] <= 0)
                weights[i] = utils::min(0, weights[i]+lr*truncated_k*g);
        }
    }

    inline double prediction(Vote * v) {
        /*cout << "mu=" << *mu << endl;
        cout << "user bias=" << b_user[v->user] << endl;
        cout << "item bias=" << b_item[v->item] << endl;
        cout << "theta user: ";
        for (int i=0; i<K; i++) {
            cout << theta_user[v->user][i] << " ";
        }
        cout << endl << "theta item: ";
        for (int i=0; i<K; i++) {
            cout << theta_item[v->item][i] << " ";
        }
        cout << endl;
        utils::pause();*/
        return  *mu + b_user[v->user] + b_item[v->item] 
                    + utils::dot(theta_user[v->user], theta_item[v->item], K);
    }
   
    inline double predictionWithoutFactorProd(Vote * v) {
        return *mu + b_item[v->item] + b_user[v->user];
    }

    inline double predictionWithoutUserBias(Vote * v) {
        return  *mu + b_item[v->item] 
                    + utils::dot(theta_user[v->user], theta_item[v->item], K);
    }
    
    inline double predictionWithoutItemBias(Vote * v) {
        return  *mu + b_user[v->user] 
                    + utils::dot(theta_user[v->user], theta_item[v->item], K);
    }
    
    inline double predictionWithoutAverage(Vote * v) {
        return  b_user[v->user] + b_item[v->item] 
                    + utils::dot(theta_user[v->user], theta_item[v->item], K);
    }

    void evalRmseError(double& train, double& valid, double& test) {
        train = 0.0, valid = 0.0, test = 0.0;
        for (vector<Vote*>::iterator it = train_votes.begin();
                it != train_votes.end(); it++)
            train += utils::square(prediction(*it) - (*it)->value) ;
        for (vector<Vote*>::iterator it = vali_votes.begin();
                it != vali_votes.end(); it++)
            valid += utils::square(prediction(*it) - (*it)->value) ;
        for (vector<Vote*>::iterator it = test_votes.begin();
                it != test_votes.end(); it++)
            test += utils::square(prediction(*it) - (*it)->value) ;
        train = sqrt(train/train_votes.size());
        valid = sqrt(valid/vali_votes.size());
        test = sqrt(test/test_votes.size());
    } 

    double evalPerplexity(double& train, double& valid, double& test) {
        return 0.0;
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
        for (int u=0; u<n_users; u++)
            fwrite(W, sizeof(double), NW, f);
        fclose(f);
    }

    void loadModelPara() {
        // total number of paramters to be learned
        NW = 1 + (n_users+n_items)*(K+1) + (n_users+n_items+1)*K + 2*K*n_words;
        W = new double[NW];
        
        double * g = W;
        int ind = 0;
        
        mu = g + ind;
        ind++;

        b_user = g + ind;
        ind += n_users;
        b_item = g + ind;
        ind += n_items;
        
        theta_user = new double*[n_users];
        for (int u = 0; u < n_users; u++) {
            theta_user[u] = g + ind;
            ind += K;
        }
        theta_item = new double*[n_items];
        for (int i=0; i < n_items; i++) {
            theta_item[i] = g + ind;
            ind += K;
        }
        gamma_user = new double*[n_users];
        for (int u = 0; u < n_users; u++) {
            gamma_user[u] = g + ind;
            ind += K;
        }
        gamma_item = new double*[n_items];
        for (int i=0; i < n_items; i++) {
            gamma_item[i] = g + ind;
            ind += K;
        }

        background_topic = g + ind;
        ind += K;
        topic_words = new double*[K];
        for (int k=0; k < K; k++) {
            topic_words[k] = g + ind;
            ind += n_words;
        }
        vb_word_topics = new double*[n_words];
        for (int i=0; i < n_words; i++) {
            vb_word_topics[i] = g + ind;
            ind += K;
        }
        
        FILE* f = utils::fopen_(model_path, "r");
        utils::fread_(W, sizeof(double), NW, f);
        fclose(f);
    }
};
