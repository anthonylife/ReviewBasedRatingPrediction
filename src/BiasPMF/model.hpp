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
// Date: 2014/6/13                                               //
// Model Implementation (BiasMF).                                //
// PMF, NMF, SVD (Bias MF), SVD++, and review extended SVD.      //
///////////////////////////////////////////////////////////////////

#include "../utils.hpp"
#include "corpus.hpp"

//#define DYNAMIC_LR
#define MINVAL 1e-5
#define BTOPIC_PEAKVAL 1e-2

//#define NONNEG_SVD
//#define NONNEG_SVDPP


using namespace std;
using namespace __gnu_cxx;

class MF{
    //******Model parameter needed to be specified by User*******//
    /// Method control variable                                  //
    int niters;                                                  //
    int inner_niters;       // number of inner iterations        //
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
    double reg_tu;          // reg hyper for user transform mat  //
    double reg_ti;          // reg hyper for item transform mat  //
    double psai_u;          // gaussian hyper for user factor    //
    double psai_i;          // gaussian hyper for item factor    //
    double psai_i2;         // gaussian hyper for item factor    //
    double sigma_u;         // gaussian hyper for user bias      //
    double sigma_i;         // gaussian hyper for item bias      //
    double sigma_a;         // gaussian hyper for average        //
    int max_words;          // maximal number of words used      //
    double alpha;           // relative weight of two objs       //
    double kappa;           // peak value of exp func            //
    ///////////////////////////////////////////////////////////////

    //*******Model Parameter needed to be learned by Model*******//
    double ** theta_user;       // user latent factor            //
    double ** theta_item;       // item latent factor            //
    double ** theta_item2;      // second item latent factor     //
    double ** theta_term;       // term factor                   //
    double ** gamma_user;       // user topic factor             //
    double ** gamma_item;       // item topic factor             //
    double * tf_user;           // user transform factor         //
    double * tf_item;           // item transform factor         //
    double * b_user;            // user rating bias              //
    double * b_item;            // item rating bias              //
    double * mu;                // total rating average          //
    double ** topic_words;      // word distribution of topics   //
    double * background_topic;  // backgroud topic distribution  //
    double ** vb_word_topics;   // topic distribution of word    //
    double * svdpp_val;         // caching temporary values      //
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

    int train_method;
    bool restart_tag;
    char* trdata_path;
    char* vadata_path;
    char* tedata_path;
    char* model_path;

public:
    MF(char* trdata_path, char* vadata_path, char* tedata_path,
            char* model_path, int niters, double delta, double lr, int K,
            double psai_u, double psai_i, double psai_i2, double sigma_u,
            double sigma_i, double sigma_a,
            int max_words, int train_method, bool restart_tag){
        this->trdata_path = trdata_path;
        this->vadata_path = vadata_path;
        this->tedata_path = tedata_path;
        this->model_path  = model_path;
        this->niters      = niters;
        this->delta       = delta;
        this->lr          = lr;
        this->K           = K;
        this->psai_u      = psai_u;
        this->psai_i      = psai_i;
        this->psai_i2     = psai_i2;
        this->sigma_u     = sigma_u;
        this->sigma_i     = sigma_i;
        this->sigma_a     = sigma_a;
        this->max_words   = max_words;
        this->train_method= train_method;
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
            ntraining_puser[(*it)->user] += (*it)->words.size();
            ntraining_pitem[(*it)->item] += (*it)->words.size();
        }
        for (vector<Vote*>::iterator it = corp->TE_V->begin();
                it != corp->TE_V->end(); it++)
            test_votes.push_back(*it);
        for (vector<Vote*>::iterator it = corp->VA_V->begin();
                it != corp->VA_V->end(); it++)
            vali_votes.push_back(*it);
       
        svdpp_val = new double[K];
        if (restart_tag == true) {
            printf("Para initialization from restart.\n");
            modelParaInit();
        } else {
            printf("Para loading from trained model.\n");
            loadModelPara();
        }
        printf("Finishing all initialization.\n");
    }
    
    ~MF() {
        delete corp;
        delete W;
        delete[] train_votes_puser;
        delete[] train_votes_pitem;
        
        if (theta_user) {
            delete[] theta_user;
            delete[] theta_item;
            delete[] theta_item2;
            delete[] theta_term;
            delete[] gamma_user;
            delete[] gamma_item;
            delete tf_user;
            delete tf_item;
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
        NW = 1 + (n_users+n_items)*(K+1) + n_items*K + (n_users+n_items+1+2)*K + 2*K*n_words + n_words*K;
        W = new double[NW];
        memset(W, 0, sizeof(double)*NW);

        allocMemForPara(W, &mu, &b_user, &b_item, &theta_user,
                &theta_item, &theta_item2, &gamma_user, &gamma_item,
                &tf_user, &tf_item, &background_topic,
                &topic_words, &vb_word_topics, &theta_term);

        // set mu to the average
        /*for (vector<Vote*>::iterator it=train_votes.begin();
                it!=train_votes.end(); it++)
            *mu += (*it)->value;
        *mu /= train_votes.size();*/
        *mu = 0;
        
        // set user and item rating bias
        /*for (vector<Vote*>::iterator it=train_votes.begin();
                it!=train_votes.end(); it++) {
            b_user[(*it)->user] += (*it)->value - *mu;
            b_item[(*it)->item] += (*it)->value - *mu;
        }
        for (int u=0; u<n_users; u++)
            b_user[u] /= train_votes_puser[u].size();
        for (int i=0; i<n_items; i++) 
            b_item[i] /= train_votes_pitem[i].size();*/
        for (int u=0; u<n_users; u++)
            b_user[u] = 0.0;
        for (int i=0; i<n_items; i++) 
            b_item[i] = 0.0;
        
        // randomly init user bias, item bias and average
        /*utils::muldimPosUniform(mu, 1, 1.0);
        utils::muldimPosUniform(b_user, n_users, 1.0);
        utils::muldimPosUniform(b_item, n_items, 1.0);*/

        // randomly init user and item latent factor and topic factor
        for (int u=0; u<n_users; u++) {
            utils::muldimGaussrand(theta_user[u], K);
            //utils::muldimUniform(theta_user[u], K);
            //utils::muldimZero(theta_user[u], K);
            //utils::muldimPosUniform(theta_user[u], K, 1.0);
            //utils::muldimGaussrand(gamma_user[u], K);
            //utils::muldimUniform(gamma_user[u], K);
            //utils::muldimZero(gamma_user[u], K);
            utils::muldimPosUniform(gamma_user[u], K, 1);
        }
        utils::muldimGaussrand(tf_user, K);
        for (int i=0; i<n_items; i++) {
            utils::muldimGaussrand(theta_item[i], K);
            utils::muldimGaussrand(theta_item2[i], K);
            //utils::muldimUniform(theta_item[i], K);
            //utils::muldimZero(theta_item[i], K);
            //utils::muldimPosUniform(theta_item[i], K, 1.0);
            //utils::muldimGaussrand(gamma_item[i], K);
            //utils::muldimUniform(gamma_item[i], K);
            //utils::muldimZero(gamma_item[i], K);
            utils::muldimPosUniform(gamma_item[i], K, 1);
        }
        utils::muldimGaussrand(tf_item, K);
       
        // background topic factor initialization
        //for (int i=0; i<K; i++)
            //background_topic[i] = 1.0/K;
        //utils::muldimZero(background_topic, K);
        //utils::muldimGaussrand(background_topic, K);
        utils::muldimPosUniform(background_topic, K, 1);

        // topic words dictionary with K bases
        for (int k=0; k<K; k++)
            for (int i=0; i<n_words; i++)
                topic_words[k][i] = 1.0/n_words;
        for (int w=0; w<n_words; w++)
            utils::muldimGaussrand(theta_term[w], K);

    }

    void allocMemForPara(double* g,
                         double** gmu,
                         double** gb_user,
                         double** gb_item,
                         double*** gtheta_user,
                         double*** gtheta_item,
                         double*** gtheta_item2,
                         double*** ggamma_user,
                         double*** ggamma_item,
                         double** gtf_user,
                         double** gtf_item,
                         double** gbackground_topic,
                         double*** gtopic_words,
                         double*** gword_topics,
                         double*** gtheta_term) {
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
        *gtheta_item2 = new double*[n_items];
        for (int i=0; i < n_items; i++) {
            (*gtheta_item2)[i] = g + ind;
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
        *gtf_user = g + ind;
        ind += K;
        *gtf_item = g + ind;
        ind += K;
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
        *gtheta_term = new double*[n_words];
        for (int w=0; w<n_words; w++) {
            (*gtheta_term)[w] = g + ind;
            ind += K;
        }
    }

    void freeMemForPara(double** gmu,
                        double** gb_user,
                        double** gb_item,
                        double*** gtheta_user,
                        double*** gtheta_item,
                        double*** gtheta_item2,
                        double*** ggamma_user,
                        double*** ggamma_item,
                        double** gtf_user,
                        double** gtf_item,
                        double** gbackground_topic,
                        double*** gtopic_words,
                        double*** gword_topics) {
        delete *gmu;
        delete *gb_user;
        delete *gb_item;
        delete *gbackground_topic;
        delete[] (*gtheta_user);
        delete[] (*gtheta_item);
        delete[] (*gtheta_item2);
        delete[] (*ggamma_user);
        delete[] (*ggamma_item);
        delete (*gtf_user);
        delete (*gtf_item);
        delete[] (*gtopic_words);
        delete[] (*gword_topics);
    }

    void train() {
        if (train_method == 0) {
            // PMF
            pmf();
        } else if (train_method == 1) {
            // Non-negative MF
            nmf();
        } else if (train_method == 2) {
            // Bias MF
            biasPmf();
        } else if (train_method == 3) {
            // Bias MF + implicit info
            //svdpp();
            svdpp1();
        } else if (train_method == 4) {
            // Bias MF
            reviewExtendedBiasPmf();
        } else {
            printf("Invalid choice of learning method!\n");
            exit(1);
        }
        saveModelPara();
    }

    void pmf() {
        double train_rmse, valid_rmse, test_rmse;
        double obj_new, obj_old, best_valid, cur_valid, obj_test;
        int user, item, rating;
        double *gtheta_user, *gtheta_item;
        double res;
        int cur_iter, complete;
        bool converged; 
        double lr_c = lr;

        gtheta_user = new double[K];
        gtheta_item = new double[K];
            
        printf("Running the model of PMF ......\n");
        printf("Before SGD learning, evaluation results are following:\n");
        evalRmseError(train_rmse, valid_rmse, test_rmse, 0);
        printf("\tValid RMSE=%.6f, Test RMSE=%.6f;", valid_rmse, test_rmse);
        cur_iter = 0;
        obj_old = 1e5;
        best_valid = 1e5;
        converged = false;
        timeval start_t, end_t;
        
        utils::tic(start_t);
        while(!converged && cur_iter < niters) {
            random_shuffle(train_votes.begin(), train_votes.end());
            complete = 0;
#ifdef DYNAMIC_LR
            lr = lr_c/(cur_iter+10);
#endif
            for (vector<Vote*>::iterator it=train_votes.begin();
                    it!=train_votes.end(); it++) {
                res = prediction(*it, 0) - (*it)->value;
                user = (*it)->user;
                item = (*it)->item;
                for (int k=0; k<K; k++) {
                    // compute gradient of user latent factor
                    gtheta_user[k] = -theta_item[item][k]*res - psai_u*theta_user[user][k];
                    // compute gradient of item latent factor
                    gtheta_item[k] = -theta_user[user][k]*res - psai_i*theta_item[item][k];
                    // update user latent factor
                    theta_user[user][k] += lr*gtheta_user[k];
                    // update item latent factor
                    theta_item[item][k] += lr*gtheta_item[k]; 
                }
            }
            //printf("\n");
            //utils::toc(start_t, end_t, true);
            evalRmseError(train_rmse, valid_rmse, test_rmse, 0);
            printf("Current iteration: %d, Train RMSE=%.6f, ", cur_iter+1, train_rmse);
            printf("Valid RMSE=%.6f, Test RMSE=%.6f\n", valid_rmse, test_rmse);
            
            cur_valid = valid_rmse;
            if (cur_valid < best_valid) {
                best_valid = cur_valid;
                obj_test = test_rmse;
                for (vector<Vote*>::iterator it=train_votes.begin();
                        it!=train_votes.end(); it++)
                    best_vali_predictions[*it] = prediction(*it, 0);
                for (vector<Vote*>::iterator it=vali_votes.begin();
                        it!=vali_votes.end(); it++)
                    best_vali_predictions[*it] = prediction(*it, 0);
                for (vector<Vote*>::iterator it=test_votes.begin();
                        it!=test_votes.end(); it++)
                    best_vali_predictions[*it] = prediction(*it, 0);
            }
            //utils::pause();
            if (cur_iter == 0)
                obj_old = train_rmse;
            else {
                obj_new = train_rmse;
                //if (obj_new >= obj_old || fabs(obj_new-obj_old) < delta)
                //    converged = true;
                obj_old = obj_new;
            }
            cur_iter += 1;
        }
        printf("PMF learning finish...\n");
        printf("RMSE on test dataset when validation results achieve best: %f\n", obj_test);
        saveModelPara();

        // release memory
        delete[] gtheta_user;
        delete[] gtheta_item;
    }
    
    void nmf() {
        double train_rmse, valid_rmse, test_rmse;
        double obj_new, obj_old, best_valid, cur_valid, obj_test;
        int user, item, rating;
        double *gtheta_user, *gtheta_item;
        double res;
        int cur_iter, complete;
        bool converged; 
        double lr_c = lr;

        gtheta_user = new double[K];
        gtheta_item = new double[K];
            
        printf("Running the model of NMF ......\n");
        printf("Before SGD learning, evaluation results are following:\n");
        evalRmseError(train_rmse, valid_rmse, test_rmse, 0);
        printf("\tValid RMSE=%.6f, Test RMSE=%.6f;", valid_rmse, test_rmse);
        cur_iter = 0;
        obj_old = 1e5;
        best_valid = 1e5;
        converged = false;
        timeval start_t, end_t;
        
        utils::tic(start_t);
        while(!converged && cur_iter < niters) {
            random_shuffle(train_votes.begin(), train_votes.end());
            complete = 0;
#ifdef DYNAMIC_LR
            lr = lr_c/(cur_iter+10);
#endif
            for (vector<Vote*>::iterator it=train_votes.begin();
                    it!=train_votes.end(); it++) {
                res = prediction(*it, 0) - (*it)->value;
                user = (*it)->user;
                item = (*it)->item;
                for (int k=0; k<K; k++) {
                    // compute gradient of user latent factor
                    gtheta_user[k] = -theta_item[item][k]*res - psai_u*theta_user[user][k];
                    // compute gradient of item latent factor
                    gtheta_item[k] = -theta_user[user][k]*res - psai_i*theta_item[item][k];
                    // update user latent factor
                    theta_user[user][k] += lr*gtheta_user[k];
                    utils::max(0.0, theta_user[user][k]);
                    // update item latent factor
                    theta_item[item][k] += lr*gtheta_item[k]; 
                    utils::max(0.0, theta_item[item][k]);
                }
            }
            //printf("\n");
            //utils::toc(start_t, end_t, true);
            evalRmseError(train_rmse, valid_rmse, test_rmse, 0);
            printf("Current iteration: %d, Train RMSE=%.6f, ", cur_iter+1, train_rmse);
            printf("Valid RMSE=%.6f, Test RMSE=%.6f\n", valid_rmse, test_rmse);
            
            cur_valid = valid_rmse;
            if (cur_valid < best_valid) {
                best_valid = cur_valid;
                obj_test = test_rmse;
                for (vector<Vote*>::iterator it=train_votes.begin();
                        it!=train_votes.end(); it++)
                    best_vali_predictions[*it] = prediction(*it, 0);
                for (vector<Vote*>::iterator it=vali_votes.begin();
                        it!=vali_votes.end(); it++)
                    best_vali_predictions[*it] = prediction(*it, 0);
                for (vector<Vote*>::iterator it=test_votes.begin();
                        it!=test_votes.end(); it++)
                    best_vali_predictions[*it] = prediction(*it, 0);
            }
            //utils::pause();
            if (cur_iter == 0)
                obj_old = train_rmse;
            else {
                obj_new = train_rmse;
                //if (obj_new >= obj_old || fabs(obj_new-obj_old) < delta)
                //    converged = true;
                obj_old = obj_new;
            }
            cur_iter += 1;
        }
        printf("NMF learning finish...\n");
        printf("RMSE on test dataset when validation results achieve best: %f\n", obj_test);
        saveModelPara();

        // release memory
        delete[] gtheta_user;
        delete[] gtheta_item;
    }

    void biasPmf() {
        double train_rmse, valid_rmse, test_rmse;
        double obj_new, obj_old, best_valid, cur_valid;
        int user, item, rating;
        double *gtheta_user, *gtheta_item;
        double res;
        int cur_iter, complete;
        bool converged; 
        double lr_c = lr;

        gtheta_user = new double[K];
        gtheta_item = new double[K];
            
        printf("Running the model of SVD (BiasMF) ......\n");
        printf("Before SGD learning, evaluation results are following:\n");
        evalRmseError(train_rmse, valid_rmse, test_rmse, 1);
        printf("\tValid RMSE=%.6f, Test RMSE=%.6f;", valid_rmse, test_rmse);
        cur_iter = 0;
        obj_old = 1e5;
        best_valid = 1e5;
        converged = false;
        timeval start_t, end_t;
        
        utils::tic(start_t);
        while(!converged && cur_iter < niters) {
            random_shuffle(train_votes.begin(), train_votes.end());
            complete = 0;
#ifdef DYNAMIC_LR
            lr = lr_c/(cur_iter+10);
#endif
            for (vector<Vote*>::iterator it=train_votes.begin();
                    it!=train_votes.end(); it++) {
                res = prediction(*it, 1) - (*it)->value;
                user = (*it)->user;
                item = (*it)->item;
                for (int k=0; k<K; k++) {
                    // compute gradient of user latent factor
                    gtheta_user[k] = -theta_item[item][k]*res - psai_u*theta_user[user][k];
                    // compute gradient of item latent factor
                    gtheta_item[k] = -theta_user[user][k]*res - psai_i*theta_item[item][k];
                    // update user latent factor
                    theta_user[user][k] += lr*gtheta_user[k];
                    // update item latent factor
                    theta_item[item][k] += lr*gtheta_item[k]; 
                }
                // compute user bias gradient and update 
                b_user[user] += lr*(-res-sigma_u*b_user[user]);
                // compute item bias gradient and update 
                b_item[item] += lr*(-res-sigma_i*b_item[item]);
                // compute gradient of average para and update
                //*mu += lr*(-res-sigma_a*(*mu));
                *mu += lr*(-res);
                
            }
            //printf("\n");
            //utils::toc(start_t, end_t, true);
            evalRmseError(train_rmse, valid_rmse, test_rmse, 1);
            printf("Current iteration: %d, Train RMSE=%.6f, ", cur_iter+1, train_rmse);
            printf("Valid RMSE=%.6f, Test RMSE=%.6f\n", valid_rmse, test_rmse);
            
            cur_valid = valid_rmse;
            if (cur_valid < best_valid) {
                best_valid = cur_valid;
                for (vector<Vote*>::iterator it=train_votes.begin();
                        it!=train_votes.end(); it++)
                    best_vali_predictions[*it] = prediction(*it, 1);
                for (vector<Vote*>::iterator it=vali_votes.begin();
                        it!=vali_votes.end(); it++)
                    best_vali_predictions[*it] = prediction(*it, 1);
                for (vector<Vote*>::iterator it=test_votes.begin();
                        it!=test_votes.end(); it++)
                    best_vali_predictions[*it] = prediction(*it, 1);
            }
            //utils::pause();
            if (cur_iter == 0)
                obj_old = train_rmse;
            else {
                obj_new = train_rmse;
                //if (obj_new >= obj_old || fabs(obj_new-obj_old) < delta)
                //    converged = true;
                obj_old = obj_new;
            }
            cur_iter += 1;
        }
        printf("BiasMF learning finish...\n");
        saveModelPara();

        // release memory
        delete[] gtheta_user;
        delete[] gtheta_item;
    }
    
    void svdpp() {
        double train_rmse, valid_rmse, test_rmse;
        double obj_new, obj_old, best_valid, cur_valid;
        int user, item, rating;
        double *gtheta_user, *gtheta_item, *gtheta_item2;
        double res, sqrt_num;
        int cur_iter, complete;
        bool converged; 
        double lr_c = lr;

        gtheta_user = new double[K];
        gtheta_item = new double[K];
        gtheta_item2 = new double[K];

        printf("Running the model of SVDPP (BiasMF+implicit feedback) ......\n");
        printf("Before SGD learning, evaluation results are following:\n");
        evalRmseError(train_rmse, valid_rmse, test_rmse, 2);
        printf("\tValid RMSE=%.6f, Test RMSE=%.6f;", valid_rmse, test_rmse);
        cur_iter = 0;
        obj_old = 1e5;
        best_valid = 1e5;
        converged = false;
        timeval start_t, end_t;
        
        utils::tic(start_t);
        while(!converged && cur_iter < niters) {
            random_shuffle(train_votes.begin(), train_votes.end());
            complete = 0;
#ifdef DYNAMIC_LR
            lr = lr_c/(cur_iter+10);
#endif
            for (vector<Vote*>::iterator it=train_votes.begin();
                    it!=train_votes.end(); it++) {
                res = prediction(*it, 2) - (*it)->value;
                user = (*it)->user;
                item = (*it)->item;
                sqrt_num = 1.0/sqrt(train_votes_puser[user].size());
                for (int k=0; k<K; k++) {
                    // compute gradient of user latent factor
                    gtheta_user[k] = -theta_item[item][k]*res - psai_u*theta_user[user][k];
                    // compute gradient of item latent factor
                    gtheta_item[k] = -theta_user[user][k]*res - psai_i*theta_item[item][k];
                    // update implicit item latent factor2
                    for (vector<Vote*>::iterator it1=train_votes_puser[user].begin();
                            it1!=train_votes_puser[user].end(); it1++)
                        theta_item2[(*it1)->item][k] += lr*(-theta_item[item][k]*res*sqrt_num - psai_i2*theta_item2[(*it1)->item][k]);
                    // update user latent factor
                    theta_user[user][k] += lr*gtheta_user[k];
                    // update item latent factor
                    theta_item[item][k] += lr*gtheta_item[k];
                }
                // compute user bias gradient and update 
                b_user[user] += lr*(-res-sigma_u*b_user[user]);
                // compute item bias gradient and update 
                b_item[item] += lr*(-res-sigma_i*b_item[item]);
                // compute gradient of average para and update
                //*mu += lr*(-res-sigma_a*(*mu));
                *mu += lr*(-res);
                
            }
            //printf("\n");
            //utils::toc(start_t, end_t, true);
            evalRmseError(train_rmse, valid_rmse, test_rmse, 2);
            printf("Current iteration: %d, Train RMSE=%.6f, ", cur_iter+1, train_rmse);
            printf("Valid RMSE=%.6f, Test RMSE=%.6f\n", valid_rmse, test_rmse);
            
            cur_valid = valid_rmse;
            if (cur_valid < best_valid) {
                best_valid = cur_valid;
                for (vector<Vote*>::iterator it=train_votes.begin();
                        it!=train_votes.end(); it++)
                    best_vali_predictions[*it] = prediction(*it, 2);
                for (vector<Vote*>::iterator it=vali_votes.begin();
                        it!=vali_votes.end(); it++)
                    best_vali_predictions[*it] = prediction(*it, 2);
                for (vector<Vote*>::iterator it=test_votes.begin();
                        it!=test_votes.end(); it++)
                    best_vali_predictions[*it] = prediction(*it, 2);
            }
            //utils::pause();
            if (cur_iter == 0)
                obj_old = train_rmse;
            else {
                obj_new = train_rmse;
                //if (obj_new >= obj_old || fabs(obj_new-obj_old) < delta)
                //    converged = true;
                obj_old = obj_new;
            }
            cur_iter += 1;
        }
        printf("BiasMF learning finish...\n");
        saveModelPara();

        // release memory
        delete[] gtheta_user;
        delete[] gtheta_item;
    }
    
    void svdpp1() {
        double train_rmse, valid_rmse, test_rmse;
        double obj_new, obj_old, best_valid, cur_valid;
        int user, item, rating;
        double *gtheta_user, *gtheta_item, *gtheta_item2;
        double res, sqrt_num;
        int cur_iter, complete;
        bool converged; 
        double lr_c = lr;

        gtheta_user = new double[K];
        gtheta_item = new double[K];
        gtheta_item2 = new double[K];

        printf("Running the model of SVDPP (BiasMF+implicit feedback) ......\n");
        printf("Before SGD learning, evaluation results are following:\n");
        evalRmseError(train_rmse, valid_rmse, test_rmse, 2);
        printf("\tValid RMSE=%.6f, Test RMSE=%.6f;", valid_rmse, test_rmse);
        cur_iter = 0;
        obj_old = 1e5;
        best_valid = 1e5;
        converged = false;
        timeval start_t, end_t;
        
        utils::tic(start_t);
        while(!converged && cur_iter < niters) {
            random_shuffle(train_votes.begin(), train_votes.end());
            complete = 0;
#ifdef DYNAMIC_LR
            lr = lr_c/(cur_iter+10);
#endif
            for (vector<Vote*>::iterator it=train_votes.begin();
                    it!=train_votes.end(); it++) {
                res = prediction(*it, 2) - (*it)->value;
                user = (*it)->user;
                item = (*it)->item;
                sqrt_num = 1.0/sqrt(train_votes_puser[user].size());
                for (int k=0; k<K; k++) {
                    // compute gradient of user latent factor
                    gtheta_user[k] = -theta_item[item][k]*res - psai_u*theta_user[user][k];
                    // compute gradient of item latent factor
                    gtheta_item[k] = -theta_user[user][k]*res - psai_i*theta_item[item][k];
                    // update implicit item latent factor2
                    theta_item2[item][k] += lr*(-theta_item[item][k]*res*sqrt_num - psai_i2*theta_item2[item][k]);
                    // update user latent factor
                    theta_user[user][k] += lr*gtheta_user[k];
                    // update item latent factor
                    theta_item[item][k] += lr*gtheta_item[k];
                }
                // compute user bias gradient and update 
                b_user[user] += lr*(-res-sigma_u*b_user[user]);
                // compute item bias gradient and update 
                b_item[item] += lr*(-res-sigma_i*b_item[item]);
                // compute gradient of average para and update
                //*mu += lr*(-res-sigma_a*(*mu));
                *mu += lr*(-res);
                
            }
            //printf("\n");
            //utils::toc(start_t, end_t, true);
            evalRmseError(train_rmse, valid_rmse, test_rmse, 2);
            printf("Current iteration: %d, Train RMSE=%.6f, ", cur_iter+1, train_rmse);
            printf("Valid RMSE=%.6f, Test RMSE=%.6f\n", valid_rmse, test_rmse);
            
            cur_valid = valid_rmse;
            if (cur_valid < best_valid) {
                best_valid = cur_valid;
                for (vector<Vote*>::iterator it=train_votes.begin();
                        it!=train_votes.end(); it++)
                    best_vali_predictions[*it] = prediction(*it, 2);
                for (vector<Vote*>::iterator it=vali_votes.begin();
                        it!=vali_votes.end(); it++)
                    best_vali_predictions[*it] = prediction(*it, 2);
                for (vector<Vote*>::iterator it=test_votes.begin();
                        it!=test_votes.end(); it++)
                    best_vali_predictions[*it] = prediction(*it, 2);
            }
            //utils::pause();
            if (cur_iter == 0)
                obj_old = train_rmse;
            else {
                obj_new = train_rmse;
                //if (obj_new >= obj_old || fabs(obj_new-obj_old) < delta)
                //    converged = true;
                obj_old = obj_new;
            }
            cur_iter += 1;
        }
        printf("BiasMF learning finish...\n");
        saveModelPara();

        // release memory
        delete[] gtheta_user;
        delete[] gtheta_item;
    }
    
    void reviewExtendedBiasPmf() {
        double train_rmse, valid_rmse, test_rmse;
        double obj_new, obj_old, best_valid, cur_valid;
        int user, item, rating;
        double *gtheta_user, *gtheta_item, *gtheta_term;
        double res;
        int cur_iter, complete;
        bool converged; 
        double lr_c = lr;

        gtheta_user = new double[K];
        gtheta_item = new double[K];
        gtheta_term = new double[K];
            
        printf("Running the model of review extended BiasMF ......\n");
        printf("Before SGD learning, evaluation results are following:\n");
        evalRmseError(train_rmse, valid_rmse, test_rmse, 3);
        printf("\tValid RMSE=%.6f, Test RMSE=%.6f;", valid_rmse, test_rmse);
        cur_iter = 0;
        obj_old = 1e5;
        best_valid = 1e5;
        converged = false;
        timeval start_t, end_t;
        
        utils::tic(start_t);
        while(!converged && cur_iter < niters) {
            random_shuffle(train_votes.begin(), train_votes.end());
            complete = 0;
#ifdef DYNAMIC_LR
            lr = lr_c/(cur_iter+10);
#endif
            for (vector<Vote*>::iterator it=train_votes.begin();
                    it!=train_votes.end(); it++) {
                res = prediction(*it, 3) - (*it)->value;
                user = (*it)->user;
                item = (*it)->item;
                for (int k=0; k<K; k++) {
                    // compute gradient of user latent factor
                    gtheta_user[k] = -theta_item[item][k]*res - psai_u*theta_user[user][k];
                    // compute gradient of item latent factor
                    gtheta_item[k] = -theta_user[user][k]*res - psai_i*theta_item[item][k];
                    // compute gradient of term latent factor and update
                    if ((*it)->words.size() > 0) {
                        for (vector<int>::iterator it1=(*it)->words.begin();
                                it1!=(*it)->words.end(); it1++)
                            theta_term[*it1][k]+= lr*(-theta_user[user][k]*res/(*it)->words.size()-psai_i2*theta_term[*it1][k]);
                    }
                    // update user latent factor
                    theta_user[user][k] += lr*gtheta_user[k];
                    // update item latent factor
                    theta_item[item][k] += lr*gtheta_item[k];

                }
                // compute user bias gradient and update 
                b_user[user] += lr*(-res-sigma_u*b_user[user]);
                // compute item bias gradient and update 
                b_item[item] += lr*(-res-sigma_i*b_item[item]);
                // compute gradient of average para and update
                //*mu += lr*(-res-sigma_a*(*mu));
                *mu += lr*(-res);
                
            }
            //printf("\n");
            //utils::toc(start_t, end_t, true);
            evalRmseError(train_rmse, valid_rmse, test_rmse, 3);
            printf("Current iteration: %d, Train RMSE=%.6f, ", cur_iter+1, train_rmse);
            printf("Valid RMSE=%.6f, Test RMSE=%.6f\n", valid_rmse, test_rmse);
            
            cur_valid = valid_rmse;
            if (cur_valid < best_valid) {
                best_valid = cur_valid;
                for (vector<Vote*>::iterator it=train_votes.begin();
                        it!=train_votes.end(); it++)
                    best_vali_predictions[*it] = prediction(*it, 3);
                for (vector<Vote*>::iterator it=vali_votes.begin();
                        it!=vali_votes.end(); it++)
                    best_vali_predictions[*it] = prediction(*it, 3);
                for (vector<Vote*>::iterator it=test_votes.begin();
                        it!=test_votes.end(); it++)
                    best_vali_predictions[*it] = prediction(*it, 3);
            }
            //utils::pause();
            if (cur_iter == 0)
                obj_old = train_rmse;
            else {
                obj_new = train_rmse;
                //if (obj_new >= obj_old || fabs(obj_new-obj_old) < delta)
                //    converged = true;
                obj_old = obj_new;
            }
            cur_iter += 1;
        }
        printf("Review extended BiasMF learning finish...\n");
        saveModelPara();

        // release memory
        delete[] gtheta_user;
        delete[] gtheta_item;
        delete[] gtheta_term;
    }
    
    void gdBatchLearning() {
        double train_rmse, valid_rmse, test_rmse;
        double train_perp, valid_perp, test_perp;
    }
    
    void coordinateBatchLearning() {
        double train_rmse, valid_rmse, test_rmse;
        double train_perp, valid_perp, test_perp;
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
    }
    
    void calDocTopic(double ** doc_topic, double *exp_gu, double *exp_gi,
            double *exp_bt, double& exp_sum_gu, double&exp_sum_gi,
            double& exp_sum, Vote * v) {
        exp_sum = 0.0;
        exp_sum_gu = 0.0;
        exp_sum_gi = 0.0;
        for (int k=0; k<K; k++) {
            exp_gu[k] = exp(kappa*gamma_user[v->user][k]);
            exp_sum_gu += exp_gu[k];
            exp_gi[k] = exp(kappa*gamma_item[v->item][k]);
            exp_sum_gi += exp_gi[k];
            exp_bt[k] = exp(kappa*background_topic[k]);
            (*doc_topic)[k] = exp_gu[k]*exp_gi[k]*exp_bt[k];
            exp_sum += (*doc_topic)[k];
        }
        if (exp_sum == 0) {
            printf("exp_sum=0\n");
            exit(1);
        }
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

    void truncatedGradient(double * weights, double g, int ndim) {
        for (int i=0; i<ndim; i++) {
            if (weights[i] <= truncated_theta && weights[i] > 0)
                weights[i] = utils::max(0, weights[i]-lr*truncated_k*g);
#ifdef TOPIC_PARA_NONNEG
            if (weights[i] <= 0)
                weights[i] = 0.0;
#else
            if (weights[i] >= -truncated_theta && weights[i] < 0)
                weights[i] = utils::min(0, weights[i]+lr*truncated_k*g);
#endif
        }
    }

    inline double prediction(Vote * v, int tag) {
        if (tag == 0)
            return utils::dot(theta_user[v->user], theta_item[v->item], K);
        else if (tag == 1)
            return *mu + b_user[v->user] + b_item[v->item] 
                   + utils::dot(theta_user[v->user], theta_item[v->item], K);
        else if (tag == 2) {
            double first = *mu + b_user[v->user] + b_item[v->item] 
                   + utils::dot(theta_user[v->user], theta_item[v->item], K);
            memset(svdpp_val, 0.0, sizeof(double)*K);
            for (vector<Vote*>::iterator it=train_votes_puser[v->user].begin();
                    it!=train_votes_puser[v->user].end(); it++) {
                for (int k=0; k<K; k++)
                    svdpp_val[k] += theta_item2[(*it)->item][k];
            }
            return first+1.0/sqrt(train_votes_puser[v->user].size())*utils::dot(theta_user[v->user], svdpp_val, K);
        } else if (tag == 3) {
            double first = *mu + b_user[v->user] + b_item[v->item] 
                   + utils::dot(theta_user[v->user], theta_item[v->item], K);
            memset(svdpp_val, 0.0, sizeof(double)*K);
            if (v->words.size() == 0)
                return first;
            for (vector<int>::iterator it=v->words.begin();
                    it!=v->words.end(); it++) {
                for (int k=0; k<K; k++)
                    svdpp_val[k] += theta_term[*it][k];
            }
            return first+1.0/v->words.size()*utils::dot(theta_user[v->user], svdpp_val, K);
        } else {
            printf("Invalid choice of tag!\n");
            exit(1);
        }
    }
   
    inline double predictionWithoutFactorProd(Vote * v) {
        return *mu + b_item[v->item] + b_user[v->user];
    }
    
    inline double predictionWithoutFactorProd(Vote * v, int k) {
        return *mu + b_item[v->item] + b_user[v->user]
                + utils::dot(theta_user[v->user], theta_item[v->item], K, k);
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

    void evalRmseError(double& train, double& valid, double& test, int tag) {
        train = 0.0, valid = 0.0, test = 0.0;
        for (vector<Vote*>::iterator it = train_votes.begin();
                it != train_votes.end(); it++)
            train += utils::square(prediction(*it, tag) - (*it)->value) ;
        for (vector<Vote*>::iterator it = vali_votes.begin();
                it != vali_votes.end(); it++)
            valid += utils::square(prediction(*it, tag) - (*it)->value) ;
        for (vector<Vote*>::iterator it = test_votes.begin();
                it != test_votes.end(); it++)
            test += utils::square(prediction(*it, tag) - (*it)->value) ;
        //cout << "Train: " << train << ", Size: " << train_votes.size() << endl;
        train = sqrt(train/train_votes.size());
        //cout << "Valid: " << valid << ", Size: " << vali_votes.size() << endl;
        valid = sqrt(valid/vali_votes.size());
        //cout << "Test: " << test << ", Size: " << test_votes.size() << endl;
        test = sqrt(test/test_votes.size());
    } 

    void evalPerplexity(double& train, double& valid, double& test) {
        int word_cnt = 0;
        double exp_sum = 0.0, word_log_prob;
        double * doc_topic_prob = new double[K];
        
        train = 0.0, word_cnt = 0;
        for (vector<Vote*>::iterator it = train_votes.begin();
                it != train_votes.end(); it++) {
            word_cnt += (*it)->words.size();
            calDocTopic(&doc_topic_prob, exp_sum, *it);
            for (int k=0; k<K; k++)
                doc_topic_prob[k] /= exp_sum;
            for (vector<int>::iterator it1 = (*it)->words.begin();
                    it1!=(*it)->words.end(); it1++) {
                word_log_prob = 0.0;
                for (int k=0; k<K; k++)
                    word_log_prob += doc_topic_prob[k]*topic_words[k][*it1];
                train += log(word_log_prob);
                //printf("word_log_prob: %f, ", log(word_log_prob));
                //utils::pause();
            }
        }
        train = exp(-train/word_cnt);
    
        valid = 0.0, word_cnt = 0;
        for (vector<Vote*>::iterator it = vali_votes.begin();
                it != vali_votes.end(); it++) {
            word_cnt += (*it)->words.size();
            calDocTopic(&doc_topic_prob, exp_sum, *it);
            for (int k=0; k<K; k++)
                doc_topic_prob[k] /= exp_sum;
            for (vector<int>::iterator it1 = (*it)->words.begin();
                    it1!=(*it)->words.end(); it1++) {
                word_log_prob = 0.0;
                for (int k=0; k<K; k++)
                    word_log_prob += doc_topic_prob[k]*topic_words[k][*it1];
                valid += log(word_log_prob);
            }
        }
        valid = exp(-valid/word_cnt);
        
        test = 0.0, word_cnt = 0;
        for (vector<Vote*>::iterator it = test_votes.begin();
                it != test_votes.end(); it++) {
            word_cnt += (*it)->words.size();
            calDocTopic(&doc_topic_prob, exp_sum, *it);
            for (int k=0; k<K; k++)
                doc_topic_prob[k] /= exp_sum;
            for (vector<int>::iterator it1 = (*it)->words.begin();
                    it1!=(*it)->words.end(); it1++) {
                word_log_prob = 0.0;
                for (int k=0; k<K; k++)
                    word_log_prob += doc_topic_prob[k]*topic_words[k][*it1];
                test += log(word_log_prob);
            }
        }
        test = exp(-test/word_cnt);
    }

    void outputTopicWords(int topk) {
        for (int k=0; k<K; k++)
            outputTopicWords(k, topk);
        printf("\n");
    }

    void outputTopicWords(int topic, int topk) {
        vector<Wordprob>* results = new vector<Wordprob>();
        for (int w=0; w<n_words; w++) {
            Wordprob wordprob;
            wordprob.id = w;
            wordprob.prob = topic_words[topic][w];
            results->push_back(wordprob);
        }
        sort(results->begin(), results->end(), utils::greaterCmp);
        printf("Topic %d--> ", topic);
        int i=0;
        for (vector<Wordprob>::iterator it=results->begin();
                it!=results->end(); it++) {
            printf("%s:%f ", corp->rword_ids[it->id].c_str(), it->prob);
            i++;
            if (i == topk)
                break;
        }
        printf("||||||");
        delete results;
    }
            
    void outputTransformWeight() {
        vector<Wordprob>* results = new vector<Wordprob>();
        for (int k=0; k<K; k++) {
            Wordprob wordprob;
            wordprob.id = k;
            wordprob.prob = tf_user[k];
            results->push_back(wordprob);
        }
        sort(results->begin(), results->end(), utils::greaterCmp);
        printf("\nUser transform vec: ");
        for (vector<Wordprob>::iterator it=results->begin();
                it!=results->end(); it++)
            printf("%d: %.4f, ", it->id, it->prob);
        
        results->clear();

        for (int k=0; k<K; k++) {
            Wordprob wordprob;
            wordprob.id = k;
            wordprob.prob = tf_item[k];
            results->push_back(wordprob);
        }
        sort(results->begin(), results->end(), utils::greaterCmp);
        printf("\nItem transform vec: ");
        for (vector<Wordprob>::iterator it=results->begin();
                it!=results->end(); it++)
            printf("%d: %.4f, ", it->id, it->prob);
        printf("\n");
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
        // total number of paramters to be learned
        NW = 1 + (n_users+n_items)*(K+1) + (2*n_users+2*n_items+1)*K + 2*K*n_words;
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
        tf_user = g + ind;
        ind += K;
        tf_item = g + ind;
        ind += K;

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
