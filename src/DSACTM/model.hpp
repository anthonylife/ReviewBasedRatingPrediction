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

using namespace std;
using namespace __gnu_cxx;

class DSACTM{
    //******Model parameter needed to be specified by User*******//
    /// Method control variable                                  //
    int niters;                                                  //
                                                                 // 
    // Online learning                                           //
    int minibatch;                                               //
    double lr;              // learning rate for online learning //
                                                                 //
    /// Hyper-parameter setting                                  //
    int K;               // latant factor dimension              //
    double lambda_u;        // laplace hyper for user topic      //
    double lambda_i;        // laplace hyper for item topic      //
    double lambda_b;        // laplace hyper for backgroud topic //
    double psai_u;          // gaussian hyper for user factor    //
    double psai_i;          // gaussian hyper for item factor    //
    double sigma_u;         // gaussian hyper for user bias      //
    double sigma_i;         // gaussian hyper for item bias      //
    double sigma_a;         // gaussian hyper for average        //
    int max_words;          // maximal number of words used      //
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
            char* model_path, int niters, int minibatch, double lr,
            int K, double lambda_u, double lambda_i, double lambda_b,
            double psai_u, double psai_i, double sigma_u, double sigma_i,
            double sigma_a, int max_words, int tr_method, bool restart_tag){
        this->trdata_path = trdata_path;
        this->vadata_path = vadata_path;
        this->tedata_path = tedata_path;
        this->model_path  = model_path;
        this->niters      = niters;
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
            printf("Model para init.\n")
            modelParaInit();
        } else {
            printf("Model para loading.\n")
            loadModelPara();
        }
        printf("Finishing all initialization.\n");
    }
    
    ~DSACTM() {
        delete corp;
        delete[] train_votes_puser;
        delete[] train_votes_pitem;
        
        if (theta_user) {
            delete[] theta_user;
            delete[] theta_item;
            delete[] gamma_user;
            delete[] gamma_item;
            delete[] topic_words;
            delete mu;
            delete b_user;
            delete b_item;
            delete background_topic;
        }

        train_votes.clear();
        vector<Vote*>(train_votes).swap(train_votes);
        vali_votes.clear();
        vector<Vote*>(vali_votes).swap(vali_votes);
        vector<Vote*>(test_votes).swap(test_votes);

    }

    void modelParaInit() {
        // total number of paramters to be learned
        NW = 1 + (n_users+n_items)*(K+1) + (n_users+n_items+1)*K + K*n_words;
        W = new double[NW];
        memset(W, 0, sizeof(double)*NW);

        allocMemForPara(W, &mu, &b_user, &b_item, &theta_user, &theta_item,
                &gamma_user, &gamma_item, &background_topic, &topic_words);

        // set mu to the average
        /*for (vector<Vote*>::iterator it=train_votes.begin();
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
            b_user[u] /= train_votes_puser[u];
        for (int i=0; i<n_items; i++) 
            b_item[i] /= train_votes_pitem[i];*/

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
            utils::muldimPosUniform(theta_item[u], K, 1.0);
            //utils::muldimGaussrand(gamma_item[i], K);
            //utils::muldimUniform(gamma_item[i], K);
            //uitls::muldimZero(gamma_item[i], K);
            utils::muldimPosUniform(gamma_item[u], K, 1.0);
        }
       
        // background topic factor initialization
        //for (int i=0; i<K; i++)
            //background_topic[i] = 1.0/K;
        //uitls::muldimZero(background_topic, K);
        utils::muldimPosUniform(theta_item[u], K, 1.0);

        // topic words dictionary with K bases
        for (int i=0; i<K; i++)
            for (int j=0; j<n_items; j++)
                topic_words = 1.0/n_items;
    }

    void allocMemForPara(double* g,
                         double** g_mu,
                         double** g_b_user,
                         double** g_b_item,
                         double*** g_theta_user,
                         double*** g_theta_item,
                         double*** g_gamma_user,
                         double*** g_gamma_item,
                         double** g_background_topic,
                         double*** g_topic_words) {
        int ind = 0;
        
        *g_mu = g + ind;
        ind++;

        *g_b_user = g + ind;
        ind += n_users;
        *g_b_item = g + ind;
        ind += n_items;
        
        *g_theta_user = new double*[n_users];
        for (int u = 0; u < n_users; u++) {
            (*g_theta_user)[u] = g + ind;
            ind += K;
        }
        *g_theta_item = new double*[n_items];
        for (int i=0; i < n_items; i++) {
            (*g_theta_item)[i] = g + ind;
            ind += K;
        }
        *g_gamma_user = new double*[n_users];
        for (int u = 0; u < n_users; u++) {
            (*g_gamma_user)[u] = g + ind;
            ind += K;
        }
        *g_gamma_item = new double*[n_items];
        for (int i=0; i < n_items; i++) {
            (*g_gamma_item)[i] = g + ind;
            ind += K;
        }

        *g_background_topic = g + ind;
        ind += K;
        *g_topic_words = new double*[K];
        for (int k=0; k < K; k++) {
            (*g_topic_words)[k] = g + ind;
            g += n_words;
        }
    }

    void freeMemForPara(double** g_mu,
                        double** g_b_user,
                        double** g_b_item,
                        double*** g_theta_user,
                        double*** g_theta_item,
                        double*** g_gamma_user,
                        double*** g_gamma_item,
                        double** g_background_topic,
                        double*** g_topic_words) {
        delete *g_mu;
        delete *g_b_user;
        delete *g_b_item;
        delete *g_background_topic;
        delete[] (*g_theta_user);
        delete[] (*g_theta_item);
        delete[] (*g_gamma_user);
        delete[] (*g_gamma_item);
        delete[] (*g_topic_words);
    }

    void train() {
        if (tr_method == 0) {
            // SGD
            online_learning();
        } else if (tr_method == 1) {
            // minibatch SGD
            minibatch_learning();
        } else if (tr_method == 2) {
            // GD + VB
            batch_learning();
        } else if (tr_method == 3) {
            // Coordinate descent (ALS+VB)
            coordinate_learning();
        } else {
            printf("Invalid choice of learning method!\n");
            exit(1);
        }
    }

    void online_learning() {
        double train_rmse, valid_rmse, test_rmse;
        double train_perp, valid_perp, test_perp;
    
        for ()
    }
    
    void minibatch_learning() {
        double train_rmse, valid_rmse, test_rmse;
        double train_perp, valid_perp, test_perp;
    }
    
    void batch_learning() {
        double train_rmse, valid_rmse, test_rmse;
        double train_perp, valid_perp, test_perp;
    }
    
    void coordinate_learning() {
        double train_rmse, valid_rmse, test_rmse;
        double train_perp, valid_perp, test_perp;
    }

    inline double prediction(Vote * v) {
        return  *mu + b_user[v->user] + b_item[v->item] 
                    + utils::dot(theta_user[v->user], theta_item[v->item]);
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
        test = sqrt(test, test_votes.size());
    } 

    double evalPerplexity(double& train, double& valid, double& test){
    
    }

    void saveModel() {
        FILE* f = utils::fopen_(user_factor_path, "w");
        for (int u=0; u<n_users; u++)
            fwrite(user_factor[u], sizeof(double), ndim, f);
        fclose(f);
    }

    void loadModel() {
        int num_para=0;
        if (bias_tag)
            num_para = (n_users+n_pois)*ndim+n_pois;
        else
            num_para = (n_users+n_pois)*ndim;
        
        double * factor = new double[num_para];
        int ind = 0;
        for (int u=0; u<n_users; u++) {
            user_factor[u] = factor+ind;
            ind += ndim;
        }
        for (int p=0; p<n_pois; p++) {
            poi_factor[p] = factor+ind;
            ind += ndim;
        }
        if (bias_tag) {
            poi_bias = factor+ind;
            ind += n_pois;
        }
        
        FILE* f = utils::fopen_(user_factor_path, "r");
        for (int u=0; u<n_users; u++)
            utils::fread_(user_factor[u], sizeof(double), ndim, f);
        fclose(f);
        f = utils::fopen_(poi_factor_path, "r");
        for (int p=0; p<n_pois; p++)
            utils::fread_(poi_factor[p], sizeof(double), ndim, f);
        fclose(f);
        if (bias_tag) {
            f = utils::fopen_(poi_bias_path, "r");
            utils::fread_(poi_bias, sizeof(double), n_pois, f);
            fclose(f);
        }
    }
};
