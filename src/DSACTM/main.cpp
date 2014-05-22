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
// Running corresponding model (DSACTM)                          //
///////////////////////////////////////////////////////////////////

//#include "model.hpp"
//#include "model1.hpp"
#include "model2.hpp"

using namespace std;


void print_usage_and_exit() {
    // print usage information
    printf("*****Dual Sparse Additive Collaborative Topic Model for Rating Prediction****\n");
    printf("Authors: anthonylife, xxx.gmail.com, Computer Science Department, XXX University.\n");
    printf("usage:\n");
    printf("      ./main [options]\n");
    printf("      --help:           print help information\n");

    printf("\n");
    printf("      -d:      data set choice\n");
    printf("      -r:      whether to restart train or using existing model\n");
    printf("      -tm:     which method to train the model\n");

    printf("******************************************************************************\n");
    exit(0);
}

int ArgPos(char *str, int argc, char **argv) {
    int a;
    for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
        if (a == argc - 1) {
            printf("Argument missing for %s\n", str);
            exit(1);
        }
        return a;
    }
    return -1;
}


int main(int argc, char **argv) {
    //***Method variables needed to be specified by User (Default values)****//
    string trdata_path      =   "../../data/yelp_train.dat";
    string vadata_path      =   "../../data/yelp_vali.dat";
    string tedata_path      =   "../../data/yelp_test.dat";
    string model_path       =   "model.out";
    int    niters           =   50;
    double delta            =   1e-5;
    int    minibatch        =   32;
    double lr               =   0.01;
    double K                =   40;
    double lambda_u         =   1;
    double lambda_i         =   1;
    double lambda_b         =   0.1;
    double reg_ut           =   0.1;
    double reg_it           =   0.1;
    double reg_bt           =   0.1;
    double psai_u           =   0.1;
    double psai_i           =   0.1;
    double sigma_u          =   0.1;
    double sigma_i          =   0.1;
    double sigma_a          =   0.1;
    double max_words        =   5000;
    double alpha            =   0.1;
    int    inner_niters     =   5;
    int    tr_method        =   0;
    bool   restart_tag      =   false;
    int    truncated_k      =   10;                 // [5, 30]
    double truncated_theta  =   0.3;
    string submission_path  =   "../../results/dsactm_result1.dat";
    ///////////////////////////////////////////////////////////////////////////// 
    
    int i;
    int data_num=-1;
    char *b=NULL;
    if (argc == 1) {
        printf("DSACTM v 0.1a\n");
        print_usage_and_exit();
        return 0;
    }
    if ((i = ArgPos((char *)"--help", argc, argv)) > 0) print_usage_and_exit();
    if ((i = ArgPos((char *)"-d", argc, argv)) > 0) data_num = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-r", argc, argv)) > 0) b = argv[i + 1];
    if ((i = ArgPos((char *)"-tm", argc, argv)) > 0) tr_method = atoi(argv[i + 1]);
    //if ((i = ArgPos((char *)"-niter", argc, argv)) > 0) niters = atoi(argv[i + 1]);
    //if ((i = ArgPos((char *)"-minibatch", argc, argv)) > 0)
    //    minibatch = atoi(argv[i + 1]);
    
    if (data_num!=0 && data_num!=1) {
        printf("Invalid choice of dataset!\n");
        exit(1);
    } else if (data_num==0) {
        trdata_path = "../../data/yelp_train.dat";
        vadata_path = "../../data/yelp_vali.dat";
        tedata_path = "../../data/yelp_test.dat";
        submission_path = "../../results/dsactm_result1.dat";
    } else {
        trdata_path = "../../data/amazonfood_train.dat";
        vadata_path = "../../data/amazonfood_vali.dat";
        tedata_path = "../../data/amazonfood_test.dat";
        submission_path = "../../results/dsactm_result2.dat";
    }
    
    if (strcmp(b, (char *)"True") == 0)
        restart_tag = true;
    else if (strcmp(b, (char *)"False") == 0)
        restart_tag = false;
    else {
        printf("Invalid input of para -r\n");
        exit(1);
    }
   
    timeval start_t, end_t;
    utils::tic(start_t);
    DSACTM *dsactm = new DSACTM((char *)trdata_path.c_str(),
                                (char *)vadata_path.c_str(),
                                (char *)tedata_path.c_str(),
                                (char *)model_path.c_str(),
                                niters,
                                delta,
                                minibatch,
                                lr,
                                K,
                                lambda_u,
                                lambda_i,
                                lambda_b,
                                reg_ut,
                                reg_it,
                                reg_bt,
                                psai_u,
                                psai_i,
                                sigma_u,
                                sigma_i,
                                sigma_a,
                                max_words,
                                alpha,
                                inner_niters,
                                truncated_k,
                                truncated_theta,
                                tr_method,
                                restart_tag);
    if (restart_tag)
        dsactm->train();
    dsactm->submitPredictions((char *)submission_path.c_str());
    utils::toc(start_t, end_t);

    return 0;
}
