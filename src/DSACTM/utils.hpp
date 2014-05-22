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

#include<iostream>
#include<algorithm>
#include<fstream>
#include<sstream>
#include<vector>
#include<map>
#include<set>
#include<ext/hash_set>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include<sys/time.h>
#include<omp.h>
#include<armadillo>

#define SPEED_MATINV

using namespace __gnu_cxx;
namespace __gnu_cxx
{
    template<> struct hash<const std::string> {
        size_t operator()(const std::string& s) const { 
            return hash<const char*>()( s.c_str() );
        }
    };
    template<> struct hash<std::string> {
        size_t operator()(const std::string& s) const { 
            return hash<const char*>()( s.c_str() );
        }
    };
}

// Data associated with a rating
struct vote
{
    int user; // ID of the user
    int item; // ID of the item
    float value; // Rating

    std::string vote_time; // Unix-time of the rating
    std::vector<int> words; // IDs of the words in the review
};
typedef struct vote Vote;


namespace utils{
    /// Task-specific functions. (e.g. loading reviews )
    std::vector<Vote*>* loadReviewData(char* data_path,
                    std::map<std::string, int>* word_ids,
                    std::map<std::string, int>* user_ids,
                    std::map<int, std::string>* ruser_ids,
                    std::map<std::string, int>* item_ids,
                    std::map<int, std::string>* ritem_ids,
                    int& n_users, int& n_items);

    
    /// basic data io
    FILE * fopen_(const char* p, const char* m);
    void fread_(double * M, size_t size, size_t count, FILE* stream);
    std::ifstream* ifstream_(const char* p);
    std::ofstream* ofstream_(const char* p);
    
    void write_submission(std::vector<std::vector<std::string> >* recommendation_result, char* submission_path);
   

    /// time measurement
    void tic(timeval &start_t);
    void toc(timeval &start_t, timeval &end_t);
    void toc(timeval &start_t, timeval &end_t, bool newline);


    /// mathematical functions
    inline double logitLoss(double x) {
        return (1-1.0/(1+exp(-x)));
    };
    inline double dot(double * factor1, double * factor2, int ndim) {
        double result = 0.0;
        for (int i=0; i<ndim; i++)
            result += factor1[i]*factor2[i];
        return result;
    };
    inline double dot(double * factor1, double * factor2, int ndim, int k) {
        double result = 0.0;
        for (int i=0; i<ndim; i++) {
            if (i == k)
                continue;
            result += factor1[i]*factor2[i];
        }
        return result;
    };
    inline void outerDotAccum(double * factor1, double * factor2,
            double ** factor3, int ndim) {
        for (int k1=0; k1<ndim; k1++)
            for (int k2=0; k2<ndim; k2++)
                factor3[k1][k2] += factor1[k1]*factor2[k2];
    };
    inline void vecProdNumAccum(double * factor1, double num,
            double * factor2, int ndim) {
        for (int i=0; i<ndim; i++)
            factor2[i] += factor1[i]*num;
    };
    inline void matAddDiagnoal(double ** factor, double lambda, int ndim) {
        for (int i=0; i<ndim; i++)
            factor[i][i] += lambda;
    };
    inline void vecAddVec(double * factor1, double * factor2, int ndim){
        for (int i=0; i<ndim; i++)
            factor1[i] += factor2[i];
    }
    inline void vecAddVec(double * factor1, double * factor2,
            double scale2, int ndim){
        for (int i=0; i<ndim; i++)
            factor1[i] += scale2*factor2[i];
    }
    inline void matProdVec(double ** factor1, double * factor2,
            double * result, int ndim) {
        for (int i1=0; i1<ndim; i1++) {
            result[i1] = 0.0;
            for (int i2=0; i2<ndim; i2++)
                result[i1] += factor1[i1][i2]*factor2[i2];
        }
    }
    inline double max(double x1, double x2) {
        if (x1 > x2)
            return x1;
        else
            return x2;
    };
    inline double min(double x1, double x2) {
        if (x1 > x2)
            return x2;
        else
            return x1;
    };
    inline double square(double x) {
        return x * x;
    };
    inline double gsquare(double x) {
        return 2 * x;
    };
    inline double soft(double x, double delta) {
        if (x >= 0)
            return utils::max(x-delta, 0.0);
        else
            return utils::min(x+delta, 0.0);
    };

    void soft(double *x, double delta, int ndim);
    
    void matrixInversion(double **A, int order, double **Y);
    int getMinor(double **src, double **dest, int row, int col, int order);
    double calcDeterminant( double **mat, int order);
    
    void project_beta(double *beta, const int &nTerms, 
							const double &dZ, const double &epsilon); 
    void project_beta1(double *beta, const int &nTerms, double min_val);

    void project_beta2(double *beta, const int &n_terms);

    void normalize(double * factor1, double * factor2, int ndim);

    void normalize(double * factor, int ndim, double min_val);

    /// random number generator
    double gaussrand(double ep, double var);
    void muldimGaussrand(double ** factor, int ndim);
    void muldimUniform(double ** factor, int ndim);
    void muldimZero(double ** factor, int ndim);
    void muldimGaussrand(double * factor, int ndim);
    void muldimUniform(double * factor, int ndim);
    void muldimZero(double * factor, int ndim);
    void muldimPosUniform(double * factor, int ndim, int max_val);
    std::vector<std::string>* genNegSamples(std::vector<std::string>* data, std::set<std::string>* filter_samples, int nsample);
    std::vector<std::string>* genSamples(std::vector<std::string>* data, int nsample);
    std::vector<std::string>* genSamples(std::vector<std::string>* data, hash_set<std::string>* filter_samples, int nsample);
    std::vector<std::string>* genSamples(std::vector<std::string>* data, std::string filter_sample, int nsample);


    // sorting functions
    void quickSort(double *arr, int left, int right, bool reverse); 


    // char array string split function
    std::vector<char *> split_str(char * in_str, char sep);
    // extract substr
    char * sub_str(int s_idx, int e_idx, char * raw_str);
    // string split function
    std::vector<std::string> split_str(std::string in_str, char sep);
    
    // count the line of file
    int cnt_file_line(std::string in_file);
    
    // allocate matrix memory
    int ** alloc_matrix(int xdim, int ydim);
    // allocate vector memory
    int * alloc_vector(int ndim);

    void pause();
}


//================Specific Functions=================
std::vector<Vote*>* utils::loadReviewData(char* data_path,
                     std::map<std::string, int>* word_ids,
                     std::map<std::string, int>* user_ids,
                     std::map<int, std::string>* ruser_ids,
                     std::map<std::string, int>* item_ids,
                     std::map<int, std::string>* ritem_ids,
                     int& n_users, int& n_items) {
    std::string u_name;
    std::string i_name;
    float value;
    std::string vote_time;
    int nw;
    int n_read = 0;
    std::string line, s_word;
    vector<Vote*>* V = new std::vector<Vote*>();
    
    Vote* v = new Vote();
    std::ifstream* in = utils::ifstream_(data_path);
    while (std::getline(*in, line)) {
        std::stringstream ss(line);
        ss >> u_name >> i_name >> value >> vote_time >> nw;
        for (int w = 0; w < nw; w++) {
            ss >> s_word;
            if (word_ids->find(s_word) != word_ids->end())
                v->words.push_back((*word_ids)[s_word]);
        }
        if (user_ids->find(u_name) == user_ids->end()) {
            (*ruser_ids)[n_users] = u_name;
            (*user_ids)[u_name] = n_users++;
        }
        v->user = (*user_ids)[u_name];
        if (item_ids->find(i_name) == item_ids->end()) {
            (*ritem_ids)[n_items] = i_name;
            (*item_ids)[i_name] = n_items++;
        }
        v->item = (*item_ids)[i_name];
        v->value = value;
        v->vote_time = vote_time;
        V->push_back(v);
        v = new Vote();
      
        n_read++;
        if (n_read % 100000 == 0) {
            printf(".");
            fflush( stdout);
        }
    }
    printf("\r");
    in->close();
    return V;
}


//================Basic Functions===============
// matrix inversioon
// the result is put in Y
#ifdef SPEED_MATINV
double *temp = new double[(40-1)*(40-1)];
double **minor = new double*[40-1];
#endif

void utils::matrixInversion(double **A, int order, double **Y) {
    // get the determinant of a
    double det = 1.0/utils::calcDeterminant(A,order);

    // memory allocation
#ifndef SPEED_MATINV
    double *temp = new double[(order-1)*(order-1)];
    double **minor = new double*[order-1];
#endif
    for(int i=0;i<order-1;i++)
        minor[i] = temp+(i*(order-1));

    for(int j=0;j<order;j++)
    {
        for(int i=0;i<order;i++)
        {
            // get the co-factor (matrix) of A(j,i)
            utils::getMinor(A,minor,j,i,order);
            Y[i][j] = det*utils::calcDeterminant(minor,order-1);
            if( (i+j)%2 == 1)
                Y[i][j] = -Y[i][j];
        }
    }

    // release memory
    //delete [] minor[0];
    delete [] temp;
    delete [] minor;
}

// calculate the cofactor of element (row,col)
int utils::getMinor(double **src, double **dest, int row, int col, int order) {
    // indicate which col and row is being copied to dest
    int colCount=0,rowCount=0;

    for(int i = 0; i < order; i++ )
    {
        if( i != row )
        {
            colCount = 0;
            for(int j = 0; j < order; j++ )
            {
                // when j is not the element
                if( j != col )
                {
                    dest[rowCount][colCount] = src[i][j];
                    colCount++;
                }
            }
            rowCount++;
        }
    }

    return 1;
}

// Calculate the determinant recursively.
double utils::calcDeterminant( double **mat, int order) {
    // order must be >= 0
	// stop the recursion when matrix is a single element
    if( order == 1 )
        return mat[0][0];

    // the determinant value
    double det = 0;

    // allocate the cofactor matrix
    double **minor;
    minor = new double*[order-1];
    for(int i=0;i<order-1;i++)
        minor[i] = new double[order-1];

    for(int i = 0; i < order; i++ )
    {
        // get minor of element (0,i)
        utils::getMinor( mat, minor, 0, i , order);
        // the recusion is here!

        det += (i%2==1?-1.0:1.0) * mat[0][i] * utils::calcDeterminant(minor,order-1);
        //det += pow( -1.0, i ) * mat[0][i] * CalcDeterminant( minor,order-1 );
    }

    // release memory
    for(int i=0;i<order-1;i++)
        delete [] minor[i];
    delete [] minor;

    return det;
}


void utils::soft(double *x, double delta, int ndim) {
    for (int i=0; i<ndim; i++) {
        if (x[i] >= 0)
            x[i] = utils::max(x[i]-delta, 0.0);
        else
            x[i] = utils::min(x[i]+delta, 0.0);
    }
}


void utils::project_beta( double *beta, const int &nTerms, 
							const double &dZ, const double &epsilon ) {
    std::vector<int> U(nTerms);
	double * mu_ = new double[nTerms];

    for ( int i=0; i<nTerms; i++ ) {
		mu_[i] = beta[i] - epsilon;
		U[i] = i + 1;
	}
	//double dZVal = dZ - epsilon * nTerms; // make sure dZVal > 0

	/* project to a simplex. */
	double s = 0;
	int p = 0;
	while ( !U.empty() ) {
		int nSize = U.size();
		int k = U[ rand()%nSize ];

		/* partition U. */
        std::vector<int> G, L;
		int deltaP = 0;
		double deltaS = 0;
		for ( int i=0; i<nSize; i++ ) {
			int j = U[i];

			if ( mu_[j-1] >= mu_[k-1] ) {
				if ( j != k ) G.push_back( j );
				deltaP ++;
				deltaS += beta[j-1];
			} else L.push_back( j );
		}

		if ( s + deltaS - (p + deltaP) * mu_[k-1] < dZ ) {
			s += deltaS;
			p += deltaP;
			U = L;
		} else {
			U = G;
		}
	}

	double theta = (s - dZ) / p;
	for ( int i=0; i<nTerms; i++ ) {
		beta[i] = utils::max(mu_[i] - theta, 0.0) + epsilon;
	}
    delete mu_;
    U.clear();
    std::vector<int>(U).swap(U);
}


// project beta to simplex ( N*log(N) ). (ICML-09, efficient L1 ball)
void utils::project_beta1( double *beta, const int &nTerms, double min_val) {
	double * mu_ = new double[nTerms];
	// copy. (mu for temp use)
	for ( int i=0; i<nTerms; i++ ) {
		mu_[i] = beta[i];
	}
	// sort m_mu.
	quickSort(mu_, 0, nTerms-1, true);
	
    // find rho.
	int rho = 0;
	double dsum = 0;
	for ( int i=0; i<nTerms; i++ ) {
		dsum += mu_[i];

		if ( mu_[i] - (dsum-1)/(i+1) > 0 )
			rho = i;
	}

	double theta = 0;
	for ( int i=0; i<=rho; i++ ) {
		theta += mu_[i];
	}
	theta = (theta-1) / (rho+1);

	for ( int i=0; i<nTerms; i++ ) {
		beta[i] = max(0.0, beta[i] - theta)+min_val;
	}
    double normalization = 0.0;
    for (int i=0; i<nTerms; i++)
        normalization += beta[i];
    for (int i=0; i<nTerms; i++)
        beta[i] /= normalization;
    delete mu_;
}


void utils::project_beta2( double *beta, const int &n_terms) {
    double normalization = 0.0;
    for (int i=0; i<n_terms; i++)
        normalization += beta[i];
    for (int i=0; i<n_terms; i++)
        beta[i] /= normalization;
}


void utils::normalize(double * factor1, double * factor2, int ndim) {
    double normalization = 0.0;
    for (int i=0; i<ndim; i++)
        normalization += factor2[i];
    for (int i=0; i<ndim; i++)
        factor1[i] = factor2[i]/normalization;
}

void utils::normalize(double * factor, int ndim, double min_val) {
    double normalization = 0.0;
    for (int i=0; i<ndim; i++)
        normalization += factor[i];
    
    for (int i=0; i<ndim; i++) {
        factor[i] = factor[i]/normalization;
        if (factor[i] < min_val)
            factor[i] = min_val;
    }
    
    normalization = 0.0;
    for (int i=0; i<ndim; i++)
        normalization += factor[i];
    
    for (int i=0; i<ndim; i++)
        factor[i] = factor[i]/normalization;
}


FILE * utils::fopen_(const char* p, const char* m) {
    FILE *f = fopen(p, m);
    if (!f) {
        printf("Failed to open %s\n", p);
        exit(1);
    }
    return f;
}


void utils::fread_(double * M, size_t size, size_t count, FILE* stream) {
    int r_size = fread(M, size, count, stream);
    if (r_size == 0) {
        printf("Fail to read data!\n");
        exit(1);
    }
}


std::ifstream* utils::ifstream_(const char* p){
    std::ifstream *in = new std::ifstream(p);
    if (!in) {
        printf("Failed to open %s\n", p);
        exit(1);
    }
    return in;
}


std::ofstream* utils::ofstream_(const char* p){
    std::ofstream *out = new std::ofstream(p);
    if (!out) {
        printf("Failed to open %s\n", p);
        exit(1);
    }
    return out;
}


void utils::write_submission(std::vector<std::vector<std::string> >* recommendation_result,
        char* submission_path) {
    int idx = 0;
    std::ofstream* out = utils::ofstream_(submission_path);

    for (std::vector<std::vector<std::string> >:: iterator it=recommendation_result->begin();
            it!=recommendation_result->end(); it++) {
        if (it->size() == 0) {
            *out << idx << std::endl;
        } else {
            *out << idx << "\t";
            for (std::vector<std::string>:: iterator it1=it->begin(); it1!=it->end()-1; it1++) {
                *out << *it1 << ",";
            }
            *out << *(it->end()-1) << std::endl;
        }
        idx++;
    }
    out->close();
}


void utils::tic(timeval &start_t) {
    gettimeofday(&start_t, 0);
}


void utils::toc(timeval &start_t, timeval &end_t){
    gettimeofday(&end_t, 0);
    double timeuse = 1000000*(end_t.tv_sec-start_t.tv_sec)+end_t.tv_usec-start_t.tv_usec;
    printf("Time cost: %f(us), %f(s), %f(min)!\n", timeuse,
            timeuse/(1000*1000), timeuse/(1000*1000*60));
}


void utils::toc(timeval &start_t, timeval &end_t, bool newline){
    gettimeofday(&end_t, 0);
    double timeuse = 1000000*(end_t.tv_sec-start_t.tv_sec)+end_t.tv_usec-start_t.tv_usec;
    printf("Time cost: %f(us), %f(s), %f(min)!", timeuse,
            timeuse/(1000*1000), timeuse/(1000*1000*60));
    if (newline)
        printf("\n");
}


double utils::gaussrand(double ep, double var) {
    double V1 = 0.0, V2=0.0, S=0.0;
    int phase = 0;
    double X;
                     
    if ( phase == 0 ) {
        do {
            double U1 = (double)rand() / RAND_MAX;
            double U2 = (double)rand() / RAND_MAX;
            V1 = 2 * U1 - 1;
            V2 = 2 * U2 - 1;
            S = V1 * V1 + V2 * V2;
        } while(S >= 1 || S == 0);
                                         
        X = V1 * sqrt(-2 * log(S) / S);
    } else
    X = V2 * sqrt(-2 * log(S) / S);
    phase = 1 - phase;
    X = X*var + ep;
    return X;
}


void utils::muldimGaussrand(double ** factor, int ndim) {
    *factor = new double[ndim];
    for (int i=0; i<ndim; i++)
        (*factor)[i] = utils::gaussrand(0.0, 0.1);
}


void utils::muldimGaussrand(double * factor, int ndim) {
    for (int i=0; i<ndim; i++)
        factor[i] = utils::gaussrand(0.0, 0.1);
}


void utils::muldimUniform(double ** factor, int ndim) {
    *factor = new double[ndim];
    for (int i=0; i<ndim; i++)
        (*factor)[i] = 2*float(rand())/RAND_MAX-1;
}


void utils::muldimUniform(double * factor, int ndim) {
    for (int i=0; i<ndim; i++)
        factor[i] = 2*float(rand())/RAND_MAX-1;
}


void utils::muldimZero(double ** factor, int ndim) {
    *factor = new double[ndim];
    for (int i=0; i<ndim; i++)
        (*factor)[i] = 0.0;
}


void utils::muldimZero(double * factor, int ndim) {
    for (int i=0; i<ndim; i++)
        factor[i] = 0.0;
}


void utils::muldimPosUniform(double * factor, int ndim, int max_val) {
    for (int i=0; i<ndim; i++)
        factor[i] = float(rand())*max_val/RAND_MAX;
}


std::vector<std::string>* utils::genNegSamples(std::vector<std::string>* data,
        std::set<std::string>* filter_samples, int nsample) {
    std::vector<std::string>* neg_samples = NULL;
    std::vector<std::string>::iterator it;
  
    it = data->begin();
    /*while(it != data->end()) {
        if (filter_samples->find(*it) != filter_samples->end())
            it = data->erase(it);
        else
            it++;
    }*/
    neg_samples = utils::genSamples(data, nsample);
    return neg_samples;
}


std::vector<std::string>* utils::genSamples(std::vector<std::string>* data,
        hash_set<std::string>* filter_samples, int nsample) {
    int sampled_num=0;
    std::vector<std::string>* samples = new std::vector<std::string>();

    std::random_shuffle(data->begin(), data->end());
    for (std::vector<std::string>::iterator it = data->begin(); it!=data->end(); it++) {
        if (filter_samples->find(*it) == filter_samples->end()) {
            samples->push_back(*it);
            sampled_num++;
            if (sampled_num == nsample)
                break;
        }
    }
    return samples;
}


std::vector<std::string>* utils::genSamples(std::vector<std::string>* data,
        std::string filter_sample, int nsample) {
    int sampled_num=0;
    std::vector<std::string>* samples = new std::vector<std::string>();

    std::random_shuffle(data->begin(), data->end());
    for (std::vector<std::string>::iterator it = data->begin(); it!=data->end(); it++) {
        if (*it != filter_sample) {
            samples->push_back(*it);
            sampled_num++;
            if (sampled_num == nsample)
                break;
        }
    }
    return samples;
}


std::vector<std::string>* utils::genSamples(std::vector<std::string>* data, int nsample) {
    int sampled_num;
    //int sample_id;
    std::vector<std::string>* samples = new std::vector<std::string>();
    
    sampled_num = 0;
    /*while (sampled_num < nsample) {
        if (data->size() == 0)
            break;
        sample_id = rand()%data->size();
        samples->push_back((*data)[sample_id]);
        data->erase(data->begin()+sample_id);
        sampled_num++;
    }*/
    std::random_shuffle(data->begin(), data->end());
    for (std::vector<std::string>::iterator it = data->begin(); it!=data->end(); it++) {
        samples->push_back(*it);
        sampled_num++;
        if (sampled_num == nsample)
            break;
    }
        
    return samples;
}


void utils::quickSort(double *arr, int left, int right, bool reverse) 
{
	int i = left, j = right;
	double tmp;
	double pivot = arr[(left + right) / 2];

	/* partition */
	while (i <= j) 
	{
		
        if (reverse) {
            while( arr[i] > pivot )
			    i ++;
		    while( arr[j] < pivot )
			    j --;
        } else {
            while( arr[i] < pivot )
			    i ++;
		    while( arr[j] > pivot )
			    j --;
        }

		if (i <= j) {
			tmp = arr[i];
			arr[i] = arr[j];
			arr[j] = tmp;

			i ++;
			j --;
		}
	};

	/* recursion */
	if (left < j)
		quickSort(arr, left, j, reverse);

	if (i < right)
		quickSort(arr, i, right, reverse);
}


std::vector<char *> utils::split_str(char * in_str, char sep){
    std::vector<char *> str_seg;
    
    int str_len = strlen(in_str);
    if(in_str[str_len - 1] == '\n')
        str_len = str_len - 1;
    
    int s_idx, e_idx;
    s_idx = 0;
    for (int i = 0; i < str_len; i++){
        if(in_str[i] == sep){
            e_idx = i-1;
            str_seg.push_back(utils::sub_str(s_idx, e_idx, in_str));
            s_idx = i+1;
        }
    }
    if (s_idx < str_len)
        str_seg.push_back(utils::sub_str(s_idx, str_len-1, in_str));

    return str_seg;
}

char * utils::sub_str(int s_idx, int e_idx, char * raw_str){
    //char new_str[e_idx+1-s_idx+1];  // first +1: right number, second +1: "\0"
    char * new_str = new char[e_idx+1-s_idx+1];
    memset(new_str, 0, e_idx+1-s_idx+1);

    for (int i=s_idx; i<=e_idx; i++)
        new_str[i-s_idx] = raw_str[i];

    return new_str;
}

std::vector<std::string> utils::split_str(std::string in_str, char sep){
    std::vector<std::string> str_seg;
    
    int str_len = in_str.length();
    if (in_str[str_len-1] == '\n')
        str_len = str_len - 1;
    
    int s_idx, e_idx;
    s_idx = 0;
    for (int i=0; i<str_len; i++){
        if(in_str[i] == sep){
            e_idx = i-1;
            str_seg.push_back(in_str.substr(s_idx, e_idx+1-s_idx));
            s_idx = i+1;
        }
    }
    if (s_idx < str_len)
        str_seg.push_back(in_str.substr(s_idx, str_len-s_idx));

    return str_seg;
}


int * utils::alloc_vector(int ndim){
    int * tmp_vector = NULL;
    
    tmp_vector = new int[ndim];
    for (int i=0; i<ndim; i++){
        tmp_vector[i] = 0;
    }

    return tmp_vector;
}

int ** utils::alloc_matrix(int xdim, int ydim){
    int ** tmp_matrix = NULL;

    tmp_matrix = new int*[xdim];
    for (int i=0; i<xdim; i++){
        tmp_matrix[i] = new int[ydim];
        for (int j=0; j<ydim; j++){
            tmp_matrix[i][j] = 0;
        }
    }

    return tmp_matrix;
}


void utils::pause(){
    printf("Pausing the program, type any character to restart...\n");
    getchar();
}

