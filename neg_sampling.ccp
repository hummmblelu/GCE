// [[Rcpp::depends(RcppArmadillo)]]
#include <iostream>
#include <random>
#include <RcppArmadilloExtensions/sample.h>

using namespace Rcpp ;

// Enables supplying an arma probability
template <class T> 
T sample(const T &x, const int size, const bool replace, arma::vec &prob_){
  return sample_main(x, size, replace, prob_);
}

//[[Rcpp::export]]
arma::vec neg_samples_n(int num_samples,
                        arma::vec word_prob,
                        int leave_out_i){
  
  arma::vec out_vec(num_samples);
  
  double denom = 1;
  double frac;
  double cumprob;
  double u;
  int i;
  for(int m = (num_samples-1); m >= 0; m--){
    double frac = 1 / (denom-word_prob[leave_out_i]);
    denom = denom - word_prob[leave_out_i];
    
    u = (double)rand() / (RAND_MAX + 1.0);
    u = u-(word_prob[leave_out_i]);
    if(u < 0){
      u = 0;
    }
    // Rcpp::Rcout << "u: " <<  u << std::endl;
    
    word_prob[leave_out_i] = 0;
    
    // if(word_prob[leave_out_i] > 0.001){
    //   double adjust_denom = (1 - word_prob[leave_out_i]);
    //   Rcpp::Rcout << "adjust_denom: " << adjust_denom << std::endl;
    //   for(int n = (word_prob.n_elem - 1); n >=0; n--){
    //     word_prob[n] = word_prob[n]/adjust_denom;
    //     word_prob[leave_out_i] = 0;
    //     Rcpp::Rcout << "word_prob: " << word_prob << std::endl;
    //   }
    // }else{
    //   word_prob[leave_out_i] = 0;
    // }
    
    // //generate uniform random variable
    // std::default_random_engine generator;
    // std::uniform_real_distribution<double> distribution(0.0,1.0);
    // double u = distribution(generator);
    
    i = 0;
    cumprob = word_prob[i] * frac;
    while(u >= cumprob){
      i += 1;
      cumprob = cumprob + word_prob[i] * frac;
      
      // Rcpp::Rcout << "i: " << i << std::endl;
      // Rcpp::Rcout << "cumprob: " << cumprob << std::endl;
    }
    
    out_vec[m] = i;
    leave_out_i = i;
  }  
  
  // Rcpp::Rcout << out_vec << std::endl;
  return (out_vec);
}

//[[Rcpp::export]]
arma::mat neg_sampling_mtx(arma::vec x_vec,
                           int num_samples,
                           arma::vec word_prob,
                           int s) {
  
  srand(s);
  int N = x_vec.n_elem;
  arma::mat out_mtx(N, num_samples);
  out_mtx.zeros();
  
  int v;
  //arma::vec ret;
  arma::vec word_prob_sample;
  arma::vec vocabs_sample;
  for(int n = (N-1); n>=0; n--){
    v = x_vec[n];
    
    arma::vec ret = neg_samples_n(num_samples, word_prob, (v-1));
    
    // Rcpp::Rcout << n << std::endl;
    
    for(int i = (num_samples -1); i >=0; i--) {
      out_mtx(n, i) = ret[i] + 1;
    }
    
    //free memory
  }
  
  return(out_mtx);
}
