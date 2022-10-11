#include <RcppArmadillo.h>
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]

//[[Rcpp::export]]
arma::mat c_eff_cpp(arma::mat alpha,
                    Rcpp::List context_val,
                    Rcpp::List context_index) {
  
  int n_len = context_val.length(); // length of observed sequence
  int p_len = alpha.n_cols; // dimension of embedding vectors
  
  arma::mat out_mtx(n_len, p_len);
  out_mtx.zeros();
  
  int w_len, t; // w_len: context width, t: item indicator
  double val_context, norm_const;
  for(int n = (n_len - 1); n >=0; n--) {
    arma::vec context_val_n = context_val[n];
    arma::vec context_index_n = context_index[n];
    w_len = context_val_n.n_elem;
    norm_const = 1/double(w_len);
    
    for(int w = (w_len - 1); w >=0; w--){
      t = context_index_n[w] - 1;
      val_context = context_val_n[w];
      
      for(int p = (p_len - 1); p >=0; p--){
        // Rcpp::Rcout << alpha(v, p) << std::endl;
        out_mtx(n,p) += alpha(t, p) * val_context * norm_const;
      }
    }
  }
  return(out_mtx);
}

//[[Rcpp::export]]
double log_norm_lkhd_cpp(arma::vec x_val,
                         arma::vec x_index,
                         arma::mat R,
                         arma::mat phi,
                         double sigma){
  double out_val = 0;
  int n_len = x_index.n_elem;
  
  int n, t, w_len;
  arma::mat R_n, phi_t;
  double val, natr_param, add_val, norm_const;
  for(int n = (n_len-1); n >=0; n--){
    t = (x_index[n]-1);
    val = x_val[n];
    R_n = R.row(n);
    phi_t = phi.col(t);
    
    natr_param = arma::accu(R_n * phi_t);
    add_val = R::dnorm(val, natr_param, sigma, 1);
    
    // Rcpp::Rcout << "R_n: "<< R_n << std::endl;
    // Rcpp::Rcout << "phi_t: "<< phi_t << std::endl;
    // 
    // Rcpp::Rcout << "val: "<< val << std::endl;
    // Rcpp::Rcout << "natr_param: "<< natr_param << std::endl;
    // Rcpp::Rcout << "add_val: "<< add_val << std::endl;
    
    out_val += add_val;
  }
  
  return(out_val);
}

//[[Rcpp::export]]
arma::mat grad_phi_cpp(arma::vec x_val,
                       arma::vec x_index,
                       arma::mat phi,
                       arma::mat R,
                       double sigma) {
  
  int n_len = x_val.n_elem;
  int t_len = phi.n_cols;
  int p_len = phi.n_rows;
  arma::mat grad_phi(p_len, t_len);
  grad_phi.zeros();
  
  int t;
  arma::vec phi_t;
  arma::mat R_n;
  double val_n, mu, d_mu, d_phi;
  for(int n = (n_len-1); n >= 0; n--){
    t = x_index[n] - 1; //index of the item (index difference for R and c++)
    val_n = x_val[n]; //value at the n-th position
    R_n = R.row(n).t();
    phi_t = phi.col(t);
    mu = arma::accu(R_n.t() * phi_t);
    d_mu = (val_n - mu) / (sigma * sigma);
    
    // if(n == 0){
    //   Rcpp::Rcout << "mu: "<< mu << std::endl;
    //   Rcpp::Rcout << "val_n: "<< val_n << std::endl;
    //   Rcpp::Rcout << "d_mu: "<< d_mu << std::endl;
    // }
    
    for(int p = (p_len - 1); p >=0; p--){
      d_phi = R_n[p];
      grad_phi(p, t) += (d_mu * d_phi);
    }
  }
  return(grad_phi);
}

//[[Rcpp::export]]
arma::mat grad_alpha_cpp(arma::vec x_val,
                         arma::vec x_index,
                         Rcpp::List context_val,
                         Rcpp::List context_index,
                         arma::mat phi,
                         arma::mat alpha, 
                         arma::mat R, 
                         double sigma){
  
  int n_len = x_index.n_elem;
  int t_len = alpha.n_rows;
  int p_len = alpha.n_cols;
  
  arma::mat out_grad(t_len, p_len);
  out_grad.zeros();
  
  int w_len, t_center, t_context;
  arma::mat R_n, phi_t;
  // arma::mat phi_t;
  double mu, d_mu, d_alpha, val_center, val_context, norm_const;
  
  for(int n = (n_len-1); n >=0; n--){
    arma::vec context_val_n = context_val[n];
    arma::vec context_index_n = context_index[n];
    w_len = context_val_n.n_elem;
    norm_const = (1/double(w_len));
    
    t_center = x_index[n]-1;
    val_center = x_val[n];
    // d_z = x_d_z[n];
    R_n = R.row(n);
    phi_t = phi.col(t_center);
    mu = arma::accu(R_n * phi_t);
    
    d_mu = (val_center - mu) / (sigma * sigma);
    
    // if(n == 0){
    //   Rcpp::Rcout << "context_val: "<< context_val_n << std::endl;
    //   Rcpp::Rcout << "context_index_n: "<< context_index_n << std::endl;
    //   Rcpp::Rcout << "t_center: "<< t_center << std::endl;
    //   Rcpp::Rcout << "val_center: "<< val_center << std::endl;
    //   Rcpp::Rcout << "phi_t: "<< phi_t << std::endl;
    //   Rcpp::Rcout << "natr_param: "<< natr_param << std::endl;
    // }
    
    for(int w = (w_len-1); w >=0; w--){
      t_context = context_index_n[w]-1;
      val_context = context_val_n[w];
      // if(n == 0){
      //   Rcpp::Rcout << "t_context: "<< t_context << std::endl;
      // }
      for(int p = (p_len - 1); p >=0; p--){
        d_alpha = norm_const * phi_t[p] * val_context;
        
        // if(n == 0 & p == 0){
        //   Rcpp::Rcout << "d_alpha: "<< d_alpha << std::endl;
        // }
        
        // out_grad(t_context, p) += (d_mu * d_alpha * d_z);
        out_grad(t_context, p) += (d_mu * d_alpha);
      }
    }
  }
  
  return(out_grad) ;
}

// [[Rcpp::export]]
double log_dplacketluce_wt_lkhd_cpp(arma::vec x, arma::vec score) {
  double out = 0;
  int len = score.n_elem;
  arma::uvec order = sort_index(x, "descend");
  
  arma::vec x_ord = x(order);
  arma::vec score_ord = score(order);
  arma::vec exp_score = exp(score);
  arma::vec exp_score_ord = exp_score(order);
  
  double numer, denom;
  int tie_start, num_ties;
  for(int i = 0; i < len; i++) {
    if(i == 0){
      num_ties = 1;
      numer = score_ord[i];
      denom = 0;
      for(int j = i; j < len; j++){
        denom += exp_score_ord[j];
      }
    }else{
      if(x_ord[i] == x_ord[i-1]){
        num_ties = num_ties + 1;
        numer += score_ord[i];
      }else{
        // Rcpp::Rcout << "num_ties: "<< num_ties << std::endl;
        // Rcpp::Rcout << "numer: "<< numer << std::endl;
        // Rcpp::Rcout << "denom: "<< denom << std::endl;
        // Rcpp::Rcout << "add: "<< (numer/num_ties) - log(denom) << std::endl;
        out += ((numer/num_ties) - log(denom));
        
        num_ties = 1;
        numer = score_ord[i];
        denom = 0;
        for(int j = i; j < len; j++){
          denom += exp_score_ord[j];
        }
      }
    }
    
    // Rcpp::Rcout << "num_ties: "<< num_ties << std::endl;
    // Rcpp::Rcout << "numer: "<< numer << std::endl;
    // Rcpp::Rcout << "denom: "<< denom << std::endl;
  }
  // Rcpp::Rcout << "num_ties: "<< num_ties << std::endl;
  // Rcpp::Rcout << "numer: "<< numer << std::endl;
  // Rcpp::Rcout << "denom: "<< denom << std::endl;
  // Rcpp::Rcout << "add: "<< (numer/num_ties) - log(denom) << std::endl;
  out += ((numer/num_ties) - log(denom));
  // Rcpp::Rcout << "add: "<< (1/num_ties) * numer - log(denom) << std::endl;
  // out += (1/num_ties) * numer - log(denom);
  
  return out;
}


// [[Rcpp::export]]
arma::vec grad_log_placketluce_wt_cpp(arma::vec x, arma::vec score) {
  int len = score.n_elem;
  arma::vec out_vec(len);
  out_vec.zeros();
  arma::uvec order = sort_index(x, "descend");
  
  arma::vec x_ord = x(order);
  arma::vec score_ord = score(order);
  arma::vec exp_score = exp(score);
  arma::vec exp_score_ord = exp_score(order);
  
  double denom_s = arma::accu(exp_score);
  double denom_inv = 0;
  double num_ties = 1;
  int l_ind = 0;
  for(int i = 0; i < len; i++) {
    if(i == 0){
      num_ties = 1;
    }else{
      if(x_ord[i] == x_ord[i-1]){
        num_ties = num_ties + 1;
      }else{
        // Rcpp::Rcout << "num_ties: "<< num_ties << std::endl;
        denom_inv += (1/denom_s);
        
        // Rcpp::Rcout << "denom_s: "<< denom_s << std::endl;
        // Rcpp::Rcout << "denom_inv: "<< denom_inv << std::endl;
        
        for(int j = l_ind; j < i; j++){
          // Rcpp::Rcout << "j: "<< j << std::endl;
          // Rcpp::Rcout << "(1/num_ties): "<< (1/num_ties) << std::endl;
          // Rcpp::Rcout << "exp_score_ord[j]: "<< exp_score_ord[j] << std::endl;
          // Rcpp::Rcout << "denom_inv: "<< denom_inv << std::endl;
          // 
          out_vec[j] = (1/num_ties) - (exp_score_ord[j] * denom_inv);
          denom_s -= exp_score_ord[j];
        }
        l_ind = i;
        
        num_ties = 1;
      }
    }
  }
  // Rcpp::Rcout << "num_ties: "<< num_ties << std::endl;
  denom_inv += (1/denom_s);
  // Rcpp::Rcout << "denom_inv: "<< denom_inv << std::endl;
  for(int j = l_ind; j < len; j++){
    out_vec[j] = 1/num_ties - exp_score_ord[j] * denom_inv;
  }
  
  
  return out_vec(sort_index(order, "ascend"));
}


//[[Rcpp::export]]
arma::mat grad_phi_var_cpp(arma::vec x_val,
                            arma::vec x_index,
                            arma::vec x_d_z,
                            arma::mat phi,
                            arma::mat R,
                            double lambda_phi) {
  
  int n_len = x_val.n_elem;
  int t_len = phi.n_cols;
  int p_len = phi.n_rows;
  arma::mat grad_phi(p_len, t_len);
  grad_phi.zeros();
  
  // gradient
  grad_phi = -1 * phi * lambda_phi;
  
  int t;
  arma::vec phi_t;
  arma::mat R_n;
  double val_n, d_z, d_phi;
  for(int n = (n_len-1); n >= 0; n--){
    t = x_index[n] - 1; //index of the item (index difference for R and c++)
    val_n = x_val[n]; //value at the n-th position
    d_z = x_d_z[n];
    R_n = R.row(n).t();
    phi_t = phi.col(t);
    // mu = arma::accu(R_n.t() * phi_t);
    // d_mu = (val_n - mu) / (sigma * sigma);
    
    // if(n == 0){
    //   Rcpp::Rcout << "mu: "<< mu << std::endl;
    //   Rcpp::Rcout << "val_n: "<< val_n << std::endl;
    //   Rcpp::Rcout << "d_mu: "<< d_mu << std::endl;
    // }
    
    for(int p = (p_len - 1); p >=0; p--){
      d_phi = R_n[p];
      grad_phi(p, t) += (d_phi * d_z);
    }
  }
  return(grad_phi);
}


//[[Rcpp::export]]
arma::mat grad_alpha_var_cpp(arma::vec x_val,
                             arma::vec x_index,
                             arma::vec x_d_z,
                             Rcpp::List context_val,
                             Rcpp::List context_index,
                             arma::mat phi,
                             arma::mat alpha,
                             arma::mat R, 
                             double lambda_alpha){
  
  int n_len = x_index.n_elem;
  int t_len = alpha.n_rows;
  int p_len = alpha.n_cols;
  
  arma::mat out_grad(t_len, p_len);
  out_grad.zeros();
  
  // gradient of prior
  out_grad = -1 * lambda_alpha * alpha;
  
  int w_len, t_center, t_context;
  arma::mat R_n, phi_t;
  double d_z, d_alpha, val_center, val_context, norm_const;
  
  for(int n = (n_len-1); n >=0; n--){
    d_z = x_d_z[n];
    arma::vec context_val_n = context_val[n];
    arma::vec context_index_n = context_index[n];
    w_len = context_val_n.n_elem;
    norm_const = (1/double(w_len));
    
    t_center = x_index[n]-1;
    val_center = x_val[n];
    // d_z = x_d_z[n];
    R_n = R.row(n);
    phi_t = phi.col(t_center);
    // if(n == 0){
    //   Rcpp::Rcout << "context_val: "<< context_val_n << std::endl;
    //   Rcpp::Rcout << "context_index_n: "<< context_index_n << std::endl;
    //   Rcpp::Rcout << "t_center: "<< t_center << std::endl;
    //   Rcpp::Rcout << "val_center: "<< val_center << std::endl;
    //   Rcpp::Rcout << "phi_t: "<< phi_t << std::endl;
    //   Rcpp::Rcout << "natr_param: "<< natr_param << std::endl;
    // }
    
    for(int w = (w_len-1); w >=0; w--){
      t_context = context_index_n[w]-1;
      val_context = context_val_n[w];
      // if(n == 0){
      //   Rcpp::Rcout << "t_context: "<< t_context << std::endl;
      // }
      for(int p = (p_len - 1); p >=0; p--){
        d_alpha = norm_const * phi_t[p] * val_context;
        
        // if(n == 0 & p == 0){
        //   Rcpp::Rcout << "d_alpha: "<< d_alpha << std::endl;
        // }
        
        // out_grad(t_context, p) += (d_mu * d_alpha * d_z);
        out_grad(t_context, p) += (d_z * d_alpha);
      }
    }
  }
  
  return(out_grad) ;
}
