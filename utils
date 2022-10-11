library(Rcpp)
library(RcppArmadillo)

sourceCpp("neg_sampling.cpp")
sourceCpp("cpp_funs.cpp")

update_gce_model<- function(gc_model, epochs = 5, n_minibatches = 1000, n_neg_samples = 10, samp_pow = 0.75,
                           beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8, lambda_phi = 0, lambda_alpha = 0, 
                           alpha_learn = 0.01, trace_indices = NULL){
  # total iterations
  total_iter<- epochs * n_minibatches
  
  # meta data
  meta<- gc_model$meta
  var_num<- meta$var_num
  
  data<- gc_model$data
  
  var_params<- gc_model$var_params
  phi<- var_params$phi
  alpha<- var_params$alpha
  
  log_lkhd<- c()
  
  # initialize optim parameters
  m_phi<- 0
  v_phi<- 0
  m_hat_phi<- 0
  v_hat_phi<- 0
  
  var_num<- meta$var_num
  m_alpha<- lapply(1:var_num, function(x) 0)
  v_alpha<- lapply(1:var_num, function(x) 0)
  m_hat_alpha<- lapply(1:var_num, function(x) 0)
  v_hat_alpha<- lapply(1:var_num, function(x) 0)
  
  # preparation for negative sampling
  if(n_neg_samples > 0){
    item_cnt<- table(data$item_vec)

    item_freq<- rep(0, length(meta$item_list))
    item_freq[as.numeric(names(item_cnt))]<- item_cnt

    item_prob<- item_freq / sum(item_freq)
    item_prob<- item_prob^(samp_pow)
    item_prob<- item_prob/sum(item_prob)
  #   
  #   print("val_mtx")
  #   zero_mtx<- matrix(0, meta$var_num, (n_neg_samples) * meta$n_len)
  #   print("combine")
  #   # val_mtx[,1:meta$n_len]<- data$val_mtx 
  #   # val_mtx<- cbind(data$val_mtx, matrix(0, meta$var_num, n_neg_samples * meta$n_len))
  #   val_mtx_pool<- cbind(data$val_mtx, zero_mtx)
  #   print(pryr::mem_used())
  #   
  #   print("context_val")
  #   context_val_neg_samples<- lapply(data$context_val, function(x){
  #     replicate(n_neg_samples, x, simplify = F)
  #   })
  #   context_val_neg_samples<- unlist(context_val_neg_samples, recursive = F)
  #   context_val_pool<- append(data$context_val, context_val_neg_samples)
  #   print(pryr::mem_used())
  #   
  #   print("context_index")
  #   context_index_neg_samples<- lapply(data$context_index, function(x){
  #     replicate(n_neg_samples, x, simplify = F)
  #   })
  #   context_index_neg_samples<- unlist(context_index_neg_samples, recursive = F)
  #   context_index_pool<- append(data$context_index, context_index_neg_samples)
  #   gc()
  #   print(pryr::mem_used())
  }
  gc()
   
  iter_t<- 1
  for(epoch_iter in 1:epochs){
    samp_indices<- sample(1:n_minibatches, meta$n_len, replace = T)
    
    if(n_neg_samples > 0){
      neg_samps_item<- neg_sampling_mtx(data$item_vec, n_neg_samples, item_prob, 1)
      item_vec<- c(data$item_vec, as.vector(t(neg_samps_item)))
    }
    
    for(minibatche_iter in 1:n_minibatches){
      # subsetting
      min_indices<- which(samp_indices == minibatche_iter)
      
      if(n_neg_samples > 0){
        item_vec<- c(data$item_vec[min_indices], as.vector(t(apply(neg_samps_item, 2, function(x) x[min_indices]))))
        
        context_val<- data$context_val[rep(min_indices, 1+n_neg_samples)]
        context_index<- data$context_index[rep(min_indices, 1+n_neg_samples)]
        
        # context_val_neg_samples<- lapply(data$context_val[min_indices], function(x){
        #   replicate(n_neg_samples, x, simplify = F)
        # })
        # context_val_neg_samples<- unlist(context_val_neg_samples, recursive = F)
        # context_val<- append(data$context_val[min_indices], context_val_neg_samples)
        
        # context_index_neg_samples<- lapply(data$context_index[min_indices], function(x){
        #   replicate(n_neg_samples, x, simplify = F)
        # })
        # context_index_neg_samples<- unlist(context_index_neg_samples, recursive = F)
        # context_index<- append(data$context_index[min_indices], context_index_neg_samples)
        
        zero_mtx<- matrix(0, meta$var_num, (n_neg_samples) * length(min_indices))
        val_mtx<- cbind(data$val_mtx[,min_indices], zero_mtx)
        
      }else{
        item_vec<- data$item_vec[min_indices]
        context_val<- data$context_val[min_indices]
        context_index<- data$context_index[min_indices]
        val_mtx<- data$val_mtx[,min_indices]
      }
      
      # simulate latent variable
      phi_item_vec<- phi[,item_vec]
      R<- lapply(alpha, function(a) c_eff_cpp(a, context_val, context_index))
      z_simu<- lapply(R, function(r) apply(t(phi_item_vec) * r, 1, sum) + rnorm(length(item_vec), 0, 1))
      
      # calculate gradient
      grad_list<- lapply(1:var_num, function(j){
        d_z<- grad_log_placketluce_wt_cpp(val_mtx[j,], z_simu[[j]])
        
        grad_phi<- grad_phi_var_cpp(z_simu[[j]],
                                    item_vec,
                                    d_z,
                                    phi,
                                    R[[j]],
                                    lambda_phi)
        
        grad_alpha<- grad_alpha_var_cpp(z_simu[[j]],
                                        item_vec,
                                        d_z,
                                        context_val,
                                        context_index,
                                        phi,
                                        alpha[[j]],
                                        R[[j]], 
                                        lambda_phi)
        
        # print(str(grad_alpha))
        
        list(grad_phi = grad_phi, grad_alpha = grad_alpha)
      })
      
      phi_grad<- -1 * Reduce("+", lapply(grad_list, function(x) x$grad_phi))
      
      alpha_grad<- lapply(grad_list, function(x) -1 * x$grad_alpha)
      
      # update parameters
      # updata phi
      m_phi = beta_1 * m_phi + (1-beta_1) * phi_grad
      v_phi = beta_2 * v_phi + (1-beta_2) * (phi_grad * phi_grad)
      m_hat_phi = m_phi / (1 - beta_1^iter_t)
      v_hat_phi = v_phi / (1 - beta_2^iter_t)
      phi = phi - alpha_learn * m_hat_phi / (sqrt(v_hat_phi) + epsilon)
      
      # update alpha
      for(j in 1:var_num){
        m_alpha[[j]] = beta_1 * m_alpha[[j]] + (1-beta_1) * alpha_grad[[j]]
        v_alpha[[j]] = beta_2 * v_alpha[[j]] + (1-beta_2) * (alpha_grad[[j]] * alpha_grad[[j]])
        m_hat_alpha[[j]] = m_alpha[[j]] / (1 - beta_1^iter_t)
        v_hat_alpha[[j]] = v_alpha[[j]] / (1 - beta_2^iter_t)
        alpha[[j]] = alpha[[j]] - alpha_learn * m_hat_alpha[[j]] / (sqrt(v_hat_alpha[[j]]) + epsilon)
      }
      
      if(length(trace_indices) > 0 & (minibatche_iter %% 10 == 0)){
        if(n_neg_samples > 0){
          item_vec<- c(data$item_vec[trace_indices], as.vector(t(apply(neg_samps_item, 2, function(x) x[trace_indices]))))
          
          # context_val_neg_samples<- lapply(data$context_val[trace_indices], function(x){
          #   replicate(n_neg_samples, x, simplify = F)
          # })
          # context_val_neg_samples<- unlist(context_val_neg_samples, recursive = F)
          # context_val<- append(data$context_val[trace_indices], context_val_neg_samples)
          context_val<- data$context_val[rep(trace_indices, (n_neg_samples + 1))]
          
          # context_index_neg_samples<- lapply(data$context_index[trace_indices], function(x){
          #   replicate(n_neg_samples, x, simplify = F)
          # })
          # context_index_neg_samples<- unlist(context_index_neg_samples, recursive = F)
          # context_index<- append(data$context_index[trace_indices], context_index_neg_samples)
          context_index<- data$context_index[rep(trace_indices, (n_neg_samples + 1))]
          
          zero_mtx<- matrix(0, meta$var_num, (n_neg_samples) * length(trace_indices))
          val_mtx<- cbind(data$val_mtx[,trace_indices], zero_mtx)
          
        }else{
          item_vec<- data$item_vec[trace_indices]
          context_val<- data$context_val[trace_indices]
          context_index<- data$context_index[trace_indices]
          val_mtx<- data$val_mtx[,trace_indices]
        }
        # trace likelihood
        
        phi_item_vec<- phi[,item_vec]
        R<- lapply(alpha, function(a) c_eff_cpp(a, context_val, context_index))
        z_simu<- lapply(R, function(r) apply(t(phi_item_vec) * r, 1, sum) + rnorm(length(item_vec), 0, 1))
        
        log_lkhd<- c(log_lkhd, sum(sapply(1:var_num, function(j) log_dplacketluce_wt_lkhd_cpp(val_mtx[j,], z_simu[[j]]))))
      }
      
      
      if(iter_t %% 10 == 0) print(paste(iter_t, "/", total_iter, " iterations finished", sep = ""))
      iter_t<- iter_t + 1
    }
  }
  
  var_params$phi<- phi
  var_params$alpha<- alpha
  
  gc_model$var_params<- var_params
  gc_model$log_lkhd<- log_lkhd
  
  return(gc_model)
}
