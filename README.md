# GCE
R implementation for the paper Gaussian Copula Embeddings

To use the model, please go the the directory and run 

```
source("utils.R")
```

Specify size embedding vectors
```
p_len<- 100
```

Specify training data
```
data_train<- list(item_vec = item_vec,
                  val_mtx = val_mtx)
data_train$context_index = context_index
data_train$context_val = lapply(data_train$context_index, function(x) rep(1, length(x)))

meta_train<- list(item_list = item_list,
                  n_len = length(item_vec),
                  var_num = nrow(val_mtx),
                  p_len = p_len)
```
item_vec: unique item index

item_list: list of center items

context_index: list of context

val_mtx: variable matrix of the central item; rows are corresponding to the variable and columns are corresponding to the position



Set up variational parameters and training model
```
var_params<- list(phi = matrix(rnorm(item_len * p_len), p_len, item_len),
                  alpha = lapply(1:var_num, function(i) matrix(rnorm(item_len * p_len), item_len, p_len)))

gce_model_init<- list(meta = meta_train, data = data_train, var_params = var_params)
gce_model_trained<- update_gce_model(gce_model_init, epochs = 2, n_minibatches = 1000, 
                                     n_neg_samples = 5, trace_indices  = 1:500, alpha_learn = 0.01)
```

Chien Lu, and Jaakko Peltonen. Gaussian Copula Embeddings. Conference on Neural Information Processing Systems (NeurIPS), 2022.

