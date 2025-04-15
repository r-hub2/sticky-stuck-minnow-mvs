# This file is part of mvs: Methods for High-Dimensional Multi-View Learning
# Copyright (C) 2018-2024  Wouter van Loon
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 2.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

adaptive_weights <- function(X, y, nfolds, lower.limits, ...){
  # Calculates (Nonnegative) Adaptive Lasso (NAdaL) weights, using ridge initialization 
  # X: input matrix X or Z
  # y: outcome variable
  # ...: additional arguments to pass to cv.glmnet
  
  folds <- kFolds(y, nfolds)
  
  ridge.obj <- glmnet::cv.glmnet(x=X, y=y, alpha = 0, nfolds = nfolds, lower.limits = lower.limits, ...)
  
  ridge.coefs <- coef(ridge.obj, s="lambda.min")[-1]
  
  if(all(ridge.coefs <= lower.limits)){
    stop("All adaptive weights are infinite, removing all views from the model. This suggests that either (1) none of the views are predictive of the outcome or (2) the model is misspecified.")
  }else{
    inf.weights <- which(ridge.coefs <= lower.limits)
    ridge.weights <- 1/abs(ridge.coefs)
    ridge.weights[inf.weights] <- 1e+05
    
    out <- list(ridge.weights = ridge.weights, inf.weights = inf.weights)
  }
  
  return(out)
}

optimize_over_gamma <- function(X, y, weights, gamma.seq, ...){
  # Optimizes glmnet over a gamma sequence
  # X: input matrix X or Z
  # y: outcome variable
  # weights: a list of weights as produced by adaptive_weights()
  # ...: additional arguments to pass to cv.glmnet
  
  model.list <- vector("list", length(gamma.seq))
  cve <- rep(NA, length(gamma.seq))
  
  for(i in 1:length(gamma.seq)){
    model.list[[i]] <- cv.glmnet(X, y, exclude=weights$inf.weights, penalty.factor=weights$ridge.weights^gamma.seq[i], ...)
    cve[i] <- min(model.list[[i]]$cvm)
  }
  
  out <- model.list[[which.min(cve)]]
  attr(out, "gamma") <- gamma.seq[which.min(cve)]
  
  return(out)
}