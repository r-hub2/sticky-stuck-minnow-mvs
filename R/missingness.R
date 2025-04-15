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

none <- function(x){
  !any(x)
}

check_partial_missings <- function(x){
  apply(x, 1, function(x){
    if(all(is.na(x)) || none(is.na(x))){
      FALSE
    }else{
      TRUE
    }
  })
}

nasum <- Vectorize(function(a, b){
  if(is.na(a) & is.na(b)){
    NA
  }else if(is.na(a)){
    b
  }else if(is.na(b)){
    a
  }
  else{
    a + b
  }
})

`%+%` <- function(a, b){
  if(is.null(dim(a)) && is.null(dim(b))){
    nasum(a,b)
  }else if(identical(dim(a), dim(b))){
    array(nasum(a,b), dim=dim(a))
  }else{
    stop("non-conformable arguments")
  }
}

impute_mean <- function(x){
  for(i in 1:ncol(x)){
    if(anyNA(x[,i])){
      x[is.na(x[,i]), i] <- mean(x[, i], na.rm=TRUE)
    }
  }
  return(x)
}

impute_forest <- function(x, y, verbose = TRUE, ...){
  Z_df <- data.frame(x)
  Z_df$y <- factor(y)
  Z_imputed <- missForest::missForest(Z_df, verbose = verbose, ...)$ximp
  Z_imputed <- data.matrix(Z_imputed)
  Z_imputed <- Z_imputed[, - ncol(Z_imputed)]
  attr(Z_imputed, "imputation_method") <- "missForest"
  attr(Z_imputed, "number_of_trees") <- ifelse("ntree" %in% names(list(...)), list(...)$ntree, formals(missForest::missForest)$ntree)
  attr(Z_imputed, "additional_arguments_passed_to_missForest") <- list(...)
  return(Z_imputed)
}

impute_mice <- function(x, y, ...){
  zy <- cbind(x, y)
  mice_obj <- mice::mice(zy, ...)
  Z_imputed <- mice::complete(mice_obj, action="all")
  if(mice_obj$m > 1){
    Z_array <- array(unlist(Z_imputed), dim=c(nrow(zy), ncol(zy), mice_obj$m))
    Z_mean <- apply(Z_array, c(1,2), mean)
    Z_imputed <- Z_mean[, - ncol(Z_mean)]
  } 
  else{
    Z_mean <- mice::complete(mice_obj)
    Z_imputed <- Z_mean[, - ncol(Z_mean)]
  }
  names(Z_imputed) <- NULL
  Z_imputed <- data.matrix(Z_imputed)
  attr(Z_imputed, "imputation_method") <- mice_obj$method
  attr(Z_imputed, "number_of_imputations") <- mice_obj$m
  attr(Z_imputed, "additional_arguments_passed_to_mice") <- list(...)
  return(Z_imputed)
}
