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

#' Multi-View Stacking
#'
#' Fit a multi-view stacking model with two or more levels.
#' @param x input matrix of dimension nobs x nvars.
#' @param y outcome vector of length nobs.
#' @param views a matrix of dimension nvars x (levels - 1), where each entry is an integer describing to which view each feature corresponds.
#' @param type a character vector of length 1 or length \code{levels}, specifying the type(s) of learner to be used at each level of MVS. Use type "StaPLR" when the desired learner(s) is/are penalized GLM(s); see \code{\link[mvs]{StaPLR}} for supported families. Use type "RF" for random forests.
#' @param levels (optional) an integer >= 2, specifying the number of levels in the MVS procedure. The default is to infer the number of levels from the supplied \code{views} argument.
#' @param alphas a numeric vector of length \code{levels} specifying the value of the alpha parameter to use at each level.
#' @param relax either a logical vector of length \code{levels} specifying whether model relaxation (e.g. the relaxed lasso) should be employed at each level, or a single TRUE or FALSE to enable or disable relaxing across all levels. Defaults to FALSE.
#' @param adaptive either a logical vector of length \code{levels} specifying whether adaptive weights (e.g. the adaptive lasso) should be employed at each level, or a single TRUE or FALSE to enable or disable adaptive weights across all levels. Note that using adaptive weights is generally only sensible if alpha > 0. Defaults to FALSE.
#' @param nnc a binary vector specifying whether to apply nonnegativity constraints or not (1/0) at each level.
#' @param na.action character specifying what to do with missing values (NA). Options are "pass", "fail", "mean", "mice", and "missForest". Options "mice" and "missForest" requires the respective R package to be installed. Defaults to "fail".
#' @param na.arguments (optional) a named list of arguments to pass to the imputation function (e.g. to \code{mice} or \code{missForest}).
#' @param parallel whether to use foreach to fit the learners and obtain the cross-validated predictions at each level in parallel. Executes sequentially unless a parallel back-end is registered beforehand.
#' @param seeds (optional) a vector specifying the seed to use at each level.
#' @param progress whether to show a progress bar (only supported when parallel = FALSE).
#' @param ... additional arguments to pass to the learning algorithm. See e.g. \code{\link[mvs]{StaPLR}}. Note that these arguments are passed to the the learner at every level of the MVS procedure.
#' @return An object of S3 class "MVS".
#' @keywords TBA
#' @import foreach
#' @export
#' @author Wouter van Loon <w.s.van.loon@fsw.leidenuniv.nl>
#' @examples \donttest{ 
#' set.seed(012)
#' n <- 1000
#' X <- matrix(rnorm(8500), nrow=n, ncol=85)
#' beta <- c(rep(10, 55), rep(0, 30)) * ((rbinom(85, 1, 0.5)*2)-1)
#' eta <- X %*% beta
#' p <- 1 /(1 + exp(-eta))
#' y <- rbinom(n, 1, p)
#'
#' ## 2-level MVS with ridge for baselearners and lasso for meta learner
#' views <- c(rep(1,45), rep(2,20), rep(3,20))
#' fit <- MVS(x=X, y=y, views=views)
#' 
#' ## 2-level MVS with random forest for base learners and lasso for meta learner
#' fit <- MVS(x=X, y=y, views=views, type = c("RF", "StaPLR"))
#' new_X <- matrix(rnorm(2*85), nrow=2)
#' predict(fit, new_X)
#' 
#' ## 3-level MVS
#' bottom_level <- c(rep(1:3, each=15), rep(4:5, each=10), rep(6:9, each=5))
#' top_level <- c(rep(1,45), rep(2,20), rep(3,20))
#' views <- cbind(bottom_level, top_level)
#' fit <- MVS(x=X, y=y, views=views, levels=3, alphas=c(0,1,1), nnc=c(0,1,1))
#' coefficients <- coef(fit)
#' predict(fit, new_X)
#' }
MVS <- function(x, y, views, type="StaPLR", levels=NULL, alphas=c(0,1), nnc=c(0,1), parallel=FALSE, 
                seeds=NULL, progress=TRUE, relax = FALSE, adaptive = FALSE, na.action = "fail", na.arguments = NULL, ...){
  
  staplr.args <- names(list(...))
  
  if(!is.null(staplr.args)){
    if(any(c("correct.for", "skip.meta", "skip.cv") %in% staplr.args)){
      stop("StaPLR arguments 'correct.for', 'skip.meta' and 'skip.cv' are not supported for use with MVS().")
    }
  }
  
  if (!is.matrix(views)) views <- matrix(views, ncol = 1L)
  
  if (is.null(levels)) levels <- ncol(views) + 1L
  
  if (levels > 1L && length(type) == 1L) type <- rep(type, times = levels)
  
  if (levels != length(alphas)){
    warning("Length of argument alphas not equal to the number of levels.")
  }
  
  if (levels != length(nnc)){
    warning("Length of argument nnc not equal to the number of levels.")
  }
  
  if (levels > 2L && na.action == "pass"){
    warning("na.action = pass, but levels > 2. Passing missingness through multiple levels is not recommended.")
  }
    
  if(length(relax)==1){
    relax <- rep(relax, levels)
  }
  
  if(length(adaptive)==1){
    adaptive <- rep(adaptive, levels)
  }

  pred_functions <- vector("list", length=ncol(views)+1)
  ll <- c(-Inf, 0)[nnc + 1]

  if(progress){
    message("Level 1 \n")
  }

  ## Fit lowest-level baselearners
  arg_list <- list(X = x, 
                   y=y,
                   views=views[,1L],
                   type=type[1L], 
                   ...)
  if (type[levels] == "StaPLR") {
    arg_list <- append(arg_list, list(
      alpha1=alphas[1L], 
      ll1=ll[1L],
      seed=seeds[1L],
      progress=progress, 
      parallel=parallel, 
      relax.base = relax[1L],
      penalty.weights.base = translate_adaptive_argument(adaptive[1L]),
      na.action=na.action, 
      na.arguments=na.arguments))
  } else if (type[levels] == "RF") {
    arg_list <- append(arg_list, list(
      na.action=na.action, 
      na.arguments=na.arguments))
  }
  pred_functions[[1L]] <- do.call(learn, arg_list)

  ## Fit intermediate-level learners
  if(levels > 2){
    for(i in 2L:ncol(views)){
      if(progress) message(paste("Level", i, "\n"))
      arg_list <- list(X = pred_functions[[i-1L]]$CVs, 
                       y=y,
                       views=condense(views, level=i),
                       type=type[i], 
                       ...)
      if (type[levels] == "StaPLR") {
        arg_list <- append(arg_list, list(
          alpha1=alphas[i], 
          ll1=ll[i],
          seed=seeds[i],
          progress=progress, 
          parallel=parallel, 
          relax.base = relax[i],
          penalty.weights.base = translate_adaptive_argument(adaptive[i]),
          na.action=na.action, 
          na.arguments=na.arguments))
      } else if (type[levels] == "RF") {
        arg_list <- append(arg_list, list(
          na.action=na.action, 
          na.arguments=na.arguments))
      }
      pred_functions[[i]] <- do.call(learn, arg_list)
    }
  }

  if(progress){
    message(paste("Level", ncol(views)+1, "\n"))
  }

  ## Fit meta learner
  arg_list <- list(X = pred_functions[[ncol(views)]]$CVs, 
                   y=y,
                   views=rep(1,ncol(pred_functions[[ncol(views)]]$CVs)),
                   type=type[levels], 
                   generate.CVs=FALSE,
                   ...)
  if (type[levels] == "StaPLR") {
    arg_list <- append(arg_list, list(
      alpha1=alphas[ncol(views)+1], 
      ll1=ll[ncol(views)+1],
      seed=seeds[ncol(views)+1],
      progress=progress, 
      parallel=parallel, 
      relax.base = relax[ncol(views)+1],
      penalty.weights.base = translate_adaptive_argument(adaptive[ncol(views)+1]),
      na.action=na.action, 
      na.arguments=na.arguments))
  } else if (type[levels] == "RF") {
    arg_list <- append(arg_list, list(
      na.action=na.action, 
      na.arguments=na.arguments))
  }
  
  if(arg_list$na.action != "pass"){
    pred_functions[[ncol(views)+1]] <- do.call(learn, arg_list)
  }

  for(i in 1:length(pred_functions)){
    if(!is.null(pred_functions[[i]])){
      pred_functions[[i]]$meta <- NULL
      names(pred_functions[[i]])[1] <- "models"
    }
  }

  names(pred_functions) <- paste("Level", 1:(ncol(views)+1))
  attr(pred_functions, "type") <- type
  class(pred_functions) <- "MVS"

  return(pred_functions)

}

#' @rdname MVS
#' @export
mvs <- MVS

#' Make predictions from an "MVS" object.
#'
#' Make predictions from a "MVS" object.
#' @param object An object of class "MVS".
#' @param newx Matrix of new values for x at which predictions are to be made. Must be a matrix.
#' @param predtype The type of prediction returned by the meta-learner. Supported are types "response", "class" and "link".
#' @param cvlambda Values of the penalty parameters at which predictions are to be made. Defaults to the values giving minimum cross-validation error.
#' @param ... Further arguments to be passed to \code{\link[glmnet]{predict.cv.glmnet}}.
#' @return A matrix of predictions.
#' @keywords TBA
#' @export
#' @author Wouter van Loon <w.s.van.loon@fsw.leidenuniv.nl>
#' @examples \donttest{ 
#' set.seed(012)
#' n <- 1000
#' X <- matrix(rnorm(8500), nrow=n, ncol=85)
#' top_level <- c(rep(1,45), rep(2,20), rep(3,20))
#' bottom_level <- c(rep(1:3, each=15), rep(4:5, each=10), rep(6:9, each=5))
#' views <- cbind(bottom_level, top_level)
#' beta <- c(rep(10, 55), rep(0, 30)) * ((rbinom(85, 1, 0.5)*2)-1)
#' eta <- X %*% beta
#' p <- 1 /(1 + exp(-eta))
#' y <- rbinom(n, 1, p)
#'
#' fit <- MVS(x=X, y=y, views=views, type="StaPLR", levels=3, alphas=c(0,1,1), nnc=c(0,1,1))
#' coefficients <- coef(fit)
#'
#' new_X <- matrix(rnorm(2*85), nrow=2)
#' predict(fit, new_X)}
predict.MVS <- function(object, newx, predtype = "response", cvlambda = "lambda.min",
                        ...){
  
  x <- newx

  for(i in 1:length(object)){

    Z <- matrix(NA, nrow=nrow(newx), ncol=length(object[[i]]$models))

    if(i < length(object)){
      pt <- object[[i]]$metadat
    } else pt <- predtype

    for(j in 1:ncol(Z)){
      if (inherits(object[[i]]$models[[j]], "randomForest")) {
        Z[,j] <- if (is.factor(object[[i]]$models[[j]]$y)) {
          predict(object[[i]]$models[[j]], x[, object[[i]]$view == j, drop=FALSE], type = "prob", ...)[ , 2L]
        } else {
          predict(object[[i]]$models[[j]], x[, object[[i]]$view == j, drop=FALSE], type = "response", ...)
        }
      } else if (inherits(object[[i]]$models[[j]], c("cv.glmnet", "glmnet"))) {
        Z[,j] <- predict(object[[i]]$models[[j]], x[, object[[i]]$view == j, drop=FALSE],
                        s = cvlambda, type = pt, ...)
      }
    }
    x <- Z
  }

  return(x)

}


#' Extract coefficients from an "MVS" object.
#'
#' Extract coefficients at each level from an "MVS" object at the CV-optimal values of the penalty parameters.
#' @param object An object of class "MVS".
#' @param cvlambda By default, the coefficients are extracted at the CV-optimal values of the penalty parameters. Choosing "lambda.1se" will extract them at the largest values within one standard error of the minima.
#' @param ... Further arguments to be passed to \code{\link[glmnet]{coef.cv.glmnet}}.
#' @return An object of S3 class "MVScoef".
#' @keywords TBA
#' @export
#' @author Wouter van Loon <w.s.van.loon@fsw.leidenuniv.nl>
#' @examples \donttest{ 
#' set.seed(012)
#' n <- 1000
#' X <- matrix(rnorm(8500), nrow=n, ncol=85)
#' top_level <- c(rep(1,45), rep(2,20), rep(3,20))
#' bottom_level <- c(rep(1:3, each=15), rep(4:5, each=10), rep(6:9, each=5))
#' views <- cbind(bottom_level, top_level)
#' beta <- c(rep(10, 55), rep(0, 30)) * ((rbinom(85, 1, 0.5)*2)-1)
#' eta <- X %*% beta
#' p <- 1 /(1 + exp(-eta))
#' y <- rbinom(n, 1, p)
#'
#' fit <- MVS(x=X, y=y, views=views, type="StaPLR", levels=3, alphas=c(0,1,1), nnc=c(0,1,1))
#' coefficients <- coef(fit)
#'
#' new_X <- matrix(rnorm(2*85), nrow=2)
#' predict(fit, new_X)}

coef.MVS <- function(object, cvlambda = "lambda.min", ...){

  out <- vector("list", length(object))

  for(i in 1:length(object)){
    out[[i]] <- vector("list", length(object[[i]]$models))
  }

  for(i in 1:length(object)){
    for(j in 1:length(object[[i]]$models)){
      if(inherits(object[[i]]$models[[j]], "cv.glmnet")){
        out[[i]][[j]] <- coef(object[[i]]$models[[j]], s=cvlambda, ...)
      }else{
        out[[i]][[j]] <- NA
      }
    }
  }
  names(out) <- paste("Level", 1:length(object))
  attr(out, "type") <- attr(object, "type")
  class(out) <- "MVScoef"

  return(out)
}


#' Calculate feature importance from an "MVS" object.
#'
#' Calculate feature importance at each level from an "MVS" object based on random forests.
#' @param x An object of class "MVS".
#' @param ... Further arguments to be passed to \code{\link[randomForest]{importance}}.
#' @return An object of S3 class "MVSimportance".
#' @keywords TBA
#' @export
#' @author Wouter van Loon <w.s.van.loon@fsw.leidenuniv.nl>
#' @examples \donttest{ 
#' set.seed(012)
#' n <- 1000
#' X <- matrix(rnorm(8500), nrow=n, ncol=85)
#' views <- c(rep(1,45), rep(2,20), rep(3,20))
#' beta <- c(rep(10, 55), rep(0, 30)) * ((rbinom(85, 1, 0.5)*2)-1)
#' eta <- X %*% beta
#' p <- 1 /(1 + exp(-eta))
#' y <- rbinom(n, 1, p)
#' 
#' ## 2-level MVS with random forest
#' fit <- MVS(x=X, y=y, views=views, type = "RF")
#' importance(fit)}
importance.MVS <- function(x, ...){
  
  out <- vector("list", length(x))
  
  for(i in 1:length(x)){
    out[[i]] <- vector("list", length(x[[i]]$models))
  }
  
  for(i in 1:length(x)){
    for(j in 1:length(x[[i]]$models)){
      if(inherits(x[[i]]$models[[j]], "randomForest")){
        out[[i]][[j]] <- randomForest::importance(x[[i]]$models[[j]], ...)
      }else{
        out[[i]][[j]] <- NA
      }
    }
  }
  names(out) <- paste("Level", 1:length(x))
  attr(out, "type") <- attr(x, "type")
  class(out) <- "MVSimportance"
  
  return(out)
}
