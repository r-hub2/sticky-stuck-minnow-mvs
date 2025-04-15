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

utils::globalVariables(c("k", "v"))

#' Stacked Penalized Logistic Regression
#' 
#
#'
#' Fit a two-level stacked penalized (logistic) regression model with a single base-learner and a single meta-learner. Stacked penalized regression models with a Gaussian or Poisson outcome can be fitted using the family argument. 
#' @param x input matrix of dimension nobs x nvars
#' @param y outcome vector of length nobs
#' @param view a vector of length nvars, where each entry is an integer describing to which view each feature corresponds.
#' @param view.names (optional) a character vector of length nviews specifying a name for each view.
#' @param family Either a character string representing one of the built-in families, or else a \code{glm()} family object. 
#'                 For more information, see \code{family} argument's documentation in \code{\link[glmnet]{glmnet}}. Note
#'                 that "multinomial", "mgaussian", "cox", or 2-column responses with "binomial" family are not yet supported. 
#' @param correct.for (optional) a matrix with nrow = nobs, where each column is a feature which should be included directly into the meta.learner. By default these features are not penalized (see penalty.weights.meta) and appear at the top of the coefficient list.
#' @param alpha1 (base) alpha parameter for glmnet: lasso(1) / ridge(0)
#' @param alpha2 (meta) alpha parameter for glmnet: lasso(1) / ridge(0)
#' @param relax.base logical indicating whether relaxed lasso should be employed for fitting the base learners. If \code{TRUE}, then CV is done with respect to the mixing parameter gamma as well as lambda.
#' @param relax.meta logical indicating whether relaxed lasso should be employed for fitting the meta learner. If \code{TRUE}, then CV is done with respect to the mixing parameter gamma as well as lambda.
#' @param relax logical, whether relaxed lasso should be used at base and meta level.
#' @param nfolds number of folds to use for all cross-validation.
#' @param na.action character specifying what to do with missing values (NA). Options are "pass", "fail", "mean", "mice", and "missForest". Options "mice" and "missForest" requires the respective R package to be installed. Defaults to "pass".
#' @param na.arguments (optional) a named list of arguments to pass to the imputation function (e.g. to \code{mice} or \code{missForest}).
#' @param seed (optional) numeric value specifying the seed. Setting the seed this way ensures the results are reproducible even when the computations are performed in parallel.
#' @param std.base should features be standardized at the base level?
#' @param std.meta should cross-validated predictions be standardized at the meta level?
#' @param ll1 lower limit(s) for each coefficient at the base-level. Defaults to -Inf.
#' @param ul1 upper limit(s) for each coefficient at the base-level. Defaults to Inf.
#' @param ll2 lower limit(s) for each coefficient at the meta-level. Defaults to 0 (non-negativity constraints). Does not apply to correct.for features.
#' @param ul2 upper limit(s) for each coefficient at the meta-level. Defaults to Inf. Does not apply to correct.for features.
#' @param cvloss loss to use for cross-validation.
#' @param metadat which attribute of the base learners should be used as input for the meta learner? Allowed values are "response", "link", and "class".
#' @param cvlambda value of lambda at which cross-validated predictions are made. Defaults to the value giving minimum internal cross-validation error.
#' @param cvparallel whether to use 'foreach' to fit each CV fold (DO NOT USE, USE OPTION parallel INSTEAD).
#' @param lambda.ratio the ratio between the largest and smallest lambda value.
#' @param fdev sets the minimum fractional change in deviance for stopping the path to the specified value, ignoring the value of fdev set through glmnet.control. Setting fdev=NULL will use the value set through glmnet.control instead. It is strongly recommended to use the default value of zero. 
#' @param penalty.weights.meta (optional) either a vector of length nviews containing different penalty factors for the meta-learner, or "adaptive" to calculate the weights from the data. The default value NULL implies an equal penalty for each view. The penalty factor is set to 0 for \code{correct.for} features.
#' @param penalty.weights.base (optional) either a list of length nviews, where each entry is a vector containing different penalty factors for each feature in that view, or "adaptive" to calculate the weights from the data. The default value NULL implies an equal penalty for each view. Note that using adaptive weights at the base level is generally only sensible if \code{alpha1} > 0.
#' @param gamma.seq a sequence of gamma values over which to optimize the adaptive weights. Only used when \code{penalty.weights.meta="adaptive"} or \code{penalty.weights.base="adaptive"}.
#' @param parallel whether to use foreach to fit the base-learners and obtain the cross-validated predictions in parallel. Executes sequentially unless a parallel backend is registered beforehand.
#' @param skip.version whether to skip checking the version of the glmnet package.
#' @param skip.meta whether to skip training the metalearner.
#' @param skip.cv whether to skip generating the cross-validated predictions.
#' @param progress whether to show a progress bar (only supported when parallel = FALSE).
#' @return An object with S3 class "StaPLR".
#' @keywords TBA
#' @import foreach
#' @import glmnet 
#' @importFrom utils getTxtProgressBar setTxtProgressBar txtProgressBar packageVersion
#' @importFrom stats na.omit
#' @export
#' @author Wouter van Loon <w.s.van.loon@fsw.leidenuniv.nl>
#' @examples \donttest{
#' set.seed(012)
#' n <- 1000
#' cors <- seq(0.1,0.7,0.1)
#' X <- matrix(NA, nrow=n, ncol=length(cors)+1)
#' X[,1] <- rnorm(n)
#'
#' for(i in 1:length(cors)){
#'   X[,i+1] <- X[,1]*cors[i] + rnorm(n, 0, sqrt(1-cors[i]^2))
#' }
#'
#' beta <- c(1,0,0,0,0,0,0,0)
#' eta <- X %*% beta
#' p <- exp(eta)/(1+exp(eta))
#' y <- rbinom(n, 1, p) ## create binary response
#' view_index <- rep(1:(ncol(X)/2), each=2)
#' 
#' # Stacked penalized logistic regression
#' fit <- StaPLR(X, y, view_index)
#' coef(fit)$meta
#'
#' new_X <- matrix(rnorm(16), nrow=2)
#' predict(fit, new_X)
#' 
#' # Stacked penalized linear regression
#' y <- eta + rnorm(100) ## create continuous response
#' fit <- StaPLR(X, y, view_index, family = "gaussian")
#' coef(fit)$meta
#' coef(fit)$base
#' new_X <- matrix(rnorm(16), nrow=2)
#' predict(fit, new_X)
#' 
#' # Stacked penalized Poisson regression
#' y <- ceiling(eta + 4) ## create count response
#' fit <- StaPLR(X, y, view_index, family = "poisson")
#' coef(fit)$meta
#' coef(fit)$base
#' new_X <- matrix(rnorm(16), nrow=2)
#' predict(fit, new_X)
#' }
StaPLR <- function(x, y, view, view.names = NULL, family = "binomial", correct.for = NULL, alpha1 = 0, alpha2 = 1, 
                   relax = FALSE, nfolds = 10, na.action = "fail", na.arguments = NULL, seed = NULL,
                   std.base = FALSE, std.meta = FALSE, ll1 = -Inf, ul1 = Inf,
                   ll2 = 0, ul2 = Inf, cvloss = "deviance", metadat = "response", cvlambda = "lambda.min",
                   cvparallel = FALSE, lambda.ratio = 1e-4, fdev=0, penalty.weights.meta = NULL, penalty.weights.base = NULL, gamma.seq=c(0.5, 1, 2), parallel = FALSE, 
                   skip.version = TRUE, skip.meta = FALSE, skip.cv = FALSE, progress = TRUE,
                   relax.base = FALSE, relax.meta = FALSE){
  
  # Check na.action argument
  na.action <- match.arg(na.action, c("pass", "fail", "mean", "mice", "missForest", "remove"))
  if(na.action == "fail" && anyNA(x)){
    stop("Missing values detected in x. Either remove or impute missing values, or choose a different na.action")
  }else if(na.action == "mean" && anyNA(x)){
    pass <- FALSE
  }else if(na.action == "mice" && anyNA(x)){
    if(!requireNamespace("mice")){
      stop("Package `mice` is required, but not installed.")
    }
    pass <- FALSE
  }else if(na.action == "missForest" && anyNA(x)){
    if(!requireNamespace("missForest")){
      stop("Package `missForest` is required, but not installed.")
    }
    pass <- FALSE
  }else if(na.action == "remove" && anyNA(x)){
    x <- na.omit(x)
    y <- y[-attr(x, "na.action")]
  }else if(na.action == "pass" && anyNA(x)){
    pass <- TRUE
  }else{
    pass <- FALSE
  }

  # Set the glmnet.control parameter fdev.
  if(!is.null(fdev)){
    glmnet_control_default <- glmnet.control()
    on.exit(do.call(glmnet.control, glmnet_control_default))
    glmnet.control(fdev=fdev)
  }
  
  # Check current version of glmnet
  if(!skip.version){
    versionMessage <- paste0("Found glmnet version ", packageVersion("glmnet"), ".\n To skip this check use: StaPLR(..., skip.version=TRUE). \n")
    message(versionMessage)
  }
  
  # check family and response type
  if (family %in% c("multinomial", "mgaussian", "cox"))
    stop("StaPLR does not yet support multinomial, multivariate gaussian or survival responses.")
  if (is.matrix(y) && ncol(y) > 1L)
    stop("StaPLR does not yet support response variables comprising multiple columns.")

  #object initialization
  V <- length(unique(view))
  n <- if (is.matrix(y)) nrow(y) else length(y)
    
  if(V==1 && !skip.meta){
    warning("Only 1 view was provided. Training of the meta-learner will be skipped!")
    skip.meta <- TRUE
  }

  # SEQUENTIAL PROCESSING
  if(!parallel){

    if(!is.null(seed)){
      set.seed(seed)
      folds <- if (is.matrix(y)) kFolds(y[, 1], nfolds) else kFolds(y, nfolds)
      base.seeds <- sample(.Machine$integer.max/2, size = V)
      z.seeds <- matrix(sample(.Machine$integer.max/2, size = V*nfolds), nrow=nfolds, ncol=V)
      meta.seed <- sample(.Machine$integer.max/2, size=1)
    } else folds <- if (is.matrix(y)) kFolds(y[, 1], nfolds) else kFolds(y, nfolds)

    if(progress){
      message("Training learner on each view...")
      pb <- txtProgressBar(min=0, max=V, style=3)
    }
    
    cv.base <- foreach(v=(1:V)) %do% {
      if(progress){
        setTxtProgressBar(pb, v)
      }
      if(!is.null(seed)){
        set.seed(base.seeds[v])
      }
      if(anyNA(x[, view == v])){
        if(any(check_partial_missings(x[, view == v]))){
          warning("Partially missing observations found in view ", v,
                  ". These observations will be treated as if all their values for view ", v,
                  " are missing. It may be more efficient to perform feature-level imputation. The partially missing observations are: ",
                  paste(which(check_partial_missings(x[, view == v])), collapse = ", "))
        }
        x_train <- na.omit(x[, view == v])
        y_train <-  y[-attr(x_train, "na.action")]
      }else{
        x_train <- x[, view == v]
        y_train <- y
      }
      if(is.null(penalty.weights.base)){
        glmnet::cv.glmnet(x_train, y_train, nfolds = nfolds, family = family,
                          type.measure = cvloss, alpha = alpha1,
                          standardize = std.base, lower.limits = ll1,
                          upper.limits = ul1, parallel = cvparallel, lambda.min.ratio = lambda.ratio,
                          relax = relax.base)
      }else if(inherits(penalty.weights.base, "list") && length(penalty.weights.base) == V){
        if(ncol(x_train) != length(penalty.weights.base[[v]])){
          stop("The length of each penalty weight vector should be equal to the number of features in that view.")
        }
        glmnet::cv.glmnet(x_train, y_train, nfolds = nfolds, family = family,
                          type.measure = cvloss, alpha = alpha1,
                          standardize = std.base, lower.limits = ll1,
                          upper.limits = ul1, parallel = cvparallel, lambda.min.ratio = lambda.ratio,
                          relax = relax.base, penalty.factor = penalty.weights.base[[v]])
      }else if(identical(penalty.weights.base, "adaptive")){
        weights <- adaptive_weights(x_train, y_train, nfolds = nfolds, type.measure = cvloss, 
                                    family = family, standardize = std.base, lower.limits = ll1,
                                    upper.limits = ul1, parallel = cvparallel, lambda.min.ratio=lambda.ratio)
        
        optimize_over_gamma(x_train, y_train, weights, gamma.seq, nfolds = nfolds, type.measure = cvloss, alpha = alpha1,
                            family = family, standardize = std.base, lower.limits = ll1, upper.limits = ul1, parallel = cvparallel,
                            lambda.min.ratio=lambda.ratio, relax = relax.base)
      }else{
        stop("penalty.weights.base must be either NULL, adaptive, or a list of length nviews, with each entry a numeric vector containing penalty weights for each feature within that view.")
      }
    }
    
    
    if(!skip.cv){
      if(progress){
        message("\n Calculating cross-validated predictions...")
        pb <- txtProgressBar(min=0, max=V*nfolds, style=3)
      }

      Z <- foreach(v=(1:V), .combine=cbind) %:%
        foreach(k=(1:nfolds), .combine="%+%") %do% {
          if(progress){
            setTxtProgressBar(pb, getTxtProgressBar(pb)+1)
          }
          if(!is.null(seed)){
            set.seed(z.seeds[k, v])
          }
          if(anyNA(x[, view == v])){
            x_train <- na.omit(x[, view == v])
            y_train <-  y[-attr(x_train, "na.action")]
            cvfolds <- folds[-attr(x_train, "na.action")]
          }else{
            x_train <- x[, view == v]
            y_train <- y
            cvfolds <- folds
          }
          if(is.null(penalty.weights.base)){
            cvf <- glmnet::cv.glmnet(x_train[cvfolds != k, ], y = y_train[cvfolds != k], 
                                     nfolds = nfolds, family = family,
                                     type.measure = cvloss, alpha = alpha1,
                                     standardize = std.base, lower.limits = ll1,
                                     upper.limits = ul1, parallel = cvparallel, lambda.min.ratio = lambda.ratio,
                                     relax = relax.base)
            newy <- rep(NA, length(y)) 
            newy[folds == k] <- predict(cvf, newx = x[folds == k, view == v], s = cvlambda, type = metadat)
            return(newy)
          }else if(inherits(penalty.weights.base, "list") && length(penalty.weights.base) == V){
            if(ncol(x_train) != length(penalty.weights.base[[v]])){
              stop("The length of each penalty weight vector should be equal to the number of features in that view.")
            }
            cvf <- glmnet::cv.glmnet(x_train[cvfolds != k, ], y = y_train[cvfolds != k], 
                                     nfolds = nfolds, family = family,
                                     type.measure = cvloss, alpha = alpha1,
                                     standardize = std.base, lower.limits = ll1,
                                     upper.limits = ul1, parallel = cvparallel, lambda.min.ratio = lambda.ratio,
                                     relax = relax.base, penalty.factor = penalty.weights.base[[v]])
            newy <- rep(NA, length(y)) 
            newy[folds == k] <- predict(cvf, newx = x[folds == k, view == v], s = cvlambda, type = metadat)
            return(newy)
          }else if(identical(penalty.weights.base, "adaptive")){
            weights <- adaptive_weights(x_train[cvfolds != k, ], y_train[cvfolds != k], nfolds = nfolds, type.measure = cvloss, 
                                        family = family, standardize = std.base, lower.limits = ll1,
                                        upper.limits = ul1, parallel = cvparallel, lambda.min.ratio=lambda.ratio)
            
            cvf <- optimize_over_gamma(x_train[cvfolds != k, ], y_train[cvfolds != k], weights, gamma.seq, nfolds = nfolds, type.measure = cvloss, alpha = alpha1,
                                       family = family, standardize = std.base, lower.limits = ll1, upper.limits = ul1, parallel = cvparallel,
                                       lambda.min.ratio=lambda.ratio, relax = relax.base)
            newy <- rep(NA, length(y)) 
            newy[folds == k] <- predict(cvf, newx = x[folds == k, view == v], s = cvlambda, type = metadat)
            return(newy)
          }else{
            stop("penalty.weights.base must be either NULL, adaptive, or a list of length nviews, with each entry a numeric vector containing penalty weights for each feature within that view.")
          }
          
        }
      
      if(na.action == "mean" && anyNA(Z)){
        Z <- impute_mean(Z)
      }else if(na.action == "mice" && anyNA(Z)){
        if(!is.null(na.arguments)){
          na.arguments <- c(list(x=Z, y=y), na.arguments)
        }else{
          na.arguments <- list(x=Z, y=y)
        }
        Z <- do.call(impute_mice, na.arguments)
      }else if(na.action == "missForest" && anyNA(Z)){
        if(!is.null(na.arguments)){
          na.arguments <- c(list(x=Z, y=y), na.arguments)
        }else{
          na.arguments <- list(x=Z, y=y)
        }
        Z <- do.call(impute_forest, na.arguments)
      }
      
      dimnames(Z) <- NULL
      if(!is.null(view.names)){
        colnames(Z) <- view.names
      }
    } else{
      Z <- NULL
      skip.meta <- TRUE
    }

    if(progress && !skip.meta && !pass){
      message("\n Training meta learner...")
    }
    if(!is.null(seed)){
      set.seed(meta.seed)
    }
    if(skip.meta || pass){
      cv.meta <- NULL
    } else if(is.null(correct.for) && is.null(penalty.weights.meta)){
      cv.meta <- glmnet::cv.glmnet(Z, y, nfolds = nfolds, type.measure = cvloss, alpha = alpha2, 
                                   family = family, standardize = std.meta, lower.limits = ll2,
                                   upper.limits = ul2, parallel = cvparallel, lambda.min.ratio=lambda.ratio,
                                   relax = relax.meta)
    } else if(is.null(correct.for) && !is.null(penalty.weights.meta)){
      if(identical(penalty.weights.meta,"adaptive")){
        weights <- adaptive_weights(Z, y, nfolds = nfolds, type.measure = cvloss, 
                                    family = family, standardize = std.meta, lower.limits = ll2,
                                    upper.limits = ul2, parallel = cvparallel, lambda.min.ratio=lambda.ratio
                                    )
        cv.meta <- optimize_over_gamma(Z, y, weights, gamma.seq, nfolds = nfolds, type.measure = cvloss, alpha = alpha2, 
                                       family = family, standardize = std.meta, lower.limits = ll2,
                                       upper.limits = ul2, parallel = cvparallel, lambda.min.ratio=lambda.ratio,
                                       relax = relax.meta)
      }else{
        cv.meta <- glmnet::cv.glmnet(Z, y, nfolds = nfolds, type.measure = cvloss, alpha = alpha2,
                                     family = family, standardize = std.meta, lower.limits = ll2,
                                     upper.limits = ul2, parallel = cvparallel, lambda.min.ratio=lambda.ratio, 
                                     penalty.factor=penalty.weights.meta, relax = relax.meta)
        }
      }else{
      if(is.null(penalty.weights.meta)){
        penalty.weights.meta <- c(rep(0, ncol(correct.for)), rep(1, ncol(Z)))
        ll2 <- c(rep(-Inf, ncol(correct.for)), rep(ll2, ncol(Z)))
        ul2 <- c(rep(Inf, ncol(correct.for)), rep(ul2, ncol(Z)))
        Z <- cbind(correct.for, Z)
        cv.meta <- glmnet::cv.glmnet(Z, y, family = family, nfolds = nfolds, type.measure = cvloss, alpha = alpha2,
                                     standardize = std.meta, lower.limits = ll2,
                                     upper.limits = ul2, parallel = cvparallel, lambda.min.ratio=lambda.ratio, 
                                     penalty.factor=penalty.weights.meta, relax = relax.meta)
      }else if(identical(penalty.weights.meta,"adaptive")){
        #stop("Adaptive weights are not currently supported if correct.for is used.")
        weights <- adaptive_weights(Z, y, nfolds = nfolds, type.measure = cvloss, 
                                    family = family, standardize = std.meta, lower.limits = ll2,
                                    upper.limits = ul2, parallel = cvparallel, lambda.min.ratio=lambda.ratio
        )
        ridge.weights <- c(rep(0, ncol(correct.for)), weights$ridge.weights)
        inf.weights <- weights$ridge_weights + ncol(correct.for)
        weights <- list(ridge.weights = ridge.weights, inf.weights = inf.weights)
        ll2 <- c(rep(-Inf, ncol(correct.for)), rep(ll2, ncol(Z)))
        ul2 <- c(rep(Inf, ncol(correct.for)), rep(ul2, ncol(Z)))
        Z <- cbind(correct.for, Z)
        cv.meta <- optimize_over_gamma(Z, y, weights, gamma.seq, nfolds = nfolds, type.measure = cvloss, alpha = alpha2, 
                                       family = family, standardize = std.meta, lower.limits = ll2,
                                       upper.limits = ul2, parallel = cvparallel, lambda.min.ratio=lambda.ratio,
                                       relax = relax.meta)
      }else{
        penalty.weights.meta <- c(rep(0, ncol(correct.for)), penalty.weights.meta)
        ll2 <- c(rep(-Inf, ncol(correct.for)), rep(ll2, ncol(Z)))
        ul2 <- c(rep(Inf, ncol(correct.for)), rep(ul2, ncol(Z)))
        Z <- cbind(correct.for, Z)
        cv.meta <- glmnet::cv.glmnet(Z, y, family = family, nfolds = nfolds, type.measure = cvloss, alpha = alpha2,
                                     standardize = std.meta, lower.limits = ll2,
                                     upper.limits = ul2, parallel = cvparallel, lambda.min.ratio=lambda.ratio, 
                                     penalty.factor=penalty.weights.meta, relax = relax.meta)
      }
    }

  }

  # PARALLEL PROCESSING
  if(parallel){

    if(!is.null(seed)){
      set.seed(seed)
      folds <- kFolds(y, nfolds)
      base.seeds <- sample(.Machine$integer.max/2, size = V)
      z.seeds <- matrix(sample(.Machine$integer.max/2, size = V*nfolds), nrow=nfolds, ncol=V)
      meta.seed <- sample(.Machine$integer.max/2, size=1)
    } else folds <- kFolds(y, nfolds)

    cv.base <- foreach(v=(1:V)) %dopar% {
      if(!is.null(seed)){
        set.seed(base.seeds[v])
      }
      if(anyNA(x[, view == v])){
        if(any(check_partial_missings(x[, view == v]))){
          warning("Partially missing observations found in view ", v,
                  ". These observations will be treated as if all their values for view ", v,
                  " are missing. It may be more efficient to perform feature-level imputation. The partially missing observations are: ",
                  paste(which(check_partial_missings(x[, view == v])), collapse = ", "))
        }
        x_train <- na.omit(x[, view == v])
        y_train <-  y[-attr(x_train, "na.action")]
      }else{
        x_train <- x[, view == v]
        y_train <- y
      }
      if(is.null(penalty.weights.base)){
        glmnet::cv.glmnet(x_train, y_train, nfolds = nfolds, family = family,
                          type.measure = cvloss, alpha = alpha1,
                          standardize = std.base, lower.limits = ll1,
                          upper.limits = ul1, parallel = cvparallel, lambda.min.ratio = lambda.ratio,
                          relax = relax.base)
      }else if(inherits(penalty.weights.base, "list") && length(penalty.weights.base) == V){
        if(ncol(x_train) != length(penalty.weights.base[[v]])){
          stop("The length of each penalty weight vector should be equal to the number of features in that view.")
        }
        glmnet::cv.glmnet(x_train, y_train, nfolds = nfolds, family = family,
                          type.measure = cvloss, alpha = alpha1,
                          standardize = std.base, lower.limits = ll1,
                          upper.limits = ul1, parallel = cvparallel, lambda.min.ratio = lambda.ratio,
                          relax = relax.base, penalty.factor = penalty.weights.base[[v]])
      }else if(identical(penalty.weights.base, "adaptive")){
        weights <- adaptive_weights(x_train, y_train, nfolds = nfolds, type.measure = cvloss, 
                                    family = family, standardize = std.base, lower.limits = ll1,
                                    upper.limits = ul1, parallel = cvparallel, lambda.min.ratio=lambda.ratio)
        
        optimize_over_gamma(x_train, y_train, weights, gamma.seq, nfolds = nfolds, type.measure = cvloss, alpha = alpha1,
                            family = family, standardize = std.base, lower.limits = ll1, upper.limits = ul1, parallel = cvparallel,
                            lambda.min.ratio=lambda.ratio, relax = relax.base)
      }else{
        stop("penalty.weights.base must be either NULL, adaptive, or a list of length nviews, with each entry a numeric vector containing penalty weights for each feature within that view.")
      }
    }

    if(!skip.cv){
      Z <- foreach(v=(1:V), .combine=cbind) %:%
        foreach(k=(1:nfolds), .combine="%+%") %dopar% {
          if(!is.null(seed)){
            set.seed(z.seeds[k, v])
          }
          if(anyNA(x[, view == v])){
            x_train <- na.omit(x[, view == v])
            y_train <-  y[-attr(x_train, "na.action")]
            cvfolds <- folds[-attr(x_train, "na.action")]
          }else{
            x_train <- x[, view == v]
            y_train <- y
            cvfolds <- folds
          }
          if(is.null(penalty.weights.base)){
            cvf <- glmnet::cv.glmnet(x_train[cvfolds != k, ], y = y_train[cvfolds != k], 
                                     nfolds = nfolds, family = family,
                                     type.measure = cvloss, alpha = alpha1,
                                     standardize = std.base, lower.limits = ll1,
                                     upper.limits = ul1, parallel = cvparallel, lambda.min.ratio = lambda.ratio,
                                     relax = relax.base)
            newy <- rep(NA, length(y)) 
            newy[folds == k] <- predict(cvf, newx = x[folds == k, view == v], s = cvlambda, type = metadat)
            return(newy)
          }else if(inherits(penalty.weights.base, "list") == "list" && length(penalty.weights.base) == V){
            if(ncol(x_train) != length(penalty.weights.base[[v]])){
              stop("The length of each penalty weight vector should be equal to the number of features in that view.")
            }
            cvf <- glmnet::cv.glmnet(x_train[cvfolds != k, ], y = y_train[cvfolds != k], 
                                     nfolds = nfolds, family = family,
                                     type.measure = cvloss, alpha = alpha1,
                                     standardize = std.base, lower.limits = ll1,
                                     upper.limits = ul1, parallel = cvparallel, lambda.min.ratio = lambda.ratio,
                                     relax = relax.base, penalty.factor = penalty.weights.base[[v]])
            newy <- rep(NA, length(y)) 
            newy[folds == k] <- predict(cvf, newx = x[folds == k, view == v], s = cvlambda, type = metadat)
            return(newy)
          }else if(identical(penalty.weights.base, "adaptive")){
            weights <- adaptive_weights(x_train[cvfolds != k, ], y_train[cvfolds != k], nfolds = nfolds, type.measure = cvloss, 
                                        family = family, standardize = std.base, lower.limits = ll1,
                                        upper.limits = ul1, parallel = cvparallel, lambda.min.ratio=lambda.ratio)
            
            cvf <- optimize_over_gamma(x_train[cvfolds != k, ], y_train[cvfolds != k], weights, gamma.seq, nfolds = nfolds, type.measure = cvloss, alpha = alpha1,
                                       family = family, standardize = std.base, lower.limits = ll1, upper.limits = ul1, parallel = cvparallel,
                                       lambda.min.ratio=lambda.ratio, relax = relax.base)
            newy <- rep(NA, length(y)) 
            newy[folds == k] <- predict(cvf, newx = x[folds == k, view == v], s = cvlambda, type = metadat)
            return(newy)
          }else{
            stop("penalty.weights.base must be either NULL, adaptive, or a list of length nviews, with each entry a numeric vector containing penalty weights for each feature within that view.")
          }
        }
      
      if(na.action == "mean" && anyNA(Z)){
        Z <- impute_mean(Z)
      }else if(na.action == "mice" && anyNA(Z)){
        if(!is.null(na.arguments)){
          na.arguments <- c(list(x=Z, y=y), na.arguments)
        }else{
          na.arguments <- list(x=Z, y=y)
        }
        Z <- do.call(impute_mice, na.arguments)
      }else if(na.action == "missForest" && anyNA(Z)){
        if(!is.null(na.arguments)){
          na.arguments <- c(list(x=Z, y=y), na.arguments)
        }else{
          na.arguments <- list(x=Z, y=y)
        }
        Z <- do.call(impute_forest, na.arguments)
      }
      
      dimnames(Z) <- NULL
      if(!is.null(view.names)){
        colnames(Z) <- view.names
      }
    } else{
      Z <- NULL
      skip.meta <- TRUE
    }

    if(!is.null(seed)){
      set.seed(meta.seed)
    }
    if(skip.meta || pass){
      cv.meta <- NULL
    } else if(is.null(correct.for) && is.null(penalty.weights.meta)){
      cv.meta <- glmnet::cv.glmnet(Z, y, nfolds = nfolds, type.measure = cvloss, alpha = alpha2, 
                                   family = family, standardize = std.meta, lower.limits = ll2,
                                   upper.limits = ul2, parallel = cvparallel, lambda.min.ratio=lambda.ratio,
                                   relax = relax.meta)
    } else if(is.null(correct.for) && !is.null(penalty.weights.meta)){
      if(identical(penalty.weights.meta,"adaptive")){
        weights <- adaptive_weights(Z, y, nfolds = nfolds, type.measure = cvloss, 
                                    family = family, standardize = std.meta, lower.limits = ll2,
                                    upper.limits = ul2, parallel = cvparallel, lambda.min.ratio=lambda.ratio
        )
        cv.meta <- optimize_over_gamma(Z, y, weights, gamma.seq, nfolds = nfolds, type.measure = cvloss, alpha = alpha2, 
                                       family = family, standardize = std.meta, lower.limits = ll2,
                                       upper.limits = ul2, parallel = cvparallel, lambda.min.ratio=lambda.ratio,
                                       relax = relax.meta)
      }else{
        cv.meta <- glmnet::cv.glmnet(Z, y, nfolds = nfolds, type.measure = cvloss, alpha = alpha2,
                                     family = family, standardize = std.meta, lower.limits = ll2,
                                     upper.limits = ul2, parallel = cvparallel, lambda.min.ratio=lambda.ratio, 
                                     penalty.factor=penalty.weights.meta, relax = relax.meta)
      }
    }else{
      if(is.null(penalty.weights.meta)){
        penalty.weights.meta <- c(rep(0, ncol(correct.for)), rep(1, ncol(Z)))
        ll2 <- c(rep(-Inf, ncol(correct.for)), rep(ll2, ncol(Z)))
        ul2 <- c(rep(Inf, ncol(correct.for)), rep(ul2, ncol(Z)))
        Z <- cbind(correct.for, Z)
        cv.meta <- glmnet::cv.glmnet(Z, y, family = family, nfolds = nfolds, type.measure = cvloss, alpha = alpha2,
                                     standardize = std.meta, lower.limits = ll2,
                                     upper.limits = ul2, parallel = cvparallel, lambda.min.ratio=lambda.ratio, 
                                     penalty.factor=penalty.weights.meta, relax = relax.meta)
      }else if(identical(penalty.weights.meta,"adaptive")){
        #stop("Adaptive weights are not currently supported if correct.for is used.")
        weights <- adaptive_weights(Z, y, nfolds = nfolds, type.measure = cvloss, 
                                    family = family, standardize = std.meta, lower.limits = ll2,
                                    upper.limits = ul2, parallel = cvparallel, lambda.min.ratio=lambda.ratio
        )
        ridge.weights <- c(rep(0, ncol(correct.for)), weights$ridge.weights)
        inf.weights <- weights$ridge_weights + ncol(correct.for)
        weights <- list(ridge.weights = ridge.weights, inf.weights = inf.weights)
        ll2 <- c(rep(-Inf, ncol(correct.for)), rep(ll2, ncol(Z)))
        ul2 <- c(rep(Inf, ncol(correct.for)), rep(ul2, ncol(Z)))
        Z <- cbind(correct.for, Z)
        cv.meta <- optimize_over_gamma(Z, y, weights, gamma.seq, nfolds = nfolds, type.measure = cvloss, alpha = alpha2, 
                                       family = family, standardize = std.meta, lower.limits = ll2,
                                       upper.limits = ul2, parallel = cvparallel, lambda.min.ratio=lambda.ratio,
                                       relax = relax.meta)
      }else{
        penalty.weights.meta <- c(rep(0, ncol(correct.for)), penalty.weights.meta)
        ll2 <- c(rep(-Inf, ncol(correct.for)), rep(ll2, ncol(Z)))
        ul2 <- c(rep(Inf, ncol(correct.for)), rep(ul2, ncol(Z)))
        Z <- cbind(correct.for, Z)
        cv.meta <- glmnet::cv.glmnet(Z, y, family = family, nfolds = nfolds, type.measure = cvloss, alpha = alpha2,
                                     standardize = std.meta, lower.limits = ll2,
                                     upper.limits = ul2, parallel = cvparallel, lambda.min.ratio=lambda.ratio, 
                                     penalty.factor=penalty.weights.meta, relax = relax.meta)
      }
    }
  }


  # create output list
  out <- list(
    "base" = cv.base,
    "meta" = cv.meta,
    "CVs" = Z,
    #"x" = x,
    #"y" = y,
    "view" = view,
    "metadat" = metadat
  )

  class(out) <- "StaPLR"

  # return output
  if (progress) message("DONE")
  return(out)
}

#' @rdname StaPLR
#' @export
staplr <- StaPLR

#' Make predictions from a "StaPLR" object.
#'
#' Make predictions from a "StaPLR" object.
#' @param object Fitted "StaPLR" model object.
#' @param newx Matrix of new values for x at which predictions are to be made. Must be a matrix.
#' @param newcf Matrix of new values of correction features, if correct.for was specified during model fitting.
#' @param predtype The type of prediction returned by the meta-learner.
#' @param cvlambda Values of the penalty parameters at which predictions are to be made. Defaults to the values giving minimum cross-validation error.
#' @param ... Further arguments to be passed to \code{\link[glmnet]{predict.cv.glmnet}}.
#' @return A matrix of predictions.
#' @keywords TBA
#' @importFrom stats predict
#' @export
#' @author Wouter van Loon <w.s.van.loon@fsw.leidenuniv.nl>
#' @examples \donttest{ 
#' set.seed(012)
#' n <- 1000
#' cors <- seq(0.1,0.7,0.1)
#' X <- matrix(NA, nrow=n, ncol=length(cors)+1)
#' X[,1] <- rnorm(n)
#'
#' for(i in 1:length(cors)){
#'   X[,i+1] <- X[,1]*cors[i] + rnorm(n, 0, sqrt(1-cors[i]^2))
#' }
#'
#' beta <- c(1,0,0,0,0,0,0,0)
#' eta <- X %*% beta
#' p <- exp(eta)/(1+exp(eta))
#' y <- rbinom(n, 1, p)
#' view_index <- rep(1:(ncol(X)/2), each=2)
#'
#' fit <- StaPLR(X, y, view_index)
#' coef(fit)$meta
#'
#' new_X <- matrix(rnorm(16), nrow=2)
#' predict(fit, new_X)}

predict.StaPLR <- function(object, newx, newcf = NULL, predtype = "response", 
                           cvlambda = "lambda.min", ...){

  V <- length(unique(object$view))
  n <- nrow(newx)
  metadat <- object$metadat
  Z <- matrix(NA, n, V)
  for (v in 1:V){
    Z[,v] <- predict(object$base[[v]], newx[, object$view == v, drop=FALSE], s = cvlambda, type = metadat)
  }
  if(!is.null(newcf)){
    Z <- cbind(newcf, Z)
  }
  colnames(Z) <- colnames(object$CVs)
  out <- predict(object$meta, Z, s = cvlambda, type = predtype, ...)
  return(out)
}



#' Extract coefficients from a "StaPLR" object.
#'
#' Extract base- and meta-level coefficients from a "StaPLR" object at the CV-optimal values of the penalty parameters.
#' @param object Fitted "StaPLR" model object.
#' @param cvlambda By default, the coefficients are extracted at the CV-optimal values of the penalty parameters. Choosing "lambda.1se" will extract them at the largest values within one standard error of the minima.
#' @param ... Further arguments to be passed to \code{\link[glmnet]{coef.cv.glmnet}}.
#' @return An object with S3 class "StaPLRcoef".
#' @keywords TBA
#' @importFrom stats coef
#' @export
#' @author Wouter van Loon <w.s.van.loon@fsw.leidenuniv.nl>
#' @examples \donttest{
#' set.seed(012)
#' n <- 1000
#' cors <- seq(0.1,0.7,0.1)
#' X <- matrix(NA, nrow=n, ncol=length(cors)+1)
#' X[,1] <- rnorm(n)
#'
#' for(i in 1:length(cors)){
#'   X[,i+1] <- X[,1]*cors[i] + rnorm(n, 0, sqrt(1-cors[i]^2))
#' }
#'
#' beta <- c(1,0,0,0,0,0,0,0)
#' eta <- X %*% beta
#' p <- exp(eta)/(1+exp(eta))
#' y <- rbinom(n, 1, p)
#' view_index <- rep(1:(ncol(X)/2), each=2)
#'
#' fit <- StaPLR(X, y, view_index)
#' coef(fit)$meta
#'
#' new_X <- matrix(rnorm(16), nrow=2)
#' predict(fit, new_X)}

coef.StaPLR <- function(object, cvlambda = "lambda.min", ...){

  out <- list(
    "base" = lapply(object$base, function(x) coef(x, s=cvlambda, ...)),
    "meta" = coef(object$meta, s=cvlambda),
    "metadat" = object$metadat
  )

  class(out) <- "StaPLRcoef"

  return(out)
}


#' Make predictions from a "StaPLRcoef" object.
#'
#' Predict using a "StaPLRcoef" object. A "StaPLRcoef" object can be considerably smaller than a full "StaPLR" object for large data sets.
#' @param object Extracted StaPLR coefficients as a "StaPLRcoef" object.
#' @param newx Matrix of new values for x at which predictions are to be made. Must be a matrix.
#' @param view a vector of length nvars, where each entry is an integer describing to which view each feature corresponds.
#' @param newcf Matrix of new values of correction features, if correct.for was specified during model fitting.
#' @param predtype The type of prediction returned by the meta-learner. Allowed values are "response", "link", and "class".
#' @param ... Not currently used.
#' @return A matrix of predictions.
#' @keywords TBA
#' @export
#' @author Wouter van Loon <w.s.van.loon@fsw.leidenuniv.nl>
#' @examples \donttest{
#' set.seed(012)
#' n <- 1000
#' cors <- seq(0.1,0.7,0.1)
#' X <- matrix(NA, nrow=n, ncol=length(cors)+1)
#' X[,1] <- rnorm(n)
#'
#' for(i in 1:length(cors)){
#'   X[,i+1] <- X[,1]*cors[i] + rnorm(n, 0, sqrt(1-cors[i]^2))
#' }
#'
#' beta <- c(1,0,0,0,0,0,0,0)
#' eta <- X %*% beta
#' p <- exp(eta)/(1+exp(eta))
#' y <- rbinom(n, 1, p)
#' view_index <- rep(1:(ncol(X)/2), each=2)
#'
#' fit <- StaPLR(X, y, view_index)
#' coefficients <- coef(fit)
#'
#' new_X <- matrix(rnorm(16), nrow=2)
#' predict(coefficients, new_X, view_index)}

predict.StaPLRcoef <- function(object, newx, view, newcf = NULL, predtype = "response",
                               ...){

  V <- length(unique(view))
  n <- nrow(newx)
  metadat <- object$metadat
  Z <- matrix(NA, n, V)
  for (v in 1:V){
    Z[,v] <- as.matrix(cbind(1, newx[, view == v, drop=FALSE]) %*% object$base[[v]])
  }

  if(metadat == "response"){
    Z <- 1/(1+exp(-Z))
  } else if(metadat == "class"){
    Z <- 1*(1/(1+exp(-Z)) > 0.5)
  } else if(metadat != "link"){
    stop("metadat should be one of 'response', 'class' or 'link'.")
  }

  if(!is.null(newcf)){
    Z <- cbind(newcf, Z)
  }

  out <- as.matrix(cbind(1, Z) %*% object$meta)

  if(predtype == "response"){
    out <- 1/(1+exp(-out))
  } else if(predtype == "class"){
    out <- 1*(1/(1+exp(-out)) > 0.5)
  } else if(predtype != "link"){
    stop("predtype should be one of 'response', 'class' or 'link'.")
  }

  return(out)

}
