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

#' Function for fitting random forests with multi-view stacking
#'
#' A wrapper function around randomForest from package of the same name that allows to use it in function MVS. 
#' @param x input matrix of dimension nobs x nvars
#' @param y outcome vector of length nobs
#' @param view a vector of length nvars, where each entry is an integer describing to which view each feature corresponds.
#' @param view.names (optional) a character vector of length nviews specifying a name for each view.
#' @param skip.meta whether to skip training the metalearner.
#' @param skip.cv whether to skip generating the cross-validated predictions.
#' @param na.action character specifying what to do with missing values (NA). Options are "pass", "fail", "mean", "mice", and "missForest". Options "mice" and "missForest" requires the respective R package to be installed. Defaults to "pass".
#' @param na.arguments (optional) a named list of arguments to pass to the imputation function (e.g. to \code{mice} or \code{missForest}).
#' @param progress whether to show a progress bar (only supported when parallel = FALSE).
#' @param ... Additional arguments to be passed to function \code{\link[randomForest]{randomForest}}.
#' @return An object with S3 class "RF".
#' @keywords TBA
#' @import randomForest
#' @export
#' @author Marjolein Fokkema <m.fokkema@fsw.leidenuniv.nl>
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
#' # Stacked random forest
#' fit <- RF(X, y, view_index, skip.meta = FALSE, skip.cv = FALSE)
#' 
#' # Stacked random forest
#' y <- eta + rnorm(100) ## create continuous response
#' fit <- RF(X, y, view_index,skip.meta = FALSE, skip.cv = FALSE)
#' }
RF <- function(x, y, view, view.names = NULL, 
               skip.meta = FALSE, skip.cv = FALSE, 
               na.action = "fail", na.arguments = NULL,
               progress = TRUE, ...) {
  
  ## When fitting an MVS model and using cv.glmnet, and !skip.cv, view-level predictions Z for training data04
  ## are cross-validated predictions. These should be OOB predictions for RF.
  ## For prediction with new observations, the 'standard' cv.glmnet predictions should be returned at the view level.
  ## This requires running cv.glmnet twice, but for RF, it should suffice to run randomForest once.
  #set.seed(1)
  #airq <- airquality[complete.cases(airquality), ]
  #rf <- randomForest(Ozone ~ ., data=airq)
  #plot(predict(rf), predict(rf, newdata = airq), xlab = "OOB predictions (no newdata)",
  #     ylab = "predictions (newdata supplied)")
  #cor(predict(rf), airq$Ozone) ## OOB predictions do not overfit
  #cor(predict(rf, newdata = airq), airq$Ozone) ## 'standard' predictions overfit
  
  
  ## check response type
  if (is.matrix(y) && ncol(y) > 1L)
    stop("MVS does not yet support response variables comprising multiple columns.")
  ## TODO: Check if y is numeric, or a factor. In the latter case, check that there are only two levels.
  task <- ifelse(length(unique(y)) == 2L, "classification", "regression")
  if (task == "classification") {
    y <- factor(y)
    response_type <- "prob"
  } else {
    response_type <- "response"
  }
  
  ## check if data is complete
  if (sum(is.na(y)) > 0L) stop("There are missing values in y. Random-forest style MVS does not suport missings in y (yet).")
  if (any(sum(apply(x, 2L, function(col) sum(is.na(col)) > 0L)))) stop("There are missing values in x. Random-forest style MVS does not suport missings in x (yet).")
  
  ## object initialization
  V <- length(unique(view))
  n <- if (is.matrix(y)) nrow(y) else length(y)

  ## Fit base and meta learners
  if (!skip.cv) { 
    if (progress) {
      message("Training learner on each view...")
      pb <- txtProgressBar(min=0, max=V, style=3)
    }
    cv.base <- foreach(v=(1:V)) %do% {
      if (progress) setTxtProgressBar(pb, v)
      x_train <- x[, view == v]
      y_train <- y
      randomForest::randomForest(x = x_train, y = y_train, ...)
    }
    if (progress) message("\n Calculating out-of-bag predictions...")
    Z <- if (response_type == "prob") {
      sapply(cv.base, function(x) predict(x, type = "prob")[ , 2])
    } else {
      sapply(cv.base, predict, type = response_type)      
    }
    if (na.action == "mean" && anyNA(Z)) {
      Z <- impute_mean(Z)
    } else if (na.action == "mice" && anyNA(Z)) {
      if (!is.null(na.arguments)) {
        na.arguments <- c(list(x=Z, y=y), na.arguments)
      } else {
        na.arguments <- list(x=Z, y=y)
      }
      Z <- do.call(impute_mice, na.arguments)
    } else if (na.action == "missForest" && anyNA(Z)) {
      if (!is.null(na.arguments)) {
        na.arguments <- c(list(x=Z, y=y), na.arguments)
      } else {
        na.arguments <- list(x=Z, y=y)
      }
      Z <- do.call(impute_forest, na.arguments)
    }
    dimnames(Z) <- NULL
    if (!is.null(view.names)) colnames(Z) <- view.names
  }

  if (progress && !skip.meta) message("\n Training meta learner...")
  if (skip.meta) {
    cv.meta <- NULL
  } else {
    ## TODO: It is odd to assign the meta learner to cv.base, 
    ## but this is to keep in line with behavior when type = "StaPLR"
    cv.base <- list()
    cv.base[[1L]] <- randomForest::randomForest(x = x, y = y, ...)
    cv.meta <- Z <- NULL
  }

  # create output list
  out <- list(
    "base" = cv.base,
    "meta" = cv.meta,
    "CVs" = Z,
    "view" = view#,
    #"metadat" = metadat ## TODO: if type = "StaPLR", metadat always seems to be "response"?  
  )

  class(out) <- "RF"

  # return output
  if (progress) message("DONE")
  return(out)
}
