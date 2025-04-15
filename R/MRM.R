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

#' Minority Report Measure
#'
#' Calculate the Minority Report Measure (MRM) for each view in a (hierarchical) multi-view stacking model.
#' 
#' The Minority Report Measure (MRM) considers the view-specific sub-models at a given level of the hierarchy as members of a committee making predictions of the outcome variable. For each view, the MRM quantifies how much the final prediction of the stacked model changes if the prediction of the corresponding sub-model changes from \code{a} to \code{b}, while keeping the predictions corresponding to the other views constant at \code{constant}. For more information about the MRM see <doi:10.3389/fnins.2022.830630>.  
#' @param fit an object of class \code{\link[mvs]{MVS}}.
#' @param constant the value at which to keep the predictions of the other views constant. The recommended value is the mean of the outcome variable. 
#' @param level the level at which to calculate the MRM. In a 3-level MVS model, \code{level = 2} (the default) is generally the level for which one would want to calculate the MRM. Note that calculating the MRM for \code{level = 1} (the feature level) is possible, but generally not sensible except under specific conditions.  
#' @param a the start of the interval over which to calculate the MRM. Defaults to 0.
#' @param b the end of the interval over which to calculate the MRM. Defaults to 1.
#' @param cvlambda denotes which values of the penalty parameters to use for calculating predictions. This corresponds to the defaults used during model fitting. 
#' @return A numeric vector of a length equal to the number of views at the specified level, containing the values of the MRM for each view.
#' @keywords TBA
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
#' ## 3-level MVS
#' bottom_level <- c(rep(1:3, each=15), rep(4:5, each=10), rep(6:9, each=5))
#' top_level <- c(rep(1,45), rep(2,20), rep(3,20))
#' views <- cbind(bottom_level, top_level)
#' fit <- MVS(x=X, y=y, views=views, levels=3, alphas=c(0,1,1), nnc=c(0,1,1))
#' MRM(fit, constant=mean(y))
#' }

MRM <-function(fit, constant, level=2, a=0, b=1, cvlambda="lambda.min"){
  # Input: an object of class MVS
  # Output: the minority report measure for all views at the specified level.
  
  if(!inherits(fit, "MVS")){
    stop("fit must be an object of class MVS.")
  }
  
  if(!(level %in% 1:length(fit))){
    stop("Level out of bounds. Must be an integer between 1 and the number of levels of the model.")
  }
  
  if(level == 1){
    warning("Calculating the MRM for the first (i.e. feature) level is only sensible if all features share the same range, and if that range is properly specified using parameters a and b.")
  }
  
  out <- rep(NA, ncol=length(fit[[level]]$view))
  
  for(i in 1:length(fit[[level]]$view)){
    out[i] <- mrm_one(fit=fit, level=level, v=i, a=a, b=b, z=constant, cvlambda=cvlambda)
  }
  
  return(out)
  
}

#' @rdname MRM
#' @export
mrm <- MRM

mrm_one <- function(fit, level=2, v, a, b, z, cvlambda="lambda.min"){
  
  if(inherits(fit[[level]], "RF")){
    stop("Calculation of the MRM is not (yet) supported for random forests.")
  }
  
  lvl_index <- fit[[level]]$view # view index at the desired level
  x0 <- rep(z, length(lvl_index)) # set all predictions to the value of z
  x1 <- x0 # idem
  x0[v] <- a # set the prediction for view v to a
  x1[v] <- b # set the prediction for view v to b

  for(j in level:length(fit)){ # for every level, starting at the desired level
    lvl_index <- fit[[j]]$view # view index at the desired level
    z0 <- c() # initialize a vector of predictions for the next level (which is of varying dimension)
    z1 <- c() # idem
    for(i in 1:length(unique(lvl_index))){ # for each view
      z0[i] <- predict(fit[[j]]$models[[i]], x0[lvl_index == i], s = cvlambda, type=fit[[j]]$metadat) 
      z1[i] <- predict(fit[[j]]$models[[i]], x1[lvl_index == i], s = cvlambda, type=fit[[j]]$metadat)
    }
    x0 <- z0 
    x1 <- z1
  }
  
  minority_report <- z1 - z0
  
  return(minority_report)
}
