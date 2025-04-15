test_that("missing_data_StaPLR",{
  set.seed(012)
  n <- 100
  cors <- seq(0.1,0.7,0.1)
  X <- matrix(NA, nrow=n, ncol=length(cors)+1)
  X[,1] <- rnorm(n)
  for(i in 1:length(cors)){
    X[,i+1] <- X[,1]*cors[i] + rnorm(n, 0, sqrt(1-cors[i]^2))
  }
  beta <- c(1,0,0,0,0,0,0,0)
  eta <- X %*% beta
  p <- exp(eta)/(1+exp(eta))
  y <- rbinom(n, 1, p)
  view_index <- rep(1:(ncol(X)/2), each=2)
  X[1:50, 1:2] <- NA
  
  StaPLR_fit_missing <- StaPLR(X, y, view_index, seed=123, na.action="pass")
  StaPLR_fit_mean <- StaPLR(X, y, view_index, seed=123, na.action="mean")
  StaPLR_fit_mice <- StaPLR(X, y, view_index, seed=123, na.action="mice", na.arguments = list(m=10, method="mean"))
  StaPLR_fit_forest <- StaPLR(X, y, view_index, seed=123, na.action="missForest", na.arguments = list(ntree=200))
  
  expect_equal(which(is.na(StaPLR_fit_missing$CVs)), 1:50)
  expect_equal(mean(StaPLR_fit_mean$CVs[1:50,1]), mean(StaPLR_fit_mean$CVs[51:100,1]), tolerance = 1e-06)
  expect_equal(attr(StaPLR_fit_mice$CVs, "number_of_imputations"), 10)
  expect_equal(attr(StaPLR_fit_forest$CVs, "number_of_trees"), 200)
  
  X[1,1] <- 1
  expect_warning(StaPLR(X, y, view_index, seed=123, na.action="pass"))
  expect_error(StaPLR(X, y, view_index, seed=123, na.action="fail"))
})

test_that("missing_data_MVS",{
  set.seed(012)
  n <- 100
  X <- matrix(rnorm(8500), nrow=n, ncol=85)
  top_level <- c(rep(1,45), rep(2,20), rep(3,20))
  bottom_level <- c(rep(1:3, each=15), rep(4:5, each=10), rep(6:9, each=5))
  views <- cbind(bottom_level, top_level)
  beta <- c(rep(10, 55), rep(0, 30)) * ((rbinom(85, 1, 0.5)*2)-1)
  eta <- X %*% beta
  p <- 1 /(1 + exp(-eta))
  y <- rbinom(n, 1, p)
  
  X[1:50, bottom_level %in% 1:2] <- NA
  
  expect_error(MVS(x=X, y=y, views=views, type="StaPLR", levels=3, alphas=c(0,1,1), nnc=c(0,1,1)))
  MVS_fit <- MVS(x=X, y=y, views=views, type="StaPLR", levels=3, alphas=c(0,1,1), nnc=c(0,1,1), na.action="mean")
  expect_equal(MVS_fit[[1]]$CVs[1,1:2], colMeans((MVS_fit[[1]]$CVs[51:100, 1:2])), tolerance = 1e-06)
  MVS_fit_pass <- MVS(x=X, y=y, views=bottom_level, type="StaPLR", levels=2, alphas=c(0,1), nnc=c(0,1), na.action="pass")
  expect_null(MVS_fit_pass$`Level 2`)
})