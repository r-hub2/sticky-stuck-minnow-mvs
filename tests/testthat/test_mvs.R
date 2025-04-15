test_that("MVS",{
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
  
  MVS_fit <- MVS(x=X, y=y, views=views, type="StaPLR", levels=3, alphas=c(0,1,1), nnc=c(0,1,1))
  MVS_fit_adaptive <- MVS(x=X, y=y, views=views, type="StaPLR", levels=3, alphas=c(0,1,1), nnc=c(0,1,1), adaptive=c(F,F,T))
  expect_equal(coef(MVS_fit)$`Level 3`[[1]][1:4], c(-1.996115,  3.917804,  0,  0), tolerance = 1e-03)
  expect_equal(mrm(MVS_fit, mean(y)), c(0.5257742, 0.6750250, 0.6129507, 0, 0, 0, 0, 0, 0), tolerance = 1e-03)
  expect_equal(coef(MVS_fit_adaptive)$`Level 3`[[1]][1:4], c(-1.904398,  3.739364, 0, 0), tolerance = 1e-03)
})