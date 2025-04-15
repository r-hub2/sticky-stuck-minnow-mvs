test_that("StaPLR",{
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
  
  StaPLR_fit <- StaPLR(X, y, view_index) # regular StaPLR model
  StaPLR_fit_adaptive <- StaPLR(X, y, view_index, penalty.weights.meta="adaptive") # adaptive StaPLR model
  StaPLR_fit_bad_prior <- StaPLR(X, y, view_index, penalty.weights.meta=c(4:1)^10) # StaPLR model with a bad choice of weights
  
  correction_X <- matrix(rnorm(2*n), nrow=n)
  
  StaPLR_fit_correction <- StaPLR(X, y, view_index, correct.for=correction_X) # regular StaPLR model with correction features
  StaPLR_fit_adaptive_correction <- StaPLR(X, y, view_index, penalty.weights.meta="adaptive", correct.for=correction_X) # adaptive StaPLR model with correction features
  StaPLR_fit_bad_prior_correction <- StaPLR(X, y, view_index, penalty.weights.meta=c(4:1)^10, correct.for=correction_X) # StaPLR model with a bad choice of weights and correction features
  
  StaPLR_fit_base_adaptive <- StaPLR(X, y, view_index, penalty.weights.base = "adaptive", alpha1=1) # StaPLR with adaptive weights at the base level
  prior_weights <- list(c(1, 1e4), c(1, 1e12), c(1, 1e12), c(1e12, 1))
  StaPLR_fit_base_prior <- StaPLR(X, y, view_index, penalty.weights.base = prior_weights, alpha1=1) # StaPLR model with predefined weights at the base level
  
  expect_equal(coef(StaPLR_fit)$meta[1:5], c(-2.091923, 2.532491, 0, 1.569726, 0), tolerance = 1e-03)
  expect_equal(coef(StaPLR_fit_adaptive)$meta[1:5], c(-1.785526, 3.239912, 0, 0, 0), tolerance = 1e-03)
  expect_equal(coef(StaPLR_fit_bad_prior)$meta[1:5], c(-2.024544, 0, 0, 3.919528, 0), tolerance = 1e-03)
  expect_equal(coef(StaPLR_fit_correction)$meta[1:7], c(-2.5334672, -0.2898884, 0.2477464, 3.1250834, 0, 2.1763337, 0), tolerance = 1e-03)
  expect_equal(coef(StaPLR_fit_adaptive_correction)$meta[1:7], c(-2.5794505, -0.3053677, 0.2505100, 2.9661588, 0, 2.4821950, 0), tolerance = 1e-03)
  expect_equal(coef(StaPLR_fit_bad_prior_correction)$meta[1:7], c(-2.1673419, -0.2876726, 0.2885247, 0, 0, 3.4494399, 0.8946129), tolerance = 1e-03)
  expect_equal(coef(StaPLR_fit_base_adaptive)$meta[1:5], c(-2.484853, 3.524436, 0, 1.619360, 0), tolerance = 1e-03)
  expect_equal(coef(StaPLR_fit_base_prior)$meta[1:5], c(-1.600930, 2.765358, 0, 0, 0.000000), tolerance = 1e-03)
  
  expect_error(StaPLR(X, y, view_index, penalty.weights.base = "test", alpha1=1)) # not a valid option
  expect_error(StaPLR(X, y, view_index, penalty.weights.base = as.list(1:4), alpha1=1)) # wrong list dimensions
  })