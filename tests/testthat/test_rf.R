test_that("Function MVS works with type='RF' and predict method works correctly.",{
  
  set.seed(012)
  n <- 1000
  X <- matrix(rnorm(8500), nrow=n, ncol=85)
  beta <- c(rep(10, 55), rep(0, 30)) * ((rbinom(85, 1, 0.5)*2)-1)
  eta <- X %*% beta
  p <- 1 /(1 + exp(-eta))
  y <- rbinom(n, 1, p)
  ## 2-level MVS
  views <- c(rep(1,45), rep(2,20), rep(3,20))
  
  ## check 'standard' RF MVS
  set.seed(013)
  fit <- MVS(x=X, y=y, views=views, type = c("RF"))
  
  ## check baselearners
  expect_equal(fit[[1]]$models[[1]]$confusion[ , 1], c("0"=451, "1"=66))
  expect_equal(fit[[1]]$models[[2]]$confusion[ , 1], c("0"=442, "1"=69))  
  expect_equal(fit[[1]]$models[[3]]$confusion[ , 1], c("0"=439, "1"=85))
  ## check metalearner
  expect_equal(fit[[2]]$models[[1]]$confusion[ , 1], c("0"=435, "1"=65))
  ## check predict method
  expect_equal(predict(fit, newx = X[1:2,]), matrix(c(1, 0), ncol = 1), 
               tolerance = 1e-03)
  
  ## check MVS combining RF for base, and glmnet for meta learner
  set.seed(014)
  fit <- MVS(x=X, y=y, views=views, type = c("RF", "StaPLR"))
  
  ## check baselearners
  expect_equal(fit[[1]]$models[[1]]$confusion[ , 1], c("0"=443, "1"=74))
  expect_equal(fit[[1]]$models[[2]]$confusion[ , 1], c("0"=447, "1"=70))  
  expect_equal(fit[[1]]$models[[3]]$confusion[ , 1], c("0"=433, "1"=83))
  ## check metalearner  
  expect_equal(fit[[2]]$models[[1]]$index, 
               matrix(c(65,29), dimnames = list(c("min", "1se"), c("Lambda"))))
  ## check predict method
  expect_equal(predict(fit, newx = X[1:2,]), matrix(c(0.9883, 0.0029), ncol = 1), 
               tolerance = 1e-04)
  ##check coef method
  expect_true(all(is.na(c(unlist(coef(fit)$`Level 1`)))))
  expect_equal(coef(fit)$`Level 2`[[1]][1:4], c(-7.254392, 3.466286, 11.122735, 0),
               tolerance = 1e-04)
  ##check importance method
  expect_true(is.na(unlist(importance(fit)$`Level 2`)))
  expect_equal(importance(fit)$`Level 1`[[3]][1:20], c(13.75300, 41.65184, 42.47944,
                                                       12.15466, 33.81692, 33.90427,
                                                       11.42747, 17.09531, 20.56260,
                                                       13.60281, 19.06030, 12.09567,
                                                       16.47032, 46.67850, 14.60348,
                                                       37.22340, 14.79609, 14.21468,
                                                       41.29158, 42.18340), 
               tolerance = 1e-04)
})