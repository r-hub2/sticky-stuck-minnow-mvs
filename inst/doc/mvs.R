## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----setup, include = FALSE---------------------------------------------------
library(mvs)

## ----mvs-diag, out.width = '70%', echo = FALSE, fig.cap="A simple graphic representation of a multi-view stacking model including 3 views: structural MRI, functional MRI, and genetic information. A sub-model is fitted on each view separately, and the predictions of these sub-models are combined by the meta-learner into a single prediction. Note that the *n* persons are the same persons for each view."----
knitr::include_graphics("img/mvs_diagram.png")

## ----staplr-diag, out.width = '70%', echo = FALSE, fig.cap="A simple graphic representation of how StaPLR can perform automatic view selection. In this (hypothetical) example, functional MRI was discarded from the model because it was not sufficiently predictive of the outcome in the presence of the other two views."----
knitr::include_graphics("img/staplr_diagram.png")

## ----alg, out.width = '70%', echo = FALSE, fig.cap="", fig.align='center'-----
knitr::include_graphics("img/algorithm_1.png")

## ----staplr-flow, out.width = '70%', echo = FALSE, fig.cap="The MVS algorithm represented as a flow diagram. StaPLR denotes the special case where all learners are penalized logistic regression learners. Figure adapted from [@StaPLR4]"----
knitr::include_graphics("img/staplr_flow.png")

## ----missing-data, out.width = '70%', echo = FALSE, fig.cap="A simple graphic representation of meta-level imputation. Assume, for example, that the three views consist of, respectively, 100, 1000 and 10,000 features. Now, say that there are 10 observations which have missing values on view $X^{(2)}$. Then in traditional imputation we would have to impute 10 × 1000 = 10,000 values whereas in list-wise deletion 10 × (100 + 10,000) = 101,000 values would be deleted even though they were observed. However, in meta-level imputation only 10 values have to be imputed, and no observed values are deleted. Figure adapted from [@StaPLR4]."----
knitr::include_graphics("img/missing_data.png")

## ----eval=F-------------------------------------------------------------------
# install.packages("mvs")

## ----eval=F-------------------------------------------------------------------
# devtools::install_gitlab("wsvanloon/mvs@develop")

## ----eval=F-------------------------------------------------------------------
# library(mvs)

## -----------------------------------------------------------------------------
set.seed(123)
n <- 100
X <- matrix(rnorm(8500), nrow=n, ncol=85)
b <- c(rep(10, 65), rep(0, 20)) * ((rbinom(85, 1, 0.5)*2)-1)
eta <- X %*% b
p <- 1 /(1 + exp(-eta))
y <- rbinom(n, 1, p)
views <- c(rep(1,45), rep(2,20), rep(3,20))

## ----results='hide', message=FALSE--------------------------------------------
fit <- MVS(x=X, y=y, views=views, alphas=c(0,1), family="binomial")

## ----eval=F-------------------------------------------------------------------
# fit <- MVS(x=X, y=y, views=views)

## -----------------------------------------------------------------------------
coef(fit)$'Level 2'

## -----------------------------------------------------------------------------
new_X <- matrix(rnorm(2*85), nrow=2)

## -----------------------------------------------------------------------------
predict(fit, new_X)

## -----------------------------------------------------------------------------
predict(fit, new_X, predtype="class")

## ----eval=FALSE---------------------------------------------------------------
# fit <- MVS(x=X, y=y, views=views, type="RF")

## ----results='hide', message=FALSE--------------------------------------------
fit <- MVS(x=X, y=y, views=views, type=c("RF", "StaPLR"))

## -----------------------------------------------------------------------------
coef(fit)

## ----R.options=list(max.print=5)----------------------------------------------
importance(fit)

## ----eval=F-------------------------------------------------------------------
# library(doParallel)
# registerDoParallel(cores = detectCores())

## ----eval=F-------------------------------------------------------------------
# fit <- MVS(x=X, y=y, views=views, parallel=TRUE)

## -----------------------------------------------------------------------------
set.seed(123)
n <- 100
X <- matrix(rnorm(8500), nrow=n, ncol=85)
b <- c(rep(0, 15), rep(10, 40), rep(0, 30)) * ((rbinom(85, 1, 0.5)*2)-1)
eta <- X %*% b
p <- 1 /(1 + exp(-eta))
y <- rbinom(n, 1, p)

sub_views <- c(rep(1:3, each=15), rep(4:5, each=10), rep(6:9, each=5))
top_views <- c(rep(1,45), rep(2,20), rep(3,20))

## -----------------------------------------------------------------------------
views <- cbind(sub_views, top_views)

## ----results='hide', message=FALSE--------------------------------------------
fit <- MVS(x=X, y=y, views=views, levels=3, alphas=c(0,1,1), nnc=c(0,1,1))

## -----------------------------------------------------------------------------
coef(fit)$'Level 3'

## -----------------------------------------------------------------------------
coef(fit)$'Level 2'

## -----------------------------------------------------------------------------
MRM(fit, constant = mean(y), level=2)

## -----------------------------------------------------------------------------
set.seed(123)
n <- 100
X <- matrix(rnorm(8500), nrow=n, ncol=85)
b <- c(rep(10, 65), rep(0, 20)) * ((rbinom(85, 1, 0.5)*2)-1)
eta <- X %*% b
p <- 1 /(1 + exp(-eta))
y <- rbinom(n, 1, p)
views <- c(rep(1,45), rep(2,20), rep(3,20))

## -----------------------------------------------------------------------------
X[1:50, 1:45] <- NA

## ----message=FALSE, error=TRUE------------------------------------------------
try({
fit <- MVS(x=X, y=y, views=views)
})

## ----message=FALSE, results='hide'--------------------------------------------
fit <- MVS(x=X, y=y, views=views, na.action="mice")

## -----------------------------------------------------------------------------
coef(fit)$'Level 2'

## -----------------------------------------------------------------------------
attributes(fit$'Level 1'$CVs)

## ----message=FALSE, results='hide'--------------------------------------------
fit <- MVS(x=X, y=y, views=views, na.action="mice", na.arguments=list(m = 10))

## -----------------------------------------------------------------------------
attributes(fit$'Level 1'$CVs)

## ----message=FALSE, warning=FALSE, results='hide'-----------------------------
fit <- MVS(x=X, y=y, views=views, na.action="pass")

## ----R.options=list(max.print=30)---------------------------------------------
fit$`Level 1`$CVs

