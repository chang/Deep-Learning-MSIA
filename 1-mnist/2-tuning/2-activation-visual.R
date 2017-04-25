require(ggplot2)
require(tidyr)

sigmoid <- function(x, k) {
  return(1 / (1 + exp(k * -x)))
}

dat <- data.frame(x = runif(500) * 20 - 10)
dat$k1 <- sigmoid(dat$x, k = 1)
dat$k0.5 <- sigmoid(dat$x, k = 0.5)
dat$k2 <- sigmoid(dat$x, k = 2)

dat <- gather(dat, k1:k2, key="k", value="y")

ggplot(dat) + 
  geom_point(aes(x, y, color = k), size = 1, alpha = .3) +
  ggtitle("Comparison of sigmoid functions")
  