library(rvinecopulib)
files <- list.files(path=train_dat)

df <- read.csv("C:\\Users\\TJR\\Documents\\Projects\\statarb\\statarb\\scrap_book\\train\\returns.csv")
df <- df[, !(colnames(df) %in% c("Date"))]

u <- pseudo_obs(df)

cvine <- cvine_structure(4:1)
plot(cvine)

fit <- vine(u, copula_controls = list(structure = cvine))
print(summary(fit))

d = dvine(u, fit)
plot(d)

p = pvine(u, fit)
plot(p)

