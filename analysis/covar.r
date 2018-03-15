test_data = read.csv('../data/matrices/c3mat.csv')

mod_min = lm(test_data[, 1] ~  test_data[, 2] + test_data[, 3])
mod_gk1 = lm(test_data[, 1] ~  test_data[, 2] + test_data[, 3] + test_data[, 5])
mod_gk2 = lm(test_data[, 1] ~  test_data[, 2] + test_data[, 3] + test_data[, 6])
mod_full = lm(test_data[, 1] ~  test_data[, 2] + test_data[, 3] + test_data[, 5] + test_data[, 6])

mods = list(mod_min, mod_gk1, mod_gk2, mod_full)

car::vif(mod_min)
print(paste0('gk1 VIF: ', round(car::vif(mod_gk1)[[3]], 1)))
print(paste0('gk2 VIF: ', round(car::vif(mod_gk2)[[3]], 1)))
car::vif(mod_full)

print('Model adjusted R^2')
for (i in 1:length(mods)){
  print(summary(mods[[i]])$adj.r.squared)
}
