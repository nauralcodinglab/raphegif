test_data = read.csv('../data/matrices/c0mat.csv')
test_data

mod = lm(test_data[, 1] ~  test_data[, 2] + test_data[, 3] + test_data[, 5] + test_data[, 6])
mod
car::Anova(mod, type = 'III')
car::vif(mod)
# According to vif, I should leave out V and IA...
# However, if non-inactivating conductance is removed, vifs decrease quite a bit.

plot(mod$fitted.values)
