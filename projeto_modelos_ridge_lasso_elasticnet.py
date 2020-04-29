from sklearn.model_selection import RandomizedSearchCV,RepeatedKFold
from scipy.stats import laplace,norm,uniform
from sklearn.linear_model import Ridge,Lasso,ElasticNet
import numpy as np
from plotnine import *
import pandas as pd

laplace=laplace(loc=2,scale=1)
x=laplace.rvs(size=100)
normal=norm(loc=0,scale=1)
e=normal.rvs(size=100)

beta_0=1.4
beta_1=5.2
beta_2=1.7
X=np.column_stack((x,x**2))
y=beta_0+beta_1*x+beta_2*(x**2)+e

particao=RepeatedKFold(n_splits=10,n_repeats=3)

modelo1=Ridge()
hiperp={"alpha":uniform(loc=0,scale=5)}
otimizacao=RandomizedSearchCV(modelo1,param_distributions=hiperp,n_iter=100
                              ,cv=particao,scoring="neg_mean_absolute_error")
otimizacao.fit(X,y)
otimizacao.best_params_
otimizacao.best_score_
otimizacao.cv_results_["param_alpha"].data
otimizacao.cv_results_["mean_test_score"]
otimizacao.cv_results_["std_test_score"]
otimizacao.best_estimator_.coef_
otimizacao.best_estimator_.intercept_

dados=pd.DataFrame({"alpha":otimizacao.cv_results_["param_alpha"].data,
                   "erro":-otimizacao.cv_results_["mean_test_score"],
                   "desvio-padrao":otimizacao.cv_results_["std_test_score"]
                   ,"indice":np.arange(1,51)})
#Ver depois
ggplot(dados)+geom_line(aes(x="indice",y="erro"))
ggplot(dados)+geom_line(aes(x="indice",y="desvio-padrao"))


modelo2=Lasso()
hiperp={"alpha":uniform(loc=0,scale=5)}
otimizacao=RandomizedSearchCV(modelo2,param_distributions=hiperp,n_iter=100,
                              cv=particao,scoring="neg_mean_absolute_error")
otimizacao.fit(X,y)
otimizacao.best_params_
otimizacao.best_score_
otimizacao.cv_results_["param_alpha"].data
otimizacao.cv_results_["mean_test_score"]
otimizacao.cv_results_["std_test_score"]
otimizacao.best_estimator_.coef_
otimizacao.best_estimator_.intercept_

dados=pd.DataFrame({"alpha":otimizacao.cv_results_["param_alpha"].data,
                   "erro":-otimizacao.cv_results_["mean_test_score"],
                   "desvio-padrao":otimizacao.cv_results_["std_test_score"]
                   ,"indice":np.arange(1,51)})
#Ver depois
ggplot(dados)+geom_line(aes(x="indice",y="erro"))
ggplot(dados)+geom_line(aes(x="indice",y="desvio-padrao"))

#+annotate("text",x=11,y=0.89,label="alpha=0.037"))

modelo3=ElasticNet()
hiperp={"l1_ratio":uniform(loc=0,scale=5)}
otimizacao=RandomizedSearchCV(modelo3,param_distributions=hiperp,n_iter=100,
                              cv=particao,scoring="neg_mean_absolute_error")
otimizacao.fit(X,y)
otimizacao.best_params_
otimizacao.best_score_
otimizacao.cv_results_["param_l1_ratio"].data
otimizacao.cv_results_["mean_test_score"]
otimizacao.cv_results_["std_test_score"]
otimizacao.best_estimator_.coef_
otimizacao.best_estimator_.intercept_

dados=pd.DataFrame({"l1_ratio":otimizacao.cv_results_["param_l1_ratio"].data,
                   "erro":-otimizacao.cv_results_["mean_test_score"],
                   "desvio-padrao":otimizacao.cv_results_["std_test_score"]
                   ,"indice":np.arange(1,101)})
#Ver depois
ggplot(dados)+geom_line(aes(x="indice",y="erro"))
ggplot(dados)+geom_line(aes(x="indice",y="desvio-padrao"))