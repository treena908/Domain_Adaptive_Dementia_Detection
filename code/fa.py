from sklearn.linear_model import RidgeClassifier
from utils import make_classification_da
# from _fa import FA
# from _dann import DANN
from _fmmd import fMMD

from data_handler import get_target_source_data
# Xs2, ys2, Xt, yt = make_classification_da()
# Xs1, ys1, Xt1, yt1 = make_classification_da()
Xs, ys, Xt, yt = make_classification_da()
# model = DANN(lambda_=0.1, Xt=Xt, metrics=["acc"], random_state=0)
# model.fit(Xs, ys, epochs=100, verbose=0)
# model.score(Xt, yt)
print(Xs.shape)
print(ys.shape)
print(Xt.shape)
print(yt.shape)

model = fMMD(RidgeClassifier(), Xt=Xt, kernel="linear", random_state=0, verbose=0)
model.fit(Xs, ys)
print(model.score(Xt, yt))

# print(type(ys1))
# src = ['ccc', 'adrc']
# tgt = ['pitt']
# Xt, yt, lt, Xs, ys, ls = get_target_source_data(source=src, target=tgt)
# Xt=Xt.to_numpy()
# yt=yt.to_numpy()
# Xs[0]=Xs[0].to_numpy()
# Xs[1]=Xs[1].to_numpy()
# ys[0]=ys[0].to_numpy()
# ys[1]=ys[1].to_numpy()
#
#
# Xs=[]
# Xs.append(Xs1)
# Xs.append(Xs2)
# ys=[]
# ys.append(ys1)
# ys.append(ys2)
# print(type(Xs))
# print(type(ys))
# print()

#
# model = FA(RidgeClassifier(), Xt=Xt, yt=yt, random_state=0)
# x=model.fit_transform()
# print('hi')
# model.fit(Xs, ys)
# #Fit transform...
# #Previous shape: (100, 2)
# #New shape: (110, 6)
# #Fit Estimator...
# score=model.score(Xt[11:], yt[11:])
# # print(Xt[11:], yt[11:])
# print(score)
# print(model.predict(Xt[11:], yt[11:]))
