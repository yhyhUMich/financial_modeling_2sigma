import kagglegym
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge

import xgboost as xgb
import math
import time
from operator import itemgetter


# train/public split, use for 
config_local = {'localrun': True, 'usepublic': True, 'vmode': False, 'all_features': False}
# validation mode (random non-stratified 80/20 split)
# in *some* versions vmode will auto-stop, others will need xgbcap
config_vmode = {'localrun': True, 'usepublic': False, 'vmode': True, 'all_features': False}
# just before submission run
config_presubmit = {'localrun': True, 'usepublic': True, 'vmode': False, 'all_features': False}



config = config_presubmit
xgbcap = 200 # reduce after a t/p run for vmode/LB runs

# I used individual variables earlier, fix later? ;)
localrun = config['localrun']
usepublic = config['usepublic']
vmode = config['vmode']
all_features = config['all_features'] # use w/vmode for feature selection.  run twice, cutting # of rounds to peak round

#function for computing R Score from R_Squared
def r_score(y_true, y_pred, sample_weight=None, multioutput=None):
    r2 = r2_score(y_true, y_pred, sample_weight=sample_weight,
                  multioutput=multioutput)
    r = (np.sign(r2)*np.sqrt(np.abs(r2)))
    if r <= -1:
        return -1
    else:
        return r


# Function XGBOOST ########################################################
def xgb_obj_custom_r(y_pred, dtrain):
    y_true = dtrain.get_label()
    y_mean = np.mean(y_true)
    y_median = np.median(y_true)
    c1 = y_true
    #c1 = y_true - y_mean
    #c1 = y_true - y_median
    grad = 2*(y_pred-y_true)/(c1**2)
    hess = 2/(c1**2)
    return grad, hess

def xgb_eval_custom_r(y_pred, dtrain):
    #y_pred = np.clip(y_pred, -0.075, .075)
#    y_pred[y_pred > .075] = .075
#    y_pred[y_pred < -.075] = -.075
    y_true = dtrain.get_label()
    ybar = np.sum(y_true)/len(y_true)
    ssres = np.sum((y_true - y_pred) ** 2)
    sstot = np.sum((y_true - ybar)**2)
    r2 = 1 - ssres/sstot
    error = np.sign(r2) * np.absolute(r2)**0.5
    return 'error', error

#读取整个dataset, 并按照50%比例区分trainset和testset, env中包含所需要的数据及一些相关信息
env = kagglegym.make()
#observation 中包含 所有的trainset, 和testset中第一天的数据
o = env.reset()

excl = ['id', 'sample', 'y', 'timestamp']
#原dataset中所有的original signals
basecols = [c for c in o.train.columns if c not in excl]

#features I need for building the final model
rcol_orig = ['y_prev_pred', 'y_prev_pred_mstd5', 'Dtechnical_20', 'technical_43_prev', 'technical_7', 'y_prev_pred_avg_diff', 'y_prev_pred_avgT0', 'y_prev_pred_mavg5', 'technical_7_prev', 'technical_20', 'technical_40_prev', 'y_prev_pred_avgT1', 'Dtechnical_40', 'Dtechnical_30', 'technical_40', 'fundamental_36_prev', 'fundamental_5', 'fundamental_8_prev', 'technical_35', 'Dtechnical_21', 'fundamental_36', 'fundamental_43_prev', 'fundamental_46', 'fundamental_18', 'Dtechnical_35', 'Dtechnical_0', 'Dfundamental_45', 'fundamental_48', 'fundamental_1_prev', 'Dtechnical_27', 'Dfundamental_50', 'Dfundamental_18', 'fundamental_16', 'Dfundamental_48', 'Dtechnical_6', 'fundamental_40_prev', 'fundamental_26_prev', 'Dfundamental_8', 'Dtechnical_19', 'fundamental_25', 'fundamental_8', 'fundamental_10_prev', 'technical_35_prev', 'technical_14_prev', 'fundamental_1', 'Dtechnical_37', 'Dfundamental_49', 'Dtechnical_18', 'Dfundamental_42', 'fundamental_41_prev', 'fundamental_62_prev', 'technical_12', 'technical_17_prev', 'technical_27_prev', 'Dtechnical_17', 'derived_0', 'fundamental_33_prev', 'fundamental_32', 'fundamental_17', 'Dtechnical_32', 'technical_32_prev', 'Dfundamental_22', 'fundamental_22', 'fundamental_35', 'Dfundamental_32', 'Dtechnical_7', 'Dfundamental_1', 'technical_28_prev', 'Dtechnical_28', 'Dfundamental_25', 'Dfundamental_63', 'Dtechnical_5', 'technical_44', 'Dfundamental_33', 'derived_2_prev']
rcol = rcol_orig.copy()

#如果all_features = True, 用所有原signal, signal_prev, Dsignal, y_prev_pred_avg_diff, y_prev_pred_avgT0, y_prev_pred_mavg5, y_prev_pred_avgT1, as the rcol to be features of the final input
if all_features:
    rcol = []
    for c in basecols:
        rcol.append(c)
        rcol.append(c + '_prev')
        rcol.append('D' + c)
        
    rcol += ['y_prev_pred_avg_diff', 'y_prev_pred_avgT0', 'y_prev_pred_mavg5', 'y_prev_pred_avgT1']
    rcol_orig = rcol.copy()

#backy_fset 用来train y_pre_model 的features
backy_fset = []
for c in ['technical_13', 'technical_20', 'technical_30']:
    backy_fset.append(c)
    backy_fset.append(c + '_prev')
    backy_fset.append('D' + c)

for f in backy_fset:
    if f not in rcol:
        rcol.append(f)

#从rcol 中 得到original signal
def get_basecols(rcol):
    duse = {}

    for r in rcol:
        if 'y' in r:
            continue

        if 'D' in r:
            duse[r[1:]] = True
        elif '_prev' in r:
            duse[r[:-5]] = True
        elif r in basecols:
            duse[r] = True

    return [k for k in duse.keys()]

#basecols_touse 从rcol中得到original signal
basecols_touse = get_basecols(rcol)

if vmode:
    #全部的数据集作为trainset
    train = pd.read_hdf('C:\\Users\\yuan\\Desktop\\financial_modeling_2sigma\\train.h5')
else:
    #50%的数据集作为trainset
    train = o.train.copy()


#preparing the median value for the using columns in the original data, would be used later for dealing with missing value
d_mean = o.train[basecols_touse].median(axis=0)
for c in basecols_touse:
    d_mean[c + '_prev'] = d_mean[c]
    d_mean['D' + c] = 0

median = {t[0]:t[1] for t in zip(d_mean.index, d_mean.values)}
median['y'] = 0


class DataPrep:
    
    def __init__(self, yprev_model = None, keepinput = True, cols = rcol, seed = 2017):
        # yprev_model是否需要对y本身的feature_engineer, keepinput是否储存处理好的input, cols需要features的名字
        #上一个timestamp的数据        
        self.previnput = None
        
        self.prevavg = 0
        
        #所需要计算的features
        self.cols = cols.copy()
        
        #各种transformed features 所对应原dataset的features
        self.basecols = get_basecols(self.cols)
        
        #在原dateset中需要保留的columns
        self.keepcols = ['y', 'id', 'timestamp'] + self.basecols
        
        #是否保存处理好的输入
        self.allinput = [] if keepinput else None
        
        #对前一日所有id对应y的预测的均值
        self.dayavg = []
        
        #对前一日y的预测的模型
        self.yprev_model = yprev_model
        
        np.random.seed(seed)
        self.random_state = np.random.get_state()
        
    def procday(self, day_in):
        #取 train中的每一天的数据生成features
        if 'y' not in day_in and 'y' in self.keepcols:
            self.keepcols.remove('y')
        
        day = day_in[self.keepcols].copy()
        
        #前一天里有该id然而在下一天中没有
        notinnew = []
        
        if self.previnput is not None:
            olen = len(day)
            day = pd.merge(day, self.previnput, on='id', how = 'left', suffixes=['', '_prev'])
            notinnew = self.previnput[~self.previnput.id.isin(day_in.id)].copy()
            #print(day.iloc[0].timestamp, len(notinnew))
        else:
            for c in self.basecols:
                #full-like = copy shape and dtype, but initialize with given value  
                #只初始化 basecol 为 0
                day[c + '_prev'] = np.full_like(day[c], 0, dtype=np.float32)
                #day[c + '_prev'] = np.zeros_like(day[c], dtype=np.float32)
        
        #Computing D_signal which is the difference between today and previous day
        for c in self.cols:
            if c == 'y_prev_pred':
                continue

            if c[0] == 'D':
                day[c] = day[c[1:]] - day[c[1:] + '_prev']
        
        #store today's day to be used in the next iteration
        self.previnput = day_in[self.keepcols].copy()
        
        #如果 id(t-1) 没出现在 id(t)中, 那么id(t-1) 仍算 id(t+1。。)的previous input
        if len(notinnew) > 0:
            self.previnput = self.previnput.append(notinnew[self.keepcols])
        
        #采用对Y本身的feature engine
        if self.yprev_model:
            #对前一日y的预测值            
            day['y_prev_pred'] = self.yprev_model.predict(day[backy_fset].fillna(d_mean).values.reshape(-1,len(backy_fset)))
            
            avg = day.y_prev_pred.mean()
                        
            self.dayavg.append(avg)
            #computing MA(5) for the avg of y_pre_pred
            day['y_prev_pred_mavg5'] = np.ma.average(np.array(self.dayavg[-5:]))#, weights=range(1, len(self.dayavg[-10:]) + 1))
            
            # (y_pre_pred - mavg5)
            day['y_prev_pred_min5'] = day.y_prev_pred - day.y_prev_pred_mavg5
            
            # (avg - mavg5)
            day['y_prev_pred_mavg5d'] = avg - np.ma.average(np.array(self.dayavg[-5:]))
            
            #std(last 5 day avg)
            day['y_prev_pred_mstd5'] = np.std(np.array(self.dayavg[-5:]))
            
            #MA(9) for the avg of y_pre_pred
            day['y_prev_pred_mavg9'] = np.ma.average(np.array(self.dayavg[-9:]))#, weights=range(1, len(self.dayavg[-10:]) + 1))
            #MA(20) for the avg of y_pre_pred            
            day['y_prev_pred_mavg20'] = np.ma.average(np.array(self.dayavg[-20:]))
            #MA(40) for the avg of y_pre_pred                        
            day['y_prev_pred_mavg40'] = np.ma.average(np.array(self.dayavg[-40:]))
            #avg_t-1
            day['y_prev_pred_avgT1'] = self.prevavg
            #avg_t            
            day['y_prev_pred_avgT0'] = avg
            #avg_t - avg_t-1            
            day['y_prev_pred_avg_diff'] = avg - self.prevavg

            self.prevavg = avg
        
        
        np.random.set_state(self.random_state)
        day['random'] = np.random.random(len(day))
        self.random_state = np.random.get_state()
        
        #保存每一天所有的股票计算处理好的输入features
        if self.allinput is not None:
            self.allinput.append(day.copy())

        return day
    
    def run(self, df):
        assert self.allinput is not None
        
        for g in df.groupby('timestamp'):
            #using everyday data
            self.procday(g[1])
            
        #all_input 就是我们处理好的输入
        return pd.concat(self.allinput)


#only use backy_fset to train ypmodel.
yptrain = DataPrep(keepinput=True, cols=backy_fset).run(train)

#inplace means the change do make change on the underlying data
#sort first by each id, then by the timestamp
yptrain.sort_values(['id', 'timestamp'], inplace=True)

#n_jobs means multithreading n_jobs = -1 means all cpu
ypmodel = LinearRegression(n_jobs=-1)

#cut the extreme y, why? train the model without the extreme y like discard the outlier? 
low_y_cut = -0.0725
high_y_cut = 0.075
mask = np.logical_and(yptrain.y > low_y_cut, yptrain.y < high_y_cut)
yptraina = yptrain[mask]

#dealing with missing value, replacing the missing value with the mean value of the whole column's mean
#remember both none and nan would be detect by isnull
#use today's signal to predict yesterday's return, kind of like back-lock prediction
X=yptraina[backy_fset].fillna(d_mean).values.reshape(-1,len(backy_fset))
Y=yptraina.y_prev.fillna(0)
ypmodel.fit(X,Y)
print(len(yptraina), ypmodel.coef_, ypmodel.intercept_)

preds = ypmodel.predict(yptrain[backy_fset].fillna(d_mean).values.reshape(-1, len(backy_fset)))
print(r_score(yptrain.y_prev.fillna(0), preds))

d_mean['y'] = 0

start = time.time()

print('beginning train processing')
#here we produce the input feature set for the final model, we use all rcols as the bas
train = DataPrep(keepinput = True, yprev_model = ypmodel).run(train)
endt = time.time()
print(endt - start)


#all the columns except y,id,timestamp, however y_prev, timestamp_prev 在里面
dcol = [c for c in train.columns if c not in excl]

if usepublic:
    data_all = pd.read_hdf('C:\\Users\\yuan\\Desktop\\financial_modeling_2sigma\\train.h5')

    #public = data_all[data_all.timestamp > 905]
    #准备两份input feature, 一份用全部的data, 一份用906开始之后的data
    allpublic = DataPrep(yprev_model = ypmodel, keepinput=True).run(data_all)
    
    public = DataPrep(yprev_model = ypmodel, keepinput=True).run(data_all[data_all.timestamp > 905])

#看一下y_pre_pred预测的效果
print(r_score(train.y_prev.fillna(0), train.y_prev_pred))
print(r_score(public.y_prev.fillna(0), public.y_prev_pred))

train.sort_values(['id', 'timestamp'], inplace=True)


print('preparing xgb now')
#xtrain, xvalid = train_test_split(train, test_size = 0.2, random_state = 2017)

if not vmode: # submission-style, use all data
    #rthreshold 用来控制随机取dataset的数量
    rthreshold = 1.0
    xtmask = train.random <= rthreshold
    #去掉一些extreme value, outlier?
    xtmask = np.logical_and(xtmask, train['y'] > -.0189765)
    xtmask = np.logical_and(xtmask, train['y'] < .0189765)
    
    #在这里用extremevalue 做validset 算下来相当于 68% / 32%
    xtrain = train[xtmask]
    xvalid = train[~xtmask]
else: # use older split to compare with say the .0202/.0203 subs
    #按照80%/20%比例随机来分 trainset, testset
    xtrain, xvalid = train_test_split(train, test_size = 0.2, random_state = 2017)
    
    #去掉train中的outlier?
    xtrain = xtrain[np.abs(xtrain['y']) < 0.018976588919758796]
    
    
#rcol_orig 原始设定要用的features
#rcol rcol_orig加上了backy_fset     
#train.columns rcol加上了所有的rcol中的signal的pre和D, 以及y_prev_pred类signals和y_pre, id_pre

#取原rcol_orig 中的features_cols
cols_to_use = [c for c in rcol if c in xtrain.columns and c in rcol_orig] 

                                                     
# Convert to XGB format
to_drop = ['timestamp', 'y']

train_xgb = xgb.DMatrix(data=xtrain[cols_to_use], label=xtrain['y'])
valid_xgb = xgb.DMatrix(data=xvalid[cols_to_use], label=xvalid['y'])

evallist = [(valid_xgb, 'valid'), (train_xgb, 'train')]

if usepublic:
    public_xgb = xgb.DMatrix(data=public[cols_to_use], label=public['y'])
    evallist = [(train_xgb, 'train'), (valid_xgb, 'xvalid'), (public_xgb, 'public')]

print('xtrain+valid')


#XGB model training

#set basic parameter
params = {
    'objective': 'reg:linear'
    ,'eta': 0.04
    ,'max_depth': 3
    ,'subsample': 0.9
    #, 'colsample_bytree': 1
    ,'min_child_weight': 3072 # 2 **11
    #,'gamma': 100
    ,'seed': 10
    #, 'base_score': xtrain.y.mean()
}

model = []
#根据不同seed,train不同的models
for seed in [10000]:
    params['seed'] = seed
    model.append(xgb.train(
                    params.items()
                  , dtrain=train_xgb
                  , num_boost_round=300 # 240 was best_ntree_limit from a train/public split run
                  , evals=evallist
                  , early_stopping_rounds=50
                  , maximize=True #since we set the feval, it means the larger the score the better
                  , verbose_eval=10
                  , feval=xgb_eval_custom_r #feval means customized evaluation metric
                  ))



print('xgb done, linear now')

lin_features = ['Dtechnical_20', 'technical_20', 'Dtechnical_21']

#fill missing value to the trainset
def prep_linear(df, c = lin_features):
    df_tmp = df.fillna(d_mean)
    m2mat = np.zeros((len(df), len(c)))
    for i in range(len(c)):
        m2mat[:,i] = df_tmp[c[i]].values
    
    return m2mat

# Observed with histograns:
#https://www.kaggle.com/bguberfain/two-sigma-financial-modeling/univariate-model-with-clip/run/482189
low_y_cut = -0.075
high_y_cut = 0.075
# cut y in trainset, since there are two edges in the end of y. 
traincut = train[np.logical_and(train.y > low_y_cut, train.y < high_y_cut)][['y'] + lin_features].copy().fillna(d_mean)

model2 = LinearRegression(n_jobs=-1)
model2.fit(prep_linear(traincut), traincut.y)

print('linear done')



if vmode:
    
    #在valid set上预测
    preds_xgb = model[0].predict(valid_xgb, ntree_limit=model[0].best_ntree_limit)
    preds_linear = model2.predict(prep_linear(xvalid))
    
    #ensemble??    
    preds = (preds_xgb * 0.7) + (preds_linear * 0.3)
    #preds = preds_xgb
    
    rs_xgb = kagglegym.r_score(xvalid.y, preds_xgb)
    rs_lr = kagglegym.r_score(xvalid.y, preds_linear)
    rs = kagglegym.r_score(xvalid.y, preds)
    
    print("validation score", rs)
    
    
    output = xvalid[['id', 'timestamp', 'y']].copy()
    output['y_hat'] = preds
    output['y_hat_xgb'] = preds_xgb
    output['y_hat_linear'] = preds_linear
    
    
start = time.time()   
#准备 input data，创建环境，先不生成
dprep = DataPrep(yprev_model = ypmodel, keepinput=localrun)
        
if localrun:
    env = kagglegym.make()
    o = env.reset()
    
while True:
    #处理test集
    test_preproc = o.features.copy()
    
    #if c in basecols:
        #test_preproc.fillna(d_mean, inplace=True)
    
    #生成第一天的testdata, testdata没有y
    #注意第一没有timestamp_prev
    test = dprep.procday(test_preproc)
    
    #test.fillna(0, inplace=True)
    
    test_xgb = xgb.DMatrix(data=test.drop(['id', 'timestamp'], axis=1)[cols_to_use])

    #对该日的testdata的预测值
    xgbpreds = np.zeros(len(test), dtype=np.float64)
    
    #采用不同 seed值生成不同的model, 然后对结果取平均
    for m in model:
        xgbpreds += m.predict(test_xgb, ntree_limit=m.best_ntree_limit)
    xgbpreds /= len(model)
    
    #ensemble xgbmodel and linear_model
    #clip means clip the outside value to the edge    
    test_y = (xgbpreds * .7) + (model2.predict(prep_linear(test)).clip(low_y_cut, high_y_cut) * 0.3)
    
    o.target['y'] = test_y
    target = o.target
    


    # We perform a "step" by making our prediction and getting back an updated "observation":
    o, reward, done, info = env.step(target)
    if done:
        print("Public score: {}".format(info["public_score"]))
        break