# YBIGTA Project: DACON 천체유형 분류
<br>
ML 2팀: 노시영, 양정열, 안세현, 최정윤, 박솔희

![](https://github.com/sehyeona/ybigta-project/blob/master/Title.jpg)

<br>

## 프로젝트의 목적 및 의의
> 프로젝트의 목적

DACON에서 제공하는 천체 트레이닝 데이터를 활용하여 테스트 데이터의 **천체 type 예측 확률을 높이는 모델을 구축**하는 것이 프로젝트의 목표이다. 채점기준은 log loss로, 모델에서 얻은 예측 데이터를 .csv형태로 저장한 후 DACON에 제출하면 자동으로 log loss값이 계산되어 출력된다. 이 수치를 최소화하는 것을 통해 모델 예측 성과를 측정한다. 

> 프로젝트의 의의

본 프로젝트는 천문학 데이터를 기반으로 하고 있다. 천문학에 익숙하지 않아도 관련 데이터를 시각화하고 중요 변수를 파악하며, 적절한 모델 선택 및 학습시키는 과정을 통해 머신러닝에 대한 이해를 높일 수 있다.

<br>
<br>

## 목차
>
1. 변수의 의미 파악

2. Training Data 시각화

3. Training Data 전처리

4. Training Data 샘플링

5. 적합한 모델 찾기(XGBoost, CatBoost, RandomForest, LightGBM)

6. 그리드 서치

<br>
<br>
<br>

## 1. 변수의 의미 파악
> 1-1. column 살펴보기
```
df.info()
   
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 199991 entries, 0 to 199990
Data columns (total 23 columns):
id            199991 non-null int64
type          199991 non-null object
fiberID       199991 non-null int64
psfMag_u      199991 non-null float64
psfMag_g      199991 non-null float64
psfMag_r      199991 non-null float64
psfMag_i      199991 non-null float64
psfMag_z      199991 non-null float64
fiberMag_u    199991 non-null float64
fiberMag_g    199991 non-null float64
fiberMag_r    199991 non-null float64
fiberMag_i    199991 non-null float64
fiberMag_z    199991 non-null float64
petroMag_u    199991 non-null float64
petroMag_g    199991 non-null float64
petroMag_r    199991 non-null float64
petroMag_i    199991 non-null float64
petroMag_z    199991 non-null float64
modelMag_u    199991 non-null float64
modelMag_g    199991 non-null float64
modelMag_r    199991 non-null float64
modelMag_i    199991 non-null float64
modelMag_z    199991 non-null float64
dtypes: float64(20), int64(2), object(1)
memory usage: 35.1+ MB   

```

- ID : 천체의 unique ID

- type : 천체의 분류

- psfMag(Point spread function magnitudes) : 먼 천체를 한 점으로 가정하여 측정한 빛의 밝기

- fiberMag(Fiber magnitudes) : 3인치 지름의 광섬유를 사용하여 광스펙트럼을 측정한 광섬유를 통과하는 빛의 밝기

- petroMag(Petrosian Magnitudes) : 천체의 위치와 거리에 상관없이 빛의 밝기를 비교하기 위한 수치

- modelMag(Model magnitudes) : 천체 중심으로부터 특정 거리의 밝기

- FiberID:관측에 사용된 광섬유의 구분자

   *참고: u(ultraviolet), g(green), r(red), i(near-infrared),z(very-near-infrared)*

<br>

> 1-2. 종속변수 분석 

type: 천체 유형으로 예측해야 하는 변수로, 모델에서 종속변수가 된다.

```
  train=pd.read_csv('/content/gdrive/My Drive/train.csv')
  test=pd.read_csv('/content/gdrive/My Drive/test.csv')
  train['type'].unique()

array(['QSO', 'STAR_RED_DWARF', 'SERENDIPITY_BLUE', 'STAR_BHB',
       'STAR_CATY_VAR', 'SERENDIPITY_DISTANT', 'GALAXY',
       'SPECTROPHOTO_STD', 'REDDEN_STD', 'ROSAT_D', 'STAR_WHITE_DWARF',
       'SERENDIPITY_RED', 'STAR_CARBON', 'SERENDIPITY_FIRST',
       'STAR_BROWN_DWARF', 'STAR_SUB_DWARF', 'SKY', 'SERENDIPITY_MANUAL',
       'STAR_PN'], dtype=object
```
총 19종류의 천체 유형으로 분류된다

<br>
<br>
<br>

## 2. Training Data 시각화

> 2-1. 천체 type class별 분포 

```
  plt.figure(figsize=(12,8))
  ax = sns.countplot(y="type", data=df)
  plt.title('Distribution of orb types\n')
  plt.ylabel('Number of type\n')

  # Make twin axis
  ax2=ax.twiny()

  # Switch so count axis is on right, frequency on left
  ax2.xaxis.tick_top()
  ax.xaxis.tick_bottom()

  # Also switch the labels over
  ax.xaxis.set_label_position('bottom')
  ax2.xaxis.set_label_position('top')

  ax2.set_xlabel('Frequency [%]')
```

![](https://github.com/sehyeona/ybigta-project/blob/master/visualization1.png)

<br>

> 2-2. feature의 분포 

```

print(len(features))
for col in features :
    plt.figure(figsize=(12,4))
    sns.distplot(df[col])
    plt.title('Distribution of %s\n'%col)
    
```

![](https://github.com/sehyeona/ybigta-project/blob/master/%EB%B3%80%EC%88%98%EB%B6%84%ED%8F%AC1.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EB%B3%80%EC%88%98%EB%B6%84%ED%8F%AC2.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EB%B3%80%EC%88%98%EB%B6%84%ED%8F%AC3.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EB%B3%80%EC%88%98%EB%B6%84%ED%8F%AC4.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EB%B3%80%EC%88%98%EB%B6%84%ED%8F%AC5.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EB%B3%80%EC%88%98%EB%B6%84%ED%8F%AC6.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EB%B3%80%EC%88%98%EB%B6%84%ED%8F%AC7.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EB%B3%80%EC%88%98%EB%B6%84%ED%8F%AC8.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EB%B3%80%EC%88%98%EB%B6%84%ED%8F%AC9.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EB%B3%80%EC%88%98%EB%B6%84%ED%8F%AC10.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EB%B3%80%EC%88%98%EB%B6%84%ED%8F%AC11.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EB%B3%80%EC%88%98%EB%B6%84%ED%8F%AC21.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EB%B3%80%EC%88%98%EB%B6%84%ED%8F%AC12.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EB%B3%80%EC%88%98%EB%B6%84%ED%8F%AC13.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EB%B3%80%EC%88%98%EB%B6%84%ED%8F%AC14.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EB%B3%80%EC%88%98%EB%B6%84%ED%8F%AC15.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EB%B3%80%EC%88%98%EB%B6%84%ED%8F%AC16.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EB%B3%80%EC%88%98%EB%B6%84%ED%8F%AC17.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EB%B3%80%EC%88%98%EB%B6%84%ED%8F%AC18.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EB%B3%80%EC%88%98%EB%B6%84%ED%8F%AC19.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EB%B3%80%EC%88%98%EB%B6%84%ED%8F%AC20.png)

분포 확인 결과

 - fiberMag_u 와 psfMag_u 가 두 변수의 분포 형태가 거의 동일한 양상을 가진다.

 - 대부분의 분포들이 평균에서 매우 떨어져있는 아웃라이어를 가지고 있다.

 - 아웃라이어들은 주로 양방향으로 분포해있기 보다 한 방향으로 치우쳐서 분포하는 경향을 보인다.
 
 - 변수들의 분포 범위가 차이를 보인다.
 
<br>

> 2-3. 천체 type에 따른 변수 간 상관관계

```
for x in types:    
    plt.figure(figsize=(12,8))
    ax =  sns.heatmap(df[df['type'] == x].corr(method='pearson'), annot = True,   
                fmt = '.2f',linewidths = 1, cmap="summer")
    buttom, top = ax.get_ylim()
    ax.set_ylim(buttom + 0.5, top - 0.5)
    plt.title("Correlations when type is %s"%x)
```

![](https://github.com/sehyeona/ybigta-project/blob/master/%EC%83%81%EA%B4%80%EA%B4%80%EA%B3%841.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EC%83%81%EA%B4%80%EA%B4%80%EA%B3%842.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EC%83%81%EA%B4%80%EA%B4%80%EA%B3%843.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EC%83%81%EA%B4%80%EA%B4%80%EA%B3%844.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EC%83%81%EA%B4%80%EA%B4%80%EA%B3%845.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EC%83%81%EA%B4%80%EA%B4%80%EA%B3%846.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EC%83%81%EA%B4%80%EA%B4%80%EA%B3%847.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EC%83%81%EA%B4%80%EA%B4%80%EA%B3%848.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EC%83%81%EA%B4%80%EA%B4%80%EA%B3%849.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EC%83%81%EA%B4%80%EA%B4%80%EA%B3%8410.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EC%83%81%EA%B4%80%EA%B4%80%EA%B3%8411.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EC%83%81%EA%B4%80%EA%B4%80%EA%B3%8412.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EC%83%81%EA%B4%80%EA%B4%80%EA%B3%8413.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EC%83%81%EA%B4%80%EA%B4%80%EA%B3%8414.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EC%83%81%EA%B4%80%EA%B4%80%EA%B3%8415.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EC%83%81%EA%B4%80%EA%B4%80%EA%B3%8416.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EC%83%81%EA%B4%80%EA%B4%80%EA%B3%8417.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EC%83%81%EA%B4%80%EA%B4%80%EA%B3%8418.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EC%83%81%EA%B4%80%EA%B4%80%EA%B3%8419.png)

위 그래프에서 천체 타입별 feature 간 상관 관계를 분석 해보면 가로 세로 직선으로 진한 색상을 가진 부분들을 관찰할 수 있다. 이는 하나의 특징이 다른 모든 특징들과 매우 강한 상관 관계를 가지고 있다는 것을 의미한다. 어떠한 특징이 일대일로 강한 상관 관계를 가지고 있는것은 가능한 일이나, 하나의 특징이 다른 모든 특징들과 매우 강한 상관 관계를 가지는 것은 극히 드물다. 이에 대한 원인으로 크게 두 가지가 가능하다.

   1\)  특징이 매우 큰 outliers 값을 가지기 때문에 다른 특징들과의 수리적 계산에서 강한 상관관계를 초래한다.

   2\)  타입 샘플의 개수가 적기 때문이다.

<br>

> 2-4. 모든 변수 간 상관 관계

![](https://github.com/sehyeona/ybigta-project/blob/master/heatmap.png)

psfMag_u, fiberMag_u, petroMag_u 세 변수 간 상관 관계가 매우 높다.

<br>
<br>
<br>

## 3. Training Data 전처리

> 3-1. scaling

feature 변수 값들의 범위가 차이를 보이고 있기 때문에 머신러닝 알고리즘 모델링에 적합한 형태로 변화시키기 위해서 아래의 4가지 방법을 이용하여 스케일링을 진행하였다. 

- standardscaler : 정규분포를 이용한다.

- minmaxscaler : 최대/최소값이 각각 0, 1이다.

- maxabsscaler : 최대절대값과 0이 각각 1, 0 이 되도록하는 scaling이다.

- robustscaler : median과 IQR 사용 outlier의 영향을 최소화 한다.

  *모든 스케일러는 sklearn.preprocessing 패키지 안에 각자 이름으로 존재*

```
#training data를 각자 스케일링 

def scaling_func(df, scaler) :
    '''
    param : dataframe / scaler object
    return : scaled dataframe / fitting scaler
    '''
    scaler = scaler()
    # type과 id를 제외하고 학습
    data_for_scaling = df.drop(['id', 'type', 'fiberID'], axis = 1)
    scaler.fit(data_for_scaling)
    # 학습후 변환
    train_scaled = scaler.transform(data_for_scaling)
    # 학습후 변환한 데이터를 다시 원래 데이터로 만들기
    result = pd.DataFrame(train_scaled, columns = data_for_scaling.columns)
    result = pd.concat([df[['id','type', "fiberID"]], result], axis=1)
    return result, scaler
    
```

```
#test data 스케일링
def test_scaling_func(df, scaler) :
    # type과 id를 제외하고 학습
    data_for_scaling = df.drop(['id', 'fiberID'], axis = 1)
    scaler.fit(data_for_scaling)
    # 학습후 변환
    train_scaled = scaler.transform(data_for_scaling)
    # 학습후 변환한 데이터를 다시 원래 데이터로 만들기
    result = pd.DataFrame(train_scaled, columns = data_for_scaling.columns)
    result = pd.concat([df[['id', "fiberID"]], result], axis=1)
    return result
```

```
#4가지방법으로 스케일링
from sklearn.preprocessing import StandardScaler
std_data, std_scaler = scaling_func(data, StandardScaler)

from sklearn.preprocessing import MinMaxScaler
mm_data, mm_scaler = scaling_func(data, MinMaxScaler)

from sklearn.preprocessing import MaxAbsScaler
ma_data, ma_scaler = scaling_func(data, MaxAbsScaler)

from sklearn.preprocessing import RobustScaler
rb_data, rb_scaler= scaling_func(data, RobustScaler)
```

가장 적합한 스케일링 방법 선정을 위해 스케일링을 거친 데이터들을 catboost 알고리즘을 통해 학습 시킨 후 log loss를 통해 예측 성과를 평가하였다.

```
#모델을 설정
import catboost as cb
cb_model_std = cb.CatBoostClassifier()
cb_model_mm = cb.CatBoostClassifier()
cb_model_ma = cb.CatBoostClassifier()
cb_model_rb = cb.CatBoostClassifier()

#독립변수,종속변수 설정
X_std = std_data[data.columns[2:]]
X_mm = mm_data[data.columns[2:]]
X_ma = ma_data[data.columns[2:]]
X_rb = rb_data[data.columns[2:]]

y = data['type']

#모델에 적용시키기
cb_model_std.fit(X_std, y)
cb_model_mm.fit(X_mm, y)
cb_model_ma.fit(X_ma, y)
cb_model_rb.fit(X_rb, y)

```

```
#df형식으로 만들기(submission file과 같은 column순서로)
id_for_index = df['id']
test = df[test_data.columns[1:]]
predictions = model.predict_proba(test)
result = pd.DataFrame(data=predictions, index=id_for_index, columns=model.classes_)
sample = pd.read_csv("./data/ybigta_sdss_sample_submission.csv")
col_order = sample.columns[1:]
result = result[col_order]
return result

test_std = test_scaling_func(test_data, std_scaler)
test_mm = test_scaling_func(test_data, mm_scaler)
test_ma = test_scaling_func(test_data, ma_scaler)
test_rb = test_scaling_func(test_data, rb_scaler)
    
```

```
#csv형식으로 변환해서 저장

make_submission(test_std, cb_model_std).to_csv('./result/std_cb.csv', sep=',')
make_submission(test_mm, cb_model_mm).to_csv('./result/mm_cb.csv', sep=',')
make_submission(test_ma, cb_model_ma).to_csv('./result/ma_cb.csv', sep=',')
make_submission(test_rb, cb_model_rb).to_csv('./result/rb_cb.csv', sep=',')

```

결과:

 1\) standardscaler : 0.4778

 2\) minmaxscaler : 1.7

 3\) maxabsscaler : 1.6

 4\) robustscaler : 0.4778

산출된 결과를 통해 미세하지만 학습 결과값이 더 우수하고, 아웃라이어에 영향을 덜 받는 robustscaler를 최종 스케일링 방법을 선택하였다.

<br>

> 3-2. 상관관계가 높은 변수 처리

상관 관계 heatmap 분석을 통해 psfMag_u, fiberMag_u, petroMag_u 세 변수의 연관성이 높음을 알 수 있었다. 높은 상관계수를 갖는 변수들의 존재는 모델의 예측 성능을 하락시킬 수 있으므로 이 변수들을 합쳐서 전처리를 진행한 후, 모델 성능이 향상되었는지 측정하였다.

```
#train,test data 쪼개기(전처리 없이)

from sklearn.model_selection import train_test_split
X = train.drop('type', axis = 1)
y = train['type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=42)

#lightgbm으로 전처리 없이 돌려보기

from lightgbm import LGBMClassifier
lgbm_model = LGBMClassifier(boosting_type='gbdt', objective='binary', num_leaves=10,
                                learning_rate=0.1, n_estimators=2000, max_depth=15,
                                bagging_fraction=0.9, feature_fraction=0.9, reg_lambda=0.2)
lgbm_model.fit(X_train,y_train)

#전처리 없이 나오는 logloss
from sklearn.metrics import log_loss
y_pred_lgbm_prob = lgbm_model.predict_proba(X_test)
log_loss(y_true=y_test, y_pred=y_pred_lgbm_prob)

```

결과:0.3874938355499263

```
#psfMag_u와 fiberMag_u 그리고 petroMag_u를 평균을 내서 하나의 변수로 만들기
ultra = train.loc[:,['psfMag_u','fiberMag_u','petroMag_u']]
train['average_ultra'] = ultra.apply(np.mean,axis=1)
train['average_ultra'

#평균 변수를 추가하고 기존 psfMag_u와 fiberMag_u 그리고 petroMag_u 변수 삭제하기
X = pd.concat([X, train[['average_ultra']]], axis=1) 
X = X.drop(['psfMag_u','fiberMag_u','petroMag_u'],axis=1)

#LGBM 모델 구축하기

from lightgbm import LGBMClassifier
lgbm_model = LGBMClassifier(boosting_type='gbdt', objective='binary', num_leaves=10,
                                learning_rate=0.1, n_estimators=2000, max_depth=15,
                                bagging_fraction=0.9, feature_fraction=0.9, reg_lambda=0.2)
lgbm_model.fit(X_train,y_train)

#logloss을 측정하기
from sklearn.metrics import log_loss
y_pred_lgbm_prob = lgbm_model.predict_proba(X_test)
log_loss(y_true=y_test, y_pred=y_pred_lgbm_prob)

```

결과: log_loss = 0.38644079445636986

높은 상관 관계를 보이는 변수들을 처리한 모델이 그렇지 않은 모델보다 더 높은 예측 성과를 보인다. 

<br>

> 3-3 outlier 조작

3-3-1. type 별 아웃라이어 제거
QSO를 기준으로 feature 분포를 살펴봤을 때 fiberMag_g 변수의 최소값과 최대값이 다른 변수들에 비해 이상치를 보이는 것을 확인할 수 있다. 앞서 시각화에서 QSO와 fiberMag_g의 상관 관계가 유의미하지 않은 것으로 나왔기 때문에 이 이상치를 제거하였다.


```
qso = df[df['type']=='QSO'] #가장 갯수가 많은 QSO부터 시작
qso.describe() #fiberMag만 이상한 것을 확인할 수 있다. 이친구의 outlier 먼저 제거해준다.

q1,q3 = np.percentile(qso['fiberMag_g'],[25,75])
iqr = q3-q1
lower_bound = q1 - (iqr*1.5)
upper_bound = q1 + (iqr*1.5)
qso = qso[(qso['fiberMag_g']<upper_bound)&(qso['fiberMag_g']>lower_bound)]

temp_no_qso = df[df['type']!='QSO']
temp_yes_qso = pd.concat([temp_no_qso, qso])
temp_yes_qso
```

마찬가지로 SPECTROPHOTO_STD-fiberMag_i의 아웃라이어도 조작하였다.

```
spec_std = temp_yes_qso[temp_yes_qso['type']=='SPECTROPHOTO_STD'] #그 다음은 SPECTROPHOTO_STD
spec_std.describe() #fiberMag_i가 이상하다.
len(spec_std)

q1,q3 = np.percentile(spec_std['fiberMag_i'],[25,75])
iqr = q3-q1
lower_bound = q1 - (iqr*1.5)
upper_bound = q1 + (iqr*1.5)
spec_std = spec_std[(spec_std['fiberMag_i']<upper_bound)&(spec_std['fiberMag_i']>lower_bound)]
spec_std.describe()
len(spec_std)

temp_no_spec = temp_yes_qso[temp_yes_qso['type']!='SPECTROPHOTO_STD']
temp_yes_spec = pd.concat([temp_no_spec, spec_std])
len(temp_yes_spec)
```

<br>

> 3-3-2. isolation forest을 통한 아웃라이어 제거
각각의 type 별 아웃라이어를 제거한 후 전체 데이터 측면에서 제거하지 못한 아웃라이어 제거를 위해 isolatio forest를 사용하였다. isolation forest는  regression tree 기반으로 모든 데이터 관측치를 고립시켜 아웃라이어를 정의하는 방법이다.

```
from sklearn.ensemble import IsolationForest
clf = IsolationForest(max_samples=7000, random_state=1)
clf.fit(X)
y_pred_outliers = clf.predict(X)
out = pd.DataFrame(y_pred_outliers)
out = out.rename(columns={0: "out"})
df1 = pd.concat([X, out], 1)

df2=train[["type","fiberID","type_num"]]
df=pd.concat([df2,df1],axis=1)

df=df[df.out !=-1]
train=df.drop(columns=['out'])
```



<br>
<br>
<br>

# 4. training data 샘플링

> 4-1. sampling

시각화 단계에서 타입의 type 클래스 개수에 큰 차이가 있음을 확인하였다.  클래스 개수가 너무 작은 type은 제대로 학습 되지 않아 예측 확률이 낮아지는 문제가 발생할 수 있다. 또한 fiberID 는 사실상 카테고리 변수이지만, 시각화 단계에서 다른 변수들과 낮은 상관관계를 보였다. 즉, 범주형 변수임에도 불구하고 자유로운 sampling이 가능하다.

oversampling을 통해 개수가 너무 적은 타입들의 개수를 늘려, 타입의 개수가 작은 변수에 대한 예측 정확도를 높일 수 있다고 판단하였다.

```
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from lightgbm import LGBMClassifier
import pickle

# 데이터 타입과 개수를 입력받아 샘플링 해주는 함수 (15기 백진우 작성)

orb_type = [
    "GALAXY",
    "QSO",
    "REDDEN_STD",
    "ROSAT_D",
    "SERENDIPITY_BLUE",
    "SERENDIPITY_DISTANT",
    "SERENDIPITY_FIRST",
    "SERENDIPITY_MANUAL",
    "SERENDIPITY_RED",
    "SKY",
    "SPECTROPHOTO_STD",
    "STAR_BHB",
    "STAR_BROWN_DWARF",
    "STAR_CARBON",
    "STAR_CATY_VAR",
    "STAR_PN",
    "STAR_RED_DWARF",
    "STAR_SUB_DWARF",
    "STAR_WHITE_DWARF"
]
def sampling(df, orb_type): #df : 데이터프레임, orb_type : 천체타입 리스트
    copy_train = df.copy()
    for t in orb_type:
        num = len(df[df["type"]==t])
        print("현재 type : ", t)
        insert_row = int(input("랜덤샘플링으로 추가할 개수를 입력하세요 : "))
        print("\n")
        temp = df[df["type"]==t].sample(n = (insert_row - num), replace=True)  #t는 tpye의 종류, replace = True 옵션을 줘야 개수가 더 많아도 샘플링 가능
        copy_train = pd.concat([copy_train, temp], axis = 0) #row로 붙여넣기
    return copy_train
    
#각 항목의 갯수 세기    
count = data.groupby("type").size().reset_index(name = "count")
count.sort_values(by="count",ascending=False)


1	QSO	42500
0	GALAXY	34000
4	SERENDIPITY_BLUE	19439
10	SPECTROPHOTO_STD	13073
2	REDDEN_STD	13052
11	STAR_BHB	12069
16	STAR_RED_DWARF	9040
14	STAR_CATY_VAR	5808
6	SERENDIPITY_FIRST	5718
3	ROSAT_D	5559
5	SERENDIPITY_DISTANT	4163
13	STAR_CARBON	2875
18	STAR_WHITE_DWARF	1932
8	SERENDIPITY_RED	1312
17	STAR_SUB_DWARF	977
12	STAR_BROWN_DWARF	220
9	SKY	170
7	SERENDIPITY_MANUAL	46
15	STAR_PN	12

```

```
sampling_data = sampling(data, ["STAR_WHITE_DWARF", "SERENDIPITY_RED", "STAR_SUB_DWARF", "STAR_BROWN_DWARF", "SKY", "SERENDIPITY_MANUAL", "STAR_PN"]

현재 type :  STAR_WHITE_DWARF
랜덤샘플링으로 추가할 개수를 입력하세요 : 2800


현재 type :  SERENDIPITY_RED
랜덤샘플링으로 추가할 개수를 입력하세요 : 2500


현재 type :  STAR_SUB_DWARF
랜덤샘플링으로 추가할 개수를 입력하세요 : 2100


현재 type :  STAR_BROWN_DWARF
랜덤샘플링으로 추가할 개수를 입력하세요 : 1400


현재 type :  SKY
랜덤샘플링으로 추가할 개수를 입력하세요 : 1200


현재 type :  SERENDIPITY_MANUAL
랜덤샘플링으로 추가할 개수를 입력하세요 : 900


현재 type :  STAR_PN
랜덤샘플링으로 추가할 개수를 입력하세요 : 500

```

- no sampling -> 0.374
- 2900 2500 1800 870 300 142 (작은타입 6개) -> 0.3738814802859438

- 2100 1600 1300 880 680 230 84 (작은타입 7개) -> 0.3760891561709987

- 2100 1600 1250 850 700 300 150 (작은타입 7개) -> 0.37593517430997203(제일 좋은 결과값)

- 2800 2500 2100 1400 1200 900 500 250 (작은타입 7개) -> 0.37803517430997203(너무 oversampling하면 오히려 성능이 떨어질 수도 있음)

```
	type	count
1	QSO	42500
0	GALAXY	34000
4	SERENDIPITY_BLUE	19439
10	SPECTROPHOTO_STD	13073
2	REDDEN_STD	13052
11	STAR_BHB	12069
16	STAR_RED_DWARF	9040
14	STAR_CATY_VAR	5808
6	SERENDIPITY_FIRST	5718
3	ROSAT_D	5559
5	SERENDIPITY_DISTANT	4163
13	STAR_CARBON	2875
18	STAR_WHITE_DWARF	1932
8	SERENDIPITY_RED	1600
17	STAR_SUB_DWARF	1250
12	STAR_BROWN_DWARF	850
9	SKY	700
7	SERENDIPITY_MANUAL	250
15	STAR_PN	120

```
  샘플링 과정에서 데이터 개수가 1000개 이하인 type을 일정 비율 늘려준 모델의 성능이 가장 높은 것으로 나타났다.


<br>
<br>
<br>


## 5. 적합한 모델 찾기(XGBoost, CatBoost, RandomForest, LightGBM)
XGBoost, CatBoost, RandomForest, LightGBM 4개의 머신러닝 알고리즘을 통해 training data를 학습시키고, 가장 적합한 모델을 선정한다.

> 5-1. XGBoost

```
import xgboost as xgb
xgb_model = xgb.XGBClassifier(n_estimators=2000,
                         n_jobs=4,
                         max_depth=15,
                         learning_rate=0.05,
                         gamma = 0.02,
                         subsample = 0.9,
                         colsample_bytree=0.9,
                         missing=-999,
                         tree_method='gpu_hist')
```


> 5-2. CatBoost

```
import catboost as cb
cb_model = cb.CatBoostClassifier()
cb_model.fit(X_train_robustscaled,y_train_robustscaled)
```

> 5-3. RandomForest

```
from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=1000, criterion="entropy", random_state=True, max_leaf_nodes=38, n_jobs=-1)
rnd_clf.fit(X, y)
```

> 5-4. LightGBM

```
from lightgbm import LGBMClassifier
lgbm_model = LGBMClassifier(boosting_type='gbdt', objective='binary', num_leaves=128,
                                learning_rate=0.005, n_estimators=2000, max_depth=30,
                                bagging_fraction=0.9, feature_fraction=0.9, reg_lambda=0.2)
lgbm_model.fit(X,y)
```
위와 같이 모델을 학습시키고 출력한 log_loss값을 비교한 결과 LGBM의 예측 성과가 가장 높았으며, 두 번째로는 XGBoost가 높은 것으로 나타났다. 결과에 따라 최종 머신러닝 모델은 LGBM과 XGBoos로 선정하였다. 

<br>
<br>
<br>

## 6. 그리드 서치
> 6-1. XGBoost GridSearch

<br>


> 6-2. LightGBM GridSearch

```
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier

lgbm_model = LGBMClassifier(objective='binary')

param_dist = {'boosting_type':['rf','gbdt'],
              'n_estimators': [1500,2000,2500],
              'max_depth': [15,30,45],
              'learning_rate': [0.01,0.05,0.1],
              'bagging_freq': [1],
              'bagging_fraction':[0.9]
             }
grid_search = GridSearchCV(lgbm_model, n_jobs=6, param_grid=param_dist, cv=5, scoring="neg_log_loss", verbose=5)
grid_search.fit(X,y)


grid_search.best_estimator_

```

최종 parameter 결과는 다음과 같다.

```
LGBMClassifier(bagging_fraction=0.9, bagging_freq=1, boosting_type='rf',
               class_weight=None, colsample_bytree=1.0, importance_type='split',
               learning_rate=0.01, max_depth=15, min_child_samples=20,
               min_child_weight=0.001, min_split_gain=0.0, n_estimators=1500,
               n_jobs=-1, num_leaves=31, objective='binary', random_state=None,
               reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0,
               subsample_for_bin=200000, subsample_freq=0)
```





