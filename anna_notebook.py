#!/usr/bin/env python
# coding: utf-8

# ## Загрузка и предобработка данных

# In[1]:


import pandas as pd
import numpy as np
import datetime as DT
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import pylab
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


# In[2]:


shipments_01 = pd.read_csv('C:\\Users\\user\\Downloads\\ngwl-predict-customer-churn\\shipments\\shipments2020-01-01.csv')
shipments_03 = pd.read_csv('C:\\Users\\user\\Downloads\\ngwl-predict-customer-churn\\shipments\\shipments2020-03-01.csv')
shipments_04 = pd.read_csv('C:\\Users\\user\\Downloads\\ngwl-predict-customer-churn\\shipments\\shipments2020-04-30.csv')
shipments_06 = pd.read_csv('C:\\Users\\user\\Downloads\\ngwl-predict-customer-churn\\shipments\\shipments2020-06-29.csv')
frames = [shipments_01, shipments_03, shipments_04, shipments_06]


# In[3]:


shipments = pd.concat(frames).reset_index(drop=True)


# In[4]:


shipments


# In[5]:


user = pd.read_csv('C:\\Users\\user\\Downloads\\ngwl-predict-customer-churn\\misc\\user_profiles.csv')


# In[6]:


user


# In[7]:


user['bdate'].isna().sum()/len(user)


# In[8]:


user['gender'].isna().sum()/len(user)


# In[9]:


shipments = shipments.merge(user, how='left', on ='user_id')


# In[10]:


shipments


# In[11]:


train = pd.read_csv('C:\\Users\\user\\Downloads\\ngwl-predict-customer-churn\\train\\train.csv')


# In[12]:


addresses = pd.read_csv('C:\\Users\\user\\Downloads\\ngwl-predict-customer-churn\\misc\\addresses.csv')


# In[13]:


addresses.columns = (['ship_address_id', 'phone_id'])


# In[14]:


shipments = shipments.merge(addresses, how='left', on='ship_address_id')


# In[15]:


shipments['month'] = pd.to_datetime(shipments['order_completed_at']).dt.month


# In[16]:


shipments


# In[17]:


train['month'] = pd.to_datetime(train['order_completed_at']).dt.month


# In[18]:


prepr_train = train[['phone_id','month', 'target']]


# In[19]:


submiss = pd.read_csv('C:\\Users\\user\\Downloads\\ngwl-predict-customer-churn\\sample_submission.csv', sep=';')


# In[20]:


submiss


# In[21]:


shipments = shipments.where(shipments['phone_id'].isin(train['phone_id'].values)).dropna(how='all', axis=0)


# In[22]:


shipments = shipments.where(shipments['phone_id'].isin(submiss['Id'].values)).dropna(how='all', axis=0)


# In[54]:


shipments


# In[55]:


shipments = shipments[shipments['s.order_state'] !='cart']


# In[56]:


shipments['s.order_state'].value_counts()


# Категоризируем статус доставки на "доставлено" и "не доставлено", заказы с измененным составом отнесем к доставленным.

# In[26]:


def bool_group(order_state):
    if order_state == 'complete':
        return 1
    elif order_state == 'resumed':
        return 1
    else:
        return 0
shipments['is_complited'] = shipments['s.order_state'].apply(bool_group)


# In[27]:


shipments['is_complited'].value_counts()


# In[28]:


shipments['ship_time'] = pd.to_datetime(shipments['shipped_at']) - pd.to_datetime(shipments['order_completed_at'])


# Заполним пропуски во времени доставки нулями.

# In[29]:


shipments['ship_time'] = shipments['ship_time'].fillna(0)


# Округлим время до часов и переведем в формат int.

# In[30]:


shipments['ship_time'] = (shipments['ship_time'].dt.round('1H') / np.timedelta64(1, 'h')).astype(int) 


# In[31]:


shipment_table = shipments[['phone_id','user_id','month', 
                            'is_complited', 'ship_time', 'total_weight', 
                            'total_cost', 'promo_total', 'rate', 'order_id']]


# In[32]:


shipment_table


# Сгруппируем все данные по месяцам.

# In[33]:


grouped_01 = shipment_table[shipment_table['month'] == 1]
january = (
    grouped_01
    .groupby('phone_id')[['is_complited','ship_time', 'total_weight',
                          'total_cost', 'promo_total', 'rate', 'order_id']]
    .agg({'is_complited': 'mean',
          'ship_time': 'mean',
          'total_weight' : 'mean',
          'total_cost': 'mean',
          'promo_total': 'mean',
          'rate': 'mean',
          'order_id': 'count'})
)
january.columns = ['part_of_complited', 'mean_time', 'mean_weight', 'mean_cost',
                   'promo_mean', 'mean_rate', 'order_count']
january = january.merge(train[train['month'] == 1], 
                        how='left', on='phone_id').drop(['order_completed_at','month'],
                                                        axis=1)


# In[34]:


january


# In[35]:


january['target'].isna().sum()


# In[36]:


grouped_02 = shipment_table[shipment_table['month'] == 2]
february = (
    grouped_02.groupby('phone_id')[['is_complited','ship_time', 'total_weight',
                          'total_cost', 'promo_total', 'rate', 'order_id']]
    .agg({'is_complited': 'mean',
          'ship_time': 'mean',
          'total_weight' : 'mean',
          'total_cost': 'mean',
          'promo_total': 'mean',
          'rate': 'mean',
          'order_id': 'count'})
)
february.columns = ['part_of_complited', 'mean_time', 'mean_weight', 'mean_cost',
                   'promo_mean', 'mean_rate', 'order_count']
february = february.merge(train[train['month'] == 2], 
                        how='left', on='phone_id').drop(['order_completed_at','month'],
                                                        axis=1)


# In[37]:


february['target'].isna().sum()


# In[38]:


grouped_03 = shipment_table[shipment_table['month'] == 3]
march = (
    grouped_03.groupby('phone_id')[['is_complited','ship_time', 'total_weight',
                          'total_cost', 'promo_total', 'rate', 'order_id']]
    .agg({'is_complited': 'mean',
          'ship_time': 'mean',
          'total_weight' : 'mean',
          'total_cost': 'mean',
          'promo_total': 'mean',
          'rate': 'mean',
          'order_id': 'count'})
)
march.columns = ['part_of_complited', 'mean_time', 'mean_weight', 'mean_cost',
                   'promo_mean', 'mean_rate', 'order_count']
march = march.merge(train[train['month'] == 3], 
                        how='left', on='phone_id').drop(['order_completed_at','month'],
                                                        axis=1)


# In[39]:


march['target'].isna().sum()


# In[40]:


grouped_04 = shipment_table[shipment_table['month'] == 4]
april = (
    grouped_04.groupby('phone_id')[['is_complited','ship_time', 'total_weight',
                          'total_cost', 'promo_total', 'rate', 'order_id']]
    .agg({'is_complited': 'mean',
          'ship_time': 'mean',
          'total_weight' : 'mean',
          'total_cost': 'mean',
          'promo_total': 'mean',
          'rate': 'mean',
          'order_id': 'count'})
)
april.columns = ['part_of_complited', 'mean_time', 'mean_weight', 'mean_cost',
                   'promo_mean', 'mean_rate', 'order_count']
april = april.merge(train[train['month'] == 4], how='left',
                    on='phone_id').drop(['order_completed_at','month'], axis=1)


# In[41]:


april['target'].isna().sum()


# In[42]:


grouped_05 = shipment_table[shipment_table['month'] == 5]
may = (
    grouped_05.groupby('phone_id')[['is_complited','ship_time', 'total_weight',
                          'total_cost', 'promo_total', 'rate', 'order_id']]
    .agg({'is_complited': 'mean',
          'ship_time': 'mean',
          'total_weight' : 'mean',
          'total_cost': 'mean',
          'promo_total': 'mean',
          'rate': 'mean',
          'order_id': 'count'})
)
may.columns = ['part_of_complited', 'mean_time', 'mean_weight', 'mean_cost',
                   'promo_mean', 'mean_rate', 'order_count']
may = may.merge(train[train['month'] == 5], how='left',
                on='phone_id').drop(['order_completed_at','month'], axis=1)


# In[43]:


may['target'].isna().sum()


# In[44]:


grouped_06 = shipment_table[shipment_table['month'] == 6]
june = (
    grouped_06.groupby('phone_id')[['is_complited','ship_time', 'total_weight',
                          'total_cost', 'promo_total', 'rate', 'order_id']]
    .agg({'is_complited': 'mean',
          'ship_time': 'mean',
          'total_weight' : 'mean',
          'total_cost': 'mean',
          'promo_total': 'mean',
          'rate': 'mean',
          'order_id': 'count'})
)
june.columns = ['part_of_complited', 'mean_time', 'mean_weight', 'mean_cost',
                   'promo_mean', 'mean_rate', 'order_count']
june = june.merge(train[train['month'] == 6], how='left',
                on='phone_id').drop(['order_completed_at','month'], axis=1)


# In[45]:


june['target'].isna().sum()


# In[46]:


grouped_07 = shipment_table[shipment_table['month'] == 7]
july = (
    grouped_07.groupby('phone_id')[['is_complited','ship_time', 'total_weight',
                          'total_cost', 'promo_total', 'rate', 'order_id']]
    .agg({'is_complited': 'mean',
          'ship_time': 'mean',
          'total_weight' : 'mean',
          'total_cost': 'mean',
          'promo_total': 'mean',
          'rate': 'mean',
          'order_id': 'count'})
)
july.columns = ['part_of_complited', 'mean_time', 'mean_weight', 'mean_cost',
                   'promo_mean', 'mean_rate', 'order_count']
july = july.merge(train[train['month'] == 7], how='left',
                on='phone_id').drop(['order_completed_at','month'], axis=1)


# In[47]:


july['target'].isna().sum()


# In[48]:


grouped_08 = shipment_table[shipment_table['month'] == 8]
august = (
   grouped_08.groupby(['phone_id','user_id'])[['is_complited','ship_time', 'total_weight',
                          'total_cost', 'promo_total', 'rate', 'order_id']]
    .agg({'is_complited': 'mean',
          'ship_time': 'mean',
          'total_weight' : 'mean',
          'total_cost': 'mean',
          'promo_total': 'mean',
          'rate': 'mean',
          'order_id': 'count'})
).reset_index()
august.columns = ['phone_id','user_id','part_of_complited', 'mean_time', 'mean_weight', 'mean_cost',
                   'promo_mean', 'mean_rate', 'order_count']


# In[68]:


august


# In[50]:


mes_01 = pd.read_csv('C:\\Users\\user\\Downloads\\message_number_updated\\mes_num_01.csv')
mes_02 = pd.read_csv('C:\\Users\\user\\Downloads\\message_number_updated\\mes_num_02.csv')
mes_03 = pd.read_csv('C:\\Users\\user\\Downloads\\message_number_updated\\mes_num_03.csv')
mes_04 = pd.read_csv('C:\\Users\\user\\Downloads\\message_number_updated\\mes_num_04.csv')
mes_05 = pd.read_csv('C:\\Users\\user\\Downloads\\message_number_updated\\mes_num_05.csv')
mes_06 = pd.read_csv('C:\\Users\\user\\Downloads\\message_number_updated\\mes_num_06.csv')
mes_07 = pd.read_csv('C:\\Users\\user\\Downloads\\message_number_updated\\mes_num_07.csv')
mes_08 = pd.read_csv('C:\\Users\\user\\Downloads\\message_number_updated\\mes_num_08.csv')


# In[51]:


january = january.merge(mes_01, how='left', on='phone_id')
february = february.merge(mes_02, how='left', on='phone_id')
march = march.merge(mes_03, how='left', on='phone_id')
april = april.merge(mes_04, how='left', on='phone_id')
may = may.merge(mes_05, how='left', on='phone_id')
june = june.merge(mes_06, how='left', on='phone_id')
july = july.merge(mes_07, how='left', on='phone_id')
august = august.merge(mes_08, how='left', on='user_id')


# In[52]:


january


# Будем считать, что, поскольку данных о количествах сообщений нет, их не было вообще. Т.е. заменим пропуски нулями.

# In[53]:


january['mes_num'] = january['mes_num'].fillna(0)
february['mes_num'] = february['mes_num'].fillna(0)
march['mes_num'] = march['mes_num'].fillna(0)
april['mes_num'] = april['mes_num'].fillna(0)
may['mes_num'] = may['mes_num'].fillna(0)
june['mes_num'] = june['mes_num'].fillna(0)
july['mes_num'] = july['mes_num'].fillna(0)
august['mes_num'] = august['mes_num'].fillna(0)


# In[69]:


category_jan = pd.read_csv('C:\\Users\\user\\Downloads\\categories\\jun.csv')
category_feb = pd.read_csv('C:\\Users\\user\\Downloads\\categories\\feb.csv')
category_mar = pd.read_csv('C:\\Users\\user\\Downloads\\categories\\mar.csv')
category_apr = pd.read_csv('C:\\Users\\user\\Downloads\\categories\\apr.csv')
category_may = pd.read_csv('C:\\Users\\user\\Downloads\\categories\\mai.csv')
category_jun = pd.read_csv('C:\\Users\\user\\Downloads\\categories\\jun.csv')
category_jul = pd.read_csv('C:\\Users\\user\\Downloads\\categories\\jul.csv')
category_aug = pd.read_csv('C:\\Users\\user\\Downloads\\categories\\aug.csv')


# In[70]:


category_jan


# In[71]:


january = january.merge(category_jan.groupby('phone_id')['discount'].mean(), how='left', on='phone_id')
february = february.merge(category_feb.groupby('phone_id')['discount'].mean(), how='left', on='phone_id')
march = march.merge(category_mar.groupby('phone_id')['discount'].mean(), how='left', on='phone_id')
april = april.merge(category_apr.groupby('phone_id')['discount'].mean(), how='left', on='phone_id')
may = may.merge(category_may.groupby('phone_id')['discount'].mean(), how='left', on='phone_id')
june = june.merge(category_jun.groupby('phone_id')['discount'].mean(), how='left', on='phone_id')
july = july.merge(category_jul.groupby('phone_id')['discount'].mean(), how='left', on='phone_id')
august = august.merge(category_aug.groupby('user_id')['discount'].mean(), how='left', on='user_id')


# In[72]:


august


# In[73]:


mes_disc_01 = pd.read_csv('C:\\Users\\user\\Downloads\\message_discounts\\mes_disc_01.csv')
mes_disc_02 = pd.read_csv('C:\\Users\\user\\Downloads\\message_discounts\\mes_disc_02.csv')
mes_disc_03 = pd.read_csv('C:\\Users\\user\\Downloads\\message_discounts\\mes_disc_03.csv')
mes_disc_04 = pd.read_csv('C:\\Users\\user\\Downloads\\message_discounts\\mes_disc_04.csv')
mes_disc_05 = pd.read_csv('C:\\Users\\user\\Downloads\\message_discounts\\mes_disc_05.csv')
mes_disc_06 = pd.read_csv('C:\\Users\\user\\Downloads\\message_discounts\\mes_disc_06.csv')
mes_disc_07 = pd.read_csv('C:\\Users\\user\\Downloads\\message_discounts\\mes_disc_07.csv')
mes_disc_08 = pd.read_csv('C:\\Users\\user\\Downloads\\message_discounts\\mes_disc_08.csv')


# In[74]:


january = january.merge(mes_disc_01, how='left', on='phone_id')
february = february.merge(mes_disc_02, how='left', on='phone_id')
march = march.merge(mes_disc_03, how='left', on='phone_id')
april = april.merge(mes_disc_04, how='left', on='phone_id')
may = may.merge(mes_disc_05, how='left', on='phone_id')
june = june.merge(mes_disc_06, how='left', on='phone_id')
july = july.merge(mes_disc_07, how='left', on='phone_id')
august = august.merge(mes_disc_08, how='left', on='user_id')


# In[75]:


january = january.dropna(how='any', axis=0)
february = february.dropna(how='any', axis=0)
march = march.dropna(how='any', axis=0)
april = april.dropna(how='any', axis=0)
may = may.dropna(how='any', axis=0)
june = june.dropna(how='any', axis=0)
july = july.dropna(how='any', axis=0)
august = august.dropna(how='any', axis=0)


# In[58]:


# january = january.drop('Unnamed: 0', axis=1)
# february = february.drop('Unnamed: 0', axis=1)
# march = march.drop('Unnamed: 0', axis=1)
# april = april.drop('Unnamed: 0', axis=1)
# may = may.drop('Unnamed: 0', axis=1)
# june = june.drop('Unnamed: 0', axis=1)
# july = july.drop('Unnamed: 0', axis=1)


# In[60]:


# january.to_csv('january.csv')
# february.to_csv('february.csv')
# march.to_csv('march.csv')
# april.to_csv('april.csv')
# may.to_csv('may.csv')
# june.to_csv('june.csv')
# july.to_csv('july.csv')
# august.to_csv('august.csv')


# ## Обучение моделей

# In[77]:


log_regr = LogisticRegression()
def train_ (model, x_train, y_train, x_test, y_test):
    #x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.3, random_state = 42)
    model = model
    model.fit(x_train, y_train)
    predict = model.predict(x_test)
    f1 = f1_score(y_test, predict)
    print(f1)   


# In[78]:


train_(log_regr, april.drop(['target','phone_id'], axis=1), april['target'],
      may.drop(['target','phone_id'], axis=1), may['target'])


# In[84]:


get_ipython().system('pip install catboost')


# In[79]:


from catboost import CatBoostClassifier


# In[80]:


cat_model = CatBoostClassifier(task_type="GPU", verbose=100, 
                              random_state=42, early_stopping_rounds=100,
                              iterations=3000)    
train_(cat_model, april.drop(['target','phone_id'], axis=1), april['target'],
      may.drop(['target','phone_id'], axis=1), may['target'])


# In[81]:


from sklearn.ensemble import RandomForestClassifier


# In[82]:


forest = RandomForestClassifier(random_state=42, 
                                bootstrap = True,
                                max_depth= 5, 
                                n_estimators=100)


# In[83]:


train_(forest, april.drop(['target','phone_id'], axis=1), april['target'],
      may.drop(['target','phone_id'], axis=1), may['target'])


# In[59]:


august


# In[94]:


july


# In[84]:


forest.fit(july.drop(['target','phone_id'], axis=1), july['target'])
predicted = forest.predict(august.drop(['user_id','phone_id'], axis=1))


# In[85]:


august['predicted'] = predicted


# In[86]:


august


# In[88]:


pred = august[['phone_id','predicted']]


# In[90]:


pred.columns = ['Id', 'predicted']


# In[92]:


submiss.merge(pred, how='left',on='Id')


# In[ ]:




