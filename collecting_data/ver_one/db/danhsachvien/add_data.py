#%%
import pandas as pd
import numpy as np

df = pd.read_excel('DS VIen 02.2021.xlsx')

#%%
from sqlalchemy import create_engine
engine = create_engine('postgresql://postgres:123456@192.168.1.15:5432/laymau')
# engine = create_engine('postgresql://dat:123456@192.168.1.54:5432/laymau')

# %%
department_df = pd.DataFrame(df['Đơn vị'].unique(), columns=['name'])
department_df

# %% add data to DEPARTMENT table and then get back everything
department_df.to_sql('DEPARTMENT', engine, if_exists='append', index=False)
department_df = pd.read_sql_table('DEPARTMENT', engine)

# %%
staff_df = pd.merge(df, department_df, left_on='Đơn vị', right_on='name')
staff_df = staff_df.drop(columns=['Đơn vị', 'name', 'comment'])
staff_df.rename(columns={
    "Mã NV": 'staff_id',
    "Họ và tên": 'fullname',
    "id": 'department_id'
}, inplace=True)
staff_df

# %%
staff_df.to_sql('STAFF', engine, if_exists='append', index=False)
# %%
