# Databricks notebook source
import numpy as np
import pandas as pd
from scipy.stats import iqr
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import os
import multiprocessing
from joblib import Parallel, delayed
#from tqdm import tqdm
import warnings
# warnings.filterwarnings('ignore')

# COMMAND ----------

import rpy2

# COMMAND ----------

# MAGIC %md
# MAGIC # write spark

# COMMAND ----------

# MAGIC %md
# MAGIC ## preprocessing

# COMMAND ----------

df=pd.read_csv('/dbfs/mnt/client-112sap21p002-unc-maternal/04_data_analysis/data/UNC_rLC_mtb_pheno_dt_capped_imputed_scaled_020422.csv')
# df=pd.read_csv('/dbfs/mnt/client-112sap21p002-unc-maternal/04_data_analysis/data/UNC_rLC_mtb_pheno_dt_020422.csv')
df.shape

# COMMAND ----------

df[(i for i in df.columns if 'mtb' not in i)].head()

# COMMAND ----------

ptb_df['spont_PTB_logit']=0
ptb_df.loc[(df.spont_ptb37==1)|(df.spont_ptb34==1),'spont_PTB_logit']=1
ptb_df.loc[(df.provider_initiated_ptb==1),'spont_PTB_logit']=0
ptb_df['spont_PTB_logit'].unique()

# COMMAND ----------

# ptb_df.drop_duplicates(['PID'],inplace=False)['spont_PTB_logit'].value_counts()

# COMMAND ----------

visit1_ptb_df=ptb_df[(ptb_df.hiv==0) & (ptb_df.visit==0)]
print((visit1_ptb_df.shape))

visit2_ptb_df=ptb_df[(ptb_df.hiv==0) & (ptb_df.visit!=0)]
print((visit2_ptb_df.shape))


# COMMAND ----------

ptb_df.shape

# COMMAND ----------

# MAGIC %md ## cap outlier

# COMMAND ----------

#cap outliers
data=ptb_df
mtb_cols=[mtb for mtb in data.columns.values if 'rLC' in mtb]
inputs = tqdm(mtb_cols)
processed_result=Parallel(n_jobs=num_cores)(delayed(cap_outliers)(data[mtb_col]) for mtb_col in inputs)
for i in range(len(mtb_cols)):
    ptb_df[mtb_cols[i]]=processed_result[i]

# COMMAND ----------

dt=ptb_df

mtb_cols=[mtb for mtb in ptb_df.columns.values if 'mtb_' in mtb]
pheno_col=[mtb for mtb in ptb_df.columns.values if 'mtb_' not in mtb]

mtb_dtm = dt.loc[:, pheno_col + mtb_cols].melt(
    id_vars = pheno_col,
    value_vars = mtb_cols,
    var_name = 'metabolite',
    value_name = 'level'
    )

# COMMAND ----------

mtb_dtm = spark.createDataFrame(mtb_dtm)

# COMMAND ----------

display(mtb_dtm)

# COMMAND ----------

path='/dbfs/mnt/client-112sap21p002-unc-maternal/04_data_analysis/data/'
mtb_dtm.write.mode('overwrite').parquet('%sunc_losgistc_mtb_sponPTB_visitall_capped_imputed_042722.parquet'%(path))

# COMMAND ----------

# MAGIC %md
# MAGIC # load spark

# COMMAND ----------

path='/dbfs/mnt/client-112sap21p002-unc-maternal/04_data_analysis/data/'
mtb_dt=spark.read.parquet('%sunc_losgistc_mtb_sponPTB_visitall_capped_imputed_042722.parquet'%(path))

# COMMAND ----------

display(mtb_dt)

# COMMAND ----------

mtb_dt = mtb_dt.withColumn("spont_PTB_logit", mtb_dt["spont_PTB_logit"].cast('int'))

# COMMAND ----------

# MAGIC %run /Users/dong.liang@sapient.bio/Demo/toolkits/stats_util_stable

# COMMAND ----------

#%run /Users/Zi.Ye@sapient.bio/toolkits/stats_util_Zi_glmer

# COMMAND ----------

# MAGIC %md
# MAGIC # run ordinal regression

# COMMAND ----------

# MAGIC %run /Users/Zi.Ye@sapient.bio/toolkits/stats_util_stable_mesami

# COMMAND ----------

mtb_dt = mtb_dt.withColumn("hiv", mtb_dt["hiv"].cast('int'))
mtb_dt = mtb_dt.withColumn("age", mtb_dt["age"].cast('int'))
mtb_dt = mtb_dt.withColumn("ega", mtb_dt["egaw_ig_atvisit"].cast('int'))
mtb_dt = mtb_dt.withColumn("p17", mtb_dt["17P_1"].cast('int'))
mtb_dt = mtb_dt.withColumn("spont_PTB_logit", mtb_dt["spont_PTB_logit"].cast('int'))

# COMMAND ----------

sdf=mtb_dt
sdf.spont_PTB_logit
sdf=sdf.filter(sdf.spont_PTB_logit.isNotNull()) 
# sdf=sdf.filter(sdf.hiv==0) 
# sdf=sdf.filter(sdf.Samples=='20 min') 
sdf=sdf.filter(sdf.visit!=0) #visit two

# COMMAND ----------

sdf.display()

# COMMAND ----------

assoc = Association_spark(regression = 'logistic')
pvalues_severity = assoc.find_association(
  sdf,
  dv = 'spont_PTB_logit',
  iv = 'level', 
#   covariate = 'Azithro_x',
  covariate = 'age  + hiv',
  groupby = 'metabolite',
#   ordered_categories = ['U','DD', 'cerebral palsy-like', 'spastic diparesis'],# comment this line for other types of regression.
  spark = spark
)

# COMMAND ----------

p = pvalues_severity.toPandas()
p

# COMMAND ----------

path='/dbfs/mnt/client-112sap21p002-unc-maternal/04_data_analysis/data/'
# p.to_csv('%sunc_1st_logsitic.csv'%path)
# p.to_csv('%sunc_2nd_logsitic.csv'%path)

# COMMAND ----------

path='/dbfs/mnt/client-112sap21p002-unc-maternal/04_data_analysis/data/'
# p.to_csv('%sunc_1st_hivneg_logsitic.csv'%path)
# p.to_csv('%sunc_2nd_hivneg_logsitic.csv'%path)
# p.to_csv('%sorginal_20min_logistic_azi_df.csv'%path)
# p.to_csv('%sorginal_alltime_logistic_df.csv'%path)

p.to_csv('%sunc_all_hiv_2visit_logsitic_2covariate050622.csv'%path)
p

# COMMAND ----------

p.shape

# COMMAND ----------

# path='/dbfs/mnt/client-112sap21p002-unc-maternal/04_data_analysis/data/'
# p.to_csv('%sunc_2nd_hiv_pos_logsitic_2covariate.csv'%path)
p

# COMMAND ----------

path='/dbfs/mnt/client-112sap21p002-unc-maternal/04_data_analysis/data/'
# p.to_csv('%sunc_1st_hiv_pos_logsitic_2covariate.csv'%path)
p

# COMMAND ----------

path='/dbfs/mnt/client-112sap21p002-unc-maternal/04_data_analysis/data/'
# p.to_csv('%sunc_1st_hiv_neg_logsitic_2covariate.csv'%path)
p

# COMMAND ----------



# COMMAND ----------

Hiv_neg_1=pd.read_csv('%sunc_1st_hiv_neg_logsitic_2covariate.csv'%path)
Hiv_neg_2=pd.read_csv('%sunc_2nd_hiv_neg_logsitic_2covariate.csv'%path)
Hiv_pos_1=pd.read_csv('%sunc_1st_hiv_pos_logsitic_2covariate.csv'%path)
Hiv_pos_2=pd.read_csv('%sunc_2nd_hiv_pos_logsitic_2covariate.csv'%path)

# COMMAND ----------

Hiv_pos_1.head()

# COMMAND ----------

# MAGIC %md # run linear mixed

# COMMAND ----------

# MAGIC %run /Users/dong.liang@sapient.bio/Demo/toolkits/stats_util_stable

# COMMAND ----------

# MAGIC %run /Users/Zi.Ye@sapient.bio/toolkits/stats_util_stable_mesami

# COMMAND ----------

mtb_dt = mtb_dt.withColumn("hiv", mtb_dt["hiv"].cast('int'))
mtb_dt = mtb_dt.withColumn("age", mtb_dt["age"].cast('int'))
mtb_dt = mtb_dt.withColumn("ega", mtb_dt["egaw_ig_atvisit"].cast('int'))
mtb_dt = mtb_dt.withColumn("p17", mtb_dt["17P_1"].cast('int'))
mtb_dt = mtb_dt.withColumn("spont_PTB_logit", mtb_dt["spont_PTB_logit"].cast('int'))

# COMMAND ----------

sdf=mtb_dt
sdf.spont_PTB_logit
sdf=sdf.filter(sdf.spont_PTB_logit.isNotNull()) 
# sdf=sdf.filter(sdf.hiv==0) 
# sdf=sdf.filter(sdf.Samples=='20 min') 
# sdf=sdf.filter(sdf.visit!=0)

# COMMAND ----------

# from pyspark.sql.functions import *

# COMMAND ----------


sdf=sdf.filter(sdf.spont_PTB_logit.isNotNull()) 
sdf = sdf.withColumn('visit_2',when(sdf.visit==0,sdf.visit).otherwise(1))
sdf = sdf.withColumn("visit_2", sdf["visit_2"].cast('int'))

# COMMAND ----------

sdf.display()

# COMMAND ----------

# MAGIC %run /Users/Zi.Ye@sapient.bio/toolkits/stats_util_stable_mesami

# COMMAND ----------

assoc = Association_spark(regression = 'glmer', family = 'binomial')
pvalues_severity = assoc.find_association(
#   sdf.filter(sdf.metabolite=='rLC_pos_mtb_2806207'),
#     sdf.filter(sdf.metabolite=='rLC_neg_mtb_3343398'),
    sdf,
  dv = 'spont_PTB_logit',
  iv = 'level',
  covariate = 'visit_2 + hiv +(1|PID)',
  groupby = 'metabolite',
  spark = spark
)

p = pvalues_severity.toPandas()
p

# COMMAND ----------

p=pd.DataFrame(p)

# COMMAND ----------

print(p[p.Warnings!='no'].shape)
p.Warnings.unique()

# COMMAND ----------

p['WarningsType'] = p['Warnings'].str.replace(r'(\d+)', '')
p['WarningsType'] = p['WarningsType'].str.replace(r'(\n)', '')
p['WarningsType'].value_counts()

# COMMAND ----------

p['Converge']='No'
Converge=(i for i in p.Warnings if 'Model failed to converge with max' in i)
p.loc[p.Warnings.isin(Converge),'Converge']='Yes'

# COMMAND ----------

p['unidentifiable']='No'
unidentifiable=(i for i in p.Warnings if 'Model is nearly unidentifiable' in i)
p.loc[p.Warnings.isin(unidentifiable),'unidentifiable']='Yes'

# COMMAND ----------

p['negative']='No'
negative=(i for i in p.Warnings if 'scaled gradient' in i)
p.loc[(p.Warnings.isin(negative)),'negative']='Yes'

# COMMAND ----------

p[p.Warnings!='no']

# COMMAND ----------

p[p.metabolite=='rLC_pos_mtb_2806207']

# COMMAND ----------

p.to_csv('/dbfs/mnt/client-112sap21p002-unc-maternal/04_data_analysis/data/unc_allvisti_allhiv_glmer_042922.csv')
p

# COMMAND ----------

pnowarning=p.loc[p.Warnings=='no',['metabolite','Estimate','Std_Error','z_value','P','Support','N','Error']]
pnowarning.shape

# COMMAND ----------

pnowarning

# COMMAND ----------

pnowarning.to_csv('/dbfs/mnt/client-112sap21p002-unc-maternal/04_data_analysis/data/unc_allvisti_allhiv_glmer_no_warning.csv')
pnowarning

# COMMAND ----------


