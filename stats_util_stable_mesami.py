# Databricks notebook source
import pandas as pd
import numpy as np
from pyspark.sql.functions import *
from pyspark.sql.types import *
import scipy.stats
from rpy2 import robjects as ro
from rpy2.robjects import pandas2ri, Formula


class Association_spark(object):
    def __init__(self, regression = 'logistic', family = None):
        """For performing large-scale regression-based association analysis in PySpark

        Args:
            regression  (str): type of regressions: [linear, logistic, ordinal logistic, lmer, glm]
            family (str): family name of regression, only used with glm Regression (glm(formula, family=familytype(link=linkfunction), data=))
                          Family	        Default Link Function
                          binomial	        (link = "logit")
                          gaussian	        (link = "identity")
                          Gamma	            (link = "inverse")  
                          inverse.gaussian	(link = "1/mu^2")
                          poisson	        (link = "log")
                          quasi	            (link = "identity", variance = "constant")
                          quasibinomial	    (link = "logit")
                          quasipoisson	    (link = "log")
  
        Returns:
            object: It returns an instance of Association_spark class.
        """   
        
        self.regression = regression
        self.family = family
        self.supported_regression_types = ['linear', 'logistic', 'ordinal logistic', 'glm', 'lmer', 'glmer']
    

    def find_association(self, sdf, dv, iv, groupby = 'metabolite', spark = spark, ordered_categories = ['1.0', '2.0', '3.0', '4.0', '5.0'], **args):
        """For performing large-scale regression-based association analysis in PySpark

        Args:
            sdf (spark dataframe): A spark data frame in long format
            dv (str): Dependent variable in your regeression model
            iv (str): Independent variable in your regression model
            groupby (str, optional): The group name by which you want to partition. Defaults to "metabolite".
            spark (str, optional): The instance of a spark engine. This argument is optional. Defaults to "spark".
            ordered_categories ([str], optional): A list of ordinal labels. This argument is only required in ordinal logistic regression. Defaults to ['1.0', '2.0', '3.0', '4.0', '5.0'].

        Returns:
            spark dataframe: It returns the regression analysis result for each metabolite. 
        """   
        
        covariate = args['covariate']
        schema_de = StructType([ \
            StructField(groupby,StringType(),True), \
            StructField('Estimate',DoubleType(),True), \
            StructField('Std_Error', DoubleType(),False), \
            StructField('z_value',DoubleType(),False), \
            StructField('P', DoubleType(),False), \
            StructField('Support', IntegerType(),False), \
            StructField('N', IntegerType(),False), \
            StructField('Error', StringType(),False) \
            ])
        
        schema_lmer = StructType([ \
            StructField(groupby,StringType(),True), \
            StructField('Estimate',DoubleType(),True), \
            StructField('Std_Error', DoubleType(),False), \
            StructField('df',DoubleType(),False), \
            StructField('z_value',DoubleType(),False), \
            StructField('P', DoubleType(),False), \
            StructField('Support', IntegerType(),False), \
            StructField('N', IntegerType(),False), \
            StructField('Error', StringType(),False) \
            ])
        
        schema_glmer = StructType([ \
            StructField(groupby,StringType(),True), \
            StructField('Estimate',DoubleType(),True), \
            StructField('Std_Error', DoubleType(),False), \
            StructField('z_value',DoubleType(),False), \
            StructField('P', DoubleType(),False), \
            StructField('Support', IntegerType(),False), \
            StructField('N', IntegerType(),False), \
            StructField('Error', StringType(),False), \
            StructField('Warnings', StringType(),False) \
            ])
        
        schema_dict = {
          'lmer': schema_lmer, 
          'glmer': schema_glmer
        }
        schema = schema_dict.get(self.regression, schema_de)
#         schema = schema_lmer if self.regression == 'lmer' | self.regression == 'glmer' else schema_de
       
        regression_analysis = get_regression(self.regression, self.family, schema, dv, iv, groupby, covariate, ordered_categories)   
        return sdf.groupby(groupby).apply(regression_analysis)



def get_regression(regression, family, schema, dv, iv, groupby, covariate, ordered_categories):

  
    @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
    def glmeRegression_R_spark(key, df):
        pandas2ri.activate()
        R = ro.r
        R('library(lme4)')
        R('library(lmerTest)')
        
        
        N = len(df)
        df = df.dropna(subset = [dv.replace('`', '')]) 
        support = len(df)
        
        
        covariate_list = [x for x in covariate.split('+') if x !='']
        formula = Formula( ' + '.join([x for x in [f'{dv} ~ {iv}', f'{covariate}'] if x !='']))
        error='no'
        warnings='no'

        try:
            M = R.glmer(formula, data = df, family = R(family))
        except Exception as e:
            error += '\\' + str(e)
            res = [np.nan] * 4
            
        try:
            res = R.summary(M).rx2('coefficients')[1].tolist()
            warning = R.summary(M).rx2('optinfo').rx2('conv').rx2('lme4').rx2('messages')
#      
            if warning:
#               if len(warning)>1:
#                 err = '_'.join([i for i in warning])
# #               err='no'
#               else:
#               warnings = R.summary(M).rx2('optinfo').rx2('conv').rx2('lme4').rx2('messages')[1]
#               print(warning)
#               print(len(warning))
# #               print(warning.shape)
#               if len(warning)>1:
# #                 err = '_'.join([i for i in warning])
#                 err=warning[0]
#               else:

              err = 'warning'
#               err=str(type(warning))
              wtype=str(type(warning))
              if wtype=='<class \'rpy2.robjects.vectors.ListVector\'>':

                pandas2ri.activate()
                R = ro.r
                R('library(lme4)')
                R('library(lmerTest)')
                warning1 = R.summary(M).rx2('optinfo').rx2('conv').rx2('lme4').rx2('messages')[0]
                warning2 = R.summary(M).rx2('optinfo').rx2('conv').rx2('lme4').rx2('messages')[1]
#                 err = 'multiple list warnings'
                err=warning1[0]+' ; ' +warning2[0]
              if wtype=='<class \'rpy2.robjects.vectors.StrVector\'>':
#                 err = warning[1]
#                 pandas2ri.activate()
#                 R = ro.r
#                 R('library(lme4)')
#                 R('library(lmerTest)')
#                 warning = R.summary(M).rx2('optinfo').rx2('conv').rx2('lme4').rx2('messages')[1]
                err = str(warning)
#                 err = 'multiple string warnings'
            else:
              err='no'
            warnings=err
        except Exception as e:
            error += '\\' +str(e)
            
            res = [np.nan] * 4
  
        pdf = pd.DataFrame([list(key) + res + [support, N, error,warnings]], columns = [groupby, 'Estimate', 'Std_Error', 'z_value', 'P', 'Support', 'N', 'Error','Warnings'])  
        pdf.fillna(value = 9999, inplace = True)
        return pdf  
  
    
    @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
    def lmeRegression_R_spark(key, df):
        pandas2ri.activate()
        R = ro.r
        R('library(lme4)')
        R('library(lmerTest)')
        
        
        N = len(df)
        df = df.dropna(subset = [dv.replace('`', '')]) 
        support = len(df)
        
        
        covariate_list = [x for x in covariate.split('+') if x !='']
        formula = Formula( ' + '.join([x for x in [f'{dv} ~ {iv}', f'{covariate}'] if x !='']))
        error = 'no'
        
        try:
            M = R.lmer(formula, data = df)
        except Exception as e:
            error = str(e)
            res = [np.nan] * 5

        try:
            res = R.summary(M).rx2('coefficients')[1].tolist()
        except Exception as e:
            error = str(e)
            res = [np.nan] * 6
  
        pdf = pd.DataFrame([list(key) + res + [support, N, error]], columns = [groupby, 'Estimate', 'Std_Error', 'df', 'z_value', 'P', 'Support', 'N', 'Error'])  
        return pdf
    
  
    @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
    def glmRegression_R_spark(key, df):
        pandas2ri.activate()
        R = ro.r 
        N = len(df)
        df = df.dropna(subset = [dv.replace('`', '')]) 
        support = len(df)
        
        covariate_list = [x for x in covariate.split('+') if x !='']
        formula = Formula( ' + '.join([x for x in [f'{dv} ~ {iv}', f'{covariate}'] if x !='']))
        error = 'no'
        
        try:
            M = R.glm(formula, data=df, family = R(family))

        except Exception as e:
            error = str(e)
            res = [np.nan] * 4
        try:
            R.summary(M).rx2('coefficients')[len(covariate_list) + 1]
            res = R.summary(M).rx2('coefficients')[1].tolist()
        except Exception as e:
            error = str(e)
            res = [np.nan] * 4

        pdf = pd.DataFrame([list(key) + res + [support, N, error]], columns = [groupby, 'Estimate', 'Std_Error', 'z_value', 'P', 'Support', 'N', 'Error'])
        pdf.fillna(value = 9999, inplace = True)
        return pdf
    

    @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
    def linearRegression_R_spark(key, df):
        pandas2ri.activate()
        R = ro.r 
        N = len(df)
#         dv = dv.split('(')[1].split(')')[0]
#         df = df.dropna(subset = [dv.split('(')[1].split(')')[0].replace('`', '')]) 
        df = df.dropna(subset = [dv.replace('`', '')])
        support = len(df) 

        covariate_list = [x for x in covariate.split('+') if x !='']
        formula = Formula( ' + '.join([x for x in [f'{dv} ~ {iv}', f'{covariate}'] if x !='']))
        error = 'no'

        try:
            M = R.lm(formula, data=df)

        except Exception as e:
            error = str(e)
            res = [np.nan] * 4
        try:
            R.summary(M).rx2('coefficients')[len(covariate_list) + 1]
            res = R.summary(M).rx2('coefficients')[1].tolist()
        except Exception as e:
            error = str(e)
            res = [np.nan] * 4

        pdf = pd.DataFrame([list(key) + res + [support, N, error]], columns = [groupby, 'Estimate', 'Std_Error', 'z_value', 'P', 'Support', 'N', 'Error'])
        pdf.fillna(value = 9999, inplace = True)
        return pdf


    
    @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
    def ordinalLogisticRegression_R_spark(key, df):  
        pandas2ri.activate()
        R = ro.r  
        R('library(MASS)')

        
        N = len(df)
        df = df.dropna(subset = [dv.replace('`', '')]) 
        support = len(df)


        df[dv] = df[dv].astype('string')
        df[dv] = pd.Categorical(
            df[dv],
            categories = ordered_categories
        )

        covariate_list = [x for x in covariate.split('+') if x !='']
        formula = Formula( ' + '.join([x for x in [f'{dv} ~ {iv}', f'{covariate}'] if x !='']))
        error = 'no'

        try:
            M = R.polr(formula, data=df, Hess = True)
        except Exception as e:
            error = str(e)
            res = [np.nan] * 4
            
        try:
            R.summary(M).rx2('coefficients')[len(covariate_list) + 1]
            res = R.summary(M).rx2('coefficients')[0].tolist()
            res.append(scipy.stats.t.sf(np.abs(res[2]), df.shape[0] - 1)*2)
        except Exception as e:
            error += '\\' + str(e)
            res = [np.nan] * 4

        pdf = pd.DataFrame([list(key) + res + [support, N, error]], columns = [groupby, 'Estimate', 'Std_Error', 'z_value', 'P', 'Support', 'N', 'Error'])
        pdf.fillna(value = 999, inplace = True)
        return pdf
      

    @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
    def logisticRegression_R_spark(key, df):
        pandas2ri.activate()
        R = ro.r 
        N = len(df)
        df = df.dropna(subset = [dv.replace('`', '')]) 
        support = len(df)

        covariate_list = [x for x in covariate.split('+') if x !='']
        group_key = df['metabolite'].iloc[0]
        formula = Formula( ' + '.join([x for x in [f'{dv} ~ {iv}', f'{covariate}'] if x !='']))
        error = 'no'
        
        try:
            M = R.glm(formula, data=df, family = "binomial")

        except Exception as e:
            error = str(e)
            res = [np.nan] * 4
        try:
            R.summary(M).rx2('coefficients')[len(covariate_list) + 1]
            res = R.summary(M).rx2('coefficients')[1].tolist()
        except Exception as e:
            error = str(e)
            res = [np.nan] * 4

        pdf = pd.DataFrame([list(key) + res + [support, N, error]], columns = [groupby, 'Estimate', 'Std_Error', 'z_value', 'P', 'Support', 'N', 'Error'])
#         pdf.fillna(value = 9999, inplace = True)
        return pdf

    if regression == 'ordinal logistic':
        return ordinalLogisticRegression_R_spark
    elif regression == 'logistic':
        return logisticRegression_R_spark 
    elif regression == 'linear':
        return linearRegression_R_spark
    elif regression == 'glm':
        return glmRegression_R_spark
    elif regression == 'lmer':
        return lmeRegression_R_spark
    elif regression == 'glmer':
        return glmeRegression_R_spark
    else:
        return 'Not yet implemented!'
      


def add_confint(df, alpha =0.05, n_var = 4):
    degree_freedom = df['Support'] - 1 - n_var
    tscore = scipy.stats.t.isf(alpha / 2, degree_freedom)
    margin_error = tscore * df['Std_Error']
    df['Estimate_lower_bound'] = df['Estimate'] - margin_error
    df['Estimate_upper_bound'] = df['Estimate'] + margin_error
    return df.loc[:, ['mtb', 'Estimate', 'Estimate_lower_bound', 'Estimate_upper_bound', 'Std_Error', 'z_value', 'P', 'Support', 'N']]
  
  
# def add_confint(df, alpha =0.05, n_var = 4):
#     degree_freedom = df['Support'] - 1 - n_var
#     tscore = scipy.stats.t.isf(alpha / 2, degree_freedom)
#     margin_error = tscore * df['Std. Error']
#     df['Estimate_lower_bound'] = df['Estimate'] - margin_error
#     df['Estimate_upper_bound'] = df['Estimate'] + margin_error
#     return df.loc[:, ['mtb', 'Estimate', 'Estimate_lower_bound', 'Estimate_upper_bound', 'Std. Error', 'z value', 'Pr(>|z|)', 'Support', 'N']]
    
def add_confint(df, alpha =0.05, n_var = 4):
    degree_freedom = df['Support'] - 1 - n_var
    tscore = scipy.stats.t.isf(alpha / 2, degree_freedom)
    margin_error = tscore * df['Std_Error']
    df['Estimate_lower_bound'] = df['Estimate'] - margin_error
    df['Estimate_upper_bound'] = df['Estimate'] + margin_error
    df['margin_error'] =  margin_error
    return df.loc[:, ['metabolite', 'Estimate', 'Estimate_lower_bound', 'Estimate_upper_bound', 'margin_error','Std_Error', 'z_value', 'P', 'Support', 'N']]
