from pyspark.sql import SparkSession
from sklearn.feature_selection import mutual_info_regression
import argparse
import pandas as pd
import numpy as np
from azureml.core import Run

spark= SparkSession.builder.getOrCreate()

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--input_dataset', required=True)
arg_parser.add_argument('--output_dataset', required=True)
arg_parser.add_argument('--start_test_date', required=True)

args, unknown_args = arg_parser.parse_known_args()

sdf_sensors_prepared = spark.read.parquet(args.input_dataset)

sensors = [t for t in sdf_sensors_prepared.columns if t not in ['Time']]
targets = ['Out1', 'Out2']
features = [c for c in sensors if c not in targets]

z_args = [[f for f in features]] + [[0. for _ in range(len(features))] for _ in range(len(targets))]
z = zip(*z_args)
sdf_mi = spark.createDataFrame([i for i in z], ['Feature'] + [t for t in targets])

df_sensors_prepared = sdf_sensors_prepared.toPandas().sort_values(by='Time').set_index('Time')

def mutual_information(pdf):
    X = df_sensors_prepared[pdf['Feature']]

    for target in targets:
        y = df_sensors_prepared[target]
        mi = mutual_info_regression(X=X, y=y, discrete_features=False)[0]
        pdf[target] = mi

    return pdf

sdf_mi = sdf_mi.groupby('Feature').applyInPandas(mutual_information, schema=sdf_mi.schema)

top_features = [list(sdf_mi.orderBy(target, ascending=False).select('Feature').toPandas()['Feature'])[0:5] for target in targets]
df_top_features = pd.DataFrame({'target': targets, 'task':'regression', 'features': top_features})
    
for target in targets:
    top_features = df_top_features[df_top_features['target']==target]['features'].tolist()[0]
    task = df_top_features[df_top_features['target']==target]['task'].tolist()[0]
    sdf = sdf_sensors_prepared[["`" + c + "`" for c in sdf_sensors_prepared.columns if c in top_features + [target] + ['Time']]]
    sdf_train = sdf[sdf['Time'] < args.start_test_date]
    sdf_test = sdf[sdf['Time'] >= args.start_test_date]
    sdf_train.repartition(1).write.parquet(args.output_dataset + '/train/' + task + '/' + target, mode='overwrite')
    sdf_test.repartition(1).write.parquet(args.output_dataset + '/test/' + task + '/' + target, mode='overwrite')