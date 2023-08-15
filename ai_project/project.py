import numpy as np
import pandas as pd
import time
import matplotlib
import xgboost as xgb
 
# data/SME_SCORING_MODEL_DATA_RECENT (2).xlsx
 
read_table_name = input("Enter the name of the table that you want to use: ")
 
 
df = pd.read_excel(read_table_name, decimal=',')
df['TEKLIF_TAR']=pd.to_datetime(df['TEKLIF_TAR'], format='%d.%m.%Y')
 
# adding categorical column
country_names=['Turkey','France','Kosovo','England']
rand_ind=np.random.randint(0,len(country_names), len(df))
df['Country']=np.array(country_names)[rand_ind]
 
print("\n")
print("Here is the first five rows from the table that you choose:")
print(df.head())
print("\n")
 
 
oto_remover_ans = input("Do you want automatic remover (removes fully empty columns or columns that have only one value): [YES/NO]")
case_sens_oto_ans = oto_remover_ans.lower()
 
if case_sens_oto_ans == "yes":
    gereksiz_sutunlar = []
    for sutun in df.columns:
        if df[sutun].nunique() <= 1 or df[sutun].isnull().all():
            gereksiz_sutunlar.append(sutun)
 
    ayrilan_sutunlar = [sutun for sutun in df.columns if sutun not in gereksiz_sutunlar]
 
    # Ayrılan sütunları görüntüleme
    print("\nRemoved Columns: ")
    for sutun in gereksiz_sutunlar:
        print(sutun)
 
    print("\nTable with remaining columns:")
    df = df.drop(columns=gereksiz_sutunlar)
    print(df.head())
 
    cutoff_value_tf = input("Do you want to define a cutoff value: [YES/NO]")
    last_version_cutoff = cutoff_value_tf.lower()
 
    if last_version_cutoff == "yes":
        cutoff_value = float(input("Give the cutoff value that you want:"))
        for i in ayrilan_sutunlar:
            empty_per = df[i].isnull().mean()
            if empty_per > cutoff_value:
                df.drop(i, axis=1, inplace=True)
        print(df.head())
 
    else:
        manual_ans = input("Do you want manual remove operation: [YES/NO] ")
        f_version_manualAns = manual_ans.lower()
 
        if f_version_manualAns == "yes":
            columns_to_remove = []
            while True:
                remove_column_inp = input("Please write the column name (To quit from this function write q or Q): ")
                if remove_column_inp.upper() == "Q":
                    break
                if remove_column_inp not in df:
                    print("Invalid column name!")
                else:
                    columns_to_remove.append(remove_column_inp)
 
            df = df.drop(columns=columns_to_remove)
            print("New table:")
            print(df.head())
 
else:
    print("SIKINTISIZ MANUEL KISIMDAN DEVAM!")
    manual_ans = input("Do you want manual remove operation: [YES/NO] ")
    f_version_manualAns = manual_ans.lower()
 
    if f_version_manualAns == "yes":
        columns_to_remove = []
        while True:
            remove_column_inp = input("Please write the column name (To quit from this function write q or Q): ")
            if remove_column_inp.upper() == "Q":
                break
            if remove_column_inp not in df.columns:
                print("Invalid column name!")
            else:
                columns_to_remove.append(remove_column_inp)
 
        df = df.drop(columns=columns_to_remove)
        print("New table:")
        print(df.head())
 
    else:
        print("Here is the table: ")
        print(df.head())
 
print(df.head())
 
print(df.dtypes)
 
df.head()
 
# cmusno ve krdteklifno
clist=['CMUSNO', 'KRDTEKLIFNO', 'TEKLIF_TAR']
df_new=df.drop(columns=clist)
df_cmusno_krd=df[clist]
df_cmusno_krd.head()
 
df_new.head()
 
 
# numeric and categorical values
num_cols=df_new.select_dtypes(include=['int64', 'float64', 'datetime64[ns]']).columns
cat_cols=df_new.select_dtypes(include=['object']).columns
 
num_cols=num_cols.drop('GOOD_BAD_FLAG')
 
 
num_cols
 
cat_cols
 
import sklearn as sk
from sklearn.model_selection import train_test_split
 
# separating the inputs and the target
X=df_new.drop(columns=['GOOD_BAD_FLAG'])
Y=df_new['GOOD_BAD_FLAG']
X.head()
 
 
percentage=float(input("Enter the percentage of data for the test set (ex: 20 for 20%)")) / 100
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=percentage, random_state=42)
 
print("Training set size:", len(X_train))
print("Test set size:", len(X_test))
 
 
X_train.shape
 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
 
num_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='mean')),
    ('scale',MinMaxScaler())
])
cat_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('one-hot',OneHotEncoder(handle_unknown='ignore', sparse=False))
])
 
 
col_trans = ColumnTransformer(transformers=[
    ('num_pipeline',num_pipeline,num_cols),
    ('cat_pipeline',cat_pipeline,cat_cols)
    ],
    remainder='drop',
    n_jobs=-1)
 
 
X_train_transformed=col_trans.fit_transform(X_train, y=Y_train)
print("Transformed training set shape: ", X_train_transformed.shape)
 
 
# test set
 
X_test_transformed=col_trans.transform(X_test)
 
print("Transformed test set shape: ", X_test_transformed.shape)
 
from sklearn.ensemble import RandomForestClassifier
import shap
 
 
 
 
# Başlangıç zamanını al
start_time = time.time()
 
model = RandomForestClassifier()
model.fit(X_train_transformed, Y_train)
 
importance_scores = model.feature_importances_
 
feature_importance = dict(zip(X.columns, importance_scores))
 
# features by importance in descending order
sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
 
# top 10 features
top_features = [feature for feature, _ in sorted_features[:10]]
 
for feature, importance in sorted_features[:10]:
    print(f"Feature: {feature}, Importance: {importance}")
 
# Bitiş zamanını al
end_time = time.time()
 
# Geçen süreyi hesapla
elapsed_time = end_time - start_time
 
print(f"Geçen süre: {elapsed_time:.5f} saniye")  # Saniye cinsinden geçen süre
 
 
x_train_dftmp=pd.DataFrame(X_train_transformed)
x_train_dftmp.head()
 
 
# Başlangıç zamanını al
start_time = time.time()
 
from sklearn.inspection import permutation_importance
 
model2 = RandomForestClassifier()
model2.fit(X_train_transformed, Y_train)
res=permutation_importance(model2, X_train_transformed, Y_train, n_repeats=10, random_state=42)
imp_scores=res.importances_mean
feature_imp=dict(zip(X_test.columns, imp_scores))
sorted_f=sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)
top=[feature for feature, _ in sorted_f[:10]]
for feature, importance in sorted_f[:10]:
    print(f"Feature: {feature}, Importance: {importance}")
    
# Bitiş zamanını al
end_time = time.time()
 
# Geçen süreyi hesapla
elapsed_time = end_time - start_time
 
print(f"Geçen süre: {elapsed_time:.5f} saniye")  
 
 
print(X_train_transformed)
 
 
# shap
 
start_time = time.time()
 
num_rows=X_train_transformed.shape[0]
rand_indices=np.random.choice(num_rows, size=5000, replace=False)
 
subset=X_train_transformed[rand_indices, :]
Y_train_subset=np.random.choice(Y_train, size=5000, replace=False)
 
 
test_model=RandomForestClassifier()
test_model.fit(subset, Y_train_subset)
 
explainer=shap.Explainer(test_model)
shap_values=explainer.shap_values(subset)
shap.summary_plot(shap_values,subset)
 
 
end_time = time.time()
elapsed_time = end_time - start_time
 
print(f"Geçen süre: {elapsed_time:.5f} saniye")
 
import pickle
 
 
from sklearn.model_selection import GridSearchCV
 
paramGrid={"subsample":[0.5, 0.8],'min_child_weight': [1, 5, 10], 'gamma': [0.5, 1, 1.5, 2, 5], 'subsample': [0.6, 0.8, 1.0], 'colsample_bytree': [0.6, 0.8, 1.0], 'max_depth': [3, 4, 5]}
 
 
xgmodel=xgb.XGBClassifier(eval_metric='error')
gridsearch=GridSearchCV(xgmodel, paramGrid, cv=5)
 
xtrain, x_eval, ytrain, y_eval=train_test_split(X_train_transformed, Y_train, test_size=0.2)
gridsearch.fit(X_train_transformed, Y_train, early_stopping_rounds=42, eval_set=[(x_eval, y_eval)])
 
 
print("best parameters:", gridsearch.best_params_)
print("best score:",gridsearch.best_score_)
 
best_model=gridsearch.best_estimator_
 
 
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
 
 
# precision, recall
 
best_parameters={'colsample_bytree': 0.8, 'gamma': 2, 'max_depth': 4, 'min_child_weight': 5, 'subsample': 0.8}
 
xgboostmodel=xgb.XGBClassifier(colsample_bytree= 0.8, gamma= 2, max_depth= 4, min_child_weight= 5, subsample= 0.8)
 
#
 
xgboostmodel.fit(X_train_transformed, Y_train)
 
Y_pred=xgboostmodel.predict(X_test_transformed)
 
 
print(classification_report(Y_test, Y_pred))
 
 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)
cm
 
 
 
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred, labels=xgboostmodel.classes_)
color = 'white'
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=xgboostmodel.classes_)
disp.plot()
plt.show()
 
from sklearn.metrics import recall_score
 
print('macro', recall_score(Y_test, Y_pred, average='macro') ) 
 
# #>>> recall_score(y_true, y_pred, average='micro')  
# 0.33...
# >>> recall_score(y_true, y_pred, average='weighted')  
# 0.33...
# >>> recall_score(y_true, y_pred, average=None)
# array([ 1.,  0.,  0.])
 
 
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
print(precision_score(Y_test, Y_pred, average='macro') ) 
print(accuracy_score(Y_test, Y_pred)) 
print(recall_score(Y_test, Y_pred, average='weighted') ) 
print(f1_score(Y_test, Y_pred,average='macro' ) )
 
 
 
feature_set1=[15,14,5,7,2,0,4,16,1,17]
feature_set2=[6, 8, 11, 5, 15, 16, 0,7, 3, 2]
feature_set3=[0,7, 14, 15, 11, 5, 2, 8, 6,13]
 
 
X_train1= X_train_transformed[feature_set1]
X_test1= X_test_transformed[feature_set1]
 
 
X_train2= X_train_transformed[feature_set2]
X_test2= X_test_transformed[feature_set2]
 
 
X_train3= X_train_transformed[feature_set3]
X_test3= X_test_transformed[feature_set3]
                            
estF1=xgboostmodel.fit(X_train1, Y_train)
estF2=xgboostmodel.fit(X_train2, Y_train)
estF3=xgboostmodel.fit(X_train3, Y_train)
                            
 
accuracy_list=[]
prec_list=[]
f1_list=[]
recall_list=[]
 
 
X_train_transformed=pd.DataFrame(X_train_transformed)
X_test_transformed=pd.DataFrame(X_test_transformed)
 
 
 
import pickle
 
with open('data_train.pkl', 'wb') as f:  # open a text file
    pickle.dump(X_train_transformed, f) # serialize the list
 
 
 
with open('data_test.pkl', 'wb') as f:  # open a text file
    pickle.dump(X_test_transformed, f) # serialize the list
 
 
with open('data_test.pkl', 'rb') as f:  # open a text file
    data_test=pickle.load(f) #
 
with open('data_target_test.pkl', 'wb') as f:  # open a text file
    pickle.dump(Y_test, f) # serialize the list
 
 
with open('data_target_test.pkl', 'rb') as f:  # open a text file
    data__target_test=pickle.load(f) #
 
with open('data_target_train.pkl', 'wb') as f:  # open a text file
    pickle.dump(Y_train, f) # serialize the list
 
 
with open('data_target_train.pkl', 'rb') as f:  # open a text file
    data_target_train=pickle.load(f) #
 
 
accuracy_listp=[]
prec_listp=[]
f1_listp=[]
recall_listp=[]
 
 
feature_set1=[15,14,5,7,2,0,4,16,1,17]
feature_set2=[6, 8, 11, 5, 15, 16, 0,7, 3, 2]
feature_set3=[0,7, 14, 15, 11, 5, 2, 8, 6,13]
 
 
X_train1p= X_train_transformed.iloc[:10000, feature_set1]
X_test1p= X_test_transformed.iloc[:,feature_set1]
 
estF1p=xgboostmodel.fit(X_train1, Y_train)
y_predf1=estF1.predict(X_test1)
prec_list.append(precision_score(Y_test,y_predf1))
 
 
X_train2p= X_train_transformed.iloc[:,feature_set2]
X_test2p= X_test_transformed.iloc[:,feature_set2]
estF2p=xgboostmodel.fit(X_train2, Y_train)
y_predf2=estF2.predict(X_test2)
prec_list.append(precision_score(Y_test,y_predf2))
 
 
 
 
X_train3p= X_train_transformed.iloc[:,feature_set3]
X_test3p= X_test_transformed.iloc[:,feature_set3]
                            
 
estFp3=xgboostmodel.fit(X_train3, Y_train)
 
y_predf3=estF3.predict(X_test3)
prec_list.append(precision_score(Y_test,y_predf3))
 
 
Y_test.shape
 
 
y_predf1.shape
 
X_test1.shape
 
X_test_transformed
 
 
 
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import time
import matplotlib
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
import pickle
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
 
 
 
with open('data_train.pkl', 'rb') as f:  # open a text file
    data_train=pickle.load(f) 
    
with open('data_test.pkl', 'rb') as f:  # open a text file
    data_test=pickle.load(f) #
 
 
with open('data_target_test.pkl', 'rb') as f:  # open a text file
    data__target_test=pickle.load(f)
 
with open('data_target_train.pkl', 'rb') as f:  # open a text file
    data_target_train=pickle.load(f) #
 
best_parameters={'colsample_bytree': 0.8, 'gamma': 2, 'max_depth': 4, 'min_child_weight': 5, 'subsample': 0.8}
 
xgboostmodel=xgb.XGBClassifier(colsample_bytree= 0.8, gamma= 2, max_depth= 4, min_child_weight= 5, subsample= 0.8)
#
 
 
accuracy_list=[]
prec_list=[]
f1_list=[]
recall_list=[]
 
feature_set1=[15,14,5,7,2,0,4,16,1,17]
feature_set2=[6, 8, 11, 5, 15, 16, 0,7, 3, 2]
feature_set3=[0,7, 14, 15, 11, 5, 2, 8, 6,13]
 
 
X_train1= data_train.iloc[:10000, feature_set1]
X_test1= data_test.iloc[:,feature_set1]
 
estF1=xgboostmodel.fit(X_train1, data_target_train[:10000])
y_predf1=estF1.predict(X_test1)
prec_list.append(precision_score(data__target_test, y_predf1))
accuracy_list.append(accuracy_score(data__target_test, y_predf1))
f1_list.append(f1_score(data__target_test, y_predf1))
recall_list.append(recall_score(data__target_test, y_predf1))
 
feature_set1=[15,14,5,7,2,0,4,16,1,17]
feature_set2=[6, 8, 11, 5, 15, 16, 0,7, 3, 2]
feature_set3=[0,7, 14, 15, 11, 5, 2, 8, 6,13]
 
 
X_train2= data_train.iloc[:10000, feature_set2]
X_test2= data_test.iloc[:,feature_set2]
 
estF2=xgboostmodel.fit(X_train2, data_target_train[:10000])
y_predf2=estF2.predict(X_test2)
prec_list.append(precision_score(data__target_test, y_predf2))
accuracy_list.append(accuracy_score(data__target_test, y_predf2))
f1_list.append(f1_score(data__target_test, y_predf2))
recall_list.append(recall_score(data__target_test, y_predf2))
 
 
feature_set1=[15,14,5,7,2,0,4,16,1,17]
feature_set2=[6, 8, 11, 5, 15, 16, 0,7, 3, 2]
feature_set3=[0,7, 14, 15, 11, 5, 2, 8, 6,13]
 
 
X_train3= data_train.iloc[:10000, feature_set3]
X_test3= data_test.iloc[:,feature_set3]
 
estF3=xgboostmodel.fit(X_train3, data_target_train[:10000])
y_predf3=estF3.predict(X_test3)
prec_list.append(precision_score(data__target_test, y_predf3))
accuracy_list.append(accuracy_score(data__target_test, y_predf3))
f1_list.append(f1_score(data__target_test, y_predf3))
recall_list.append(recall_score(data__target_test, y_predf3))
 
print(accuracy_list)
print(prec_list)
print(f1_list)
print(recall_list)
 
 
import matplotlib.pyplot as plt
import numpy as np
 
 
 
 
 
categories = ['SHAP','Permutation Importance', 'Random Forest']
y_values = [0.9143122293256342, 0.9373582181893174,  0.9381831305423799]
 
bar_width = 0.5
bar_positions = np.arange(len(categories))
 
color_palette = ['r','g','b']
 
plt.bar(bar_positions, y_values, width = bar_width, color = color_palette)
 
plt.xlabel('Feature Importances')
plt.ylabel('Accuracy Values')
 
plt.xticks(bar_positions, categories)
 
plt.title('Accuracy/Feature Importances')
plt.grid(True)
 
plt.ylim(0.80, 1.00)
plt.show()
 
 
read_table_name = input("Enter the name of the table that you want to use: ")
 
 
df = pd.read_excel(read_table_name, decimal=',')
 
 
 
value_counts = df['GOOD_BAD_FLAG'].value_counts()
 
 
zeros_count = value_counts[0]
ones_count = value_counts[1]
 
 
plt.figure(figsize=(6, 6))
plt.pie([zeros_count, ones_count], labels=['NUMBER OF ZEROS', 'NUMBER OF ONES'], autopct='%1.1f%%', startangle=140, colors=['skyblue', 'lightgreen'])
plt.title('Sütun: GOOD_BAD_FLAG\nNUMBER OF 0 AND 1')
plt.axis('equal')
 
plt.show()
