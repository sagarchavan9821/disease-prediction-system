from preprocessing import x,y
from sklearn.model_selection import train_test_split,GridSearchCV   
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,accuracy_score,classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

disease_columns = [
    'heart_disease', 'diabetes', 'stroke', 'kidney_disease',
    'cancer', 'alzheimers_disease', 'copd', 'liver_disease',
    'parkinsons_disease', 'tuberculosis'
]

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42
)

results = {}
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   
X_test  = scaler.transform(X_test) 

for disease in disease_columns:
    # Get single disease column
    y_train_single = y_train[disease]
    y_test_single  = y_test[disease]

    # Logistic Regression
    # lr = LogisticRegression(max_iter=1000)----------------------------------------- THIS MODEL IS NOT GIVING ME GOOD ACCU
    # lr.fit(X_train, y_train_single)
    # lr_acc = accuracy_score(y_test_single, lr.predict(X_test))

    
    # # Decision Tree

    # dt = DecisionTreeClassifier(random_state=42)----------------------------------- THIS MODEL IS NOT GIVING ME GOOD ACCU
    # dt.fit(X_train, y_train_single)
    # dt_acc = accuracy_score(y_test_single, dt.predict(X_test))
    
    # # Random Forest

    # rf = RandomForestClassifier(n_estimators=100,max_depth=None,random_state=42)------THIS MODEL IS NOT GIVING ME GOOD ACCU
    # rf.fit(X_train,y_train_single)
    # rf_acc= accuracy_score(y_test_single,rf.predict(X_test))

    # "SVM"
      
    # svm=SVC(kernel=('rbf'))
    # svm.fit(X_train,y_train_single)
    # sm_acc=accuracy_score(y_test_single,svm.predict(X_test))--------------------------THIS SVM MOdel give best accuracy 
    
    # # gradientBoostying
    # gb=GradientBoostingClassifier(n_estimators=100,learning_rate=0.1,random_state=42)___THIS MODEL IS NOT GIVING ME GOOD ACCU
    # gb.fit(X_train,y_train_single)
    # gb_acc=accuracy_score(y_test_single,gb.predict(X_test))
    
    # results[disease] = {"Logistic Regression": lr_acc, "Decision Tree": dt_acc}


    # print(f"{disease:25} | LR: {lr_acc:.2%} | DT: {dt_acc:.2%} | rf:{rf_acc:.2%} |sm:{sm_acc:.2%} | gb:{gb_acc:.2%}")

 # so i deside to take only one model that  giving me best accuracy that svm

    classifier=GridSearchCV ((SVC()),{     
    'C':[0.1],
    "kernel":['rbf'],
    'gamma':  ['scale' ] 
    
   },cv=5, return_train_score=False)

    classifier.fit(X_train,y_train_single)

    svm_model = classifier.best_estimator_
    acc = accuracy_score(y_test_single, svm_model.predict(X_test))
    print(f"  Best Params : {classifier.best_params_}")
    print(f"  Tuned SVM Accuracy: {acc:.2%}")

    #                                                                                                \
    # classifier=GridSearchCV(RandomForestClassifier(class_weight='balanced', random_state=42),{      |
    #     'n_estimators': [100, 200, 300],                                                            |
    #     'max_depth':    [5, 10, None],                                                                                                                                                                    
    #                                                                                                 |                                                                                                       |  
    # },cv=5 ,return_train_score=False, scoring='f1',                                                                                                      
    #                                                                                                 |
    # n_jobs=-1)                                                                                                      
    #                                                                                                 |
                                                                                                      
    #                                                                                                 |----------- this gives me bad ACCURaCY  
    # classifier.fit(X_train,y_train_single)                                                                                                      
    #                                                                                                 |
                                                                                                      
    #                                                                                                 |
    # rf_model =classifier.best_estimator_                                                                                                      
    #                                                                                                 |
    # acc=accuracy_score(y_test_single,rf_model.predict(X_test))                                                                                                      
    #                                                                                                 |
    # print(f"  best params:{classifier.best_params_}")                                                                                                      
    #                                                                                                 |
    # print(f"tuned rf accuracy :{acc:0.2%} ")                                                        /                                             |