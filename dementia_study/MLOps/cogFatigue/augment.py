from cf_libs import *
from cf_helper import *
from dataGen_10 import featuresBND,features6ND,features12ND,featuresBMMD,features6MMD,features12MMD,featuresBSD,features6SD,features12SD

def pipeline(clf_name,clf,traindata,testdatasets,testdatasets_names):
    def params(model_name,model,trainset,testset):
        #   prepare trainset
        features = trainset.columns.values.tolist()
        features = trainset.columns.values.tolist()
        trainset.drop(['index'], axis=1, inplace=True)
        trainset.fillna(trainset.mean(), inplace=True)
        #   prepare testset
        testset = testset.dropna()
        features = testset.columns.values.tolist()
        testset.drop(['index'], axis=1, inplace=True)
        #   seperate features and target
        xtestset,ytestset = testset.drop(['labels'], axis=1), testset['labels'].ravel()
        xtrainset = trainset.drop(['labels'], axis=1)
        ytrainset = trainset['labels'].ravel()
        #   normalize
        norm = MinMaxScaler()
        xtrainset_norm = norm.fit_transform(xtrainset)
        xtestset_norm = norm.transform(xtestset)
        xtrainset_norm = pd.DataFrame(xtrainset_norm)
        xtrainset_norm.columns = features[1:-1]
        xtrainset_norm.isnull().sum()
        #   pca
        xtrainset_norm = xtrainset_norm.values
        pca = PCA(n_components=0.99,random_state=0)
        pca.fit(xtrainset_norm)
        xtrainset_pca = pca.transform(xtrainset_norm)
        xtestset_pca = pca.transform(xtestset_norm)
        #  fit model
        model_norm = model.fit(xtrainset_norm,ytrainset)
        model_pca = model.fit(xtrainset_pca,ytrainset)
        #   predict
        ypred_norm = model_norm.predict(xtestset_norm)
        ypred_pca = model_pca.predict(xtestset_pca)
        #   evaluate
        acc_norm = accuracy_score(ytestset, ypred_norm)
        tn, fp, fn, tp = confusion_matrix(ytestset, ypred_norm).ravel()
        spec_norm = tn/(tn+fp)
        sens_norm = tp/(tp+fn)
        acc_pca = accuracy_score(ytestset, ypred_pca)
        tn, fp, fn, tp = confusion_matrix(ytestset, ypred_pca).ravel()
        spec_pca = tn/(tn+fp)
        sens_pca = tp/(tp+fn)
        perf_norm = pd.DataFrame ({"Norm-Accuracy":acc_norm, "Norm-Specificity": spec_norm, "Norm-Sensitivity": sens_norm},index=[model_name])
        perf_pca = pd.DataFrame ({"PCA-Accuracy":acc_pca, "PCA-Specificity": spec_pca, "PCA-Sensitivity": sens_pca},index=[model_name])
        display(perf_norm)
        display(perf_pca)
        return perf_norm,perf_pca

    output = []
    for i in range(len(testdatasets)):
        print("Processing",testdatasets_names[i])
        output.append(params(clf_name,clf,traindata,testdatasets[i]))
    return output

results = pipeline(clf_name="Random Forest",clf=RandomForestClassifier(random_state=0),
                   traindata=pd.concat([featuresBND,features6ND,features12ND],axis=0),
                   testdatasets=[featuresBND,features6ND,features12ND,featuresBMMD,features6MMD,features12MMD,featuresBSD,features6SD,features12SD],
                   testdatasets_names=["BND","6ND","12ND","BMMD","6MMD","12MMD","BSD","6SD","12SD"])
