# copper_modelling_regression_classification
- Download all the files in this repository
- The aim of this project is to predict copper selling price(regression) and copper status(classification) and create a user interactable app using streamlit to predict copper selling price and status.
- Create a new environment called 'ml_coppermodelling' using conda prompt.Ref link [https://www.youtube.com/watch?v=xl0N7tHiwlw&t=1806s]
- Run **copper_modelling_regressor_model_final.ipynb** file,you will get **saved_steps_regressor.pkl** file.(TO PREDICT COPPER SELLING PRICE)
- Similarly run **copper_modelling_classifier_model_final.ipynb** file,you will get **saved_steps_classifier.pkl** file.(TO PREDICT COPPER STATUS)
- The other .ipynb files(*CopperModelling_1,CopperModelling_2,copper_modelling_classifier*) contains Data preprocessing,various model training and evaluation,feature selections I did before finding out the best model.
- - ## Regression R2 score Table
|    Model             |  Train(R2-score)   |  Test(R2-score)   |
| :------------------- | -----------------  |-----------------: |
| Linear Regression    |      0.727         |0.707              |
| (Lin-reg,Yeo-Johnson)|      0.731         |0.710              |
| LassoCV              |      0.731         |0.710              |
| RidgeCV              |      0.731         |0.710              |
| ElasticNetCV         |      0.731         |0.710              |
| Polynomial Reg(deg-2)|      0.833         |0.820              |
| Polynomial Reg(deg-3)|      0.898         |0.888              |
| Polynomial Reg(deg-4)|      0.908         |0.893              |
| DecisionTree         |      0.959         |0.933              |
| RandomForest         |      0.958         |0.940              |
|ExtraTreesRegressor|1.00        |0.931  
|HistGradientBoostingRegressor|0.989        |0.969              |
- ## Classification ROCAUC score and Accuracy Table
|    Model             |  Train(ROC-AUC)   |  Test(ROC-AUC)   |Accuracy
| :------------------- | -----------------  |-----------------|-----------------: 
| RandomForestClassifier    |      0.972         |0.951             |0.871
| ExtraTreesClassifier|      0.983         |0.961              |0.895
| XGBClassifier             |      0.974         |0.938              |0.871
| HistGradientClassifier            |      0.984         |0.951              |0.899
# HistGradientBoostingRegressor and  HistGradientBoostingClassifier models performance is good
- Now launch vscode using the 'ml_coppermodelling' environment.
- On the VScode terminal,type 'streamlit run prediction_app.py'
- The app is now deployed
- ## Screenshots from the app

|![Screenshot (238)](https://github.com/Kav1n-Lal/copper_modelling_regression_classification/assets/116146011/73cde17f-507d-4cd5-86a7-8af77360db60)
![Screenshot (239)](https://github.com/Kav1n-Lal/copper_modelling_regression_classification/assets/116146011/fb726a75-f694-4035-b574-75e5d2301382)
![Screenshot (240)](https://github.com/Kav1n-Lal/copper_modelling_regression_classification/assets/116146011/2edc4b74-e8d6-491d-bd2c-302efb27a530)
![Screenshot (241)](https://github.com/Kav1n-Lal/copper_modelling_regression_classification/assets/116146011/3048b251-d78f-41db-a3b8-d1c93ad5f631)


