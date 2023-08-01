import streamlit as st  
import numpy as np 
import pandas as pd
import pickle

z=pd.read_csv('Cu_regressor_features.csv')

s=st.sidebar.selectbox('Choose one',['Copper price prediction','Status prediction'])
if s=='Copper price prediction':
    def load_model():
        with open('saved_steps_regressor.pkl','rb') as file:
            data=pickle.load(file)
        return data

    data=load_model()

    regressor=data['model']
    mean_enc=data['mean_enc']
    ordinal_enc=data['ordinal_enc']
    #def show_predict_page():
    st.title("COPPER SELLING PRICE PREDICTION")
    st.write("""### We need some information to predict selling price""")

    # Storing all user input 
    input_dict={'item_date':'','customer':'','status':'','application':'','thickness':'',
                'width':'','material_ref':'','product_ref':'','delivery date':''}

    form = st.form("form1")
    d1=form.selectbox('Enter item_date',([i for i in range(1,32)]))
    m1=form.selectbox('Enter item_month',([i for i in range(1,13)]))
    y1=form.selectbox('Enter item_year',([2020,2021]))



        





            

    cstmr=form.selectbox('Choose customerID',list(z['customer'].apply(lambda x:round(float(x))).sort_values().unique()))
    input_dict['customer']=str(cstmr)+'.0'

    sts=form.selectbox('Choose Status',list(z['status'].unique()))
    input_dict['status']=sts 

    app=form.selectbox('Choose applicationID',list(z['application'].unique()))
    input_dict['application']=str(app)

    thick=form.selectbox('Choose thickness',list(z['thickness'].apply(lambda x:(float(x))).sort_values().unique()))
    input_dict['thickness']=str(thick)

    #wid=st.selectbox('Choose width size',list(z['width'].apply(lambda x:(float(x))).sort_values().unique()))   
    #input_dict['width']=str(wid) 

    wid=form.selectbox('Choose width size',list(z['width'].apply(lambda x:(float(x))).sort_values().unique()))   
    input_dict['width']=str(wid)    

    mat=form.selectbox('Choose material_ref',list(z['material_ref'].unique()))
    input_dict['material_ref']=str(mat)

    pro=form.selectbox('Choose product_ref',list(z['product_ref'].unique()))
    input_dict['product_ref']=str(pro)

    #ok=form.form_submit_button("Submit")    

            
    d2=form.selectbox('Enter delivery date',([i for i in range(1,32)]))
    m2=form.selectbox('Enter delivery month',([i for i in range(1,13)]))
    y2=form.selectbox('Enter delivery year',([2019,2020,2021,2022]))

    ok=form.form_submit_button("Submit") #   
    if ok:
        
        if d1>29 and m1==2 and y1==2020:
            st.error('February does not have more than 29 days(item_date)')
        elif d1>28 and m1==2 and y1==2021:
            st.error('February does not have more than 28 days(item_date)')
        elif d1>30 and (m1==4 or m1==6 or m1==9 or m1==11):
            st.error('The selected month has only 30 days(item_date)')

        else:
            v=[]
            v.append(str(y1))
            
            if len(str(m1))==1:
                v.append('0'+(str(m1)))
            elif len(str(m1))==2:
                v.append(str(m1)) 

            if len(str(d1))==1:
                v.append('0'+(str(d1)))
            elif len(str(d1))==2:
                v.append(str(d1))
                
            
            v.append('.0')
                
            input_dict['item_date']=''.join(v)
        
        
        if d2>29 and m2==2 and y2==2020:
            st.error('February does not have more than 29 days(del_date)')
        elif d2>28 and m2==2 and (y2==2019,y2==2021,y2==2022):
            st.error('February does not have more than 28 days(del_date)')
        elif d2>30 and (m2==4 or m2==6 or m2==9 or m2==11):
            st.error('The selected month has only 30 days(del_date)')

        else:
            v2=[]
            v2.append(str(y2))
            
            if len(str(m2))==1:
                v2.append('0'+(str(m2)))
            elif len(str(m2))==2:
                v2.append(str(m2)) 

            if len(str(d2))==1:
                v2.append('0'+(str(d2)))
            elif len(str(d2))==2:
                v2.append(str(d2))
                
            
            v2.append('.0')
                
            input_dict['delivery date']=''.join(v2)
            
            #st.write(input_dict)
            
            A=pd.DataFrame(input_dict,index=[1])
            A=mean_enc.transform(A)
            A=ordinal_enc.transform(A)
            y_pred=regressor.predict(A)
            u=str(round(y_pred[0],2))+'/-'
            st.write(f'### Predicted Copper Selling Price is : {u}')
            
else:
    z=pd.read_csv('Cu_classifier_features.csv')
    def load_model():
        with open('saved_steps_classifier.pkl','rb') as file:
            data=pickle.load(file)
        return data

    data=load_model()

    classifier=data['model']
    mean_enc=data['mean_enc']
    scaler=data['scaler']
    
    #def show_predict_page():
    st.title("COPPER STATUS PREDICTION")
    st.write("""### We need some information to predict status""")

    # Storing all user input 
    input_dict={'item_date':'','country':'','item type':'','customer':'',
                'application':'','width':'','material_ref':'','product_ref':'',
                'delivery date':'','selling_price':''}

    form = st.form("form1")
    d1=form.selectbox('Enter item_date',([i for i in range(1,32)]))
    m1=form.selectbox('Enter item_month',([i for i in range(1,13)]))
    y1=form.selectbox('Enter item_year',([2020,2021]))



    cnty=form.selectbox('Choose country',list(z['country'].unique()))
    input_dict['country']=str(cnty)   

    ity=form.selectbox('Choose item_type',list(z['item type'].unique()))
    input_dict['item type']=str(ity)

    cstmr=form.selectbox('Choose customerID',list(z['customer'].apply(lambda x:round(float(x))).sort_values().unique()))
    input_dict['customer']=str(cstmr)+'.0'

    app=form.selectbox('Choose applicationID',list(z['application'].unique()))
    input_dict['application']=str(app)

    

    #wid=st.selectbox('Choose width size',list(z['width'].apply(lambda x:(float(x))).sort_values().unique()))   
    #input_dict['width']=str(wid) 

    wid=form.selectbox('Choose width size',list(z['width'].apply(lambda x:(float(x))).sort_values().unique()))   
    input_dict['width']=str(wid)    

    mat=form.selectbox('Choose material_ref',list(z['material_ref'].unique()))
    input_dict['material_ref']=str(mat)

    pro=form.selectbox('Choose product_ref',list(z['product_ref'].unique()))
    input_dict['product_ref']=str(pro)

    #sell=form.selectbox('Choose selling price',list(z['selling_price'].apply(lambda x:(float(x))).sort_values().unique()))
    #input_dict['selling_price']=str(sell)
    #ok=form.form_submit_button("Submit")    
    sell=form.slider('Choose selling price',min_value=-1160,max_value=6000)
    input_dict['selling_price']=str(sell)
            
    d2=form.selectbox('Enter delivery date',([i for i in range(1,32)]))
    m2=form.selectbox('Enter delivery month',([i for i in range(1,13)]))
    y2=form.selectbox('Enter delivery year',([2019,2020,2021,2022]))

    ok=form.form_submit_button("Submit") #   
    if ok:
        
        if d1>29 and m1==2 and y1==2020:
            st.error('February does not have more than 29 days(item_date)')
        elif d1>28 and m1==2 and y1==2021:
            st.error('February does not have more than 28 days(item_date)')
        elif d1>30 and (m1==4 or m1==6 or m1==9 or m1==11):
            st.error('The selected month has only 30 days(item_date)')

        else:
            v=[]
            v.append(str(y1))
            
            if len(str(m1))==1:
                v.append('0'+(str(m1)))
            elif len(str(m1))==2:
                v.append(str(m1)) 

            if len(str(d1))==1:
                v.append('0'+(str(d1)))
            elif len(str(d1))==2:
                v.append(str(d1))
                
            
            v.append('.0')
                
            input_dict['item_date']=''.join(v)
        
        
        if d2>29 and m2==2 and y2==2020:
            st.error('February does not have more than 29 days(del_date)')
        elif d2>28 and m2==2 and (y2==2019,y2==2021,y2==2022):
            st.error('February does not have more than 28 days(del_date)')
        elif d2>30 and (m2==4 or m2==6 or m2==9 or m2==11):
            st.error('The selected month has only 30 days(del_date)')

        else:
            v2=[]
            v2.append(str(y2))
            
            if len(str(m2))==1:
                v2.append('0'+(str(m2)))
            elif len(str(m2))==2:
                v2.append(str(m2)) 

            if len(str(d2))==1:
                v2.append('0'+(str(d2)))
            elif len(str(d2))==2:
                v2.append(str(d2))
                
            
            v2.append('.0')
                
            input_dict['delivery date']=''.join(v2)
            
            #st.write(input_dict)
            
            A=pd.DataFrame(input_dict,index=[1])
            A=mean_enc.transform(A)
            A=scaler.transform(A)
            y_pred=classifier.predict(A)
            u=(y_pred[0])
            if u==1:
                st.success('### Predicted Copper status : "WON"') 
            else:        
                st.error('### Predicted Copper status : "LOST"')