import pandas as pd
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import pylab
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="Agri learn",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.sidebar.title(' 👨‍🌾 Agri learn')

st.sidebar.text('Project By:')
st.sidebar.text('Yash Shah')
st.sidebar.text('Yash Joshi')
st.sidebar.text('Deep Patel')
st.write('''
## 👨‍🌾  Agri learn - Revolutionizing Farming through Advanced Crop Analytics
This app predicts the Best crop which framer should cultivate according to location.
* Prediction of crop by state
* Prediction of crop by region
* Prediction of crop by season
* Best sorting of crops by profit
* Soils composition
***
''')


df = pd.read_excel("region_crop.xlsx")

le = preprocessing.LabelEncoder()
x1 = df["SOIL"]
y1 = df["STATE"]
x = le.fit_transform(x1)
y = le.fit_transform(y1)
df1 = df.merge(pd.DataFrame({'SOIL_C':x,'STATE_C':y}),right_index=True,left_index = True)
df1.head()




def predictmodel(state,region):
    state = str(state)
    region = int(region)
    print("For State : {} and Region : {}".format(state,region))
    ordinary=df1.query('STATE == @state and Region_C == @region')
    reg_soil=list(ordinary[['STATE_C','SOIL_C']].values) #input for classifier
    #output for state and region
    soil_type=ordinary[['SOIL']].values
    print("Soil Types found are : \n",soil_type)
    return reg_soil


basic=pd.read_excel('location_Avail.xlsx')

le2 = preprocessing.LabelEncoder()
x2 = basic["SOIL"]
y2 = basic["STATE"]
a = le2.fit_transform(x2)
b = le2.fit_transform(y2)
basic1 = basic.merge(pd.DataFrame({'SOIL_C':a,'STATE_C':b}),right_index=True,left_index = True)
basic1.head()
X = basic1[['STATE_C','SOIL_C']]
Y = basic1[['Rice','Cotton','Sugarcane','Wheat','Millets','Cardamom','Ginger','Coconut','Orange','Soyabean','Maize']]
classifier = DecisionTreeClassifier()
classifier.fit(X,Y)


def crop_prediction(soil):
    

    a = classifier.predict(soil)
    print(len(a))
    print(a[0][0])
    return a

def crops_detail(state,crop_pred):
    state = str(state)
    rice_predict_yeild = ""
    rice_predict_profit = ""
    rice_predict_cost =  ""
    cotton_predict_yeild = ""
    cotton_predict_profit = ""    
    cotton_predict_cost = ""
    sugarcane_predict_yeild = ""
    sugarcane_predict_profit = ""
    sugarcane_predict_cost = ""
    wheat_predict_yeild = ""
    wheat_predict_profit = ""
    wheat_predict_cost = ""
    millets_predict_yeild = ""
    millets_predict_profit = ""
    millets_predict_cost = ""
    carda_predict_yeild = ""
    carda_predict_profit = ""
    carda_predict_cost =""
    gin_predict_yeild =""
    gin_predict_profit =""
    gin_predict_cost =""
    coco_predict_yeild =""
    coco_predict_profit =""
    coco_predict_cost =""
    orange_predict_yeild =""
    orange_predict_profit =""
    orange_predict_cost =""
    soya_predict_yeild =""
    soya_predict_profit =""
    soya_predict_cost =""
    mai_predict_yeild =""
    mai_predict_profit =""
    mai_predict_cost = ""

    result = {"crop" : [], "Profit" : []}

    for i in range(len(crop_pred)):
      for j in range(11):
        if crop_pred[i][j]:
          if j == 0:
            print("Rice")
            crop="Rice"
            result["crop"].append("Rice")
            data = pd.read_excel('cropdata.xlsx')
            data1=data.query('STATE == @state and CROP == @crop')
            regressor=LinearRegression()
            X=data1[['YEAR']]
            Y=data1['YEILD']
            Z=data1['PROFIT']
            W=data1['COST OF CULTIVATION']
            #predection of yeild
            regressor.fit(X,Y)
            #a1=regressor.score(X,Y)*100
            rice_predict_yeild = regressor.predict([[2018]])
            #predection of profit
            regressor.fit(X,Z)
            #b1=regressor.score(X,Z)*100
            rice_predict_profit = regressor.predict([[2018]])
            #prediction of cost of cultivation
            regressor.fit(X,W)
            #c1=regressor.score(X,W)*100
            rice_predict_cost = regressor.predict([[2018]])
          elif j == 1:
            print("Cotton")
            crop="Cotton"
            result["crop"].append("Cotton")
            data = pd.read_excel('cropdata.xlsx')
            data2=data.query('STATE == @state and CROP == @crop')
            regressor=LinearRegression()
            X=data2[['YEAR']]
            Y=data2['YEILD']
            Z=data2['PROFIT']
            W=data2['COST OF CULTIVATION']
            #prediction of the yeild
            regressor.fit(X,Y)
            cotton_predict_yeild = regressor.predict([[2018]])
            #prediction of the profit
            regressor.fit(X,Z)
            cotton_predict_profit = regressor.predict([[2018]])
            #prediction of cost of cultivation
            regressor.fit(X,W)
            cotton_predict_cost = regressor.predict([[2018]])
          elif j == 2:
                print("Sugarcane")
                crop="Sugarcane"
                result["crop"]="Rice"
                data = pd.read_excel('cropdata.xlsx')
                data3=data.query('STATE == @state and CROP == @crop')
                regressor=LinearRegression()
                X=data3[['YEAR']]
                Y=data3['YEILD']
                Z=data3['PROFIT']
                W=data3['COST OF CULTIVATION']
                #prediction of the yeild
                regressor.fit(X,Y)
                sugarcane_predict_yeild = regressor.predict([[2018]])
                #prediction of the profit
                regressor.fit(X,Z)
                sugarcane_predict_profit = regressor.predict([[2018]])
                #prediction of cost of cultivation
                regressor.fit(X,W)
                sugarcane_predict_cost = regressor.predict([[2018]])
          elif j == 3:
                print("Wheat")
                crop="Wheat"
                result["crop"].append("Wheat")
                data = pd.read_excel('cropdata.xlsx')
                data4=data.query('STATE == @state and CROP == @crop')
                regressor=LinearRegression()
                X=data4[['YEAR']]
                Y=data4['YEILD']
                Z=data4['PROFIT']
                W=data4['COST OF CULTIVATION']
                #prediction of yeild
                regressor.fit(X,Y)
                wheat_predict_yeild = regressor.predict([[2018]])
                #prediction of the profit
                regressor.fit(X,Z)
                wheat_predict_profit = regressor.predict([[2018]])
                #prediction of cost of cultivation
                regressor.fit(X,W)
                wheat_predict_cost = regressor.predict([[2018]])
          elif j == 4:
                print("Millets")
                crop="Millets"
                result["crop"].append("Millets")
                data = pd.read_excel('cropdata.xlsx')
                data5=data.query('STATE == @state and CROP == @crop')
                regressor=LinearRegression()
                X=data5[['YEAR']]
                Y=data5['YEILD']
                Z=data5['PROFIT']
                W=data5['COST OF CULTIVATION']
                #prediction of the yeild
                regressor.fit(X,Y)
                millets_predict_yeild = regressor.predict([[2018]])
                #prediction of the profit
                regressor.fit(X,Z)
                millets_predict_profit = regressor.predict([[2018]])
                #prediction of cost of cultivation
                regressor.fit(X,W)
                millets_predict_cost = regressor.predict([[2018]])
          elif j == 5:
                print("Cardamom")
                crop="Cardamom"
                result["crop"].append("Cardamom")
                data = pd.read_excel('cropdata.xlsx')
                data6=data.query('STATE == @state and CROP == @crop')
                regressor=LinearRegression()
                X=data6[['YEAR']]
                Y=data6['YEILD']
                Z=data6['PROFIT']
                W=data6['COST OF CULTIVATION']
                #prediction of the yeild
                regressor.fit(X,Y)
                carda_predict_yeild = regressor.predict([[2018]])
                #prediction of the profit
                regressor.fit(X,Z)
                carda_predict_profit = regressor.predict([[2018]])
                #prediction of cost of cultivation
                regressor.fit(X,W)
                carda_predict_cost = regressor.predict([[2018]])
          elif j == 6:
                print("Ginger")
                crop="Ginger"
                result["crop"].append("Ginger")
                data = pd.read_excel('cropdata.xlsx')
                data7=data.query('STATE == @state and CROP == @crop')
                regressor=LinearRegression()
                X=data7[['YEAR']]
                Y=data7['YEILD']
                Z=data7['PROFIT']
                W=data7['COST OF CULTIVATION']
                #prediction of the yeild
                regressor.fit(X,Y)
                gin_predict_yeild = regressor.predict([[2018]])
                #prediction of the profit
                regressor.fit(X,Z)
                gin_predict_profit = regressor.predict([[2018]])
                #prediction of cost of cultivation
                regressor.fit(X,W)
                gin_predict_cost = regressor.predict([[2018]])
          elif j == 7:
                print("Coconut")
                crop="Coconut"
                result["crop"].append("Coconut")
                data = pd.read_excel('cropdata.xlsx')
                data8=data.query('STATE == @state and CROP == @crop')
                regressor=LinearRegression()
                X=data8[['YEAR']]
                Y=data8['YEILD']
                Z=data8['PROFIT']
                W=data8['COST OF CULTIVATION']
                #prediction of the yeild
                regressor.fit(X,Y)
                coco_predict_yeild = regressor.predict([[2018]])
                #prediction of the profit
                regressor.fit(X,Z)
                coco_predict_profit = regressor.predict([[2018]])
                #prediction of cost of cultivation
                regressor.fit(X,W)
                coco_predict_cost = regressor.predict([[2018]])
          elif j == 8:
                print("Orange")
                crop="Orange"
                result["crop"].append("Orange")
                data = pd.read_excel('cropdata.xlsx')
                data9=data.query('STATE == @state and CROP == @crop')
                regressor=LinearRegression()
                X=data9[['YEAR']]
                Y=data9['YEILD']
                Z=data9['PROFIT']
                W=data9['COST OF CULTIVATION']
                #prediction of the yeild
                regressor.fit(X,Y)
                orange_predict_yeild = regressor.predict([[2018]])
                #prediction of the profit
                regressor.fit(X,Z)
                orange_predict_profit = regressor.predict([[2018]])
                #prediction of cost of cultivation
                regressor.fit(X,W)
                orange_predict_cost = regressor.predict([[2018]])
          elif j == 9:
                print("Soyabean")
                crop="Soyabean"
                result["crop"].append("Soyabean")
                data = pd.read_excel('cropdata.xlsx')
                data10=data.query('STATE == @state and CROP == @crop')
                regressor=LinearRegression()
                X=data10[['YEAR']]
                Y=data10['YEILD']
                Z=data10['PROFIT']
                W=data10['COST OF CULTIVATION']
                #prediction of the yeild
                regressor.fit(X,Y)
                soya_predict_yeild = regressor.predict([[2018]])
                #prediction of the profit
                regressor.fit(X,Z)
                soya_predict_profit = regressor.predict([[2018]])
                #prediction of cost of cultivation
                regressor.fit(X,W)
                soya_predict_cost = regressor.predict([[2018]])
          elif j == 10:
                print("Maize")
                crop="Maize"
                result["crop"].append("Maize")
                data = pd.read_excel('cropdata.xlsx')
                data11=data.query('STATE == @state and CROP == @crop')
                regressor=LinearRegression()
                X=data11[['YEAR']]
                Y=data11['YEILD']
                Z=data11['PROFIT']
                W=data11['COST OF CULTIVATION']
                #prediction of the yeild
                regressor.fit(X,Y)
                mai_predict_yeild = regressor.predict([[2018]])
                #prediction of the profit
                regressor.fit(X,Z)
                mai_predict_profit = regressor.predict([[2018]])
                #prediction of cost of cultivation
                regressor.fit(X,W)
                mai_predict_cost = regressor.predict([[2018]])

    ## for price sorting
    if rice_predict_yeild != "" :
        print(rice_predict_yeild)
    if rice_predict_profit != "" :
        print(rice_predict_profit)
        result["Profit"].append(int(rice_predict_profit))
    if rice_predict_cost !="" :
        print(rice_predict_cost)

        
    #displaying the prediction data (for COTTON)
    if cotton_predict_yeild != "" :
        print(cotton_predict_yeild)
    if cotton_predict_profit != "":
        print(cotton_predict_profit)
        result["Profit"].append(int(cotton_predict_profit))
    if cotton_predict_cost != "":
        print(cotton_predict_cost)

    #diplaying the prediction data (for SUGARCANE)
    if sugarcane_predict_yeild != "":
        print(sugarcane_predict_yeild)
    if sugarcane_predict_profit !="":
        print(sugarcane_predict_profit)
        result["Profit"].append(int(sugarcane_predict_profit))
    if sugarcane_predict_cost != "":
        print(sugarcane_predict_cost)

    #displaying prediction data (for WHEAT)
    if wheat_predict_yeild != "":
        print(wheat_predict_yeild)
    if wheat_predict_profit !="":
        print(wheat_predict_profit)
        result["Profit"].append(int(wheat_predict_profit))
    if wheat_predict_cost != "":
        print(wheat_predict_cost)

    #displaying prediction data(for MILLETS)
    if millets_predict_yeild != "":
        print(millets_predict_yeild)
    if millets_predict_profit !="":
        print(millets_predict_profit)
        result["Profit"].append(int(millets_predict_profit))
    if millets_predict_cost != "":
        print(millets_predict_cost)
        

    #displaying prediction data(for CARDAMOM)
    if carda_predict_yeild != "":
        print(carda_predict_yeild)
    if carda_predict_profit !="":
        print(carda_predict_profit)
        result["Profit"].append(int(carda_predict_profit))
    if carda_predict_cost != "":
        print(carda_predict_cost)

    #displaying prediction data(for GINGER)
    if gin_predict_yeild != "":
        print(gin_predict_yeild)
    if gin_predict_profit !="":
        print(gin_predict_profit)
        result["Profit"].append(int(gin_predict_profit))
    if gin_predict_cost != "":
        print(gin_predict_cost)
        
    #displaying prediction data(for COCONUT)
    if coco_predict_yeild != "":
        print(coco_predict_yeild)
    if coco_predict_profit !="":
        print(coco_predict_profit)
        result["Profit"].append(int(coco_predict_profit))
    if coco_predict_cost != "":
        print(coco_predict_cost)

    #displaying prediction data(for ORANGE)
    if orange_predict_yeild != "":
        print(orange_predict_yeild)
    if orange_predict_profit !="":
        print(orange_predict_profit)
        result["Profit"].append(int(orange_predict_profit))
    if orange_predict_cost != "":
        print(orange_predict_cost)
        
    #displaying prediction data(for SOYABEAN)
    if soya_predict_yeild != "":
        print(soya_predict_yeild)
    if soya_predict_profit !="":
        print(soya_predict_profit)
        result["Profit"].append(int(soya_predict_profit))
    if soya_predict_cost != "":
        print(soya_predict_cost)
        
    #displaying prediction data(for MAIZE)
    if mai_predict_yeild != "":
        print(mai_predict_yeild)
    if mai_predict_profit !="":
        print(mai_predict_profit)
        result["Profit"].append(int(mai_predict_profit))
    if mai_predict_cost != "":
        print(mai_predict_cost)

    return result    


basic_season=pd.read_excel('season_crop.xlsx')
le2 = preprocessing.LabelEncoder()
x2 = basic_season["SOIL"]
y2 = basic_season["STATE"]
z2 = basic_season["SEASON"]
a = le2.fit_transform(x2)
b = le2.fit_transform(y2)
c= le2.fit_transform(z2)
basic2 = basic_season.merge(pd.DataFrame({'SOIL_C':a,'STATE_C':b,'SEASON_C':c}),right_index=True,left_index = True)
basic2.head()

def season_predict(state,season,region,reg_soil):
    X = basic2[['STATE_C','SOIL_C','SEASON_C']]
    Y = basic2[['Rice','Cotton','Sugarcane','Wheat','Millets','Cardamom','Ginger','Coconut','Orange','Soyabean','Maize']]
    classifier = DecisionTreeClassifier()
    classifier.fit(X,Y)
    if state == "Jammu and Kashmir" :
      st = 0
    elif state == "Kerala" :
      st = 1
    elif state == "Maharashtra" : 
      st = 2
    elif state == "Sikkim" : 
      st = 3
    elif state == "Uttarakhand" : 
      st = 4

    pred_list = np.insert(reg_soil, 2, season, axis = 1)
    crop_pred = classifier.predict(pred_list)
    return crop_pred
         




def run():
    st.header(" 🌾 Crop Prediction")
    tweet_input = st.text_input("Enter the state:",value = "")
    print(tweet_input)


    options = ["Select an option ","North", "West", "South","East","Central","Coastal"]

    selected_option = st.selectbox("Choose your region:", options)
    tweet_region =0
    if selected_option == 'North':
        tweet_region = 1
    if selected_option == 'West':
        tweet_region = 2
    if selected_option == 'South':
        tweet_region = 3
    if selected_option == 'East':
        tweet_region =4
    if selected_option == 'Central':
        tweet_region = 5
    if selected_option == 'Coastal':
        tweet_region =6

    print(tweet_region)
    
    options1 = ["Select an option ","Kharif", "Rabi", "Zaid"]

    season_input = -1

    season_word = st.selectbox("Choose your season:", options1)

    if season_word == "Kharif":
        season_input=0
    if season_word == "Rabi":
        season_input=1
    if season_word == "Zaid":
        season_input=2

    print(season_input)
    st.write('''
    ***
    ''')
    if tweet_region: 
        soil = predictmodel("Sikkim",tweet_region)
        print(soil)
        
        classif = crop_prediction(soil)
        st.header(" 🌾 Crops for "+ str(tweet_input) + " State for region "+ str(selected_option) +" : ")

        for i in range(len(classif)):
            for j in range(11):
                if classif[i][j]:
                    if j==0:
                        print("Rice")
                        st.info("Rice")
                    if j==1:
                        print("Cotton")
                        st.info("Cotton")
                    if j==2:
                        print("Sugarcane")
                        st.info("Sugarcane")
                    if j==3:
                        print("Wheat")
                        st.info("Wheat")
                    if j==4:
                        print("Millets")
                        st.info("Millets")
                    if j==5:
                        print("Cardamom")
                        st.info("Cardamom")
                    if j==6:
                        print("Ginger")
                        st.info("Ginger")
                    if j==7:
                        print("Coconut")
                        st.info("Coconut")

                    if j==8:
                        print("Orange")
                        st.info("Orange")
                    if j==9:
                        print("Soyabean")
                        st.info("Soyabean")
                    if j==10:
                        print("Maize")
                        st.info("Maize")

        st.header(" 💰 Crops for "+ tweet_input + " State with maximum profit :")

        price = crops_detail(tweet_input,classif)
        sorted_crops = sorted(zip(price['crop'], price['Profit']), key=lambda x: x[1],reverse=True)
        print(sorted_crops)
        for crop,profit in sorted_crops:
            st.success("Crop: "+ str(crop) + " \t " +"Profit: "+ str(profit))

        st.header(" ⛅  Crops for "+ tweet_input + " State for season "+ str(season_word) +" :")
        crop_pred = season_predict(tweet_input,season_input,tweet_region,soil)

    #st.info(crop_pred)

        for i in range(len(crop_pred)):
            for j in range(11):
                if crop_pred[i][j]:
                    if j==0:
                        print("Rice")
                        st.info("Rice")
                    if j==1:
                        print("Cotton")
                        st.info("Cotton")
                    if j==2:
                        print("Sugarcane")
                        st.info("Sugarcane")
                    if j==3:
                        print("Wheat")
                        st.info("Wheat")
                    if j==4:
                        print("Millets")
                        st.info("Millets")
                    if j==5:
                        print("Cardamom")
                        st.info("Cardamom")
                    if j==6:
                        print("Ginger")
                        st.info("Ginger")
                    if j==7:
                        print("Coconut")
                        st.info("Coconut")

                    if j==8:
                        print("Orange")
                        st.info("Orange")
                    if j==9:
                        print("Soyabean")
                        st.info("Soyabean")
                    if j==10:
                        print("Maize")
                        st.info("Maize")

    if tweet_input == 'Maharashtra':                    
        image = Image.open('67fce6d5-d45d-4fba-be37-3cb2af039c30.jfif')

        st.image(image, use_column_width= True)
    if tweet_input == 'Kerala':
        image = Image.open('c6ccc894-5e77-4301-878e-2463d445c8e6.jfif')

        st.image(image, use_column_width= True)

    if tweet_input == 'Uttarakhand':
        image = Image.open('cafb356d-870c-4bc5-bf46-89d1fad90a15.jfif')

        st.image(image, use_column_width= True)
    if tweet_input == 'Jammu and Kashmir':
        image = Image.open('c2646f1f-5dac-43ca-83bc-ebef0d543013.jfif')

        st.image(image, use_column_width= True)
    if tweet_input == 'Jammu and Kashmir':
        image = Image.open('e7b6ed90-0c10-4a84-a6ea-d586801968ca.jfif')

        st.image(image, use_column_width= True)



run()