from PIL import Image
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static

df = pd.read_csv("dataset.csv")
df = df.dropna(axis = 0, how = 'any')
df['Date'] = pd.to_datetime(df.Date, dayfirst=True)
df['Date'] = df['Date'].dt.strftime('%d/%m/%Y')

df['Kids_Category'] = df['Kids_Category'].str.replace(" ", "")

df2 = pd.read_csv("data.csv")
df2['time'] = pd.to_datetime(df2.time, dayfirst=True)
df2['time'] = df2['time'].dt.strftime('%d/%m/%Y')

df2.rename(columns = {'time' : 'Date'}, inplace = True)
data = pd.merge(df, df2, on = 'Date', how = 'left')
data['temperature_median'] = round((data['temperature_max'] + data['temperature_min']) / 2, 1)

st.set_page_config(page_title = 'Webpage', page_icon = 'tada', layout = 'wide')

#----assets----
image_flowchart = Image.open('images/flow.png')
image_sales = Image.open('images/sales.png')
image_time = Image.open('images/time.png')
image_races = Image.open('images/races.png')
image_freq = Image.open('images/freq.png')
arm_top = Image.open('images/top.jpg')
arm_tail = Image.open('images/tail.jpg')
supp_conf = Image.open('images/support and confidence.png')
linear = Image.open('images/linear.png')
random = Image.open('images/random.png')
nb5 = Image.open('images/NB5.png')
nb10 = Image.open('images/NB10.png')
knn5 = Image.open('images/knn5.png')
knn10 = Image.open('images/knn10.png')
ensemble = Image.open('images/ensemble.png')
hyperNB = Image.open('images/hyper nb5.png')
hyperKNN = Image.open('images/hyper knn5.png')
roc = Image.open('images/roc curve.png')
cluster = Image.open('images/KMeansClustering.jpeg')
#---------------

#----header----
with st.container():
    st.title('Data Mining Project')
    st.header(
    """
        Members:
        Fam Yao Deng,
        Friendy,
        Quinito Norman Octaviano,
        Sanjeev Kumar Nair A/L Vasuthevan
    """
    )
    
with st.container():
    st.write('---')
    left_column, right_column = st.columns(2)
    with left_column:
        st.image(image_flowchart)
    
    with right_column:
        st.subheader('Project Flowchart')
        st.write(
            """
            The flowchart describes the workflow for this project as well as the different tasks and experiments conducted. Here the step by step that we do the analysis of the dataset:
            1. We first load the original dataset.
            2. Then we find an external dataset that includes extra data we desired and then we concatinate the both datasets.
            3. We find the missing values or NaN values in the dataset, then we drop all the rows that has a missing value.
            4. With the data available to use, we explore different relationships between variables, find the most interesting association rules, uncover patterns group.
            5. We created 4 models: 2 Regression models and 2 Classification models and compare them based on different settings such as using feature selection and SMOTE.
            6. We compare the results of the different models.
            """
        )
    st.write('---')
#---------------

#----content----
with st.container():
    left_column, right_column = st.columns(2)
    with left_column:
        st.image(image_sales)
        
    with right_column:
        st.subheader('Months with the most sales in 2015 and 2016')
        st.write(
            """
                The barplots above show the monthy sales for 2015 and 2016. 
                We can notice that the total sales are the highest nearing December in the year 2015.
                Next, in 2016, the sales are the most at the start of the year, particularly from January to March has the most monthly sales.
            """
        )
    st.write('---')
    
with st.container():
    left_column, right_column = st.columns(2)
    with left_column:
        st.image(image_time)
        
    with right_column:
        st.subheader('Frequent visiting times')
        st.write(
            """
                Based on this barplot, we observe that most of the customers prefer to visit the laundry shop at 5am to 12 noon.
                Followed by the afternoon until 5pm being the second favorite time of the day to visit the laundry shop .
                Next, 6pm until midnight is the least favorite time of day to visit laundry shop.
            """
        )
    st.write('---')
            
with st.container():
    left_column, right_column = st.columns(2)
    with left_column:    
        st.image(image_races)
        
    with right_column:
        st.subheader('Spendings based on different races and genders')
        st.write(
            """
                From the graph, we can see that out of all the races, male foreigners spend the most in laundrymats.
                We can also see that males and females for the other 3 races spend about the same amount.
                Next, we want to see the relationship between gender and race with total money spent.
                
                Null hypothesis: Gender and Race has no influence on total money spent.
                Alternative hypothesis: Gender and Race has influence on total money spent.
                The p-value which is 0.966 is greater than the alpha value of 0.05, hence we fail to reject null hypothesis.
                Therefore we can conclude that gender and race of a customer does not affect total money spent.
            """
        )  
    st.write('---') 
        
with st.container():
    left_column, right_column = st.columns(2)
    with left_column:
        st.image(image_freq)
        
    with right_column:
        st.subheader('Frequency of visits based on race')
        st.write(
            """
                We can see that each race seems to visit the laundry shop more often at the start and towards the end of the year.
                Adding up the total number of customers per race, Malaysians have the most visits at 658 between 2015-2016.
                Followed by Chinese at 632, then Indians with 627, lastly, Foreigners with the least at 599.
                Now, we check if whether or not race is associated with the frequency of visits.
                
                Null hypothesis: Race has no association with frequency of visits.
                Alternative hypothesis: Race has an association with frequency of visits.
                The p-value for the chi-square test is 0.0001, which is lesser than 0.05, hence, we reject the null hypothesis.
                We have sufficient evidence to conclude that race is associated with frequency of visits.
            """
        )
    st.write('---')  
#---------------

#----Apriori----
with st.container():
    st.subheader('Apriori')
    st.write(
            """
                For the apriori shown below, it can be seen that the top 10 of the confidence is usually customers with casual shirts and short sleeves will bring a big basket,
                and for the worst confidence it can be seen customers with no kids, usually wear a casual shirt. And in the overall comparison, 
                it can be seen that more confidence and support in the lower score appear more often compared to the higher score part, 
                this means from the data of the laundry, there are many items that appears in the shop that is not correlated with the customer, because the confidence is lower than 0.6 confidence.
            """
    )
    left_column, middle, right_column = st.columns(3)
    with left_column:
        st.image(arm_top)
        
    with right_column:
        st.image(supp_conf)
    st.write('---')  
    
    with middle:
        st.image(arm_tail)
#---------------

#----K-Means----
with st.container():
    st.subheader('K-Means Clustering')
    left_column, right_column = st.columns(2)
    with left_column:
        st.image(cluster)
        
    with right_column:
        st.write(
                """
                We did a 3D model for clustering technique from mpl\_toolkits.mplot3d from Axes3D python library. We identified 3 clusters.
                The three clusters represent low temperature, high temperature and Windspeed. It is apparent that weather has an affect on the sales. 
                Sales increases when temperature
                """)
#---------------

#----models----
with st.container():
    st.subheader('Regression Model Performance (using Boruta)')
    left_column, right_column = st.columns(2)
    with left_column:
        st.image(linear)
        
    with right_column:
        st.image(random)
    st.write('---')

with st.container():
    st.subheader('Classifier Performance (using RFECV)')
    left_column, right_column = st.columns(2)
    with left_column:
        st.markdown('NB using Top 5 Features')
        st.image(nb5)
        
    with right_column:
        st.markdown('KNN using Top 5 Features')
        st.image(knn5)
    st.write('---')
        
with st.container():
    with left_column:
        st.markdown('NB using Top 10 Features')
        st.image(nb10)
        
    with right_column:
        st.markdown('KNN using Top 10 Features')
        st.image(knn10)
        
with st.container():
    with left_column:
        st.markdown('NB with Hyperparameters')
        st.image(hyperNB)
        
    with right_column:
        st.markdown('KNN with Hyperparameters')
        st.image(hyperKNN)
#---------------

#----Ensemble----
with st.container():
    st.subheader('Ensemble Classifier')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(' ')
    with col2:
        st.image(ensemble)
    with col1:
        st.write(' ')
    st.write('---')
#---------------

#----Graphs----
with st.container():
    st.subheader('Receiver Operating Characeristic (ROC) Curve')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(' ')
        
    with col2:
        st.image(roc)
        
    with col1:
        st.write(' ')
    st.write('---')
#---------------

#----map----
with st.echo():
    location = data['latitude'].mean(), data['longitude'].mean()
    map = folium.Map(location = location, tiles='CartoDB positron', zoom_start = 14, scrollWheelZoom = False)
    for (index, row) in data.iterrows():
        iframe = folium.IFrame('Race: ' + row.loc['Race'] + '<br>' + 'Gender: ' + row.loc['Gender'] + '<br>' + 'TimeSpent_minutes: ' + 
                            str(row.loc['TimeSpent_minutes']) + '<br>' + 'TotalSpent_RM: ' + str(row.loc['TotalSpent_RM']))
        popup = folium.Popup(iframe, min_width = 185, max_width = 185)
        folium.Marker(location = [row.loc['latitude'], row.loc['longitude']], popup = popup).add_to(map)

    folium_static(map, width = 700, height = 1400)
