import streamlit as st
import numpy as np
#import joblib
import pickle

from prediction import get_prediction, ordinal_encoder, ordinal_Deencoder

# import bz2file as bz2

#model = joblib.load(r'models/TrafficSeverityClassificationModel2.joblib')
model = pickle.load(open('models/TrafficSeverityClassificationModel2.pkl', 'rb'))

# data = bz2.BZ2File('models/TrafficSeverityClassificationModel2.pbz2', 'rb')
# model = pickle.load(data)

st.set_page_config(page_title="Accident Severity Prediction App", page_icon="🚧", layout="wide")

#creating option list for dropdown menu
options_Day_of_week	= ['Monday' ,'Sunday', 'Friday', 'Wednesday', 'Saturday', 'Thursday', 'Tuesday']

options_Age_band_of_driver = ['18-30','31-50','Under 18','Over 51','Unknown']

options_Sex_of_driver	= ['Male','Female','Unknown']	

options_Educational_level = ['Above high school', 'Junior high school', 'Elementary school','High school', 'Unknown','Illiterate', 'Writing & reading']	

options_Driving_experience = ['1-2yr', 
                              'Above 10yr', 
                              '5-10yr', 
                              '2-5yr', 
                              'No Licence', 
                              'Below 1yr', 
                              'unknown']	

options_Type_of_vehicle = ['Automobile',
                           'Public (> 45 seats)', 
                           'Lorry (41?100Q)' ,
                           'Public (13?45 seats)',
                           'Lorry (11?40Q)' ,
                           'Long lorry', 
                           'Public (12 seats)'
                           'Taxi' ,
                           'Pick up upto 10Q' ,
                           'Stationwagen' ,
                           'Ridden horse' ,
                           'Other' ,
                           'Bajaj'
                           'Turbo' ,
                           'Motorcycle' ,
                           'Special vehicle' ,
                           'Bicycle']	

options_Owner_of_vehicle = ['Owner' ,'Governmental' , 'Organization', 'Other']	

options_Service_year_of_vehicle = ['Above 10yr' ,'5-10yrs' , '1-2yr' ,'2-5yrs' ,'Unknown' ,'Below 1yr']	

options_Area_accident_occured = ['Residential areas' ,'Office areas' ,'  Recreational areas',
                                 ' Industrial areas' , 'Other' ,' Church areas' ,'  Market areas',
                                 'Unknown' ,'Rural village areas', ' Outside rural areas', ' Hospital areas',
                                 'School areas' ,'Rural village areasOffice areas', 'Recreational areas']	

options_Lanes_or_Medians = [ 'Undivided Two way','other' ,'Double carriageway (median)' ,
                            'One way',
                            'Two-way (divided with solid lines road marking)',
                            'Two-way (divided with broken lines road marking)', 
                            'Unknown']	

options_Road_allignment = ['Tangent road with flat terrain' ,
                            'Tangent road with mild grade and flat terrain',
                            'Escarpments',
                            'Tangent road with rolling terrain' 'Gentle horizontal curve',
                            'Tangent road with mountainous terrain and',
                            'Steep grade downward with mountainous terrain', 
                            'Sharp reverse curve',
                            'Steep grade upward with mountainous terrain']	

options_Types_of_Junction = ['No junction', 'Y Shape', 'Crossing', 'O Shape', 'Other' ,'Unknown' ,'T Shape', 'X Shape' ]	

options_Road_surface_type = ['Asphalt roads', 'Earth roads' , 'Asphalt roads with some distress', 'Gravel roads' ,'Other']	

options_Road_surface_conditions = ['Dry' ,'Wet or damp' ,'Snow' ,'Flood over 3cm. deep']	

options_Light_conditions = ['Daylight' ,'Darkness - lights lit' ,'Darkness - no lighting','Darkness - lights unlit']	

options_Weather_conditions = ['Normal', 'Raining' ,'Raining and Windy' ,'Cloudy' ,'Other' 'Windy' 'Snow','Unknown' ,'Fog or mist']	

options_Type_of_collision = ['Collision with roadside-parked vehicles',
                             'Vehicle with vehicle collision',
                             'Collision with roadside objects',
                             'Collision with animals',
                             'Other', 
                             'Rollover', 
                             'Fall from vehicles',
                             'Collision with pedestrians', 
                             'With Train',
                             'Unknown']

options_Vehicle_movement = ['Going straight' ,
                            'U-Turn' ,
                            'Moving Backward' ,
                            'Turnover' ,
                            'Waiting to go',
                            'Getting off' ,
                            'Reversing' ,
                            'Unknown' ,
                            'Parked', 
                            'Stopping',
                            'Overtaking',
                            'Other', 
                            'Entering a junction']

options_Casualty_class = ['na' ,'Driver or rider', 'Pedestrian' ,'Passenger']

options_Sex_of_casualty = ['na', 'Male' ,'Female']

options_Age_band_of_casualty = ['na' ,'31-50' ,'18-30' ,'Under 18' ,'Over 51' ,'5']

options_Casualty_severity = ['na', '3' ,'2' ,'1']

options_Pedestrian_movement = ['Not a Pedestrian', 
                               "Crossing from driver's nearside",
                               'Crossing from nearside - masked by parked or statioNot a Pedestrianry vehicle',
                               'Unknown or other',
                               'Crossing from offside - masked by  parked or statioNot a Pedestrianry vehicle',
                               'In carriageway, statioNot a Pedestrianry - not crossing  (standing or playing)',
                               'Walking along in carriageway, back to traffic',
                               'Walking along in carriageway, facing traffic',
                                'In carriageway, statioNot a Pedestrianry - not crossing  (standing or playing) - masked by parked or statioNot a Pedestrianry vehicle']

options_Cause_of_accident = ['Moving Backward' ,
                             'Overtaking' ,                             
                             'Changing lane to the left',
                            'Changing lane to the right', 
                            'Overloading',
                            'Other',
                            'No priority to vehicle' ,
                            'No priority to pedestrian' ,
                            'No distancing',
                            'Getting off the vehicle improperly', 
                            'Improper parking', 
                            'Overspeed',
                            'Driving carelessly', 
                            'Driving at high speed', 
                            'Driving to the left',
                            'Unknown', 
                            'Overturning', 
                            'Turnover', 
                            'Driving under the influence of drugs', 
                            'Drunk driving']

options_accident_severity = ['Fatal Injury','Serius Injury','Slight Injury']


features = ['ligth_conditions','number_of_casualties','number_of_vehicles','age_band','time_minute',
            'day_of_the_week','driving_experience','road_surface_conditions','types_of_junctions', 'time_hour']

st.markdown("<h1 style='text-align: center;'>Accident Severity Prediction App 🚧</h1>", unsafe_allow_html=True)
st.markdown("This project is part of my personal portfolio.")
st.markdown("In this, an attempt is made to predict the severity of a traffic accident.")
st.markdown("The target feature is **:red[Accident_severity]** which is a multi-class variable. The task is to classify this variable based on the other 31 features.")
st.markdown("The metric used for evaluation is **:green[f1-score]**")
st.write("You can check the source code on [GitHub](https://github.com/LuisAnibalVasquez/Road-Traffic-Severity-Classification)")

def main():
    with st.form('prediction_form'):
        
        st.subheader("Enter the input for following features:")

        with st.container():
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                Day_of_week = st.selectbox("Select Day of week: ", options=options_Day_of_week)
            with col2:                
                Age_band_of_driver = st.selectbox("Select Age band of driver: ", options=options_Age_band_of_driver)
            with col3:
                Sex_of_driver = st.selectbox("Select Sex of driver: ", options=options_Sex_of_driver)
            with col4:    
                Educational_level = st.selectbox("Select Educational level: ", options=options_Educational_level)
        with st.container():   
            col6, col7, col8, col9, col10 = st.columns(5)         
            with col6:    
                Driving_experience = st.selectbox("Select Driving experience: ", options=options_Driving_experience)
            with col7:    
                Type_of_vehicle = st.selectbox("Select Type of vehicle: ", options=options_Type_of_vehicle)
            with col8:    
                Owner_of_vehicle = st.selectbox("Select Owner of vehicle: ", options=options_Owner_of_vehicle)
            with col9:    
                Service_year_of_vehicle = st.selectbox("Select Service year of vehicle: ", options=options_Service_year_of_vehicle)
            with col10:    
                Area_accident_occured = st.selectbox("Select Area accident occured: ", options=options_Area_accident_occured)
        with st.container():
            col11, col12, col13, col14, col15= st.columns(5)
            with col11:    
                Lanes_or_Medians = st.selectbox("Select Lanes or Medians: ", options=options_Lanes_or_Medians)
            with col12:    
                Road_allignment = st.selectbox("Select Road allignment: ", options=options_Road_allignment)
            with col13:    
                Types_of_Junction = st.selectbox("Select Types of Junction: ", options=options_Types_of_Junction)
            with col14:    
                Road_surface_type  = st.selectbox("Select Road surface type : ", options=options_Road_surface_type )
            with col15:   
                Pedestrian_movement = st.selectbox("Select Pedestrian movement: ", options=options_Pedestrian_movement)
        with st.container():
            col16, col17, col18, col19, col20 = st.columns(5)
            with col16:                
                Cause_of_accident = st.selectbox("Select Cause of accident: ", options=options_Cause_of_accident)
            with col17:
                Road_surface_conditions = st.selectbox("Select Road surface conditions: ", options=options_Road_surface_conditions)
            with col18:                
                Light_conditions = st.selectbox("Select Light conditions: ", options=options_Light_conditions)
            with col19:                
                Weather_conditions = st.selectbox("Select Weather conditions: ", options=options_Weather_conditions)
            with col20:                
                Type_of_collision = st.selectbox("Select Type of collision: ", options=options_Type_of_collision)
        with st.container():
            col21, col22, col23, col24, col25 = st.columns(5)
            with col21:                
                Vehicle_movement = st.selectbox("Select Vehicle movement: ", options=options_Vehicle_movement)
            with col22:                
                Casualty_class = st.selectbox("Select Casualty class: ", options=options_Casualty_class)
            with col23:                
                Sex_of_casualty = st.selectbox("Select Sex of Casualty: ", options=options_Sex_of_casualty)
            with col24:                
                Age_band_of_casualty = st.selectbox("Select Age band of casualty: ", options=options_Age_band_of_casualty)
            with col25:                
                Casualty_severity = st.selectbox("Select Casualty severity: ", options=options_Casualty_severity)
        with st.container():     
            col26, col27, col28, col29 = st.columns(4)               
            with col26:    
                Number_of_vehicles_involved = st.number_input('Number of vehicles involved: ', min_value = 1, max_value = 100, value  = 1, step =1, format = "%i")                                          
            with col27:        
                Number_of_casualties = st.number_input('Number of casualties: ', min_value = 0, max_value = 100, value  = 0, step =1, format = "%i")                         
            with col28:
                hour = st.number_input('Hour: ', min_value = 1, max_value = 24, value  = 1, step = 1, format = "%i") 
            with col29:                        
                minute = st.number_input('Minute: ', min_value = 0, max_value = 60, value  = 0, step =1, format = "%i") 


        submit = st.form_submit_button("Predict")

        if submit:
            Day_of_week = ordinal_encoder(Day_of_week,options_Day_of_week)
            Age_band_of_driver = ordinal_encoder(Age_band_of_driver,options_Age_band_of_driver)
            Sex_of_driver = ordinal_encoder(Sex_of_driver,options_Sex_of_driver)
            Educational_level = ordinal_encoder(Educational_level,options_Educational_level)
            Driving_experience = ordinal_encoder(Driving_experience,options_Driving_experience)
            Type_of_vehicle = ordinal_encoder(Type_of_vehicle,options_Type_of_vehicle)
            Owner_of_vehicle = ordinal_encoder(Owner_of_vehicle,options_Owner_of_vehicle)
            Service_year_of_vehicle = ordinal_encoder(Service_year_of_vehicle,options_Service_year_of_vehicle)
            Area_accident_occured = ordinal_encoder(Area_accident_occured,options_Area_accident_occured)
            Lanes_or_Medians = ordinal_encoder(Lanes_or_Medians,options_Lanes_or_Medians)
            Road_allignment = ordinal_encoder(Road_allignment,options_Road_allignment)
            Types_of_Junction = ordinal_encoder(Types_of_Junction,options_Types_of_Junction)
            Road_surface_type  = ordinal_encoder(Road_surface_type ,options_Road_surface_type)
            Road_surface_conditions = ordinal_encoder(Road_surface_conditions,options_Road_surface_conditions)
            Light_conditions = ordinal_encoder(Light_conditions,options_Light_conditions)
            Weather_conditions = ordinal_encoder(Weather_conditions,options_Weather_conditions)
            Type_of_collision = ordinal_encoder(Type_of_collision,options_Type_of_collision)
            Vehicle_movement = ordinal_encoder(Vehicle_movement,options_Vehicle_movement)
            Casualty_class = ordinal_encoder(Casualty_class,options_Casualty_class)
            Sex_of_casualty = ordinal_encoder(Sex_of_casualty,options_Sex_of_casualty)
            Age_band_of_casualty = ordinal_encoder(Age_band_of_casualty,options_Age_band_of_casualty)
            Casualty_severity = ordinal_encoder(Casualty_severity,options_Casualty_severity)
            Pedestrian_movement = ordinal_encoder(Pedestrian_movement,options_Pedestrian_movement)
            Cause_of_accident = ordinal_encoder(Cause_of_accident,options_Cause_of_accident)

            data = np.array([Day_of_week,
                             Age_band_of_driver,
                             Sex_of_driver,
                             Educational_level,
                             Driving_experience,
                             Type_of_vehicle,
                             Owner_of_vehicle,
                             Service_year_of_vehicle,
                             Area_accident_occured,
                             Lanes_or_Medians,
                             Road_allignment,
                             Types_of_Junction,
                             Road_surface_type,
                             Road_surface_conditions,
                             Light_conditions,
                             Weather_conditions,
                             Type_of_collision,
                             Vehicle_movement,
                             Casualty_class,
                             Sex_of_casualty,
                             Age_band_of_casualty,
                             Casualty_severity,
                             Pedestrian_movement,
                             Cause_of_accident,
                             Number_of_vehicles_involved, 
                             Number_of_casualties, 
                             hour, 
                             minute]).reshape(1,-1)




            pred = get_prediction(data=data, model=model)
            st.subheader(f"The predicted severity is:  {ordinal_Deencoder(pred[0], options_accident_severity)}")
            

if __name__ == '__main__':
    main()