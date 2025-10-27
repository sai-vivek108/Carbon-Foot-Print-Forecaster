import streamlit as st
import pandas as pd
from prediction import predict

# Set page configuration
st.set_page_config(layout="wide")

st.title("Carbon Footprint Prediction")
st.markdown("A machine learning model to predict carbon footprint usage")


# First Section
st.subheader("Personal Features")
expander1 = st.expander("Click to expand")
with expander1:
    col1, col2 = st.columns(2)
    with col1:
        body = st.radio('Body Type',('obese', 'overweight', 'underweight', 'normal') )
        gender = st.radio("Gender",("male","female"))
    with col2:
        shower = st.radio("Shower Frequency",('less frequently', 'twice a day', 'more frequently', 'daily'))
        new_clothes = st.slider("How many new clothes per Month", 0,100, step=1)
st.text('')

# Second Section
st.subheader("Travel")
expander = st.expander("Click to expand")
with expander:
    col3, col4 = st.columns(2)
    with col3:
        transport = st.radio("Transportation", ('walk/bicycle', 'private', 'public' ))
        vehicle_type = st.radio("Vehicle Type", ('PMM', 'diesel', 'electric', 'petrol', 'lpg', 'hybrid', 'publicTransport'))
    with col4:
        air_travel = st.radio("Air Travel", ('rarely', 'very frequently', 'frequently', 'never'))
        distance_travelled = st.slider("Distance Travelled Per Month ",1,10000,step=150)
st.text('')

st.subheader("Food Habits")
expander1 = st.expander("Click to expand")
with expander1:
    col7, col8 = st.columns(2)
    with col7:
        grocery_bill = st.slider("Avg. Monthly Grocery bill", 1,600, step=10)
        diet = st.radio("Diet", ('vegetarian', 'omnivore', 'vegan', 'pescatarian'))
        waste_size = st.radio('Waste bag size',('extra large', 'medium', 'large', 'small'))
        waste_count = st.slider('Waste bag count Per Week', 1,8, step =1)
        social = st.radio("Social Activity", ('often', 'sometimes', 'never'))
    with col8:
        grill = st.radio("Cooking with Grill", ('yes', 'no'))
        oven = st.radio("Cooking with Oven", ('yes', 'no'))
        stove = st.radio("Cooking with Stove", ('yes', 'no'))
        microwave = st.radio("Cooking with Microwave", ('yes', 'no'))
        airfryer = st.radio("Cooking with Airfryer", ('yes', 'no'))

st.text('')

st.subheader("Home and recycle")
container = st.expander('Click to expand')
with container:
    col5, col6 = st.columns(2)
    with col5:
        tv_pc = st.slider("TV/PC watch time Per Day", 0,24, step=1)
        internet = st.slider("Time spent on Internet Per Day", 0,24, step=1)
        heating = st.radio("Heating Source", ('natural gas', 'wood', 'coal', 'electricity'))
        energy_eff = st.radio('Energy Efficiency',('No', 'Sometimes', 'Yes'))
    with col6:
        paper = st.radio("Recycling Paper", ('yes', 'no'))
        plastic = st.radio("Recycling Plastic", ('yes', 'no'))
        glass = st.radio("Recycling Glass", ('yes', 'no'))
        metal = st.radio("Recycling Metal", ('yes', 'no'))

st.text('')
if st.button("Estimate Carbin Footprint"):
    if body=='obese':
        bt_obese=1
        bt_ow,bt_uw =0,0        
    elif body =='overweight':
        bt_ow=1
        bt_obese,bt_uw =0,0        
    elif body == 'underweight':
        bt_uw=1
        bt_obese,bt_ow =0,0            
    else:
        bt_obese,bt_ow,bt_uw  =0,0,0

    if gender=='male':
        sex=0
    else: 
        sex =1
    
    if diet=='pescatarian':
        d_pesc,d_vegan,d_veg =1,0,0
    elif diet =='vegetarian':
        d_pesc,d_vegan,d_veg =0,0,1
    elif diet=='vegan':
        d_pesc,d_vegan,d_veg =0,1,0
    else:
        d_pesc,d_vegan,d_veg =0,0,0

    if shower == 'less frequently':
        sf_lf,sf_mf, sf_td=1,0,0
    elif shower=='more frequently':
        sf_lf,sf_mf, sf_td=0,1,0
    elif shower=='twice a day':
        sf_lf,sf_mf, sf_td=0,0,1
    else:
        sf_lf,sf_mf, sf_td=0,0,0

    if heating=='electricity':
        h_elec,h_gas,_h_wood=1,0,0
    elif heating=='natural gas':
        h_elec,h_gas,_h_wood=0,1,0
    elif heating=='wood':
        h_elec,h_gas,_h_wood=0,0,1
    else:
        h_elec,h_gas,_h_wood=0,0,0
    
    if transport=='public':
        trs_pu,trs_wa=1,0
    elif transport=='walk/bicycle':
        trs_pu,trs_wa=0,1
    else:
        trs_wa,trs_pu=0,0
    
    vt_ele,vt_hyb,vt_lpg,vt_pet,vt_pmm,vt_pub=0,0,0,0,0,0
    if vehicle_type=='electric':
        vt_ele=1
    elif vehicle_type=='hybrid':
        vt_hyb=1
    elif vehicle_type=='PMM':
        vt_pmm=1
    elif vehicle_type=='petrol':
        vt_pet=1
    elif vehicle_type=='publicTransport':
        vt_pub=1
    elif vehicle_type=='lpg':
        vt_lpg=1
    
    sa_oft,sa_st=0,0
    if social=='often':
        sa_oft=1
    elif social=='sometimes':
        sa_st=1

    atf_nev,atf_rare,atf_vf=0,0,0
    if air_travel=='never':
        atf_nev=1
    elif air_travel=='rarely':
        atf_rare=1
    elif air_travel=='very frequently':
        atf_vf=1

    wb_l,wb_m,wb_s=0,0,0
    if waste_size=='large':
        wb_l=1
    elif waste_size=='medium':
        wb_m=1
    elif wb_s=='small':
        wb_s=1
    
    ee_some,ee_yes=0,0
    if energy_eff=='Sometimes':
        ee_some=1
    elif ee_yes=='Yes':
        ee_yes=1

    if grill=='yes':
        cg_grill=1
    else:
        cg_grill=0
    
    if oven=='yes':
        cg_oven=1
    else:
        cg_oven=0
    
    if stove=='yes':
        cg_stove=1
    else:
        cg_stove=0

    if microwave=='yes':
        cg_microwave=1
    else:
        cg_microwave=0

    if airfryer=='yes':
        cg_airfryer=1
    else:
        cg_airfryer=0
    
    if paper=='yes':
        r_paper=1
    else:
        r_paper=0
    
    if glass=='yes':
        r_glass=1
    else:
        r_glass=0
    
    if plastic=='yes':
        r_plastic=1
    else:
        r_plastic=0
    
    if metal=='yes':
        r_metal=1
    else:
        r_metal=0

    dict1 = {'GC_PM':(grocery_bill-173.36287)/72.386055, 'DT_PM':(distance_travelled-2042.42381)/2777.828915, 'TPW':(tv_pc-12.13766)/7.139277,
       'NCP':(new_clothes-25.05342)/14.692149, 'IT':(internet-11.88657)/7.288290, 'BT_obese':bt_obese, 'BT_overweight':bt_ow, 'BT_underweight':bt_uw, 'S':sex,
        'D_pescatarian':d_pesc, 'D_vegan':d_vegan, 'D_vegetarian':d_veg, 'SF_less_frequent':sf_lf, 'SF_more_frequent':sf_mf,
       'SF_twice_a_day':sf_td,  'HS_electric':h_elec, 'HS_gas':h_gas, 'HS_wood':_h_wood, 'T_public':trs_pu, 'T_walk_bike':trs_wa, 
       'VT_electric':vt_ele, 'VT_hybrid':vt_hyb, 'VT_lpg':vt_lpg, 'VT_petrol':vt_pet, 'VT_PMM':vt_pmm, 'VT_public':vt_pub, 
       'SA_often':sa_oft, 'SA_sometimes':sa_st, 'ATF_never':atf_nev,'ATF_rarely':atf_rare, 'ATF_very_frequent':atf_vf,
        'WB_large':wb_l, 'WB_medium':wb_m, 'WB_small':wb_s , 'WCP':waste_count, 
       'EE_sometimes':ee_some, 'EE_yes':ee_yes, 'CG_grill':cg_grill, 'CG_oven':cg_oven, 'CG_stove':cg_stove,
       'CG_microwave':cg_microwave, 'CG_airfryer':cg_airfryer, 'RP_paper':r_paper,
       'RP_plastic':r_plastic, 'RP_glass':r_glass, 'RP_metal':r_metal}
    data= pd.DataFrame(dict1, index=[0])
    result = predict(data)

    st.text(result[0])
st.text('')
