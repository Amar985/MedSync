import streamlit as st
from PIL import Image
import sys
import pandas as pd


st.set_page_config(
        layout="centered",
        initial_sidebar_state="collapsed")

st.markdown("# MedSync - Step towards hospital inventory optimization 💻")
st.sidebar.markdown("# Home 🎈")

#logo = Image.open("./project_dashboard/heatmap.jpg")

logo = Image.open("./project_dashboard/medsync_logo.png")


st.subheader("Streamlining Inventory Management for Storable Healthcare Products in a Unified Demand Environment")
st.info("Welcome to our project! ⛩️", icon="⛩️")

st.write(" ")
st.write("---")
st.write(" ")

st.subheader("About Us")
team_members = [
    {
        "name": "Amar Kumar",
        "degree": "B.tech in Computer Science and Engineering",
        "linkedin": "https://www.linkedin.com/in/meamarkumar/",
    },
    {
        "name": "Shreya Sinha",
        "degree": "B.tech in Computer Science and Engineering",
        "linkedin": "https://www.linkedin.com/in/shreya--sinha/",
    },
    
]

# Set up columns for horizontal layout
col1, col2 = st.columns(2)
cols = [col1, col2, col1, col2]

for member, col in list(zip(team_members, cols)):
    with col:
        c = st.container()
        c.write(f"### {member['name']}")
        c.markdown(f"<u>Degrees</u>: {member['degree']}", unsafe_allow_html=True)
        c.write(f"Linkedin: [{member['name']}]({member['linkedin']})")


st.write(" ")
st.write("---")
st.write(" ")

st.image(logo, use_column_width=True)

st.write(" ")
st.write("---")
st.write(" ")

st.subheader("Abstract")
st.write(
    "In this project, we propose a mathematical model and implementation based on a collaborative scheme designed to optimize the storage and distribution of medical products to hospitals given historical data.")

with open("DSN4096_Capstone_Project_Medsync.pdf", "rb") as pdf_file:
    PDFbyte = pdf_file.read()

st.download_button(label="Dowload Paper",
                    data=PDFbyte,
                    file_name="./project_dashboard/Medsync_Paper_Final_Phase.pdf",
                    mime='application/octet-stream')

with open("Medsync_ppt.pdf", "rb") as pdf_file:
    PDFbyte = pdf_file.read()

st.download_button(label="Dowload PPT",
                    data=PDFbyte,
                    file_name="./project_dashboard/Medsync_PPt.pdf",
                    mime='application/octet-stream')


st.write("---")

st.subheader("Github")
st.info('See our [Github](https://github.com/Amar985/MedSync) repository', icon="ℹ️")
st.write("---")

st.subheader("Introduction")

st.write("Robust supply chains are critical to businesses because they not only guarantee operations but also yield substantial financial gains. Robustness is crucial, particularly in medical contexts where a shortfall might have fatal consequences. Furthermore, it is critical for a business to reduce the environmental impact of its operations, particularly in the area of transportation, which accounts for 28% of all pollution as of 2021.")

st.write("---")
st.subheader("Our approach: Problem Statement")

st.write("Determining cost-optimality while meeting environmental and robustness criteria is hence the aim. To address the issue, we propose a unified demand approach, which entails an agreement among various healthcare facilities to pool their demand for medical supplies into a single order in an effort to cut costs. This order is then based on a mathematical model, which is then further implemented to solve the problem in this particular setting. To keep things simple, we take a month-level granularity of time.")
st.write("---")
st.subheader("Proposed SOlution")
st.write("Assuming the referenced economic benefits, we assume in the unified demand scenario that all hospitals combine their orders for a certain product into a single order. As a result, we take into account every product independently and create a model to optimize the procedures for every product.")
st.write("We take into consideration a distinct supplier and distribution hub for a certain product (because we are in a specific area, we can presume a specific location). Optimising costs while maintaining a specific environmental footprint and resilience score is our task. We express the environmental impact in terms of the number of orders, as stated in the problem specification and the references. This is essentially deciding when and how much to order given a certain number of orders, a robust level of demand satisfaction, and storage costs.")
st.write("A forecast for the purchase plan provides us with the demand, from which we calculate the number of units required for the upcoming year.")
st.write(" ")
st.write(" ")
st.write("---")
st.write(" ")
st.write(" ")





st.markdown("# Unified demand model 🌐")
st.sidebar.markdown("# Unified demand model 🌐")


video_file = open('./project_dashboard/animation0.mp4', 'rb')
video_bytes = video_file.read()

st.write("Simulation of the flow of products between providers and a central storage center that supplies all hospitals in a region:")
st.video(video_bytes)
st.caption("Generated with pygame")

st.write(" ")
st.write("---")
st.write(" ")

st.markdown("# Methods for obtaining the outcomes ✌️")
st.sidebar.markdown("# Methods for obtaining the outcomes ✌️")
st.write('To replicate our training and obtain the metrics reported in the paper, execute the training script in') 
code = '''src/train.py'''
st.code(code, language='python')
 
st.write('For this, generate the virtual environment and install the necessary dependencies using [poetry](https://python-poetry.org/docs/):')
code = '''poetry install'''
st.code(code, language='python')
st.write('And execute the training script. The training script has two CLI arguments, data_path requires the relative path to the training data, model defines which model to use, and has to be one of Boltzmann (for the Boltzmann ensemble) or tft (for the Temporal Fusion Transformer, GPU required!).') 
code = '''cd src
poetry run python train.py --data_path  ../data --model boltzmann'''
st.code(code, language='python')




st.markdown("# Data 📊")
st.sidebar.markdown("# Data 📊")
# Load your CSV data
csv_data = pd.read_csv("./project_dashboard/consumo_material_clean.csv")

# Display the DataFrame
st.write("CSV Data: consumo_material_clean")
st.dataframe(csv_data)

st.markdown("# Dataset Overview 🕹️")
st.sidebar.markdown("# Dataset Overview 🕹️")
csv_data = pd.read_csv("./project_dashboard/Dataset_Overview.csv", encoding='latin1')

# Display the DataFrame
st.write("### Dataset Overview")
st.write("Purchase history since 2015 for all the healthcare supplies of a group of hospitals")
st.dataframe(csv_data)


st.markdown("# Model results 📋")
st.sidebar.markdown("# Model results 📋")


st.write('Parameters of optimization:')
st.write('', r'$\beta$', ' - Resilience factor, factor by which we multiply demand to increase supply chainresilience.')
st.write('', r'$P_{max}$', ' - Number of orders ∝ CO2 emissions,a proxy for environmental impact.')
st.write('We have used different values for', r'$\beta$', ' and ', r'$P_{max}$',' to observe the effects of different environmental and robustness restrictions on the optimal cost of storage.')
heatmap = Image.open("./project_dashboard/heatmap.jpg")
st.image(heatmap, use_column_width=True)
st.caption("Heatmap of optimal costs in terms of "+r'$\beta$'+" and "+ r'$P_{max}$'+" for product 70130 (APÓSITO DE HIDROCOLOIDE-7)")
st.write(' As expected, the more robust and the fewer orders allowed (i.e. the less environmental impact) lead to increased optimal costs. We also observe that it is significantly harder to have a lesser environmental impact than to be more robust.')
st.write(' You should obtain the following results (as reported in the report): ')
Model_result = Image.open("./project_dashboard/Model result.png")
st.image(Model_result, use_column_width=True)
