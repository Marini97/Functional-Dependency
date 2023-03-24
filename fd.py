import seaborn as sns
import streamlit as st
import pandas as pd
import subprocess
import jsonlines
from io import StringIO
from PIL import Image
# to run this app, run the following command in the terminal
# streamlit run fd.py

# cache the uploaded dataset to avoid reading it every time the app is updated
@st.cache_data
def get_data(file):
    return pd.read_csv(file)

# the dataset parameter is used to check if the dataset has been changed
# if it has been changed, then we need to read the output file again
# if it hasn't been changed, then we can use the cached results
@st.cache_data
def read_output(dataset, results, fd):
    try:
        with jsonlines.open('results/data_fds') as reader:
            df_num = len(df.columns)
            for data in reader:
                
                determinant = data['determinant']['columnIdentifiers']
                for i in range(len(determinant)):
                    determinant[i] = determinant[i]['columnIdentifier']
                    
                dependant = data['dependant']['columnIdentifier']
                
                if len(determinant)+1 < df_num:
                    score = df[determinant+[dependant]].drop_duplicates().shape[0]
                    results.append({'determinants': determinant, 'dependant': dependant, 'score': score})
        
    except Exception as exception:
        fd = False

    if len(results) == 0:
        fd = False
        st.write("The **HyFD Algorithm** didn't find any **Functional Dependencies** in this dataset.")
        
    return results, fd
    
        # create a dataframe with all FDs
    rows = pd.DataFrame(columns=['Determinants', 'Dependant', 'Score'])
    for result in results:
        determinants = ", ".join(map(str,result['determinants']))
        dependant = result['dependant']
        score = result['score']
        row = pd.DataFrame({'Determinants': [determinants], 'Dependant': [dependant], 'Score': [score]})
        rows = pd.concat([rows, row], ignore_index=True)
    return rows

# get all FDs and group them by Determinants
@st.cache_data
def get_fds(results, df_num):
    rows = pd.DataFrame(columns=['Determinants', 'Dependant', 'Score'])
    for result in results:
        determinants = ", ".join(map(str,result['determinants']))
        dependant = result['dependant']
        score = result['score']
        row = pd.DataFrame({'Determinants': [determinants], 'Dependant': [dependant], 'Score': [score]})
        rows = pd.concat([rows, row], ignore_index=True)
        
    # group the FDs by Determinants and concatenate the Dependant attributes
    rows = rows.groupby('Determinants').agg({'Dependant': ', '.join, 'Score': 'mean'}).reset_index()
    # if the union of Determinants and Dependant contains all attributes, then the FD is filtered out
    for row in rows.itertuples():
        if len(row.Determinants.split(", "))+len(row.Dependant.split(", ")) > df_num:
            rows = rows.drop(row.Index)
    return rows

icon = Image.open('icon.png')
st.set_page_config(
    page_title="Functional Dependency App",
    page_icon=icon,
    layout="wide",
    initial_sidebar_state="expanded",
)
# title of the app
st.title("Functional Dependency App\n\n")

with st.sidebar:
    st.title("Select a Dataset")
    # select a dataset
    dataset = st.selectbox("Choose a Dataset from the Seaborn library",sns.get_dataset_names())
    # load dataset
    df = sns.load_dataset(dataset)

    # upload a dataset
    uploaded_file = st.file_uploader("Or upload a CSV file from your computer.")
    if uploaded_file is not None and not uploaded_file.name.endswith(".csv"):
        st.error("You need to select a CSV file.")
        uploaded_file = None
    if uploaded_file is not None:
        try:
            df = get_data(file=uploaded_file)
            dataset = uploaded_file.name
        except(pd.errors.EmptyDataError):
            st.error("The selected CSV file is empty.")
            uploaded_file = None
            
# write the selected dataset to a csv file 
if df is not None:
    # remove all files from the results folder
    subprocess.run(["rm", "-r", "results/*"])
    # write the dataset to a csv file
    df.to_csv("results/data.csv", index=False)
    st.header("Selected Dataset")
    st.dataframe(df, use_container_width=True)
    
# run Metanome through subprocess.run
# if the dataset doesn't have any FDs, then the output file will be empty
subprocess.run(["java", "-cp", "jars/*", "de.metanome.cli.App", "--algorithm", "de.metanome.algorithms.hyfd.HyFD", 
                "--file-key", "INPUT_GENERATOR", "--files", "results/data.csv", "--separator", ",", 
                "--header", "-o", "file:data"])

results = []
fd = True
results, fd = read_output(dataset, results, fd)

# write the results to the streamlit app 
if fd:
    st.header("Functional Dependencies")
    st.write("In the following dropdown menu you can see all the **Functional Dependencies** found in the dataset. " 
             +"The **Functional Dependencies** are grouped by **Determinants** and sorted by their unique number of rows. "
             +"An FD is filtered out if the union of Determinants and Dependants **contains all the attributes**.")
    
    df_num = len(df.columns)
    st.write("The dataset has **"+str(df_num)+" columns**. The HyFD algorithm "
             +"Found "+str(len(results))+" **Functional Dependencies**")
    rows = get_fds(results, df_num)    
    # sort the FDs by score
    rows = rows.sort_values(by=['Score'], ascending=False)
    selected = st.selectbox("Choose a FD to see his table normalized.  (**Determinants -> Dependants: Score**)",rows['Determinants']+" -> "+rows['Dependant']+": "+rows['Score'].astype(str))
    # get all the attributes of the selected FD based on the dataframe columns
    attributes = []
    for attr in df.columns:
        if attr in selected:
            attributes.append(attr)
    
    output = df[attributes]
    output = output.drop_duplicates()
    st.dataframe(output, use_container_width=True)
