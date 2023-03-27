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
        
    return results, fd

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
    # group the FDs by subsets of Determinants and concatenate the Dependant attributes
    for row in rows.itertuples():
        for row2 in rows.itertuples():
            if row.Index != row2.Index and set(row2.Determinants.split(", ")).issubset(set(row.Determinants.split(", "))):
                rows.loc[row.Index, 'Dependant'] = rows.loc[row.Index, 'Dependant']+", "+row2.Dependant
                rows = rows.drop(row2.Index)
    
    # if the union of Determinants and Dependant contains all attributes, then the FD is filtered out
    for row in rows.itertuples():
        if len(row.Determinants.split(", "))+len(row.Dependant.split(", ")) >= df_num:
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

df_num = len(df.columns)
rows = get_fds(results, df_num)    

# write the results to the streamlit app 
if fd and rows.shape[0] > 0:
    st.header("Functional Dependencies")
    st.write("In the following dropdown menu you can see all the **Functional Dependencies**  in the dataset. " 
             +"The **Functional Dependencies** are grouped by **Determinants** and sorted by their unique number of rows. "
             +"A FD is filtered out if the union of Determinants and Dependants **contains all the attributes**.")
    
    st.write("The dataset has **"+str(df_num)+" attributes**. The HyFD algorithm "
             +"has  "+str(len(results))+" **Functional Dependencies**.")
    
    # sort the FDs by score
    rows = rows.sort_values(by=['Score'], ascending=False)
    selected = st.selectbox("Choose a FD to see the dataset normalized.  (**Determinants -> Dependants: Score**)",rows['Determinants']+" -> "+rows['Dependant']+": "+rows['Score'].astype(str))
    # get the Determinants and Dependants attributes from the selected FD
    dependants = selected.split(" -> ")[1].split(", ")
    dependants[-1] = dependants[-1].split(":")[0]
    # get all the attributes of the selected FD based on the dataframe columns
    attr1 = []
    output1 = df.copy()
    for attr in df.columns:
        if attr in selected:
            attr1.append(attr)
        # remove the Dependants attributes from the dataframe
        if attr in dependants:
            output1 = output1.drop(attr, axis=1)
    
    output2 = df[attr1]
    output2 = output2.drop_duplicates()
    col1, col2 = st.columns(2)
    col1.dataframe(output1, use_container_width=True)
    col2.dataframe(output2, use_container_width=True)
    
else:
    st.write("The **HyFD Algorithm** didn't find any **Functional Dependencies** in this dataset.")
