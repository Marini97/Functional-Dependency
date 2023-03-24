import seaborn as sns
import streamlit as st
import pandas as pd
import subprocess
import jsonlines
from io import StringIO
# to run this app, run the following command in the terminal
# streamlit run fd.py


@st.cache_data
def get_data(file):
    return pd.read_csv(file)

# title of the app
st.title("Functional Dependency App\n\n")

with st.sidebar:
    st.title("Select a Dataset")
    # select a dataset
    dataset = st.selectbox("Choose a Dataset from the Seaborn library",sns.get_dataset_names())
    # load dataset
    df = sns.load_dataset(dataset)

    # upload a dataset
    uploaded_file = st.file_uploader("or upload a CSV file from your computer")
    if uploaded_file is not None and not uploaded_file.name.endswith(".csv"):
        st.error("You need to select a CSV file.")
        uploaded_file = None
    if uploaded_file is not None:
        try:
            df = get_data(file=uploaded_file)
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

# run Metanome through subprocess.run:
subprocess.run(["java", "-cp", "jars/*", "de.metanome.cli.App", "--algorithm", "de.metanome.algorithms.hyfd.HyFD", 
                "--file-key", "INPUT_GENERATOR", "--files", "results/data.csv", "--separator", ",", 
                "--header", "-o", "file:data"])

# if the dataset doesn't have any FDs, then it will not create a result file
# so we need to check if the result file exists
# if it doesn't exist, then we need to display a message
results = []
fd = True

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
                
    if len(results) == 0:
        fd = False
        st.write("There are no **Functional Dependencies** in this dataset.")
except:
    fd = False

# write the results to the streamlit app
# in a table like format
if fd:
    st.header("Functional Dependencies")
    st.write("The score is the number of **unique tuples** for each FD."
            +"If the union of Determinants and Dependant **contains all attributes** then the FD is filtered out.")
    
    # create a dataframe with all FDs
    rows = pd.DataFrame(columns=['Determinants', 'Dependant', 'Score'])
    for result in results:
        determinants = ", ".join(map(str,result['determinants']))
        dependant = result['dependant']
        score = result['score']
        row = pd.DataFrame({'Determinants': [determinants], 'Dependant': [dependant], 'Score': [score]})
        rows = pd.concat([rows, row], ignore_index=True)
        
    # sort the FDs by score
    rows = rows.sort_values(by=['Score'], ascending=False)
    selected = st.selectbox("Choose a FD to see his table.  (**Determinants -> Dependant: Score**)",rows['Determinants']+" -> "+rows['Dependant']+": "+rows['Score'].astype(str))
    # get the attributes of the selected FD
    attributes = selected.split(" -> ")[0].split(", ")
    attributes.append(selected.split(" -> ")[1].split(": ")[0])
    
    output = df[attributes]
    output = output.drop_duplicates()
    st.dataframe(output, use_container_width=True)