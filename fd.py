import seaborn as sns
import streamlit as st
import pandas as pd
import subprocess
import jsonlines
# to run this app, run the following command in the terminal
# streamlit run fd.py

# title of the app
st.title("Metanome Streamlit App")

# select a dataset
st.header("Select a Dataset from the dropdown menu")
dataset = st.selectbox("",sns.get_dataset_names())

# load dataset
df = sns.load_dataset(dataset)
# show the first 5 rows of the dataset
st.write("Show the first 5 rows of the selected dataset")
st.dataframe(df.head(), use_container_width=True)

# write the selected dataset to a csv file 
df.to_csv("results/data.csv", index=False)

# run Metanome through subprocess.run:
subprocess.run(["java", "-cp", "jars/*", "de.metanome.cli.App", "--algorithm", "de.metanome.algorithms.hyfd.HyFD", 
                "--file-key", "INPUT_GENERATOR", "--files", "results/data.csv", "--separator", ",", 
                "--header", "-o", "file:data"])

# read the result file 
results = []
with jsonlines.open('results/data_fds') as reader:
    for data in reader:
        determinant = data['determinant']['columnIdentifiers']
        for i in range(len(determinant)):
            determinant[i] = determinant[i]['columnIdentifier']
        dependant = data['dependant']['columnIdentifier']
        results.append({'determinant': determinant, 'dependant': dependant})

# write the results to the streamlit app
# in a table like format
st.header("Functional Dependencies")
st.write("The following table shows the functional dependencies found with the HyFD algorithm.")
df = pd.DataFrame(columns=['Determinants', 'Dependant'])

for result in results:
    determinants = ", ".join(map(str,result['determinant']))
    dependant = result['dependant']
    row = pd.DataFrame({'Determinants': [determinants], 'Dependant': [dependant]})
    df = pd.concat([df, row], ignore_index=True)

st.dataframe(df, use_container_width=True)