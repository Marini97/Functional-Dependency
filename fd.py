import seaborn as sns
import streamlit as st
import pandas as pd
import subprocess
import jsonlines
from io import StringIO
from PIL import Image


@st.cache_data
def get_data(file: str, header: bool=True) -> pd.DataFrame:
    """Read the data file and return a dataframe.

    Args:
        file : str
            File path of the data file.
        header : bool, optional
            True if the file has a header, False otherwise.

    Returns:
        df : pd.DataFrame
            A pandas dataframe with the data read from the file.
    """
    if header:
        return pd.read_csv(file)
    else:
        df = pd.read_csv(file, header=None)
        df.columns = [f"Column {i}" for i in range(df.shape[1])]
        return df


@st.cache_data
def read_json(df: pd.DataFrame, df_cols: int, results: pd.DataFrame) -> pd.DataFrame:
    """Read the output file from Metanome and return a dataframe with the results.

    Args:
        df : pd.DataFrame
            The dataframe with the data.
        df_cols : int
            The number of columns in the dataset.
        results : pd.DataFrame
            The dataframe with the results.

    Returns:
        results : pd.DataFrame
            A pandas dataframe with the results.
    """
    try:
        # read the output file from Metanome
        with jsonlines.open('results/data_fds') as lines:
            for line in lines:

                # get the determinants and dependant attributes
                determinants = line['determinant']['columnIdentifiers']
                for i in range(len(determinants)):
                    determinants[i] = determinants[i]['columnIdentifier']
                    
                dependant = line['dependant']['columnIdentifier']   
                
                # if the union of Determinants and Dependant contains all attributes, then the FD is filtered out
                if len(determinants)+1 < df_cols:
                    score = df[determinants+[dependant]].drop_duplicates().shape[0]
                    row = {'Determinants': determinants, 'Dependant': dependant, 'Score': score}
                    results.loc[len(results)] = row
                    
    except Exception as exception:
        print(exception)
    if len(results) == 0:
        return pd.DataFrame(columns=['Determinants', 'Dependant', 'Score'])
    return results


@st.cache_data
def get_fds(results: pd.DataFrame, df_cols: int) -> pd.DataFrame:
    """This function takes the results of the FD algorithm and returns a dataframe with the FDs as columns and the attributes as rows.

    Args:
        results : pd.DataFrame
            The results of the FD algorithm.
        df_cols : int
            The number of columns in the dataframe.

    Returns:
        rows : pd.DataFrame
            A dataframe with Determinants, Dependant and Score as attributes and FDs as rows.
    """
    # Convert the determinants to a string
    results['Determinants'] = [', '.join(map(str, l)) for l in results['Determinants']]
    
    # Create a new dataframe with the columns 'Determinants', 'Dependant' and 'Score'
    #rows = pd.DataFrame(columns=['Determinants', 'Dependant', 'Score'])
    rows = results.copy()
    
    # Group the dataframe by the determinants
    rows = rows.groupby('Determinants').agg({'Dependant': ', '.join, 'Score': 'max'}).reset_index()

    # Loop through all rows of the dataframe
    for row in rows.itertuples():
        # Loop through all rows of the dataframe again
        for row2 in rows.itertuples():
            # If the determinants of row2 are a subset of the determinants of row
            if row.Index != row2.Index and set(row2.Determinants.split(", ")).issubset(set(row.Determinants.split(", "))):
                # Add the dependants of row2 to the dependants of row
                rows.loc[row.Index, 'Dependant'] = rows.loc[row.Index, 'Dependant']+", "+row2.Dependant
                
    # Remove all rows with more than df_cols columns
    for row in rows.itertuples():
        if len(row.Determinants.split(", "))+len(row.Dependant.split(", ")) >= df_cols:
            rows = rows.drop(row.Index)
            
    # Return the dataframe
    return rows


icon = Image.open('icon.png')
st.set_page_config(
    page_title="Functional Dependency App",
    page_icon=icon,
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS to remove links
st.markdown("""
    <style>
    .css-15zrgzn {display: none}
    .css-10pw50 {display: none}
    .css-16a7pdm {display: none}
    </style>
    """, unsafe_allow_html=True)
    
# title of the app and the icon
col1, col2 = st.columns([2, 0.7])
col1.title("Functional Dependency App")
col2.image(icon, width=80)

# sidebar to select a dataset
with st.sidebar:
    st.title("Select a Dataset")
    # select a dataset
    dataset = st.selectbox("Choose a Dataset from the Seaborn library",sns.get_dataset_names())
    # load dataset
    df = sns.load_dataset(dataset)

    # upload a dataset
    uploaded_file = st.file_uploader("Or upload a CSV file from your computer.")
    header = True
    
    if uploaded_file is not None and not uploaded_file.name.endswith(".csv"):
        st.error("You need to select a CSV file.")
        uploaded_file = None
    if uploaded_file is not None:
        try:
            # select if header is present in the dataset
            header = st.checkbox("Does the dataset have a header?", value=False)
            df = get_data(uploaded_file, header)
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

df_cols = len(df.columns)
df_rows = df.shape[0]
results = pd.DataFrame(columns=['Determinants', 'Dependant', 'Score'])
results = read_json(df, df_cols, results)
rows = get_fds(results, df_cols)

st.header("Functional Dependencies")
# write the results to the streamlit app 
if rows.shape[0] > 0:
    st.write("In the following dropdown menu you can see all the **Functional Dependencies**  in the dataset. " 
             +"The **Functional Dependencies** are grouped by **Determinants** and sorted by their unique number of rows. ")
    
    st.success("The dataset has **"+str(df_cols)+" attributes** and **"+str(df_rows)+" rows**. " 
             +"The HyFD Algorithm has found **"+str(len(results))+" Functional Dependencies**.")
    
    st.write("An FD is filtered out if the union of Determinants and Dependants **contains ALL the attributes**.")
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
        if str(attr) in selected:
            attr1.append(attr)
        # remove the Dependants attributes from the dataframe
        if attr in dependants:
            output1 = output1.drop(attr, axis=1)
    
    output2 = df[attr1].copy()
    output2 = output2.drop_duplicates()
    col1, col2 = st.columns(2)
    
    col1.write("Table without the FD's Dependants attributes:")
    col1.dataframe(output1, use_container_width=True)
    col2.write("Table for the selected FD:")
    col2.dataframe(output2, use_container_width=True)
    
else:
    st.warning("In the selected dataset, the **HyFD Algorithm** did **NOT** find any **FD** or they got filtered out.")
    st.write("An FD is filtered out if the union of Determinants and Dependants **contains ALL the attributes**.")
