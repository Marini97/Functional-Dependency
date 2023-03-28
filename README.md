# Functional Dependency App <img src=icon.png height="50px" width="50px">

This project is part of the "Data Profiling" topic for the "Advanced Topics in Computer Science" course at the University of Roma Tre. The aim of this project is to provide a tool that can profile datasets and perform data normalization using functional dependencies (FDs).

## Workflow

The Functional Dependency App has the following workflow:

1. Accepts **CSV-files** or **Seaborn datasets** as input.
2. **Computes FDs** on the dataset using the **Metanome CLI** through subprocess.run().
3. **Extends FDs** by searching for all right-hand sides that are determined by either the same left-hand side or a subset of it.
4. **Filters FDs** out if the union of Determinants and Dependants contains **ALL** attributes.
5. **Scores FDs** based on the number of unique rows.
6. Allows the user to **select an FD** and visualize the normalized dataset based on the selected FD.

## Requirements

This app is tested with Python 3.10.2 and the following libraries:

* jsonlines==3.1.0
* pandas==1.3.5
* Pillow==9.4.0
* seaborn==0.12.2
* streamlit==1.20.0

## Installation

To install this app, you can simply clone the repository and install the required packages using pip:

```bash
git clone https://github.com/Marini97/Functional-Dependency.git
cd Functional-Dependency
pip install -r requirements.txt
```

## Usage

```bash
cd Functional-Dependency
streamlit run fd.py
```

After running the streamlit command, the App should open in the browser. If it doesn't, you can go to [http://localhost:8501/](http://localhost:8501/)

## Deployed App

This app has been deployed on the Streamlit Community Cloud and is available at the following URL: [https://functional-dependency.streamlit.app/](https://functional-dependency.streamlit.app/)
