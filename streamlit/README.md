# This is the web interaction function for measuring the single qubit

## How to use it
First you need to install streamlit
```
pip install streamlit
```
To run the scrip, cd into the streamlit file and input `streamlit run Homepage.py` in your terminal. And you can start to measure the qubit step by step by left column

## How to create a new measurement scrip
All the qubit measruement scrip are all in the `single_qubit_pyscrip` folder. You just need to put your Qick measurement scrip inside the folder first. And create a new operate scrip inside the `streamlit/pages` folder to put it in.

## Code construnction of streamlit pages

* import package and measurement funciton
* Wite a class function to acquire the Qick measurement funciont
* copy the column of streamlit funciton. If the experiment is 1D, you can reference the `3_Resonator_spectroscopy.py`. If it si 2D you can reference `4_Resonator_Punch_out.py`


