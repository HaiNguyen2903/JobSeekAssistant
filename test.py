import streamlit as st
from streamlit_modal import Modal

import streamlit.components.v1 as components

# Create 4 equal-width columns
col1, col2, col3, col4 = st.columns(4)

modal = Modal(
    "Skills you are missing", 
    key="demo-modal",
    
    # Optional
    padding=20,    # default value
    max_width=744,  # default value
)

with col1:
    if st.button("Button 1"):
        st.write("You clicked Button 1")

with col2:
    if st.button("Button 2"):
        st.write("You clicked Button 2")

with col3:
    if st.button("Button 3"):
        st.write("You clicked Button 3")
        modal.open()
        with modal.container():
            st.write("Text goes here")

with col4:
    if st.button("Button 4"):
        st.write("You clicked Button 4")

open_modal = st.button("Open")
if open_modal:
    modal.open()

if modal.is_open():
    with modal.container():
        st.write("Text goes here")