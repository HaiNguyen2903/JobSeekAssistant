import streamlit as st
from streamlit_option_menu import option_menu
import os

def load_page(page_path):
    with open(page_path, 'r') as file:
        exec(file.read(), globals())

def main():
    st.set_page_config(
        page_title="Resume-Job Matching Platform",
        page_icon="📄",
        layout="wide",
    )

    st.sidebar.title("Navigation")
    selected_page = option_menu(
        menu_title="Main Menu",
        options=["Resume Matching", "Job Listings"],
        icons=["file-earmark-person", "briefcase"],
        menu_icon="cast",
        default_index=0,
    )

    if selected_page == "Resume Matching":
        st.title("Resume Matching")
        st.markdown("Upload your resume and find the best matching jobs based on your skills and experience.")
        load_page('job_matching_page.py')

    elif selected_page == "Job Listings":
        st.title("Job Listings")
        st.markdown("Explore job opportunities and company details. Filter based on your preferences to find the perfect match for your career.")
        load_page('job_list_page.py')

if __name__ == '__main__':
    main()