# A Simple LLM Power Job Search application

The project focuses on creating a web application designed to help users secure jobs where they stand out as top candidates based on their resumes. 

Key features include extracting essential details from users’ PDF resumes, gathering pertinent data from job postings on LinkedIn and other platforms, and aligning user resumes with the most suitable job opportunities, incorporating with other advanced features such as skill gaps recommendation or draft cover letter generation.

The goal is to assist users in pinpointing the best job matches that capitalize on their strengths and experience, boosting their employment prospects.

![My Image](images/demo1.png)
![My Image](images/demo2.png)
# Installation Guide

## Install neccessary packages
In order to create a virtual environment with neccessary packages, please follow the following steps:

### Create a conda environment:
```
conda create -n {your environment}
```

### Activate the created environment:
```
conda activate {your environment}
```

### Install neccessary packages:
```
pip install -r requirements.txt
```

## Clone this project
To clone this project, run:
```
git clone https://github.com/HaiNguyen2903/JobSeekAssistant
```

Move to the repo folder:
```
cd JobSeekAssistant
```

## Prepare neccessary files

### Prepare config file
An OpenAI API key is required to initialize the config file. To create a config file, run:
```
python create_config.py --api_key {YOUR API KEY}
```

### Prepare other files

#### Installing from Google Drive
You can download and unzip the prepared file on [Google Drive](https://drive.google.com/file/d/1lGxQC2D3p6c8iPJZTl8Z_Ojix4-HJ2YI/view?usp=share_link). This file includes both ```datasets``` and ```embeddings``` folders with neccessary files. These folder shoule be in the same place with the repo folder as described below:

```
project_dir
    |__datasets/
    |       |__linkedin-jobs-2023-2024/
    |       |__job_merged.csv
    |
    |__embeddings/
    |       |__job_embeds.faiss
    |       |__id_mapping.json
    |__ ...

```

**Notes:** If you have successfully installed the aforementioned files, move to the **Streamlit Application** section and skip the manually installation guide.

#### Install manually

**Download Job Postings dataset**

The LinkedIn job postings dataset can be download from [Kaggle](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings). The dataset should be installed in a ```datasets``` folder.


**Generate Job dataset with necessary information**

The dataset includes both posting details, company details and generated job summaries. To generate the dataset, run:
```
python gen_job_data.py --save_path {your .csv save file path} --max_jobs {maximum jobs to save}
```
The default setting will save the generated file in ```.datasets/job_merged.csv```

**Create job embedding files**

In order to generate the embedding files, run:
```
python gen_job_embeddings.py --save_dir {directory to save embedding files}
```
The script generates 2 files: ```job_embeds.faiss``` to store vector embeddings and ```id_mapping.json``` to map the embedding ids to job ids.

The default setting will save the generated files in ```.embeddings/```.

In final, the project structure should be as followed:

```
project_dir
    |__datasets/
    |       |__linkedin-jobs-2023-2024/
    |       |__job_merged.csv
    |
    |__prompts/
    |__embeddings/
    |       |__job_embeds.faiss
    |       |__id_mapping.json
    |__config.json
    |__ ...

```

# Streamlit Application

In order to run the streamlit app, run:
```
streamlit run app.py
```

The application allows user to filter jobs based on certain options such as company name, job title or location. In additonally, user can also upload resume file in PDF format to search for the most suitable jobs based on their resume summary.

**Notes:** In order to upload a new resume file, please remove the current file first before uploading to avoid duplicating.