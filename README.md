# Running tutorial

## 0. Install neccessary packages
```
pip install -r requirement
```

## 1. Create the joined job dataset:=

The dataset includes posting details, company details and job summary. Replace neccessary paths and run:
```
python utils.py
```

## 2. Create embedding files
Replace neccessary paths and run:
```
python text_embedding.py
```

## 3. Run the streamlit app
Replace neccessary paths and run:
```
streamlit run app.py
```