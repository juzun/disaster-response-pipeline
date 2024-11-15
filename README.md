# Disaster Response Pipeline Project

### Instructions:
1. Install virtual environment.
    `poetry install`

2. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python disaster_response_pipeline/core/process_data.py --messages-filepath data/messages.csv --categories-filepath data/categories.csv --database-filepath data/DisasterResponse`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Run the following command in the app's directory to run your web app.
    `python run.py`

4. Go to http://0.0.0.0:3001/
