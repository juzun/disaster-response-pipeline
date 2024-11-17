# Udacity course - Disaster Response Pipeline Project

## Table of Contents

- [Introduction](#introduction)
- [Project description](#description)
- [Repository overview](#repository-overview)
- [Instructions](#instructions)
- [ML Pipeline](#ml-pipeline)
- [Results analysis](#results)


## Introduction
The task of this project was create a model classifying received messages from various disaster events. Based on this classification the message can be then forwarded to adequate disaster response agency.


## Project description
This project consists of three main parts:

- ETL Pipeline - loads data from CSV, preprocesses them and stores into SQLite database.
- ML Pipeline - builds and trains a model to classify messages, then store it (in `pickle` file).
- Web application
    - Automatically loads database and visualizes some statistics from the source data.
    - Automatically loads model and classifies user's input message.


## Repository overview
This repository contains a machine learning pipeline designed to process, classify, and visualize disaster response messages. It includes ETL (Extract, Transform, Load) and ML (Machine Learning) pipelines, a web-based user interface for exploring the results, and relevant scripts for data preprocessing and model training.

### Directory and File Structure
- `.vscode/`
  - `launch.json` - Configuration for debugging in VS Code.
- `data/`
  - `categories.csv` - Dataset with disaster categories.
  - `messages.csv` - Dataset with disaster response messages.
  - `DisasterResponse.db` - SQLite database created during the ETL process.
  - `trained_model.pkl` - Serialized model file (trained classifier) - not present in repository due to its size. Users must generate it by themselves locally.
- `disaster_response_pipeline/`
  - `core/` - Core pipeline scripts
    - `custom_transformers.py` - Custom preprocessing transformers for ML pipeline.
    - `process_data.py` - ETL pipeline for data processing.
    - `train_classifier.py` - ML pipeline to build and save the model.
  - `ui/` - Web application files.
    - `templates/` - HTML templates for the Flask application.
      - `go.html`
      - `master.html`
    - `run.py` - Main script to run the Flask web application.
- `notebooks/`
  - `ETL Pipeline Preparation.ipynb` - Notebook for ETL pipeline development.
  - `ML Pipeline Preparation.ipynb` - Notebook for ML pipeline development.
  - `run.ipynb` - Notebook for running and testing the pipeline.
- `pyproject.toml` - Project metadata and dependencies configuration for Poetry.
- `README.md` - Project documentation.
- `requirements.txt` - List of Python dependencies for other virtual environments managers.


## Instructions
1. Install virtual environment.\
    `poetry install`
    - If you are not using poetry, you can utilize `requirements.txt` file.

2. Run the following commands in the project's root directory to set up database and model.

    - To run ETL pipeline that cleans data and stores in database\
        `python disaster_response_pipeline/core/process_data.py --messages-filepath data/messages.csv --categories-filepath data/categories.csv --database-filepath data/DisasterResponse`
    - To run ML pipeline that trains classifier and saves it\
        `python disaster_response_pipeline/core/train_classifier.py --database-filepath data/DisasterResponse --model_filepath data/trained_model`
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Run the following command to run the web app.\
    `python disaster_response_pipeline/ui/run.py`

4. Go to http://0.0.0.0:3001/

5. In the web application:
    - One can see two bar graphs with counts of messages genres (direct, news, social) and counts of each target category.
    - In the text input field user can type a message and generate its classification.
        - For example one might write: "We are starving here, need food"
        - Then you should get categories: Related, Request, Aid Related, Food and Direct Report.


## ML Pipeline
During the creation of pipeline, several options were considered and compared. For example checking whether there would be difference if we added a genre to the input or the original message. These attempts failed since the performance of the model was roughly the same, but the complexity was higher and also it became less practical, since user would have to make more input. 

Comparison between model with and without a transformer analyzing whether first word is a verb or similar. Based on precision and recall metrics, the one with transforer was chosen.

After the training of the pipeline, grid search was performed as well as randomized search (both with cross validation). These were tested for a low number of parameters since it is computationally very demanding. Best results were reached using randomized search with parameters about maximal tree depth and number of trees in random forest method. All of these attempts were always less precise than just the pipeline itself. But since it was a task to include that, it is appended in the model building part.

For anyone curious you can find all of these attempts in the notebooks directory, in `ML Pipeline Preparation.ipynb` file.


## Results analysis

Following table shows classification report for the pipeline used in the code. From this table we can deduce some facts.

Support is very helpful metrics. It shows us which target is literally supported and which one is not. For example "offer" has occured (meaning its value was 1) in the training dataset only 26 times. That tells us that even if the precision or recall values were great, it wouldn't of much help, since we can't be simple sure. In the other hand, categories like "related", "aid_related", "weather_related" or "micro_avg" have very strong support, therefore we can rely on these predictions more.

Precision explains how much of predicted positive values were actually positive. Meaning, if precision is low, it could easily happen that we will clasify a message which is related to food not just to food-related agency, but also for example to clothing-related one. It is therefore important if we want to avoid these "false positives".

Recall explains kind of vice-versa situation - how much of the actual positive values were correctly classified by the model. If recall would be low for some target, then for our food case, we wouldn't classify that case as a food case. We would perhaps classify it as another class, but not the real one. These are called "false negatives". They are undesirable especially in medicine - detecting a tumor is crucial, we can't fail on discovering this real positive. On the other hand, discovering a disease which patient doesn't have is not such a big deal.

F1 score is a harmonic mean of precision and recall. It is therefore useful when both precision and recall are equally important.

In our case, I say recall is the crucial one (and if not, then perhaps f1 score).

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>precision</th>
      <th>recall</th>
      <th>f1-score</th>
      <th>support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>related</th>
      <td>0.818447</td>
      <td>0.966788</td>
      <td>0.886454</td>
      <td>4938.0</td>
    </tr>
    <tr>
      <th>request</th>
      <td>0.829008</td>
      <td>0.491848</td>
      <td>0.617396</td>
      <td>1104.0</td>
    </tr>
    <tr>
      <th>offer</th>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>26.0</td>
    </tr>
    <tr>
      <th>aid_related</th>
      <td>0.762887</td>
      <td>0.681651</td>
      <td>0.719984</td>
      <td>2714.0</td>
    </tr>
    <tr>
      <th>medical_help</th>
      <td>0.694444</td>
      <td>0.046125</td>
      <td>0.086505</td>
      <td>542.0</td>
    </tr>
    <tr>
      <th>medical_products</th>
      <td>0.921053</td>
      <td>0.100865</td>
      <td>0.181818</td>
      <td>347.0</td>
    </tr>
    <tr>
      <th>search_and_rescue</th>
      <td>0.700000</td>
      <td>0.036458</td>
      <td>0.069307</td>
      <td>192.0</td>
    </tr>
    <tr>
      <th>security</th>
      <td>0.333333</td>
      <td>0.007519</td>
      <td>0.014706</td>
      <td>133.0</td>
    </tr>
    <tr>
      <th>military</th>
      <td>0.687500</td>
      <td>0.048889</td>
      <td>0.091286</td>
      <td>225.0</td>
    </tr>
    <tr>
      <th>water</th>
      <td>0.896552</td>
      <td>0.362791</td>
      <td>0.516556</td>
      <td>430.0</td>
    </tr>
    <tr>
      <th>food</th>
      <td>0.845824</td>
      <td>0.540356</td>
      <td>0.659432</td>
      <td>731.0</td>
    </tr>
    <tr>
      <th>shelter</th>
      <td>0.837719</td>
      <td>0.319398</td>
      <td>0.462470</td>
      <td>598.0</td>
    </tr>
    <tr>
      <th>clothing</th>
      <td>0.666667</td>
      <td>0.059406</td>
      <td>0.109091</td>
      <td>101.0</td>
    </tr>
    <tr>
      <th>money</th>
      <td>0.800000</td>
      <td>0.047619</td>
      <td>0.089888</td>
      <td>168.0</td>
    </tr>
    <tr>
      <th>missing_people</th>
      <td>1.000000</td>
      <td>0.014706</td>
      <td>0.028986</td>
      <td>68.0</td>
    </tr>
    <tr>
      <th>refugees</th>
      <td>0.555556</td>
      <td>0.024038</td>
      <td>0.046083</td>
      <td>208.0</td>
    </tr>
    <tr>
      <th>death</th>
      <td>0.862745</td>
      <td>0.149660</td>
      <td>0.255072</td>
      <td>294.0</td>
    </tr>
    <tr>
      <th>other_aid</th>
      <td>0.666667</td>
      <td>0.028777</td>
      <td>0.055172</td>
      <td>834.0</td>
    </tr>
    <tr>
      <th>infrastructure_related</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>419.0</td>
    </tr>
    <tr>
      <th>transport</th>
      <td>0.767442</td>
      <td>0.111111</td>
      <td>0.194118</td>
      <td>297.0</td>
    </tr>
    <tr>
      <th>buildings</th>
      <td>0.811321</td>
      <td>0.119777</td>
      <td>0.208738</td>
      <td>359.0</td>
    </tr>
    <tr>
      <th>electricity</th>
      <td>0.800000</td>
      <td>0.031496</td>
      <td>0.060606</td>
      <td>127.0</td>
    </tr>
    <tr>
      <th>tools</th>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>40.0</td>
    </tr>
    <tr>
      <th>hospitals</th>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>71.0</td>
    </tr>
    <tr>
      <th>shops</th>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>35.0</td>
    </tr>
    <tr>
      <th>aid_centers</th>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>72.0</td>
    </tr>
    <tr>
      <th>other_infrastructure</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>281.0</td>
    </tr>
    <tr>
      <th>weather_related</th>
      <td>0.860320</td>
      <td>0.682470</td>
      <td>0.761144</td>
      <td>1814.0</td>
    </tr>
    <tr>
      <th>floods</th>
      <td>0.917647</td>
      <td>0.438202</td>
      <td>0.593156</td>
      <td>534.0</td>
    </tr>
    <tr>
      <th>storm</th>
      <td>0.783375</td>
      <td>0.498397</td>
      <td>0.609207</td>
      <td>624.0</td>
    </tr>
    <tr>
      <th>fire</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>72.0</td>
    </tr>
    <tr>
      <th>earthquake</th>
      <td>0.889113</td>
      <td>0.727723</td>
      <td>0.800363</td>
      <td>606.0</td>
    </tr>
    <tr>
      <th>cold</th>
      <td>1.000000</td>
      <td>0.078571</td>
      <td>0.145695</td>
      <td>140.0</td>
    </tr>
    <tr>
      <th>other_weather</th>
      <td>0.888889</td>
      <td>0.024242</td>
      <td>0.047198</td>
      <td>330.0</td>
    </tr>
    <tr>
      <th>direct_report</th>
      <td>0.803309</td>
      <td>0.336931</td>
      <td>0.474742</td>
      <td>1297.0</td>
    </tr>
    <tr>
      <th>micro avg</th>
      <td>0.817318</td>
      <td>0.521689</td>
      <td>0.636869</td>
      <td>20771.0</td>
    </tr>
    <tr>
      <th>macro avg</th>
      <td>0.713327</td>
      <td>0.199309</td>
      <td>0.251005</td>
      <td>20771.0</td>
    </tr>
    <tr>
      <th>weighted avg</th>
      <td>0.776080</td>
      <td>0.521689</td>
      <td>0.561011</td>
      <td>20771.0</td>
    </tr>
    <tr>
      <th>samples avg</th>
      <td>0.754421</td>
      <td>0.628630</td>
      <td>0.555601</td>
      <td>20771.0</td>
    </tr>
  </tbody>
</table>
</div>
