** Please note: the data file used by main.py is too large to be uploaded to the repo. Please contact Jenna if you want to access it. **
# NLP-based Hotel Recommendation System
### Computational Data Science (CMPT353) Final Project 

Submitted by
Setu Patel, Jenna Liebe, Luna Sang

## About the project

Many people rely on the reviews of other users like themselves, but sometimes it can be hard to compare between hotels with all the options out there. It can also be difficult to wade through the many reviews left, and discern if one bad review is trustworthy compared to all the positive reviews. With this project, we aim to simplify the process of searching for the best possible hotel by using the powers of data analysis and machine learning. Most importantly, hotel reviews obtained from real guests of those hotels are analysed using NLP based machine learning model. Our hotel recommendation system provides list of top hotels for the user's choice of city for their stay.

## Getting Started  

To get a local copy up and running follow these simple example steps.

#### Clone the repository

Run the following command to clone the repository in your preferred directory

`git clone https://github.sfu.ca/dsa133/CMPT353_Project.git`

Then enter into the project repository using `cd CMPT353_Project` command.

#### Installing dependencies

Run the following command to install all dependencies required to interact with the program.

`pip install -r requirements.txt`

Note: 
- You may need set environment variables to resolve path issues.
- On Windows, you may need to run the above command using `--user` option or check the permissions to access certain directory to allow installation.

Once all dependencies are resolved, you are all set to interact with the program

## Usage

Use the following command to run the program from the project directory

`python main.py`

You may see some warning messages from pyspark, please ignore them.

#### User Inputs

Please enter your responses as directed from the list of choices provided in the displayed message.

This is the list of expected responses from the user based on choices
- Choice of city to get the hotel recommendation for
- User choice of category of service to consider while providing the hotel recommendations
- View ratings for a specific hotel


#### Outputs

The program will train Naive Bayes model and provide hotel recommendations based on user choices. It will also show wordclouds for each of the top suggested hotels based on guests reviews to give user highlights for the reviews.

The wordcloud and the list of hotel recommendation along with some additional hotel information generated during the program run are saved in output directory for users to look at closely.

## Directory Structure

- All source code files driving the program are placed in [src](src/) directory
- The outputs such as hotel recommendations and wordclouds generated while running the program are saved in [output](output/) directory
- The cleaned data obtained after ETL operations and dataset used in ML are saved in [filtered_data](filtered_data/) directory
- Hotel review dataset used for analysis along with results of ML model predictions are placed in [predicted_data](predicted_data/) directory
- Sample wordcloud and exported csv output from program run are saved in [output](output/) directory
- Some files used to generate our report with Latex formatting are saved in [report_related](report_related/) directory
