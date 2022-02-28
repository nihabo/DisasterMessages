# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the project's root directory to run your web app.
    `python app/run.py`

3. Go to "http://<WORKSPACESPACEID>-3001.<WORKSPACEDOMAIN>" where WorkspaceID and WorkspaceDomain can be retrieved using command "env | grep WORK"
(e.g. view6914b2f4-3001.udacity-student-workspaces.com)

# Files in this repository
  app Folder
  	templates folder
  		go.html - used to show the results of the classification
  		master.html - main html file for the webapp
  	run.py - used to run the webapp, includes visualization code
  
  data Folder
  	DisasterResponse.db - Database which is storing the data used to perform Machine Learning on
  	disaster_categories.csv - Source file containing raw category data
  	disaster_messages.csv - Source file containing raw messages data
  	process_data.py - used to build up the database out of the two source files
  
  models Folder
  	classifier.pkl - contains the last trained classifier
  	train_classifier.py - used to build up the classifier based on the data preprocesses in process_data.py / stored in DisasterResponse.db

  num_category_values - statistical file containing number of items for each category
  resultsddmmYYYY_HHMMSS - containing the precision, f_score etc for last trained model + best Parameters found by GridSearch
  

  
  
  
### Remarks on the dataset
The messages are addtionally assigned to a genre out of "direct", "social" or "news". But there is an additional category "direct_report" its not clear which is the difference between this category 
  and the genre "direct".
  
Issues with special categories: 
  "Related" - almost all items have the category "related" (20042 out of 26182, see num_category_values), all of the row which are not assigned to category "related" (6122 values)
				are assigned to none other category as well --> Related column could be omitted
  			In addition there are some rows containing value 2 instead of 1. These rows should be cleaned. (Same as for value 0 all of the other categories are not assigned in this case)
  "Child_alone" - in the give dataset all message have value 0 for category child_alone. To be able to identify such messages by the trained model additional training data would be needed
  e.g. "Offer" - has a very limited number of messages being categorized as this category (only 121 out of 26k) this leads to the fact, that the model is taking is very rarely into account.
  			This leads to the fact that all messages are categorized as offer = 0 (which is shown results for offer_0 being 1.0 in any case and for offer_1 being 0, see results... file)
  

