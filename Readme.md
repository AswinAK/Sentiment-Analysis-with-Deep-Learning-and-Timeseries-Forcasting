DeepLearning With Tensor Flow:  

DATASET:  
Yelp data set available at https://www.yelp.com/dataset_challenge (Not included here as it is ~5 GB when unzipped)  

The following the deliverables for this section:  
process_reviews.py  
deeplearning.py  

DEPENDENCIES:  
tensorflow  
json  
numpy  
nltk  


HOW TO RUN OUR CODE:  
Before we run train the model, we need to get the data for training, this is done by the process_reviews.py  
This will generate the positve.txt and negative.txt file from the Yelp dataset.  
For convenience, we have included these files along with our code.  
  
So you can directly run deeplearning.py to train the neural network.  
If you dont have positive.txt and negative.txt, run process_reviews.py first to get these two files.  

  
process_reviews.py is python script to processes Yelp JSON data and create two files: positive.txt and negative.txt  
which contains negative and positive reviews for a particular type of business.  
  
To run:  
	python process_reviews.py   (to run with default parameters: collects 2500 positive and negative reviews for Restaurants)  
OR   
	python process_reviews.py [targget business category] [# of reviews required]  


deeplearning.py does the feature set creation and then trains our neural network. All the parametes like   
number of neurons, number of epochs, batch size are defined with global variables at the top of the file.  
These can be changed to train the model with different parameters.  

To run:  
	python deeplearning.py  
  
Forcasting:  
The deliverables are:  
forecast.r  
timeseries_filter.py  

timeseries_filter.py will go through the reviews data set and find out how many reviews were made each month for Restaurants  
startin from a specefic year and within a specified range of star rating  

To run:  
	python timeseries_filter.py (to run with default values: gets reviews with 0 to 2 stars, begining from the year 2009)  
OR  
	python timeseries_filter.py [begin year] [min rating] [max rating] [output file name]  

Then run the R code in R Studio  




