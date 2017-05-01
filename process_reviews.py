#Simple code that takes a given bussiness category and finds reviews related to that category of business
#and stores the reviews in two files: positive.txt and negative.txt
import json
import ijson
import time
import pprint
import csv
import sys

yelp_file_business = "yelp_academic_dataset_business.json"
yelp_review_file = "yelp_academic_dataset_review.json"
targetCategory = 'Restaurants' #Type of business to look for
max_reviews = 2500 #Number of reviews required in each of positive and negative categories 
yelp_similar_businesses = []

def getReviews():
  #This will again go through the entire file line by line and look for businesses with similar categories
  with open(yelp_file_business) as f:
    for line in f:
      temp_business = json.loads(line)
      temp_categories = temp_business["categories"]
      if not temp_categories is None:
        if targetCategory in temp_categories:   #looking for businesses that have atleas 3 categories in common
            yelp_similar_businesses.append(temp_business["business_id"])


  # the list yelp_similar_businesses will now contain all similar bussiness
  print 'No of businesses: ',len(yelp_similar_businesses)

  outfile_pos = open("positive.txt",'a')
  outfile_neg = open("negative.txt",'a')
  pos_review_count = 0
  neg_review_count = 0
  #writer = csv.writer(outfile)
  list_negative = []
  list_possitive = []
  i = 0
  #Now get reviews for the collected businesses
  with open(yelp_review_file) as f:
    for line in f:
      temp_review = json.loads(line)
      temp_business_id = temp_review["business_id"]
      temp_useful_count = int(temp_review["useful"])
      temp_rating = float(temp_review["stars"])
      if temp_business_id in yelp_similar_businesses and temp_useful_count > 1:
        #print 'restaurant found'
        if temp_rating <= 2:
          review_text = temp_review["text"].encode('utf-8')
          t2 = review_text.replace('\n', ' ').replace('\r', '')
          line = str(t2+"\n")
          outfile_neg.write(line)
          neg_review_count += 1

        elif temp_rating >= 4:
          review_text = temp_review["text"].encode('utf-8')
          t2 = review_text.replace('\n', ' ').replace('\r', '')
          line = str(t2+"\n")
          outfile_pos.write(line)
          pos_review_count += 1
        if pos_review_count >= max_reviews and neg_review_count >= max_reviews:
          break
        if pos_review_count % 1000 == 0:
              print 'found ',pos_review_count, 'positive reviews', neg_review_count, 'negative reviews...'
  print 'Positive reviews: ',pos_review_count,' Negative reviews: ',neg_review_count
  outfile_pos.close()
  outfile_neg.close()

#MAIN
if __name__ == '__main__':
  if len(sys.argv) >= 3:
    targetCategory = sys.argv[1]
    max_reviews = int(sys.argv[2])
  getReviews()
  

