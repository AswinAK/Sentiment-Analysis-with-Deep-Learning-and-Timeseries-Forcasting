import json
import ijson
import time
from pprint import pprint
import csv
import operator
from dateutil.parser import parse
import sys

MIN_RATING = 0
MAX_RATING = 2
BEGIN_YEAR = 2009
output_file_name = "timeseries.txt"

yelp_file_business = "/Users/AswinAk/Downloads/yelp_dataset_challenge_round9/yelp_academic_dataset_business.json"
yelp_review_file = "/Users/AswinAk/Downloads/yelp_dataset_challenge_round9/yelp_academic_dataset_review.json"

yelp_similar_businesses = []
target_business_object = None

def getTag(year, month):
  month_label = None
  if month == 1:
    month_label = '01'
  if month == 2:
    month_label = '02'
  if month == 3:
    month_label = '03'
  if month == 4:
    month_label = '04'
  if month == 5:
    month_label = '05'
  if month == 6:
    month_label = '06'
  if month == 7:
    month_label = '07'
  if month == 8:
    month_label = '08'
  if month == 9:
    month_label = '09'
  if month == 10:
    month_label = '10'
  if month == 11:
    month_label = '11'
  if month == 12:
    month_label = '12'

  return str(year) + '_' + str(month_label)

#This will read the json file line by line and find restaurant bussinesses in a particular city
def generateTimeSeries():
  targetCategory = 'Restaurants'
  countDict = {}

  #Getting counts of restaurants in different cities
  with open(yelp_file_business) as f:
    for line in f:
      temp_business = json.loads(line)
      temp_categories = temp_business["categories"]
      temp_city = temp_business["city"]
      if not temp_categories is None:
        if targetCategory in temp_categories:  #looking for businesses that have atleas 3 categories in common
            if countDict.has_key(temp_city):
              countDict[temp_city] = countDict[temp_city]+1
            else:
              countDict[temp_city] = 1

  #Sorting the dictionary by value(no of restaurants in this case)
  ss = sorted(countDict.items(), key=lambda x:x[1])

  #Getting business IDs of all businesses in Toronto(bottom of the list in descending order sorting)
  with open(yelp_file_business) as f:
    for line in f:
      temp_business = json.loads(line)
      temp_categories = temp_business["categories"]
      temp_city = temp_business["city"]
      if not temp_categories is None:
        if targetCategory in temp_categories and temp_city == ss[-1][0]: 
          yelp_similar_businesses.append(temp_business["business_id"])
            
  print 'highest number of businesses in ', ss[-1][0]
  print 'second highest number of businesses in ', ss[-2][0]
  print 'total no of businesses: ',len(yelp_similar_businesses) 

  #
  review_dict = {}
  review_count = 0
  start_time = time.time()
  i = 0
  with open(yelp_review_file) as f:
    for line in f:
      # i += 1
      # if i % 1000 == 0:
      #   print 'processed ',i, ' reviews'
      # if i == 5000:
      #   break
          
      temp_review = json.loads(line)
      temp_business_id = temp_review["business_id"]

      temp_date = temp_review["date"]
      date_obj = parse(temp_date)
      review_year = date_obj.year
      review_month = date_obj.month

      review_stars = int(temp_review["stars"])

      tag = getTag(review_year,review_month)
      if date_obj.year >= BEGIN_YEAR and temp_business_id in yelp_similar_businesses and review_stars >= MIN_RATING and review_stars <= MAX_RATING:
        review_count += 1
        if review_dict.has_key(tag):
          review_dict[tag] = review_dict[tag] + 1
        else:
          review_dict[tag] = 1

        
  print 'review count', review_count
  end_time = time.time()
  print'Took ', end_time - start_time, 'seconds'

  outfile_ts = open(output_file_name,'a')
  keylist = review_dict.keys()
  keylist.sort()
  for key in keylist:
    print key, '  ', review_dict[key]
    outfile_ts.write(str(review_dict[key])+"\n")
  outfile_ts.close()

if __name__ == '__main__':
  if len(sys.argv) >= 5:
    BEGIN_YEAR = int(sys.argv[1])
    MIN_RATING = int(sys.argv[2])
    MAX_RATING = int(sys.argv[3])
    output_file_name = sys.argv[4]
    print 'BEGIN: ',BEGIN_YEAR,' min: ',MIN_RATING,' max: ',MAX_RATING, 'file: ',output_file_name
  generateTimeSeries()

