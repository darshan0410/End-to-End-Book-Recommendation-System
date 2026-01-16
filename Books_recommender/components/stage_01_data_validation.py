import os
import sys
import ast 
import pandas as pd
import pickle
from Books_recommender.logger.log import logging
from Books_recommender.config.configuration import AppConfiguration
from Books_recommender.exception.exception_handler import AppException



class DataValidation:
    def __init__(self, app_config = AppConfiguration()):
        try:
            self.data_validation_config= app_config.get_data_validation_config()
        except Exception as e:
            raise AppException(e, sys) from e


    
    def preprocess_data(self):
        try:
            ratings = pd.read_csv(self.data_validation_config.ratings_csv_file, sep=",")
            books = pd.read_csv(self.data_validation_config.books_csv_file, sep=",")
            
            logging.info(f" Shape of ratings data file: {ratings.shape}")
            logging.info(f" Shape of books data file: {books.shape}")

            #Here Image URL columns is important for the poster. So, we will keep it
            books = books[['ISBN','Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher','Image-URL-L']]

            books.rename(columns={"Book-Title":'Title',
                                'Book-Author':'Author',
                                "Year-Of-Publication":'Year',
                                "Publisher":"Publisher",
                                "Image-URL-L":"Image_url"},inplace=True)

            
            ratings.rename(columns={"User-ID":'User_id',
                                'Book-Rating':'Rating'},inplace=True)

            # Lets store users who had at least rated more than 200 books
            x = ratings['User_id'].value_counts() > 200
            y = x[x].index
            ratings = ratings[ratings['User_id'].isin(y)]

            # Now join ratings with books
            ratings_with_books = ratings.merge(books, on='ISBN')
            number_rating = ratings_with_books.groupby('Title')['Rating'].count().reset_index()
            number_rating.rename(columns={'Rating':'num_of_ratings'},inplace=True)
            final_rating = ratings_with_books.merge(number_rating, on='Title')

            # Lets take those books which got at least 50 rating of user
            final_rating = final_rating[final_rating['num_of_ratings'] >= 50]

            # lets drop the duplicates
            final_rating.drop_duplicates(['User_id','Title'],inplace=True)
            logging.info(f" Shape of the final clean dataset: {final_rating.shape}")
                        
            # Saving the cleaned data for transformation
            os.makedirs(self.data_validation_config.clean_data_dir, exist_ok=True)
            final_rating.to_csv(os.path.join(self.data_validation_config.clean_data_dir,'clean_data.csv'), index = False)
            logging.info(f"Saved cleaned data to {self.data_validation_config.clean_data_dir}")


            #saving final_rating objects for web app
            os.makedirs(self.data_validation_config.serialized_objects_dir, exist_ok=True)
            pickle.dump(final_rating,open(os.path.join(self.data_validation_config.serialized_objects_dir, "final_rating.pkl"),'wb'))
            logging.info(f"Saved final_rating serialization object to {self.data_validation_config.serialized_objects_dir}")

        except Exception as e:
            raise AppException(e, sys) from e

    
    def initiate_data_validation(self):
        try:
            logging.info(f"{'='*20}Data Validation log started.{'='*20} ")
            self.preprocess_data()
            logging.info(f"{'='*20}Data Validation log completed.{'='*20} \n\n")
        except Exception as e:
            raise AppException(e, sys) from e



    