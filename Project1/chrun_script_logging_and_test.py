import os
import logging
from churn_library import *
import glob

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import(import_data):
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	try:
		df = import_data("./data/bank_data.csv")
		logging.info("Testing import_data: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing import_eda: The file wasn't found")
		raise err

	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError as err:
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err


def test_eda(perform_eda):

    try:
#         df = import_data("./data/bank_data.csv")
#         perform_eda(df)
        assert len(os.listdir('./images/eda/') ) > 0
        logging.info('Images were created: SUCCESS')
    except AssertionError as err:
        logging.error("Testing perform_eda: Plots were not created, check your data")
    
    try:
        data_verification = [os.path.getsize(file)>0 for file in glob.glob('./images/eda/*.png')]
        assert any(x == True for x in data_verification)
        logging.info('Image were created and not empty: SUCCESS')
    except AssertionError as err:
        logging.error("Testing perform_eda: Plots were created some has data size 0 kb")            

def test_encoder_helper(encoder_helper):
	'''
	test encoder helper
	'''


def test_perform_feature_engineering(perform_feature_engineering):
	'''
	test perform_feature_engineering
	'''


def test_train_models(train_models):
	'''
	test train_models
	'''


if __name__ == "__main__":
    print(test_import(import_data))
    print(test_eda(perform_eda))








