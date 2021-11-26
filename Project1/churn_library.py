# library doc string


# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from PIL import ImageDraw


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''	
    df = pd.read_csv(pth)
    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    for column_names in df.keys():
        if df[column_names].dtypes != 'object':
            plt.figure(figsize=(20,10))
            plt.ylabel(f'{column_names}')
            fig_hist = df[column_names].hist().get_figure()
            fig_hist.savefig(f'./images/eda/{column_names}_hist.png')
            plt.figure(figsize=(20,10))
            plt.ylabel(f'{column_names}')
            fig_value_counts = df[column_names].value_counts('normalize').plot(kind='bar').get_figure()
            fig_value_counts.savefig(f'./images/eda/{column_names}_value_counts.png')
            plt.figure(figsize=(20,10))
            plt.ylabel(f'{column_names}')
            fig_distplot = sns.distplot(df[column_names]).get_figure()
            fig_distplot.savefig(f'./images/eda/{column_names}_distplot.png')
    
    plt.figure(figsize=(20,10))
    fig_heatmap = sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2).get_figure()
    fig_heatmap.savefig(f'./images/eda/heatmap.png')



def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    for category in category_lst:
        category_groups = df.groupby(category).mean()[response]
        df[f'{category}_{response}'] = [category_groups.loc[val] for val in df[category]]
    return df


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    y = df['Churn']
    X = df[response]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42)
    return X_train, X_test, y_train, y_test
    

def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    blank_image_rftr = Image.new('RGB', (400, 150))
    draw_blankIm_rftr = ImageDraw.Draw(blank_image)
    draw_blankIm_rftr.text((5,5), 'Random Forest Train')
    draw_blankIm_rftr.text((15, 15), classification_report(y_train, y_train_preds_rf))
    blank_image_rftr.save('./images/eda/RandomForest_train.png')
    
    blank_image_rfte = Image.new('RGB', (400, 150))
    draw_blankIm_rfte = ImageDraw.Draw(blank_image)
    draw_blankIm_rfte.text((5,5), 'Random Forest Test')
    draw_blankIm_rfte.text((15, 15), classification_report(y_test, y_test_preds_rf))
    blank_image_rfte.save('./images/eda/RandomForest_test.png')
    
    blank_image_lrtr = Image.new('RGB', (400, 150))
    draw_blankIm_lrtr = ImageDraw.Draw(blank_image)
    draw_blankIm_lrtr.text((5,5), 'Logistic Regression Train')
    draw_blankIm_lrtr.text((15, 15), classification_report(y_train, y_train_preds_lr))
    blank_image_lrtr.save('./images/eda/LogisticRegression_train.png')
    
    blank_image_lrte = Image.new('RGB', (400, 150))
    draw_blankIm_lrte = ImageDraw.Draw(blank_image)
    draw_blankIm_lrte.text((5,5), 'Logistic Regression Tests')
    draw_blankIm_lrte.text((15, 15), classification_report(y_test, y_test_preds_lr))
    blank_image_lrte.save('./images/eda/LogisticRegression_test.png')


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    pass

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    pass