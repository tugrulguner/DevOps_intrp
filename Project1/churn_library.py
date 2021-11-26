'''
Churn Clean Code Project

Author: Tugrul Guner
Date: 25 Nov 2021
'''


# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_roc_curve, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from PIL import ImageDraw, Image


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
    # loop over DataFrame keys
    for column_names in df.keys():

        # To eliminate non-string columns
        if df[column_names].dtypes != 'object':

            # Plot and save the Histograms
            plt.figure(figsize=(20, 10))
            plt.ylabel(f'{column_names}')
            fig_hist = df[column_names].hist().get_figure()
            fig_hist.savefig(f'./images/eda/{column_names}_hist.png')

            # Plot and save the VALUE COUNTS plots
            plt.figure(figsize=(20, 10))
            plt.ylabel(f'{column_names}')
            fig_value_counts = df[column_names].value_counts(
                'normalize').plot(kind='bar').get_figure()
            fig_value_counts.savefig(
                f'./images/eda/{column_names}_value_counts.png')

            # Plot and save the Seaborn distplot
            plt.figure(figsize=(20, 10))
            plt.ylabel(f'{column_names}')
            fig_distplot = sns.distplot(df[column_names]).get_figure()
            fig_distplot.savefig(f'./images/eda/{column_names}_distplot.png')

    # Plot and save the Seaborn Heatmap
    plt.figure(figsize=(20, 10))
    fig_heatmap = sns.heatmap(
        df.corr(),
        annot=False,
        cmap='Dark2_r',
        linewidths=2).get_figure()
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

        # Groupby each category and generate new categorized columns
        category_groups = df.groupby(category).mean()[response]
        df[f'{category}_{response}'] = [category_groups.loc[val]
                                        for val in df[category]]

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

    # Prepare the target and input values
    target_value = df['Churn']
    input_value = df[response]

    # Usee train_test_split to get train and test data
    X_train, X_test, y_train, y_test = train_test_split(input_value,
                                                        target_value,
                                                        test_size=0.3,
                                                        random_state=42)

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

    # Random Forest Train Classification Report
    blank_image_rftr = Image.new('RGB', (400, 150))
    draw_blankIm_rftr = ImageDraw.Draw(blank_image_rftr)
    draw_blankIm_rftr.text((5, 5), 'Random Forest Train')
    draw_blankIm_rftr.text(
        (15, 15), classification_report(
            y_train, y_train_preds_rf))
    blank_image_rftr.save('./images/eda/RandomForest_train.png')

    # Random Forest Test Classification Report
    blank_image_rfte = Image.new('RGB', (400, 150))
    draw_blankIm_rfte = ImageDraw.Draw(blank_image_rfte)
    draw_blankIm_rfte.text((5, 5), 'Random Forest Test')
    draw_blankIm_rfte.text(
        (15, 15), classification_report(
            y_test, y_test_preds_rf))
    blank_image_rfte.save('./images/eda/RandomForest_test.png')

    # Logistic Regression Train Classification Report
    blank_image_lrtr = Image.new('RGB', (400, 150))
    draw_blankIm_lrtr = ImageDraw.Draw(blank_image_lrtr)
    draw_blankIm_lrtr.text((5, 5), 'Logistic Regression Train')
    draw_blankIm_lrtr.text(
        (15, 15), classification_report(
            y_train, y_train_preds_lr))
    blank_image_lrtr.save('./images/eda/LogisticRegression_train.png')

    # Logistic Regression Test Classification Report
    blank_image_lrte = Image.new('RGB', (400, 150))
    draw_blankIm_lrte = ImageDraw.Draw(blank_image_lrte)
    draw_blankIm_lrte.text((5, 5), 'Logistic Regression Tests')
    draw_blankIm_lrte.text(
        (15, 15), classification_report(
            y_test, y_test_preds_lr))
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

    # Calculate feature importances
    importances = model.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    # Save the Figure
    plt.savefig(f'{output_pth}Feature_Importance.png')


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
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression()

    param_grid = {
        'n_estimators': [200],
        'max_features': ['auto'],
        'max_depth': [4],
        'criterion': ['gini', 'entropy']
    }

    # Grid Search and Fit -- Random Forest
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    # Logistic Regression Fit
    lrc.fit(X_train, y_train)

    # Best estimator predictions
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # scores
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    # Plot and save Logistic Regression Model Score
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    plt.savefig('./images/results/lrc_plot.png')

    # plots
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    plot_roc_curve(
        cv_rfc.best_estimator_,
        X_test,
        y_test,
        ax=ax,
        alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig('./images/results/lrc_rfc.png')

    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    rfc_model = joblib.load('./models/rfc_model.pkl')
    lr_model = joblib.load('./models/logistic_model.pkl')

    # Plot and Save Loaded Logistic Regression Model
    plt.figure(figsize=(15, 8))
    lrc_plot = plot_roc_curve(lr_model, X_test, y_test)
    plt.savefig('./images/results/lr_model_plot.png')

    # Plot and Save Loaded Logistic Reg. and Random Forest Models
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    plot_roc_curve(rfc_model, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig('./images/results/lrc_rfc_best.png')
