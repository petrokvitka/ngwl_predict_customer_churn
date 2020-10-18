![Alt Text](https://github.com/petrokvitka/ngwl_predict_customer_churn/blob/main/peach_team_logo.gif)

# Peach team Predicting customer churn for NGWL Hackathon

This is the code we use to generate our submission.

## Preprocessing the data

The preprocessing of the data is done in three steps, whereas in each step the aim is to create a table with phone_id and corresponding information, such as other ids, extracted data or custom features.
First, the script ``extract_categories.py`` is used to detect in future the connections between likelihood of purchasing an item and belonging of this item to a certain category.

Finally, the ``data_merge_and_train.ipynb`` Jupyter notebook is used for the merging the data into one data set and training of models.

After the data set is created to optimize the training, the features are tested to be expressive or not. This can be done with the Jupyter notebook ``statistical_analysis.ipynb``.
