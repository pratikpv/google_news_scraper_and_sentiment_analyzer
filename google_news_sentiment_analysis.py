import pandas as pd
import flair
from textblob import TextBlob
import os
import datetime
import numpy as np

fmt = '%Y-%m-%d'


def get_sentiment_val_for_flair(sentiments):
    """
    parse input of the format [NEGATIVE (0.9284018874168396)] and return +ve or -ve float value
    :param sentiments:
    :return:
    """
    total_sentiment = str(sentiments)
    neg = 'NEGATIVE' in total_sentiment
    if neg:
        total_sentiment = total_sentiment.replace('NEGATIVE', '')
    else:
        total_sentiment = total_sentiment.replace('POSITIVE', '')

    total_sentiment = total_sentiment.replace('(', '').replace('[', '').replace(')', '').replace(']', '')

    val = float(total_sentiment)
    if neg:
        return -val
    return val


def add_to_dict(final_dict, input_dict):
    """
    add matching key values and store final result in final_dict
    :param final_dict:
    :param input_dict:
    :return:
    """
    for item in final_dict:
        input_dict_val = input_dict.get(item, 0)
        final_dict[item] += input_dict_val


def devide_dict_by_scaler(in_dict, val):
    """
    devide each value of dict by scaler
    :param in_dict:
    :param val:
    :return:
    """
    for item in in_dict:
        in_dict[item] /= val


def get_sentiment_report(input_filename, output_filename, start_date=None, simulate=False):
    """

    :param data_df: input data is panda dataframe, with index as date of the format fmt
    :return: another dataframe with same index as input dataframe and new columns as sentiment values
    """
    data_df = pd.read_csv(input_filename, index_col=0)
    col = data_df.columns

    # sid = SentimentIntensityAnalyzer()
    if simulate:
        flair_sentiment = None
    else:
        flair_sentiment = flair.models.TextClassifier.load('en-sentiment')
    temp_c = 0
    for row_i, row in data_df.iterrows():
        temp_c += 1
        # print(row_i)

        if start_date is not None:
            start_date_time_obj = datetime.datetime.strptime(start_date, fmt)
            current_date_time_obj = datetime.datetime.strptime(str(row_i), fmt)

            if current_date_time_obj < start_date_time_obj:
                print('Skipping record of date ', str(current_date_time_obj), ' But looking for ',
                      str(start_date_time_obj))
                continue

        total_sentiment_data_count = 0
        tb_sentiment_polarity_dict = dict()
        tb_sentiment_subjectivity_dict = dict()
        flair_sentiment_dict = dict()

        flair_sentiment_total = 0

        tb_polarity_total = 0
        tb_subjectivity_total = 0

        # sid_polarity_total = {'neg': 0., 'neu': 0., 'pos': 0., 'compound': 0.}

        for col_i in range(len(col)):
            data = (str(row[col_i]))
            # print('\t', col_i)
            if data == 'NaN':
                continue

            if simulate:
                flair_sentiment_total = 5
                tb_polarity_total = 6
                tb_subjectivity_total = 7
                total_sentiment_data_count = 9
            else:
                # sid_polarity = sid.polarity_scores(data)
                # add_to_dict(sid_polarity_total, sid_polarity)

                tb_polarity_total += TextBlob(data).sentiment[0]
                tb_subjectivity_total += TextBlob(data).sentiment[1]

                flair_s = flair.data.Sentence(data)
                flair_sentiment.predict(flair_s)
                flair_total_sentiment = flair_s.labels
                flair_val = get_sentiment_val_for_flair(flair_total_sentiment)
                flair_sentiment_total += flair_val
                total_sentiment_data_count += 1

                # print(flair_sentiment_total / total_sentiment_data_count, tb_polarity_total / total_sentiment_data_count,
                #      tb_subjectivity_total / total_sentiment_data_count)

        print(str(row_i), ' ', temp_c)
        flair_sentiment_dict[str(row_i)] = flair_sentiment_total / total_sentiment_data_count
        tb_sentiment_polarity_dict[str(row_i)] = tb_polarity_total / total_sentiment_data_count
        tb_sentiment_subjectivity_dict[str(row_i)] = tb_subjectivity_total / total_sentiment_data_count
        # devide_dict_by_scaler(sid_polarity_total, total_sentiment_data_count)
        print(flair_sentiment_dict[str(row_i)], tb_sentiment_polarity_dict[str(row_i)],
              tb_sentiment_subjectivity_dict[str(row_i)])

        flair_df = pd.DataFrame.from_dict(flair_sentiment_dict, orient='index', columns=['gnews_flair'])
        flair_df.index.name = 'date'

        tb_polarity_df = pd.DataFrame.from_dict(tb_sentiment_polarity_dict, orient='index',
                                                columns=['gnews_tb_polarity'])
        tb_polarity_df.index.name = 'date'

        tb_subjectivity_df = pd.DataFrame.from_dict(tb_sentiment_subjectivity_dict, orient='index',
                                                    columns=['gnews_tb_subjectivity'])
        tb_subjectivity_df.index.name = 'date'

        final_senti_df = pd.concat([flair_df, tb_polarity_df, tb_subjectivity_df], axis=1)

        if os.path.exists(output_filename):
            keep_header = False
        else:
            keep_header = True

        final_senti_df.to_csv(output_filename, mode='a', header=keep_header)

    return


def clean_sentiment_report(input_filename, output_filename):
    # drop duplicates and sort
    master_df = pd.read_csv(input_filename, index_col=0)
    master_df.index = pd.to_datetime(master_df.index)
    idx = np.unique(master_df.index, return_index=True)[1]
    master_df = master_df.iloc[idx]
    master_df.to_csv(output_filename)


if __name__ == "__main__":
    input_filename = 'google_news_final.csv'
    output_filename = input_filename[0:-4] + '_sentiment.csv'
    get_sentiment_report(input_filename, output_filename, simulate=False)
    clean_sentiment_report(output_filename, output_filename)
