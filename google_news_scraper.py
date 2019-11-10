import requests
from bs4 import BeautifulSoup
from newspaper import Article
import datetime
import pandas as pd
import time
import os
import numpy as np

URL = "https://www.google.com/search?q=bitcoin+cryptocurrency&hl=en&gl=us&as_drrb=b&tbas=0&tbs=cdr:1,cd_min:{min_date},cd_max:{max_date},sbd:1&tbm=nws&sxsrf=ACYBGNRfmviSo9arK1e_P_YIl5wsskZBPw:1574225634362&source=lnt&sa=X&ved=0ahUKEwj4wu29__flAhWV9Z4KHaKJAGcQpwUIIA&biw=1685&bih=863&dpr=1.1"

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/46.0.2490.80 Safari/537.36',
    'Content-Type': 'text/html',
}

max_count = 10  # max 10 news articles per day
fmt = '%m/%d/%Y'
news_cols = ['index', 'date', 'status_code', 'url', 'news_1_url', 'news_1_text',
             'news_1_publish_date', 'news_2_url', 'news_2_text', 'news_2_publish_date',
             'news_3_url', 'news_3_text', 'news_3_publish_date', 'news_4_url',
             'news_4_text', 'news_4_publish_date', 'news_5_url', 'news_5_text',
             'news_5_publish_date', 'news_6_url', 'news_6_text', 'news_6_publish_date',
             'news_7_url', 'news_7_text', 'news_7_publish_date', 'news_8_url',
             'news_8_text', 'news_8_publish_date', 'news_9_url', 'news_9_text',
             'news_9_publish_date']


def run_google_news_scrapper(**params):
    output_file_name = ''
    for key, value in params.items():
        if key == 'min_date':
            min_date = value
        if key == 'output_file':
            output_file_name = value

    news_data_dict = dict()
    columns = []
    news_data_dict['date'] = min_date
    columns.append('date')
    response = requests.get(URL.format(**params), headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    news_data_dict['status_code'] = response.status_code
    columns.append('status_code')
    if response.status_code != 200:
        print("******** fail ********** ")
        return
    # print(response.url)
    news_data_dict['url'] = response.url
    columns.append('url')
    count = 1
    for link in soup.find_all('a'):
        link_str = str(link.get('href'))
        try:
            if link_str.startswith("https://") and link_str.find('google.com') == -1 and link_str.find(
                    "https://www.youtube.com/") == -1 and link_str.find("https://www.blogger.com/") == -1:
                article = Article(link_str)
                article.download()
                article.parse()
                # print(link_str)
                # print(article.authors)
                # print(article.publish_date)
                # print(article.text)
                news_count = 'news_' + str(count)
                news_data_dict[news_count + '_url'] = link_str
                news_data_dict[news_count + '_text'] = article.text
                news_data_dict[news_count + '_publish_date'] = article.publish_date

                columns.append(news_count + '_url')
                columns.append(news_count + '_text')
                columns.append(news_count + '_publish_date')
                count += 1
                if count >= max_count:
                    break

        except:
            pass
    news_data_df = pd.DataFrame(news_data_dict, index=[0],
                                columns=news_cols)

    """
    if os.path.exists(output_file_name):
        keep_header = False
    else:
        keep_header = True

    news_data_df.to_csv(output_file_name, mode='a', header=keep_header)
    """
    news_data_df.to_csv(output_file_name)

    return news_data_dict


def google_news_scrapper(start_date, end_date, output_file_name):
    step_obj = datetime.timedelta(days=1)
    start_date_time_obj = datetime.datetime.strptime(start_date, fmt)
    end_date_time_obj = datetime.datetime.strptime(end_date, fmt)

    while start_date_time_obj <= end_date_time_obj:
        start_date = start_date_time_obj.strftime(fmt)
        print(start_date)
        run_google_news_scrapper(min_date=start_date, max_date=start_date, output_file=output_file_name)
        time.sleep(np.random.randint(2, 5))
        start_date_time_obj += step_obj


def sort_news_report(input_file_name, cleaned_output_file_name, save_index=False):
    df = pd.read_csv(input_file_name)
    df = df.set_index('date', drop=True)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().drop_duplicates(keep='first')
    df.to_csv(cleaned_output_file_name)
    if save_index:
        df_i = pd.DataFrame(df.index)
        df_i.to_csv(cleaned_output_file_name[0:-4] + '_index.csv')


def clean_news_report(input_file_name, cleaned_output_file_name, save_index=False):
    master_df = pd.read_csv(input_file_name)
    # get only given columns
    master_df = master_df[['date', 'news_1_text', 'news_2_text', 'news_3_text', 'news_4_text', 'news_5_text', 'news_6_text',
             'news_7_text', 'news_8_text', 'news_9_text']]
    master_df = master_df.set_index('date', drop=True)
    master_df.index = pd.to_datetime(master_df.index, format=fmt)
    # soft and drop duplicates
    master_df = master_df.sort_index().drop_duplicates(keep='first')
    idx = np.unique(master_df.index, return_index=True)[1]
    master_df = master_df.iloc[idx]
    master_df.to_csv(cleaned_output_file_name)

    master_df.to_csv(cleaned_output_file_name)
    if save_index:
        df_i = pd.DataFrame(master_df.index)
        df_i.to_csv(cleaned_output_file_name[0:-4] + '_index.csv')


if __name__ == "__main__":
    start_date = '01/01/2018'
    # end_date = datetime.datetime.now().strftime(fmt)
    end_date = '01/02/2019'
    news_raw_filename = 'google_news_final.csv'

    google_news_scrapper(start_date, end_date, news_raw_filename)
    news_cleaned_filename = news_raw_filename[0:-4] + '_cleaned.csv'
    clean_news_report(news_raw_filename, news_cleaned_filename)
