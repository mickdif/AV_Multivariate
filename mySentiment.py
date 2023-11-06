from doctest import debug
from gnews import GNews # scrape
from deep_translator import GoogleTranslator # translate
from textblob import TextBlob # evaluate
import numpy as np


def Sentiment_Analysis(query, language, country, start, end, debug=False):

    google_news = GNews(language=language, country=country, start_date=start, end_date=end, max_results=10)
    json_resp = google_news.get_news(query)

    polarity_tot = 0
    polarity_array = np.array('')
    well_read = 0 # articoli aperti senza errori
    
    gt=GoogleTranslator(source='it', target='en')

    for idx in range(len(json_resp)):
        article = google_news.get_full_article(json_resp[idx]['url']) # newspaper3k instance
    
        try:
            if query not in article.title:
                break
            # open and translate
            text = article.text
            text = gt.translate(text=text)
 
            # Sentiment Analysis
            # The sentiment property returns a namedtuple of the form Sentiment(polarity, subjectivity). 
            # The polarity score is a float within the range [-1.0, 1.0]. 
            # The subjectivity is a float within the range [0.0, 1.0] where 0.0 is very objective and 1.0 is very subjective. (from api reference)
            analysis = TextBlob(text)
            polarity_tot += analysis.sentiment.polarity
            polarity_array = np.append(polarity_array, analysis.sentiment.polarity)        
            
            well_read += 1 

            if debug == True: 
                print(article.title)
                print(analysis.sentiment) 
            
        except: 
            continue # ignore exceptions


    if well_read == 0:
        return 0.0
    else:
        return polarity_tot, polarity_array