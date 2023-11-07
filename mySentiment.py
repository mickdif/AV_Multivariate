from doctest import debug
from gnews import GNews # scrape
from deep_translator import GoogleTranslator # translate
from textblob import TextBlob # evaluate

class Sentiment_Analysis:
    def __init__(self, query, language, country, start, end):
        self.google_news = GNews(language=language, country=country, start_date=start, end_date=end, max_results=1000)         

        # altri esempi che danno problemi
        # self.google_news = GNews(language=language, country=country, start_date=(2019, 1, 1), end_date=(2020, 1, 1), max_results=1000) 
        # self.google_news = GNews(language=language, country=country, period="5y") 
        self.json_resp = self.google_news.get_news(query)
        self.gt = GoogleTranslator(source=language, target='en')
        self.idx = 0

    def do_Analysis(self, date, debug=False): # per singolo giorno
        polarity_tot = 0.0

        while self.idx < len(self.json_resp):
            article = self.google_news.get_full_article(self.json_resp[self.idx]['url']) # newspaper3k instance
            
            if article.publish_date  != date:
                if debug == True: print(article.publish_date, " != ", date)
                break

            self.idx += 1
            
            try:
                # commentato per mettere in risalto altri problemi
                # if self.query not in article.title: 
                # if debug == True: print(self.query, " non in ", article.title)
                    # continue

                # open and translate
                text = article.text
                text = self.gt.translate(text=text)
                
                if debug == True:
                    print("debug")
                    print(article.title)
                    print(article.publish_date)
                
            except:
                try:
                    text = article.description
                    text = self.gt.translate(text=text)
                except :
                    if debug == True: print("errore nell'apertura dell'articolo")
                    continue # ignore exceptions
                
            try:
                analysis = TextBlob(text)
                polarity_tot += analysis.sentiment.polarity
                if debug == True: print(analysis.sentiment) 
            except:
                if debug == True: print("errore nella traduzione")
                continue # ignore exceptions
           
        return polarity_tot     