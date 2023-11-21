from gnews import GNews # scrape
from deep_translator import GoogleTranslator # translate
from textblob import TextBlob # evaluate

fmt = "%d/%m/%Y"


class Sentiment_Analysis:
    def __init__(self, query, start, end):   

        self.google_news = GNews(language='it', country='it', start_date=start, end_date=end, max_results=3)
        self.json_resp = self.google_news.get_news(query)
        self.polarity_tot = 0.0
        self.gt = GoogleTranslator(source='it', target='en')
    def do_Analysis(self, debug=False):
        aperti = 0
        for idx in range(len(self.json_resp)):
            article = self.google_news.get_full_article(self.json_resp[idx]['url'])            
            try:
                if debug == True: print(article.title)
                # open and translate
                text = article.text
                text = self.gt.translate(text=text)
                if debug == True: print("art tradotto")
                
                
            except:
                try:
                    text = article.title
                    text = self.gt.translate(text=text)
                    if debug == True: print("titolo tradotto")
                except :
                    if debug == True: print("errore nell'apertura o traduzione dell'articolo")
                    continue # ignore exceptions
                
            try:
                analysis = TextBlob(text)
                self.polarity_tot += analysis.sentiment.polarity
                if debug == True: print(self.polarity_tot) 
            except:
                if debug == True: print("errore nell'analisi")
                continue # ignore exceptions
            aperti += 1.0
         
        if aperti != 0: return self.polarity_tot/aperti
        else: return 0.0
