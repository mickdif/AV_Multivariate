from gnews import GNews # scrape
from deep_translator import GoogleTranslator # translate
from textblob import TextBlob # evaluate

class Sentiment_Analysis:
    def __init__(self):   
        self.gt = GoogleTranslator(source='it', target='en')
        
    def do_Analysis(self, query, start, end, debug=False):
        google_news = GNews(language='it', country='it', start_date=start, end_date=end, max_results=3)
        json_resp = google_news.get_news(query)
        polarity_tot = 0.0
        aperti = 0
        for idx in range(len(json_resp)):
            article = google_news.get_full_article(json_resp[idx]['url'])            
            try:
                if debug == True: print(article.title)
                # open and translate
                text = article.text
                text = self.gt.translate(text=text)                
                
            except:
                try: # se non riesce col testo prova col solo titolo
                    text = article.title
                    text = self.gt.translate(text=text)
                except :
                    if debug == True: print("errore nell'apertura o traduzione dell'articolo")
                    continue # ignore exceptions
            try:
                analysis = TextBlob(text)
                polarity_tot += analysis.sentiment.polarity 
            except:
                if debug == True: print("errore nell'analisi")
                continue # ignore exceptions
            if analysis.sentiment.polarity != 0: 
                aperti += 1.0
                if debug == True: print(analysis.sentiment.polarity)
            
        if debug == True: print("Analizzati correttamente: ", aperti, " su ", len(json_resp))         
        if aperti != 0: return polarity_tot/aperti
        else: return 0.0
