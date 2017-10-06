from libbgg.apiv2 import BGG
import nltk
import numpy as np
import json
import pandas as pd
import requests
import re
from time import sleep
#217974, 199908

def get_ids(pages):
    id_list = []
    for i in range(pages):
        f = requests.get("https://boardgamegeek.com/browse/boardgame/page/"+str(i+4)+".html")
        m = re.findall('metasell/thing/(.*?)"',f.text)
        id_list.extend(m)
    
    return [int(id) for id in id_list]

def analyse_comments(filename, word):
    with open(filename) as json_data:
        gamelist = json.load(json_data)
        
    results = []
    for game in gamelist:
        try:
            rated = [comment['value'] for comment in game['comments'] if comment['rating']!='N/A']
        except:
            print(game['title'])
            continue
        freq = 100*sum(1 for i in rated if word in i.lower())/len(rated)
        results.append((game['title'],freq))
    results = pd.DataFrame(results,columns=['title','term %']).sort_values('term %',ascending=False)
    return results
    
    
    
#100*sum(1 for i in rated if "bankrupt" in i.lower())/len(rated)
def save_comments(pages,filename):
    id_list = get_ids(pages)
    json_output = []
    
    for id in id_list:
        comments = get_comments(id)
        json_output.append(comments)
        
    with open(filename, 'w') as outfile:
        json.dump(json_output,outfile)

    

def analyse(game_id):

    comments = get_comments(game_id)
    
    #strip unrated
    rated = [comment['value'] for comment in comments if comment['rating']!='N/A']
    
#    high = [comment['value'] for comment in rated if float(comment['rating']) >= 9]
#    low = [comment['value'] for comment in rated if float(comment['rating']) < 6]
    
    highstring = ' '.join(rated)
#    lowstring = ' '.join(low)

#    stopwords = nltk.corpus.stopwords.words('english')
#    stopwords.extend(
#        ['game','play','one','games','best','like','played','players','love','great','really','time','good','much','get','...','would']
#    )
#    highwords = nltk.tokenize.word_tokenize(highstring)

    return rated
#    return (game_id, highstring)
#    highworddist = nltk.FreqDist(w.lower() for w in highwords if len(w)>2 and w.lower() not in stopwords)

#    lowwords = nltk.tokenize.word_tokenize(lowstring)
#    lowworddist = nltk.FreqDist(w.lower() for w in lowwords if len(w)>2 and w.lower() not in stopwords)

#    print(highworddist.most_common(20))
#    print(lowworddist.most_common(20))
    
    
def get_comments(game_id):
    bgg = BGG()
    game_tree = bgg.boardgame(game_id,comments=True)
    game = game_tree['items']['item']
    ncomments = int(game['comments']['totalitems'])
    try:
        title = game['name']['value']
    except:
        title = game['name'][0]['value']
    comments = []
    print(title)
    for i in range(1,int(np.ceil(ncomments/100))+1):
        try:
            game_tree = bgg.boardgame(game_id,comments=True,pagesize=100,page=i)
        except:
            continue
        game = game_tree['items']['item']
        comments.extend(game['comments']['comment'])
        sleep(1)
        print(i)
        
    comments = {'id':game_id,'title':title,'comments':comments}
    return comments