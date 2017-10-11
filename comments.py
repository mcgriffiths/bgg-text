from libbgg.apiv2 import BGG
import nltk
import numpy as np
import json
import pandas as pd
import requests
import re
from time import sleep
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from statsmodels.formula.api import ols

#217974, 199908

mechanics = ['programming','action point','area control','auction','betting','card-driven','drafting',
             'cooperative','deck building','dice rolling','hand management',
             'memory','partnership','pick up','elimination','push your luck','roll and move',
             'set collection','route building','simultaneous','stock holding','take that','tile laying',
             'time track','trading','trick-taking','worker placement']

categories = ['abstract','dexterity','euro','bluffing','card game','children','city building',
              'civilization','collectible','deduction','economic','family','negotiation','party',
              'political','puzzle','racing','wargame','word game']

def get_ids(start, end):
    id_list = []
    for i in range(start, end):
        f = requests.get("https://boardgamegeek.com/browse/boardgame/page/"+str(i)+".html")                       
# ?sort=numvoters&sortdir=desc")
        m = re.findall('metasell/thing/(.*?)"',f.text)
        id_list.extend(m)
    
    return [int(id) for id in id_list]

def get_json(top):
    with open('top100.json') as json_data:
        gamelist = json.load(json_data)

    for i in range(9):
        with open('from'+str(i+1)+'00.json') as json_data:
            gamelist1 = json.load(json_data)
        gamelist.extend(gamelist1)
    
    return gamelist[:top]

def analyse_comments(word,top=1000):
    
    gamelist = get_json(top)    
    results = []
    for game in gamelist:
        rated = [comment['value'] for comment in game['comments'] if isinstance(comment,dict) and comment['rating']!='N/A']
        matches = sum(1 for comment in rated if word in comment.lower())
        freq = 100.*matches/len(rated)
        results.append((game['id'],game['title'],matches,freq))
    results = pd.DataFrame(results,columns=['id','title','matches','term %']).set_index('id')  
    print(results.query('matches>2').sort_values('term %',ascending=False))
    return results

def build_df(wordlist,top=1000):

    gamelist = get_json(top)
    
    titles = [(game['id'],game['title']) for game in gamelist]
    results = pd.DataFrame(titles,columns=['id','title']).set_index('id')
    for word in wordlist:
        game_results = []
        for game in gamelist:
            rated = [comment['value'] for comment in game['comments'] if isinstance(comment,dict) and comment['rating']!='N/A']
            matches = sum(1 for comment in rated if word.lower() in comment.lower())
            freq = 100.*matches/len(rated)
            game_results.append(freq)
        word = word.replace(" ","_").replace("-","_")
        results[word] = game_results
        results[word] = 100.*results[word]/results[word].max()
    
    return results

def add_user(df,username):
    coll = get_ratings(username)
    results = df.reset_index().merge(coll).set_index('id')
    print (results.corr().sort_values('rating')['rating'])
    return results

def add_cluster(df,clusters):
    model = KMeans(n_clusters=clusters)
    model.fit(df.select_dtypes(include=[np.number]))
    df['cluster'] = model.labels_
    return df
        
def save_comments(filename,start,end):
    id_list = [161936, 182028, 12333, 174430, 187645, 120677, 167791, 169786, 173346, 84876, 102794, 3076, 115746, 193738, 31260, 96848, 170216, 25613, 205637, 205059, 164153, 209010, 2651, 72125, 164928, 175914, 121921, 183394, 124361, 28720, 35677, 124742, 126163, 177736, 146508, 178900, 68448, 122515, 171623, 18602, 12493, 146021, 28143, 110327, 62219, 163412, 157354, 150376, 132531, 93, 40834, 103885, 159675, 37111, 201808, 172386, 2511, 36218, 180263, 144733, 102680, 42, 73439, 30549, 194655, 521, 104162, 155426, 127023, 125153, 9609, 34635, 128882, 161970, 155068, 103343, 146652, 70149, 147020, 175155, 126042, 123260, 14105, 17133, 14996, 43111, 148949, 91, 39463, 21050, 171131, 9216, 27833, 77423, 54043, 146886, 4098, 43015, 148228, 176189, 128621, 555, 104006, 31627, 215, 188, 10630, 129437, 154203, 101721, 9209, 20551, 31481, 234, 93260, 140620, 198928, 182874, 22545, 97207, 185343, 146439, 54998, 24181, 12, 7854, 38453, 59294, 172818, 822, 127060, 19857, 181304, 118048, 25021, 48726, 108745, 196340, 192291, 176494, 193037, 27708, 66589, 92828, 155873, 181279, 41114, 463, 17392, 77130, 172287, 54138, 421, 82222, 155821, 55690, 119890, 3, 45315, 1, 760, 40692, 36932, 59959, 15985, 123123, 198773, 163745, 39856, 176920, 124708, 243, 136888, 37046, 156129, 25292, 40765, 33160, 129622, 163068, 63628, 2655, 144344, 21241, 70919, 5, 25554, 19777, 160477, 182631, 121, 54625, 143693, 31594, 54, 70323, 71, 105134, 46213, 109276, 8217, 157969, 9217, 2653, 204583, 139976, 21348, 24480, 15987, 79828, 172081, 90137, 198994, 176734, 55670, 39683, 188834, 118, 147949, 176396, 163967, 128996, 25417, 712, 96913, 66188, 83330, 73171, 13122, 154809, 18833, 40354, 66362, 136063, 65781, 5404, 150658, 100901, 875, 102652, 144592, 17226, 172, 11170, 27162, 98778, 137408, 205359, 97786, 133038, 13, 58421, 183562, 146791, 169255, 105551, 192458, 138161, 63888, 128671, 203993, 127398, 42052, 36553, 62222, 170042, 195421, 34219, 116998, 182134, 3685, 200680, 62227, 125618, 90419, 158899, 171668, 209685, 141572, 166669, 151347, 475, 50, 43570, 175640, 162082, 34119, 113924, 161614, 6472, 27173, 133848, 163968, 39351, 478, 119506, 95527, 9625, 128271, 155703, 97842, 3307, 39938, 155987, 15062, 140603, 150, 9674, 1513, 15363, 159508, 18, 133473, 181521, 192153, 192457, 42776, 22827, 119432, 189932, 31999, 121288, 50750, 65532, 2346, 21790, 134726, 166384, 586, 58281, 121408, 191862, 20437, 167400, 131357, 18098, 140934, 27746, 24800, 31730, 192836, 163166, 26997, 155624, 143519, 36235, 47, 128780, 11, 79127, 191189, 30869, 135219, 1353, 113997, 151022, 153938, 144189, 111341, 160499, 148575, 20963, 177639, 9823, 160010, 183251, 70512, 4390, 199042, 137988, 150997, 10547, 372, 42215, 28181, 113294, 206718, 15364, 30380, 3201, 6249, 126792, 160495, 88, 38996, 110277, 21763, 12942, 206941, 68425, 91312, 82421, 28023, 2163, 162007, 24508, 31563, 103886, 82420, 199561, 171, 172308, 174785, 65901, 55600, 168584, 194607, 41002, 22345, 37904, 156009, 47185, 156689, 117915, 35570, 123540, 180511, 158275, 41066, 91872, 12891, 46, 13004, 503, 73761, 107529, 130960, 43528, 146278, 132028, 139898, 94, 144797, 30957, 151004, 84419, 156336, 158889, 180974, 143741, 1035, 100423, 131287, 27976, 163602, 12962, 26566, 528, 181530, 137269, 203420, 143515, 155362, 2181, 169426, 111799, 1345, 12002, 5782, 168435, 92415, 156943, 148319, 29603, 69779, 9440, 7717, 21882, 491, 69789, 483, 98351, 108784, 12902, 169124, 1041, 173442, 85897, 158600, 8125, 52461, 162286, 156566, 20100, 106217, 13642, 66356, 53953, 2453, 119788, 939, 45986, 132018, 29368, 72225, 171499, 117959, 2955, 108906, 240, 40628, 71721, 95064, 129948, 161417, 145659, 209778, 19600, 193558, 554, 89409, 199478, 170416, 131646, 147151, 202408, 1301, 67492, 191977, 8051, 168917, 174570, 171273, 38823, 22825, 165401, 1822, 904, 160018, 30645, 161533, 171669, 45, 192135, 37380, 8045, 122298, 194594, 699, 51, 173, 2398, 162886, 65244, 104710, 26990, 60, 200147, 432, 161866, 141517, 98229, 224, 55427, 66505, 156546, 156776, 160851, 169654, 104363, 204305, 172047, 16747, 72287, 38054, 177590, 72321, 135382, 153016, 154458, 27588, 59946, 551, 175117, 137649, 50768, 91514, 94246, 122522, 15817, 97903, 165838, 18100, 2093, 177678, 104020, 25568, 41, 66056, 219513, 151007, 17396, 5737, 150999, 157403, 194879, 212445, 171233, 142326, 23094, 57390, 35497, 37759, 179803, 38159, 193042, 68264, 136991, 71671, 1634, 170561, 180680, 27364, 42910, 179275, 327, 124, 25669, 4099, 152162, 127518, 121410, 154825, 34084, 165722, 17405, 8203, 91536, 157526, 159473, 1608, 230, 44163, 130176, 37628, 164265, 30618, 39953, 30381, 132372, 2507, 178336, 200077, 121297, 221, 158435, 21523, 185589, 207691, 1382, 91873, 19100, 183880, 823, 204027, 173064, 213893, 157001, 1465, 855, 131260, 8989, 123955, 833, 154086, 56692, 15126, 53093, 37836, 154246, 201921, 60435, 143884, 25643, 25821, 91080, 153065, 181687, 203427, 24068, 191231, 18258, 1261, 174660, 195539, 527, 35815, 17329, 11825, 95103, 180593, 75449, 125678, 183840, 179172, 40793, 164338, 43022, 96007, 278, 140933, 13823, 188920, 6830, 111124, 147303, 12495, 120523, 156091, 119591, 105037, 224037, 37919, 105, 1159, 134352, 144041, 145639, 47055, 104347, 20542, 22141, 42452, 12995, 102548, 175199, 75165, 16395, 109125, 154386, 159109, 176544, 137297, 106, 138649, 66, 152470, 21441, 16992, 103, 9203, 29934, 902, 10640, 35052, 33604, 157809, 72991, 10, 15, 150312, 129051, 131014, 123499, 2338, 143405, 117985, 156714, 160902, 163154, 152, 125548, 39684, 354, 43443, 122294, 112, 7480, 171879, 193949, 878, 177802, 13780, 176229, 58936, 492, 1231, 13884, 19999, 138431, 16986, 163413, 207336, 123607, 30658, 142992, 176165, 34194, 22038, 22484, 1897, 16496, 6205, 196, 195137, 55697, 24417, 200954, 423, 106662, 94362, 826, 3072, 61692, 15512, 9220, 63268, 127024, 92319, 134253, 101785, 160610, 1540, 166226, 41916, 681, 9446, 85256, 38862, 220, 26, 6411, 191051, 143986, 198953, 181796, 3800, 40270, 19237, 182340, 481, 165986, 1115, 139030, 12761, 179572, 139771, 112138, 2081, 177524, 148943, 67254, 38343, 113873, 134453, 146221, 21550, 187617, 203417, 26457, 167270, 184424, 73369, 3284, 83195, 5716, 1499, 57925, 2842, 142079, 203416, 129459, 88827, 41933, 197070, 197443, 142961, 115, 165872, 146418, 147253, 730, 71836, 22, 1403, 19622, 94104, 1915, 103185, 152765, 701, 113401, 7805, 394, 173090, 141423, 37387, 42487, 191876, 24827, 6351, 191300, 103092, 37907, 254, 24773, 242, 142852, 21920, 168, 194880, 90040, 131111, 594, 2122, 195162, 38765, 205398, 187377, 181810, 38863, 148951, 157096, 149155, 5206, 207830, 107255, 172381, 131366, 168998, 175223, 144529, 70, 172220, 87890, 158900, 131325, 29937, 31497, 138233, 35435, 135779, 182694, 37358, 180040, 34887, 204836, 111417, 104955, 122842, 2393, 39339, 143185, 201, 91984, 42124, 14254, 3565, 168609, 36522, 178054, 1662, 37141, 84, 71906, 620, 125752, 113636, 55829, 624, 3228, 204, 181524, 199, 197405, 179460, 82168, 116, 21954, 1887, 56931, 92190, 73, 3955, 2987, 180899, 172385, 10093, 136280, 38992, 1829, 18745, 170771, 38386, 33451, 8129, 169794, 23679, 2808, 157, 167513, 75476, 195544, 986, 2961, 40209, 180852, 111105, 122240, 36811]    
    id_list = id_list[start:end]
#    id_list = get_ids(start,end)
    json_output = []
    
    for id in id_list:
        comments = get_comments(id)
        json_output.append(comments)
        
    with open(filename, 'w') as outfile:
        json.dump(json_output,outfile)

def word_count(game_id):
    comments = get_comments(game_id)['comments']
    
    #strip unrated
    rated = [comment['value'] for comment in comments if comment['rating']!='N/A']
    comment_string = ' '.join(rated)

    stopwords = nltk.corpus.stopwords.words('english')
    stopwords.extend(
        ['game','play','one','games','best','like','played','players','love','great','really','time','good','much','get','...','would']
    )
    comment_words = nltk.tokenize.word_tokenize(comment_string)

    word_dist = nltk.FreqDist(w.lower() for w in comment_words if len(w)>2 and w.lower() not in stopwords)

    print(word_dist.most_common(20))
    return word_dist
    
    
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
    for i in range(int(np.ceil(ncomments/100))):
        try:
            game_tree = bgg.boardgame(game_id,comments=True,pagesize=100,page=i+1)
        except:
            continue
        game = game_tree['items']['item']
        comments.extend(game['comments']['comment'])
        sleep(1)
        print(i)
        
    comments = {'id':game_id,'title':title,'comments':comments}
    return comments

def get_ratings(username):
    bgg = BGG()
    collection = bgg.get_collection(username,rated=1,stats=1)
    coll_list = collection['items']['item']
    mycoll = []
    for game in coll_list:
        game_id = int(game['objectid'])
        title = game['name']['TEXT']
        rating = float(game['stats']['rating']['value'])
        mycoll.append((game_id,title,rating))
    coll_df = pd.DataFrame(mycoll,columns=['id','title','rating']).set_index('id')  
    return coll_df 

def add_similar(df,game_id):
    sim = cosine_similarity(df.select_dtypes(include=[np.number]))
    colname = 'similarity_'+str(game_id)
    df[colname] = sim[df.index.get_loc(game_id)]
    print (df.sort_values(colname,ascending=False)[['title',colname]])
    return df

def fit_model(df,username):
    user = add_user(df,username)
    formula = ols_formula(user, 'rating', 'title')
    model = ols(formula,user)
    results = model.fit()
    print (results.summary())
    df['prediction'] = results.predict(df)
    mycoll = get_ratings(username)
    unrated = df[~df.index.isin(mycoll.index)]
    print(unrated[['title','prediction']].sort_values('prediction',ascending=False))
    return unrated
 
def ols_formula(df, dependent_var, *excluded_cols):
    '''
    Generates the R style formula for statsmodels (patsy) given
    the dataframe, dependent variable and optional excluded columns
    as strings
    '''
    df_columns = list(df.columns.values)
    df_columns.remove(dependent_var)
    for col in excluded_cols:
        df_columns.remove(col)
    return dependent_var + ' ~ ' + ' + '.join(df_columns)
        