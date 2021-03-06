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
from sklearn.feature_extraction.text import TfidfVectorizer
from statsmodels.formula.api import ols
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
pd.set_option('display.precision',3)

#predefined list of terms derived from BGG mechanics and categories.
#would like to add to these with more descriptive terms
mechanics = ['programming','action point','area control','auction','betting','card-driven','drafting',
             'cooperative','deck building','dice rolling','hand management',
             'memory','partnership','pick up','elimination','push your luck','roll and move',
             'set collection','route building','simultaneous','stock holding','take that','tile laying',
             'time track','trading','trick-taking','worker placement']

categories = ['abstract','dexterity','euro','bluffing','card game','children','city building',
              'civilization','collectible','deduction','economic','family','negotiation','party',
              'political','puzzle','racing','wargame','word game','thematic','ameritrash']

dynamics = ['salad','non-gamer','opaque','thinky',' nasty','unforgiving',
            'brain burn','fantasy','clever','unique','pure','social',
            'simple','complex','fun','light','heavy']

#functions used in retrieving data from BGG
def get_ids(start, end):
    """returns a list of game IDs taken from BGG rankings pages 
    (start page and end page specified)"""
    id_list = []
    for i in range(start, end):
        f = requests.get("https://boardgamegeek.com/browse/boardgame/page/"+str(i)+".html")
        m = re.findall('metasell/thing/(.*?)"',f.text)
        id_list.extend(m)
    
    return [int(id) for id in id_list]

def get_comments(game_id):
    """gets coments for a single game"""
    bgg = BGG()
    game_tree = bgg.boardgame(game_id,comments=True,pagesize=10)
    
    if(len(game_tree['items'])==1):
        game_tree = bgg.boardgameexpansion(game_id,comments=True,pagesize=10)

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
            if(len(game_tree['items'])==1):
                game_tree = bgg.boardgameexpansion(game_id,comments=True,pagesize=100,page=i+1)
        except:
            continue
        game = game_tree['items']['item']
        comments.extend(game['comments']['comment'])
        sleep(1.5)
        print(i)
        
    comments = {'id':game_id,'title':title,'comments':comments}
    return comments

def get_data(game_id):
    """get data for a single game"""
    print(game_id)
    bgg = BGG()
    game_tree = bgg.boardgame(game_id,comments=False,stats=True)
    if(len(game_tree['items'])==1):
        game_tree = bgg.boardgameexpansion(game_id,comments=False,stats=True)
    game = game_tree['items']['item']  
    sleep(2)
    return game
    
def save_data(filename,retrieve_func,start,end):
    """scrapes comments from BGG for given rankings, saves to json"""
# top 1000
    id_list = [161936, 182028, 12333, 174430, 187645, 120677, 167791, 169786, 173346, 84876, 102794, 3076, 115746, 193738, 31260, 96848, 170216, 25613, 205637, 205059, 164153, 209010, 2651, 72125, 164928, 175914, 121921, 183394, 124361, 28720, 35677, 124742, 126163, 177736, 146508, 178900, 68448, 122515, 171623, 18602, 12493, 146021, 28143, 110327, 62219, 163412, 157354, 150376, 132531, 93, 40834, 103885, 159675, 37111, 201808, 172386, 2511, 36218, 180263, 144733, 102680, 42, 73439, 30549, 194655, 521, 104162, 155426, 127023, 125153, 9609, 34635, 128882, 161970, 155068, 103343, 146652, 70149, 147020, 175155, 126042, 123260, 14105, 17133, 14996, 43111, 148949, 91, 39463, 21050, 171131, 9216, 27833, 77423, 54043, 146886, 4098, 43015, 148228, 176189, 128621, 555, 104006, 31627, 215, 188, 10630, 129437, 154203, 101721, 9209, 20551, 31481, 234, 93260, 140620, 198928, 182874, 22545, 97207, 185343, 146439, 54998, 24181, 12, 7854, 38453, 59294, 172818, 822, 127060, 19857, 181304, 118048, 25021, 48726, 108745, 196340, 192291, 176494, 193037, 27708, 66589, 92828, 155873, 181279, 41114, 463, 17392, 77130, 172287, 54138, 421, 82222, 155821, 55690, 119890, 3, 45315, 1, 760, 40692, 36932, 59959, 15985, 123123, 198773, 163745, 39856, 176920, 124708, 243, 136888, 37046, 156129, 25292, 40765, 33160, 129622, 163068, 63628, 2655, 144344, 21241, 70919, 5, 25554, 19777, 160477, 182631, 121, 54625, 143693, 31594, 54, 70323, 71, 105134, 46213, 109276, 8217, 157969, 9217, 2653, 204583, 139976, 21348, 24480, 15987, 79828, 172081, 90137, 198994, 176734, 55670, 39683, 188834, 118, 147949, 176396, 163967, 128996, 25417, 712, 96913, 66188, 83330, 73171, 13122, 154809, 18833, 40354, 66362, 136063, 65781, 5404, 150658, 100901, 875, 102652, 144592, 17226, 172, 11170, 27162, 98778, 137408, 205359, 97786, 133038, 13, 58421, 183562, 146791, 169255, 105551, 192458, 138161, 63888, 128671, 203993, 127398, 42052, 36553, 62222, 170042, 195421, 34219, 116998, 182134, 3685, 200680, 62227, 125618, 90419, 158899, 171668, 209685, 141572, 166669, 151347, 475, 50, 43570, 175640, 162082, 34119, 113924, 161614, 6472, 27173, 133848, 163968, 39351, 478, 119506, 95527, 9625, 128271, 155703, 97842, 3307, 39938, 155987, 15062, 140603, 150, 9674, 1513, 15363, 159508, 18, 133473, 181521, 192153, 192457, 42776, 22827, 119432, 189932, 31999, 121288, 50750, 65532, 2346, 21790, 134726, 166384, 586, 58281, 121408, 191862, 20437, 167400, 131357, 18098, 140934, 27746, 24800, 31730, 192836, 163166, 26997, 155624, 143519, 36235, 47, 128780, 11, 79127, 191189, 30869, 135219, 1353, 113997, 151022, 153938, 144189, 111341, 160499, 148575, 20963, 177639, 9823, 160010, 183251, 70512, 4390, 199042, 137988, 150997, 10547, 372, 42215, 28181, 113294, 206718, 15364, 30380, 3201, 6249, 126792, 160495, 88, 38996, 110277, 21763, 12942, 206941, 68425, 91312, 82421, 28023, 2163, 162007, 24508, 31563, 103886, 82420, 199561, 171, 172308, 174785, 65901, 55600, 168584, 194607, 41002, 22345, 37904, 156009, 47185, 156689, 117915, 35570, 123540, 180511, 158275, 41066, 91872, 12891, 46, 13004, 503, 73761, 107529, 130960, 43528, 146278, 132028, 139898, 94, 144797, 30957, 151004, 84419, 156336, 158889, 180974, 143741, 1035, 100423, 131287, 27976, 163602, 12962, 26566, 528, 181530, 137269, 203420, 143515, 155362, 2181, 169426, 111799, 1345, 12002, 5782, 168435, 92415, 156943, 148319, 29603, 69779, 9440, 7717, 21882, 491, 69789, 483, 98351, 108784, 12902, 169124, 1041, 173442, 85897, 158600, 8125, 52461, 162286, 156566, 20100, 106217, 13642, 66356, 53953, 2453, 119788, 939, 45986, 132018, 29368, 72225, 171499, 117959, 2955, 108906, 240, 40628, 71721, 95064, 129948, 161417, 145659, 209778, 19600, 193558, 554, 89409, 199478, 170416, 131646, 147151, 202408, 1301, 67492, 191977, 8051, 168917, 174570, 171273, 38823, 22825, 165401, 1822, 904, 160018, 30645, 161533, 171669, 45, 192135, 37380, 8045, 122298, 194594, 699, 51, 173, 2398, 162886, 65244, 104710, 26990, 60, 200147, 432, 161866, 141517, 98229, 224, 55427, 66505, 156546, 156776, 160851, 169654, 104363, 204305, 172047, 16747, 72287, 38054, 177590, 72321, 135382, 153016, 154458, 27588, 59946, 551, 175117, 137649, 50768, 91514, 94246, 122522, 15817, 97903, 165838, 18100, 2093, 177678, 104020, 25568, 41, 66056, 219513, 151007, 17396, 5737, 150999, 157403, 194879, 212445, 171233, 142326, 23094, 57390, 35497, 37759, 179803, 38159, 193042, 68264, 136991, 71671, 1634, 170561, 180680, 27364, 42910, 179275, 327, 124, 25669, 4099, 152162, 127518, 121410, 154825, 34084, 165722, 17405, 8203, 91536, 157526, 159473, 1608, 230, 44163, 130176, 37628, 164265, 30618, 39953, 30381, 132372, 2507, 178336, 200077, 121297, 221, 158435, 21523, 185589, 207691, 1382, 91873, 19100, 183880, 823, 204027, 173064, 213893, 157001, 1465, 855, 131260, 8989, 123955, 833, 154086, 56692, 15126, 53093, 37836, 154246, 201921, 60435, 143884, 25643, 25821, 91080, 153065, 181687, 203427, 24068, 191231, 18258, 1261, 174660, 195539, 527, 35815, 17329, 11825, 95103, 180593, 75449, 125678, 183840, 179172, 40793, 164338, 43022, 96007, 278, 140933, 13823, 188920, 6830, 111124, 147303, 12495, 120523, 156091, 119591, 105037, 224037, 37919, 105, 1159, 134352, 144041, 145639, 47055, 104347, 20542, 22141, 42452, 12995, 102548, 175199, 75165, 16395, 109125, 154386, 159109, 176544, 137297, 106, 138649, 66, 152470, 21441, 16992, 103, 9203, 29934, 902, 10640, 35052, 33604, 157809, 72991, 10, 15, 150312, 129051, 131014, 123499, 2338, 143405, 117985, 156714, 160902, 163154, 152, 125548, 39684, 354, 43443, 122294, 112, 7480, 171879, 193949, 878, 177802, 13780, 176229, 58936, 492, 1231, 13884, 19999, 138431, 16986, 163413, 207336, 123607, 30658, 142992, 176165, 34194, 22038, 22484, 1897, 16496, 6205, 196, 195137, 55697, 24417, 200954, 423, 106662, 94362, 826, 3072, 61692, 15512, 9220, 63268, 127024, 92319, 134253, 101785, 160610, 1540, 166226, 41916, 681, 9446, 85256, 38862, 220, 26, 6411, 191051, 143986, 198953, 181796, 3800, 40270, 19237, 182340, 481, 165986, 1115, 139030, 12761, 179572, 139771, 112138, 2081, 177524, 148943, 67254, 38343, 113873, 134453, 146221, 21550, 187617, 203417, 26457, 167270, 184424, 73369, 3284, 83195, 5716, 1499, 57925, 2842, 142079, 203416, 129459, 88827, 41933, 197070, 197443, 142961, 115, 165872, 146418, 147253, 730, 71836, 22, 1403, 19622, 94104, 1915, 103185, 152765, 701, 113401, 7805, 394, 173090, 141423, 37387, 42487, 191876, 24827, 6351, 191300, 103092, 37907, 254, 24773, 242, 142852, 21920, 168, 194880, 90040, 131111, 594, 2122, 195162, 38765, 205398, 187377, 181810, 38863, 148951, 157096, 149155, 5206, 207830, 107255, 172381, 131366, 168998, 175223, 144529, 70, 172220, 87890, 158900, 131325, 29937, 31497, 138233, 35435, 135779, 182694, 37358, 180040, 34887, 204836, 111417, 104955, 122842, 2393, 39339, 143185, 201, 91984, 42124, 14254, 3565, 168609, 36522, 178054, 1662, 37141, 84, 71906, 620, 125752, 113636, 55829, 624, 3228, 204, 181524, 199, 197405, 179460, 82168, 116, 21954, 1887, 56931, 92190, 73, 3955, 2987, 180899, 172385, 10093, 136280, 38992, 1829, 18745, 170771, 38386, 33451, 8129, 169794, 23679, 2808, 157, 167513, 75476, 195544, 986, 2961, 40209, 180852, 111105, 122240, 36811]    
#1000 - 2000
#    id_list = [54361, 9910, 86246, 8098, 119, 116954, 202670, 128442, 63543, 28089, 237, 99975, 3931, 195571, 4396, 1568, 149951, 494, 205716, 361, 168788, 102435, 232, 38545, 65564, 438, 1544, 24703, 175307, 71074, 155255, 43868, 915, 166857, 192860, 2065, 171726, 24742, 15953, 81640, 15818, 91620, 25224, 15511, 13308, 141736, 31552, 442, 31105, 85005, 2397, 32125, 32666, 118174, 63759, 619, 157026, 159503, 925, 17710, 12477, 171011, 41749, 165950, 21133, 35503, 165948, 67239, 2570, 15839, 145189, 144568, 123096, 220520, 179303, 28259, 85036, 73316, 19526, 46396, 39217, 89910, 6779, 169611, 2251, 4636, 31759, 12166, 145371, 114031, 175293, 153064, 424, 124172, 27627, 134157, 144239, 31624, 66588, 903, 157413, 34707, 22348, 171662, 929, 118536, 13551, 150145, 40769, 8190, 16267, 58110, 144826, 128898, 194626, 17223, 5781, 24310, 25277, 42673, 120217, 120, 8481, 1442, 340, 12589, 33154, 9215, 35761, 68182, 118695, 40760, 22143, 148601, 5072, 41010, 249, 46255, 62220, 75358, 85325, 16772, 113289, 180592, 118063, 66214, 9027, 71882, 169416, 29972, 168681, 117, 142121, 1032, 119391, 172996, 3353, 71061, 29223, 98, 133632, 21551, 158339, 1198, 36399, 40393, 32412, 112686, 195314, 143401, 38718, 66510, 1117, 54137, 43152, 57660, 187653, 123219, 39927, 4741, 11057, 123609, 109291, 155969, 3720, 175095, 162388, 128721, 205597, 130, 127997, 18748, 9341, 20134, 49, 128667, 31483, 1107, 6901, 40237, 77, 472, 192074, 104581, 29839, 176558, 589, 15510, 1234, 88408, 65515, 398, 38531, 193737, 137330, 217372, 124052, 30367, 66171, 136056, 98315, 182189, 22877, 177478, 111732, 631, 83734, 72, 195528, 12005, 12750, 41863, 176083, 76417, 592, 163370, 15600, 94480, 83667, 27848, 92044, 144864, 38309, 166571, 174078, 99, 40529, 126100, 11971, 67185, 173101, 124965, 19764, 156858, 1155, 67928, 154182, 32674, 73070, 384, 143404, 14, 153507, 19995, 196202, 76150, 175621, 146816, 157917, 216459, 22897, 183006, 511, 9441, 122588, 36648, 443, 36400, 158572, 54307, 154509, 29073, 28, 166286, 34585, 174973, 4218, 151247, 65282, 128445, 126996, 144415, 139899, 67180, 191894, 2596, 10997, 4192, 198525, 337, 198487, 125046, 205317, 3570, 78733, 140, 183231, 172503, 15290, 466, 66849, 21632, 638, 181345, 180771, 144553, 102237, 137031, 137423, 149776, 195503, 66587, 23540, 125, 7858, 162823, 169318, 157958, 8095, 128, 691, 144270, 67888, 153097, 1295, 195043, 2476, 152242, 123239, 206051, 65907, 81453, 145645, 526, 15318, 205610, 128931, 181797, 129731, 183571, 132, 91523, 156548, 20133, 2533, 45748, 1563, 17025, 12350, 2569, 4394, 39206, 3870, 103686, 12692, 38786, 148261, 21791, 195856, 22532, 22192, 119781, 3972, 59, 39188, 37371, 6819, 61487, 715, 36946, 37196, 124969, 217, 19348, 150293, 531, 16991, 11229, 23107, 17161, 216092, 41627, 348, 1219, 118418, 156501, 158340, 798, 32484, 19427, 197572, 125921, 40990, 20, 112092, 154301, 1042, 31479, 108, 168274, 559, 145203, 2529, 103660, 7, 2603, 206803, 17240, 140535, 59753, 320, 1044, 15954, 286, 8730, 119012, 189052, 25729, 180956, 158109, 155695, 28620, 3085, 34297, 35488, 1423, 949, 1491, 96152, 936, 208895, 38506, 205507, 156138, 129050, 217449, 42939, 41429, 191572, 99392, 90305, 223, 153737, 34969, 1270, 163, 37734, 124827, 40832, 6887, 205046, 382, 131835, 111148, 55952, 75212, 932, 205158, 203780, 15369, 145633, 27225, 634, 23418, 18866, 28723, 244, 127127, 149241, 256, 133528, 180564, 191597, 189, 3236, 32944, 67877, 4491, 157789, 1334, 140343, 124968, 38713, 155463, 146312, 153318, 19643, 173156, 167698, 811, 187680, 10550, 142296, 46007, 93724, 5867, 216091, 70519, 37120, 154906, 2288, 1194, 18460, 118000, 139033, 126008, 27463, 1262, 32116, 118705, 75091, 40770, 23451, 173200, 191972, 7865, 296, 788, 40381, 508, 170587, 140717, 16366, 805, 21654, 216201, 1484, 16216, 1621, 35614, 168679, 36708, 8490, 101, 28829, 25261, 172547, 1037, 162, 36888, 31291, 3139, 40425, 2612, 8195, 770, 154003, 883, 8170, 111, 162009, 71655, 192802, 205418, 4249, 27048, 632, 143157, 35505, 23730, 140951, 35572, 95105, 176530, 169704, 80006, 5406, 188866, 3421, 23817, 1589, 129351, 37400, 16, 1111, 25951, 192455, 18985, 164949, 17393, 197831, 103649, 3202, 6068, 19841, 128554, 6050, 30241, 109779, 102104, 11168, 141090, 8924, 27356, 80771, 23686, 140125, 156496, 2516, 178570, 221965, 55253, 172552, 187273, 43307, 5622, 97915, 122, 96792, 143175, 39336, 62871, 19301, 714, 5620, 184459, 987, 147, 3041, 24762, 191963, 520, 6866, 34615, 692, 23604, 160081, 36367, 145588, 11949, 433, 171890, 135281, 193621, 140519, 8124, 4688, 206940, 18500, 14083, 132544, 8107, 132428, 38548, 136000, 1770, 112373, 177939, 19854, 90870, 195560, 207207, 759, 17484, 63091, 58, 192945, 218603, 5205, 3967, 9364, 3267, 172931, 192934, 11945, 166317, 100679, 149809, 3439, 193308, 130877, 130486, 38872, 7866, 22398, 215311, 130882, 3409, 40831, 78954, 147537, 6613, 3234, 3348, 37301, 14808, 84465, 1265, 287, 168054, 193557, 118610, 139952, 1425, 123570, 135243, 17449, 142854, 2266, 35342, 162559, 37231, 265, 39242, 299, 205045, 827, 22551, 163640, 93538, 164812, 4616, 24920, 143176, 33107, 83040, 35350, 85250, 10788, 172933, 19996, 108722, 177249, 99875, 35285, 19679, 32165, 108044, 98527, 117960, 31722, 12681, 94389, 4174, 206084, 176371, 163474, 135649, 84772, 58602, 176565, 24037, 41490, 147930, 141791, 37165, 33003, 169, 5942, 119866, 40398, 117942, 122889, 13883, 193739, 148290, 15033, 124380, 107190, 125311, 175, 219215, 41569, 1116, 8172, 20545, 131118, 3605, 153623, 155708, 103132, 56885, 119637, 44, 68228, 24070, 168433, 76674, 3655, 21641, 422, 137776, 10819, 30539, 4204, 56707, 169464, 114387, 163930, 420, 154875, 19948, 189660, 96713, 20609, 84469, 149853, 161226, 151771, 105624, 9, 140863, 105593, 2639, 148203, 7806, 23985, 552, 217085, 139562, 206504, 129320, 4583, 127095, 25794, 464, 635, 5795, 195372, 815, 74390, 97357, 1597, 32014, 1260, 2981, 79073, 182351, 34320, 181761, 1444, 173105, 79131, 156180, 101020, 65673, 1017, 155689, 25674, 20022, 122891, 150926, 200, 169147, 205494, 29109, 24509, 155, 1645, 136955, 4209, 175878, 192638, 179182, 178944, 21380, 115293, 138788, 32441, 142830, 177354, 120605, 85, 2162, 102148, 178134, 380, 152899, 130390, 66085, 175360, 52328, 128063, 196526, 3553, 153, 68247, 53168, 1293, 4370, 1758, 25234, 42743, 17053, 29410, 22278, 9408, 136440, 137397, 34373, 129736, 174614, 30356, 129122, 31133, 21892, 142402, 203430, 329, 185123, 172155, 20806, 56786, 28025, 201248, 39813, 34127, 184522, 160958, 171835, 193560, 338, 1430, 149119, 444, 191004, 6719, 25071, 31503, 152237, 147116, 208773, 25584, 68858, 131891, 31056, 2238, 35035, 2582, 139245, 84889, 541, 73863, 3128, 181260, 1585, 216597, 214, 40214, 29687, 146408, 139992, 2795, 10660, 11265, 160559, 200454, 198522, 626, 204184, 124847, 19363, 152053, 123160, 117914, 40444, 104994, 87821, 173047, 27739, 189067, 4090, 155173, 95234, 30662, 145976, 154443, 170199, 85769, 2247, 1561, 174524, 38749, 257, 101786, 437, 371, 12632, 34599, 37208, 10081, 96613, 188314, 267, 50381, 2652, 38194, 6366, 7349, 2689]
#    id_list = get_ids(start,end)
    id_list = id_list[start:end]
    json_output = []
    
    for id in id_list:
        data = retrieve_func(id)
        json_output.append(data)
    
    filename = 'data/'+filename     
    with open(filename, 'w') as outfile:
        json.dump(json_output,outfile)

def get_ratings(username):
    """gets ratings for a specified user"""
    bgg = BGG()
    collection = bgg.get_collection(username,rated=1,stats=1)
    coll_list = collection['items']['item']
    mycoll = []
    for game in coll_list:
        game_id = int(game['objectid'])
        rating = float(game['stats']['rating']['value'])
        mycoll.append((game_id,rating))
    coll_df = pd.DataFrame(mycoll,columns=['id','rating']).set_index('id')  
    return coll_df 


def get_json(top):
    """reads in the comments data files for the top x games"""
    with open('data/top100.json') as json_data:
        gamelist = json.load(json_data)

    for i in range(10):
        with open('data/from'+str(i+1)+'00.json') as json_data:
            gamelist1 = json.load(json_data)
        gamelist.extend(gamelist1)
    
    with open('data/from1500.json') as json_data:
        gamelist2 = json.load(json_data)
    gamelist.extend(gamelist2)
    return gamelist[:top]

#basic analysis functions
def analyse_comments(word,gamelist=None,wholeword=False,top=2000):
    """returns data frame of matches and % frequency of specified term in comments"""
    if (gamelist is None):
        gamelist = get_json(top)
    
    results = []
    for game in gamelist:
        rated = [comment['value'] for comment in game['comments'] 
                    if isinstance(comment,dict) 
                    and comment['rating']!='N/A' 
                    and len(comment['value'])>0]
        if(wholeword):
            r = "\\b"+word+"\\b"
            matches = sum(1 for comment in rated if re.search(r, comment.lower()))
        else:
            matches = sum(1 for comment in rated if word in comment.lower())
        if (len(rated)==0 or matches < 3):
            freq = 0
        else:
            freq = 100.*matches/len(rated)
        results.append((game['id'],game['title'],matches,len(rated),freq))
    results = pd.DataFrame(results,columns=['id','title','matches','total','term %']).set_index('id')  
    return results.sort_values('term %',ascending=False)

def build_df(wordlist,top=2000,add=False,df=None):
    """creates a data frame with % freq for each word in wordlist"""
    gamelist = get_json(top)    
    if(add):
        results = df
    else:
        titles = [(game['id'],game['title']) for game in gamelist]
        results = pd.DataFrame(titles,columns=['id','title']).set_index('id')
    
    for word in wordlist:
        freq = analyse_comments(word,gamelist)['term %']
        word = word.replace(" ","_").replace("-","_")
        results[word] = freq
        results[word] = 100.*results[word]/results[word].max()
    return results
    
def read_data():
    with open('data/bggdata_1000.json') as json_data:
        bggdata = json.load(json_data)
    with open('data/bggdata_2000.json') as json_data:
        bggdata2 = json.load(json_data)
    bggdata.extend(bggdata2)
    results = [{'count':1, 
                'id':int(game['id']), 
                'year':int(game['yearpublished']['value']),
                'weight':float(game['statistics']['ratings']['averageweight']['value']),
                'playtime':int(game['maxplaytime']['value']),
                'designer':min([int(item['id']) for item in game['link'] if item['type']=='boardgamedesigner']+[999999999])
                } for game in bggdata]
    data_df = pd.DataFrame(results).set_index('id')
    return data_df

def plot_series(word,ax=None,minyear=1990):
    comments = analyse_comments(word)
    results = read_data().join(comments)
    by_year = results.groupby('year').agg({'count':'count', 'term %': lambda ts: (ts > 1).sum(),'matches':sum, 'total':sum, 'weight':'mean'})    
    by_year['freq_game'] = 100.*by_year['term %']/by_year['count']
    by_year['freq_comment'] = 100.*by_year['matches']/by_year['total']
 #   plt.plot(by_year.loc[1990:].freq_game)
    if(ax is None):
        fig,ax = plt.subplots()
    ax.plot(by_year.loc[minyear:].freq_comment,label=word)
    return by_year   

def comparison_plot(wordlist,minyear=1990,legloc='upper right'):
    fig,ax = plt.subplots()
    for word in wordlist:
        plot_series(word,ax,minyear)
    ax.legend(loc=legloc)
    return fig,ax

def scatter_plot(word, var, xmin=0, xmax=None, ymin=0, ymax=None):
    bggdata = read_data()
    comments = analyse_comments(word)
    results = bggdata.join(comments)
    fig,ax = plt.subplots()
    ax.scatter(results[var],results['term %'])
    if(xmax):
        plt.xlim(xmin,xmax)
    if(ymax):
        plt.ylim(ymin,ymax)
    return fig,ax
    
#advanced analysis functions
def user_corrs(df,username):
    """gets a user's ratings and merges them on to a column in dataframe"""
    coll = get_ratings(username)
    titles = pd.DataFrame(df['title'])
    results = titles.join(coll)
    corrs = df.select_dtypes(include=[np.number]).corrwith(results.rating)
    return corrs.sort_values(ascending = False)

def find_clusters(df,clusters):
    """k means cluster using all numeric columns in data frame"""
    model = KMeans(n_clusters=clusters)
    model.fit(df.select_dtypes(include=[np.number]))
    results = pd.DataFrame(df['title'])
    cluster = pd.Series(model.labels_)
    results = results.assign(cluster = cluster.values)
    centers = model.cluster_centers_
    for i,center in enumerate(centers):
        biggest = list(center).index(max(list(center)))
        print(list(df.columns)[biggest+1])
#        print(df.query('cluster=='+str(i))['title'])
    return results
        
def tree_cluster(df,max_d):    
    """hierarchical cluster. print dendrogram and return games by cluster"""
    tree = linkage(df.select_dtypes(include=[np.number]), 'ward')
    
    # calculate full dendrogram
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
            tree,
#            truncate_mode = 'lastp',
#            p=12,
            leaf_rotation=90.,  # rotates the x axis labels
            leaf_font_size=8.,  # font size for the x axis labels
            )
    clusters = fcluster(tree,max_d,criterion='distance')
    results = pd.DataFrame(df['title'])
    results = results.assign(cluster = clusters)
    return results

def like_designer(name,des_id):
    """returns frequency of matches to 'name' excluding any designed by des_id"""
    comments = analyse_comments(name)
    comments = read_data().join(comments)
    comments = comments.query('designer!='+str(des_id))
    return comments.sort_values('term %',ascending=False)[['title','term %']]
 
def find_similar(df,game_id):
    """finds similar games to a specified game, based on cosine similarity"""
    sim = cosine_similarity(df.select_dtypes(include=[np.number]))
    results = pd.DataFrame(df['title'])
    similarity = pd.Series(sim[results.index.get_loc(game_id)])
    results = results.assign(sim = similarity.values)
    return results.sort_values('sim',ascending=False)

def fit_model(df,username,excluded_cols=None,sigthresh=0,numthresh=0):
    """fits linear regression model to given user's ratings,
    returns predictions for unrated"""
    ratings = get_ratings(username)
    user = df.join(ratings)
    rated = user[user.rating.notnull()]
    unrated = user[user.rating.isnull()]
    
    df_columns = list(df.columns.values)
    
    if excluded_cols is None:
        excluded_cols = []
    excluded_cols.append('title')
    
    for col in excluded_cols:
        df_columns.remove(col)
    
    sigcols = [col for col in df_columns if len(rated[rated[col] > sigthresh]) > numthresh]
    formula = 'rating ~ ' + ' + '.join(sigcols)   
#    formula = ols_formula(sigcols, 'rating', excluded_cols)
    model = ols(formula,user)
    fitresult = model.fit()
    print (fitresult.summary())

    pred = pd.Series(fitresult.predict(unrated))
    results = pd.DataFrame(unrated['title'])
    results = results.assign(prediction = pred.values)
    return results.sort_values('prediction',ascending=False)
 
def ols_formula(df, dependent_var, excluded_cols):
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
        
        
def word_count(game_id):
    """returns most common words for specific game ID"""
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

    return word_dist

def tf_idf(top, rank):
    game_list = get_json(top)
    
    corpus = []
    for game in game_list:
        corpus.append(' '.join([comment['value'] for comment in game['comments']]))
    
    tf = TfidfVectorizer(analyzer='word',ngram_range=(1,3),min_df=0,stop_words='english')
    tfidf_matrix = tf.fit_transform(corpus)
    
    feature_names = tf.get_feature_names()
    
    dense = tfidf_matrix.todense()
    game = dense[rank].tolist()[0]
    phrase_scores = [pair for pair in zip(range(0,len(game)),game) if pair[1]>0]
    
    sorted_phrase_scores = sorted(phrase_scores, key = lambda t: t[1]*-1)
    for phrase,score in [(feature_names[word_id],score) for (word_id,score) in sorted_phrase_scores][:50]:
        print('{0: <20} {1}'.format(phrase,score))


