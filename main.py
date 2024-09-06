from collections import defaultdict
import gzip
import numpy as np

import gzip
from collections import defaultdict

def readGz(path):
  for l in gzip.open(path, 'rt'):
    yield eval(l)

def readCSV(path):
  f = gzip.open(path, 'rt')
  f.readline()
  for l in f:
    yield l.strip().split(',')




#find most populular set ++++++++++++++++++++++++++++++++++++++++++++++
bookCount = defaultdict(int)
totalRead = 0

for user,book,_ in readCSV("train_Interactions.csv.gz"):
  bookCount[book] += 1
  totalRead += 1
    
mostPopular = [(bookCount[x], x) for x in bookCount]
mostPopular.sort()
mostPopular.reverse()

return1 = set()
count = 0

def findPopularSet(thred:float):

    mostPopularSet = set()
    count = 0
    for ic, i in mostPopular:
        count += ic
        mostPopularSet.add(i)
        if count > totalRead/100 * thred: return mostPopularSet




#+++++++++++++++++++++++++++++++
#Jaccard 
allRating = []

userRatings = defaultdict(list)
bookUserInter = defaultdict(list)

bookIds = set()

# ratingPerItem = defaultdict(list)
all_rating_per_user = defaultdict(set)
all_rating_per_item = defaultdict(set)
all_items_per_user = defaultdict(list)
all_entries = []

interactions = []
itemIDs = defaultdict(int)
userIDs = defaultdict(int)

iId = 0
uId = 0

for user,book,r in readCSV("train_Interactions.csv.gz"):
  r = int(r)
  allRating.append(r)
  userRatings[user].append(r)
  bookUserInter[user].append(book)
  all_rating_per_item[book].add((user,r))
  all_rating_per_user[user].add((book,r))
  all_entries.append(np.array([user,book,r]))
  all_items_per_user[user].append(book)
  interactions.append([user,book,r])

  if(user not in userIDs):
     userIDs[user] = uId
     uId += 1
  
  if(book not in itemIDs):
     itemIDs[book] = iId
     iId += 1
     
  



# itemsPerUser = defaultdict(set)
# usersPerItem = defaultdict(set)



def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    return numer / denom

# ratingPerUser = defaultdict(list)


# for u,i in allRating:
#     r = d['hours_transformed']
#     hoursPerUser[u].append(r)
#     hoursPerItem[g].append((u,r))
#     itemsPerUser[u].add(g)
#     usersPerItem[g].add(u)


#predict just based on jaccard sim
def predict_sim(u, i, rating_per_user_tr, rating_per_item_tr):
    maxSim = 0
    users_rating = set(rating_per_item_tr[i])
    for g2,_ in rating_per_user_tr[u]:
        sim = Jaccard(users_rating,set(rating_per_item_tr[g2]))
        if sim > maxSim:
            maxSim = sim

    return maxSim

pop_set = findPopularSet(58)
def predict_pop_N_sim(u, i, rpu, rpi):
   if(i in pop_set):
      return 1
   else:
      return predict_sim(u, i, rpu, rpi)
   


#________________________________________________________________________________\\
#BPR tensorflow
import tensorflow as tf
import random

items = list(itemIDs.keys())

class BPRbatch(tf.keras.Model):
    def __init__(self, K, lamb):
        super(BPRbatch, self).__init__()
        # Initialize variables
        self.betaI = tf.Variable(tf.random.normal([len(itemIDs)],stddev=0.001))
        self.gammaU = tf.Variable(tf.random.normal([len(userIDs),K],stddev=0.001))
        self.gammaI = tf.Variable(tf.random.normal([len(itemIDs),K],stddev=0.001))
        # Regularization coefficient
        self.lamb = lamb

    # Prediction for a single instance
    def predict(self, u, i):
        p = self.betaI[i] + tf.tensordot(self.gammaU[u], self.gammaI[i], 1)
        return p

    # Regularizer
    def reg(self):
        return self.lamb * (tf.nn.l2_loss(self.betaI) +\
                            tf.nn.l2_loss(self.gammaU) +\
                            tf.nn.l2_loss(self.gammaI))
    
    def score(self, sampleU, sampleI):
        u = tf.convert_to_tensor(sampleU, dtype=tf.int32)
        i = tf.convert_to_tensor(sampleI, dtype=tf.int32)
        beta_i = tf.nn.embedding_lookup(self.betaI, i)
        gamma_u = tf.nn.embedding_lookup(self.gammaU, u)
        gamma_i = tf.nn.embedding_lookup(self.gammaI, i)
        x_ui = beta_i + tf.reduce_sum(tf.multiply(gamma_u, gamma_i), 1)
        return x_ui

    def call(self, sampleU, sampleI, sampleJ):
        x_ui = self.score(sampleU, sampleI)
        x_uj = self.score(sampleU, sampleJ)
        return -tf.reduce_mean(tf.math.log(tf.math.sigmoid(x_ui - x_uj)))
    
optimizer = tf.keras.optimizers.Adam(0.1)
modelBPR = BPRbatch(5, 0.00001)

def trainingStepBPR(model, interactions):
    Nsamples = 50000
    with tf.GradientTape() as tape:
        sampleU, sampleI, sampleJ = [], [], []
        for _ in range(Nsamples):
            u,i,_ = random.choice(interactions) # positive sample
            j = random.choice(items) # negative sample
            while j in all_items_per_user[u]:
                j = random.choice(items)
            sampleU.append(userIDs[u])
            sampleI.append(itemIDs[i])
            sampleJ.append(itemIDs[j])

        loss = model(sampleU,sampleI,sampleJ)
        loss += model.reg()
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients((grad, var) for
                              (grad, var) in zip(gradients, model.trainable_variables)
                              if grad is not None)
    return loss.numpy()

for i in range(100):
    obj = trainingStepBPR(modelBPR, interactions)
    if (i % 10 == 9): print("iteration " + str(i+1) + ", objective = " + str(obj))

u,i,_ = interactions[0]
# In this case just a score (that can be used for ranking), rather than a prediction of a rating


modelBPR.predict(userIDs[u], itemIDs[i]).numpy()

def predict_tensor_flow(u, i):
    return modelBPR.predict(userIDs[u], itemIDs[i]).numpy()

users_item_test_set = defaultdict(list)

for l in open("./pairs_Read.csv"):

    if l.startswith("userID"):

        continue
    u,i = l.strip().split(',')
    
    users_item_test_set[u].append(i)

countZero = 0
countOne = 0

test_set_predict = defaultdict(int)
for u in users_item_test_set:
    game_list = users_item_test_set[u]
    scores = []
    for j in game_list:
        # pred = predict_sim(u, j, all_rating_per_user, all_rating_per_item)
        # pred = predict_pop_N_sim(u, j, all_rating_per_user, all_rating_per_item)
        pred = predict_tensor_flow(u,j)

        scores.append(pred)
        if(pred == 0):
            countZero += 1
        else:
            countOne += 1
    
    scores = np.array(scores)

    sorted_indices = np.argsort(scores)

    mid_index = len(game_list) / 2

    # print(scores[sorted_indices])
    game_list = np.array(game_list)
    game_list = game_list[sorted_indices]
    print(scores[sorted_indices])
    # for ke in range()
    
    for j in range(len(game_list)):
        if(j >= mid_index):
            test_set_predict[(u,game_list[j])] = 1
        else:
            test_set_predict[(u,game_list[j])] = 0

predictions = open("./predictions_Read.csv", 'w')

for l in open("./pairs_Read.csv"):

    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,g = l.strip().split(',')
    
   
    pred =  test_set_predict[(u,g)]
    
    
    _ = predictions.write(u + ',' + g + ',' + str(pred) + '\n')

predictions.close()

import gzip
def readGz(path):
  for l in gzip.open(path, 'rt'):
    yield eval(l)

def readCSV(path):
  f = gzip.open(path, 'rt')
  f.readline()
  for l in f:
    yield l.strip().split(',')

all_ratings = []
for user,book,r in readCSV("train_Interactions.csv.gz"):
  all_ratings.append([user,book,float(r)])

from surprise import SVDpp, accuracy
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import train_test_split
import numpy as np
from surprise.model_selection.split import KFold
from collections import defaultdict

reader = Reader(line_format='user item rating', sep=',', skip_lines=1)
data = Dataset.load_from_file("" + "train_Interactions.csv", reader=reader)
trainset, testset = train_test_split(data, test_size=.25)

# data = Dataset.load_from_df(df_ratings, reader=reader)
bestMSE = 2
best_i = 0
best_u = 0

i_mse = defaultdict(list)
u_mse = defaultdict(list)

# for i in [1,2,3,4]:
#     for u in [0.25,0.23,0.22,0.2,0.18,0.16,0.14,0.12,0.09,0.07]:

#         kf = KFold(n_splits=5)

#         accur = []

#         for trainset, testset in kf.split(data):

#     # train and test algorithm.
#             model = SVDpp(n_factors = i, reg_all = u)
#             model.fit(trainset)
#             predictions = model.test(testset)

#     # Compute and print Root Mean Squared Error
#             accur.append(accuracy.mse(predictions, verbose=False))

#         accur = np.array(accur)
    
#         mse_current = np.mean(accur)
#         if(mse_current < bestMSE):
#             best_i = i
#             best_u = u
            
#             bestMSE = mse_current

#         print("reg_all = ", u, " n_factors = ", i, "MSE = ", mse_current)
#         i_mse[i].append(mse_current)
#         u_mse[u].append(mse_current)

# print("bestMSE = ", bestMSE)
# print("best parameter reg_bi = ", best_i, " reg_bu = ", best_u)


trainset, testset = train_test_split(data, test_size=0.000000001)
model = SVDpp(reg_all=0.11,n_factors=0)
model.fit(trainset)

predictions = open("predictions_Rating.csv", 'w')
for l in open("./pairs_Rating.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,g = l.strip().split(',')
    
    # Logic...
    
    # _ = predictions.write(u + ',' + g + ',' + str(predict_rating(u,g)) + '\n')

    prediction = model.predict(u,g).est
    # prediction = float(modelLFM.predict(userIDs[u], itemIDs[i]))

    # prediction = modelLFM.predict(userIDs[u], itemIDs[i]).numpy()
    # print(prediction)
    if(prediction < 0):
        prediction = 0
    if(prediction > 5):
        prediction = 5


    _ = predictions.write(u + ',' + g + ',' + str(prediction) + '\n')

predictions.close()


