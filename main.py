
# coding: utf-8

# # In silico Voting Experiments
# 
# Implementation of cultures and voting rules
# 
# Code uploaded to the github.

# In[1]:


import pandas 
import sklearn
import math
import numpy as np
import random
random.seed(12)


# # Cultures

# In[2]:


class Сulture:
    def __init__(self, n, K):
        self.n = n
        self.K = K
        self.ctype = None
    
    def createMatrix(self):
        # matrix K x K 
        print(self.ctype)
        M = np.zeros((self.K, self.K))
        for t in range(self.n):
            M += self.createMatrixProfile()
        return M


# In[3]:


class Rousseauist_culture(Сulture):
    # We assume that 1 > 2 > ... > K
    def setParams(self, alpha, beta):
        # alpha >= 0
        # beta >= 0
        self.alpha = alpha
        self.beta = beta
        self.ctype = "Rousseauist_culture"
        
    def createMatrixProfile(self):
        tempM = np.zeros((self.K, self.K))
        for i in range(self.K):
            for j in range(i+1, self.K):
                if( random.random() < self.getP(i,j)):
                    tempM[i, j] += 1
                else:
                    tempM[j, i] += 1
        return tempM
        
    def getP(self, k1, k2):
        # Truchon and Drissi-Bakhkhat
        
        # k > k'
        if(k2 <= k1): 
            raise NameError("k >= k'")
            
        s = math.exp(self.alpha + self.beta*(k2-k1))
        return s/(1+s)


# In[4]:


# r = Rousseauist_culture(5,6)
# r.setParams(1,1)
# r.getP(4,5)
# print(r.createMatrix())


# In[17]:


class Impartial_culture(Сulture):
    def setParams(self):
        self.ctype = "Impartial_culture"
    
    def createLinearProfile(self):
        # [1,6,4,3,0,5]: 1 > 6 > 4 > 3 > 0 > 5
        profile = list(range(self.K))
        random.shuffle(profile)
        return profile
    
    def linearIntoMatrix(self, profile):
        tempM = np.zeros((self.K, self.K))
        for i in range(self.K):
            for j in range(i+1, self.K):
                tempM[profile[i], profile[j]] += 1
        return tempM
    
    def createMatrixProfile(self):
        pr = self.createLinearProfile()
        return self.linearIntoMatrix(pr)
        


# In[18]:


# i = Impartial_culture(5,6)
# i.setParams()
# i.createMatrix()


# In[19]:


class Distributive_culture(Сulture):
    def setParams(self):
        self.ctype = "Distributive_culture"


# In[20]:


# Consensual redistributive culture
class Consensual_redistributive_culture(Distributive_culture):
    def setParams(self):
        self.ctype = "Distributive_culture"
        self.cctype = "Consensual_redistributive_culture"
    
    def defineShares(self):
        Y = np.random.random((self.K))
        X = np.array([ll/sum(Y) for ll in Y])
        return X
    
    def sharesIntoLinear(self, shares):
        # [0.61, 0.85, 0.42, 0.13]: [1, 0, 2, 3]: 1 > 0 > 2 > 3 
        profile = [i for (v, i) in sorted([(v, i) for (i, v) in enumerate(shares)], reverse=True)]
        return profile
    
    def linearIntoMatrix(self, profile):
        tempM = np.zeros((self.K, self.K))
        for i in range(self.K):
            for j in range(i+1, self.K):
                tempM[profile[i], profile[j]] += 1
        return tempM
    
    def createMatrixProfile(self):
        shares = self.defineShares()
        pr = self.sharesIntoLinear(shares)
        print(pr)
        return self.linearIntoMatrix(pr)
    
    def GiniIndex(self, shares: np.array):
        growing = sorted(shares)
        accumulated = np.cumsum(growing)
        S = 0
        for i in range(len(accumulated)):
            S += ((i+1)/len(accumulated) - accumulated[i])
        S *= 2/len(accumulated)
        return S
    
    def simpleGiniIndex(self, shares: np.array):
        growing = sorted(shares)
        accumulated = np.cumsum(growing)
        S = 0
        for i in range(len(accumulated)):
            if(i==0):
                S += ( (i+0.5) /len(accumulated) - (accumulated[i]-0)/2)
            else:
                S += ( (i+0.5) /len(accumulated) - (accumulated[i]-accumulated[i-1])/2)
        S *= 2/len(accumulated)
        return S
        


# In[21]:


# crc = Consensual_redistributive_culture(4,6)
# crc.setParams()
# crc.createMatrix()


# ## Something strange happend with SimpleGini. We didn't obtained exactly the same results.
# Have to be done in future

# In[22]:


def SimulateGini(times = 100):
    vals = [3,5,11,49,99]
    gini_avgs = []
    for K in vals:
        gini_avg = 0
        for t in range(times):
            crc = Consensual_redistributive_culture(4,K)
            crc.setParams()
            shares = crc.defineShares()
            gini_avg += crc.GiniIndex(shares)
        gini_avg /= times
        print(f'Average Gini index for K = {K} is: {gini_avg}')
        gini_avgs.append(gini_avg)

def SimulateSimpleGini(times = 100):
    vals = [3,5,11,49,99,999]
    gini_avgs = []
    for K in vals:
        gini_avg = 0
        for t in range(times):
            crc = Consensual_redistributive_culture(4,K)
            crc.setParams()
            shares = crc.defineShares()
            gini_avg += crc.simpleGiniIndex(shares)
        gini_avg /= times
        print(f'Average Gini index for K = {K} is: {gini_avg}')
        gini_avgs.append(gini_avg)
#
# print('Gini:')
# SimulateGini()
# print('\nSimpleGini:')
# SimulateSimpleGini()


# In[24]:


# Inegalitarian distributive cultures
class Inegalitarian_distributive_culture(Distributive_culture):
    def setParams(self, e):
        self.ctype = "Distributive_culture"
        self.cctype = "Inegalitarian_distributive_culture"
        self.e = e
    
    def defineShares(self):
        
#         Y = [(i+1)**e / self.n for i in range(self.n)]
#         X = []
#         for i, el in enumerate(Y):
#             if(i==0):
#                 X.append(el)
#             else:
#                 X.append(Y[i]-Y[i-1])
        X = [(i+1 ** self.e) / self.n for i in range(self.K)]
        return X
        
    def sharesIntoLinear(self, shares):
        # [0.61, 0.85, 0.42, 0.13]: [1, 0, 2, 3]: 1 > 0 > 2 > 3 
        profile = [i for (v, i) in sorted([(v, i) for (i, v) in enumerate(shares)], reverse=True)]
        return profile
    
    def linearIntoMatrix(self, profile):
        tempM = np.zeros((self.K, self.K))
        for i in range(self.K):
            for j in range(i+1, self.K):
                tempM[profile[i], profile[j]] += 1
        return tempM
    
    def createMatrixProfile(self):
        shares = self.defineShares()
        pr = self.sharesIntoLinear(shares)
        print(pr)
        return self.linearIntoMatrix(pr)
    
    def GiniIndex(self, shares: np.array):
        growing = sorted(shares)
        accumulated = np.cumsum(growing)
        S = 0
        for i in range(len(accumulated)):
            S += (i/len(accumulated) - accumulated[i])
        S *= 2/len(accumulated)
        return S


# In[26]:


crc = Inegalitarian_distributive_culture(4,6)
crc.setParams(e = 2)
crc.createMatrix()


# In[98]:


class Spatial_culture(Сulture):
    def setDimention(self, d):
        self.d = d
        self.ctype = "Spatial_culture"
        
    def inequalityMeasure(self, k1, k2):
        # ??? 
        return 0


# In[47]:


def generateCultureFile(culture: Culture):
    if culture.ctype == "Rousseauist_culture":
        pass
    elif culture.ctype == "Impartial_culture":
        pass
    elif culture.ctype == "Distributive_culture":
        pass
    elif culture.ctype == "Spatial_culture":
        pass


# # Voting types

# In[ ]:



class Voting:
    def __init__(self, filepath, method):
        self.df = pd.read_csv(filepath)
        self.df = self.df.drop('Same', axis=1)
        self.alternatives = set(self.df.columns)
        self.method = method

        # All preferences for each alternative
        self.ALTR_dict = {} # was profile_dict
        for cand in self.alternatives:
            self.ALTR_dict[cand] = self.df[cand].values

        # Preferences for each voter
        self.INDV_d = {}
        for itr, row in self.df.iterrows():
            self.INDV_d[itr]=row.to_dict()

    def voting(self):
        if (self.method == 'Plurality'):
            return self.plurality_voting()
        elif (self.method == 'Borda'):
            return self.borda_voting()
        elif (self.method == 'Nanson'):
            return self.nanson_voting()
        elif (self.method == 'STV'):
            return self.stv_voting()
        elif (self.method == 'Copeland'):
            return self.copeland_voting()
        elif (self.method == 'OurMethod'):
            return self.our_voting()


    # Common fuctions:
    def reboot(self):
        # Normalize
        self.df = self.normalize_df(self.df)
        self.alternatives = set(self.df.columns)

        self.ALTR_dict = {}
        for cand in self.alternatives:
            self.ALTR_dict[cand] = self.df[cand].values

        # Preferences for each voter
        self.INDV_d = {}
        for itr, row in self.df.iterrows():
            self.INDV_d[itr]=row.to_dict()

        return self.df

    def normalize_df(self, df):
        new_df = pd.DataFrame(columns=df.columns)
        for itr, row in df.iterrows():
            new_row = {}
            sorted_dict = sorted(row.to_dict().items(), key=operator.itemgetter(1))
            for i, el in enumerate(sorted_dict):
                new_row[el[0]] = i + 1
            new_df = new_df.append(new_row, ignore_index=True)
        return new_df

    def lexicografic_order(self, winners):
        winners.sort()
        return winners[0]

    def get_winners(self, score_dict):
        winners = []
        max_val = max([i for i in score_dict.values()])
        for key in score_dict.keys():
            if(score_dict[key]==max_val):
                winners.append(key)
        return winners

    def get_losers(self, score_dict):
        losers = []
        if(self.method == 'Nanson'):
            Mean = np.mean([score_dict[cand] for cand in score_dict.keys()])
            for cand in self.alternatives:
                if(score_dict[cand]<Mean):
                    self.df = self.df.drop(cand, axis=1)
        elif(self.method == 'STV'):
            min_val = min([i for i in score_dict.values()])
            for key in score_dict.keys():
                if(score_dict[key]==min_val):
                    losers.append(key)
        elif (self.method == 'OurMethod'):
            max_val = max([i for i in score_dict.values()])
            for key in score_dict.keys():
                if (score_dict[key] == max_val):
                    losers.append(key)
        return losers

    def checkCondorset(self, show=False):
        for el1 in self.alternatives:
            Winner_Flag = True
            for el2 in self.alternatives:
                if(el1!=el2):
                    scores = [0, 0]
                    for itr, row in self.df.iterrows():
                        if(self.df[el1][itr]<self.df[el2][itr]):
                            # el1 - winner
                            scores[0] += 1
                        else:
                            scores[1] += 1
                    if(scores[0]>scores[1]):
                        pass
                    elif (scores[0] < scores[1]):
                        condorset_winner = None
                        Winner_Flag = False
                        break
            if(Winner_Flag):
                if(show):
                    print(f'Alternative {el1} is a Condorcet winner.')
                self.df = self.df.drop(el1, axis=1)
                self.df = self.reboot()
                return True
        return False

    def deleteCondorset(self, show=False):

        Cond_Win = self.checkCondorset(show)
        while (Cond_Win):
            Cond_Win = self.checkCondorset(show)

    # Plurality:
    def plurality_score(self, cand):
        return sum([el == 1 for el in self.ALTR_dict[cand]])

    def plurality_voting(self):
        score_dict = {}
        for cand in self.alternatives:
            score_dict[cand] = self.plurality_score(cand)
        winners = self.get_winners(score_dict)
        return self.lexicografic_order(winners), score_dict

    # Borda:
    def pref_into_borda(self, alternative_prefs, m):
        res_list = []
        for pref in alternative_prefs:
            res_list.append(m-pref)
        return res_list

    def borda_score(self, cand, m):
        return sum(self.pref_into_borda(self.ALTR_dict[cand], m))

    def borda_voting(self):
        score_dict = {}
        m = len(self.alternatives)
        for cand in self.alternatives:
            score_dict[cand] = self.borda_score(cand, m)
        winners = self.get_winners(score_dict)
        return self.lexicografic_order(winners), score_dict

    # Nanson
    def nanson_voting(self):

        score_dict = {}
        m = len(self.alternatives)

        for cand in self.alternatives:
            score_dict[cand] = self.borda_score(cand, m)

        winners = self.get_winners(score_dict)
        while(len(winners)!=len(self.alternatives)):
            losers = self.get_losers(score_dict)
            for loser in losers:
                self.df = self.df.drop(loser, axis=1)

            self.df = self.reboot()
            m = len(self.alternatives)

            score_dict = {}
            for cand in self.alternatives:
                score_dict[cand] = self.borda_score(cand, m)

            winners = self.get_winners(score_dict)

        winners = self.get_winners(score_dict)
        return self.lexicografic_order(winners), score_dict

    # STV
    def stv_score(self, cand):
        return sum([el == 1 for el in self.ALTR_dict[cand]])

    def stv_voting(self):
        # Get scores
        score_dict = {}
        for cand in self.alternatives:
            score_dict[cand] = self.stv_score(cand)
        # Check two or more winners
        winners = self.get_winners(score_dict)
        while(len(winners)!=len(self.alternatives)):
            losers = self.get_losers(score_dict)
            for loser in losers:
                self.df = self.df.drop(loser, axis=1)
            # Normalize
            self.df = self.reboot()

            score_dict = {}
            for cand in self.alternatives:
                score_dict[cand] = self.stv_score(cand)

            winners = self.get_winners(score_dict)

        winners = self.get_winners(score_dict)
        return self.lexicografic_order(winners), score_dict

    # Copeland
    def build_dominance_graph(self):
        graph = {}
        reverse_graph = {}
        for cand in self.alternatives:
            graph[cand] = set()
            reverse_graph[cand] = set()

        for el1 in self.alternatives:
            for el2 in self.alternatives:
                if (el1 != el2):
                    scores = [0, 0]
                    for itr, row in self.df.iterrows():
                        if (self.df[el1][itr] < self.df[el2][itr]):
                            # el1 - winner
                            scores[0] += 1
                        else:
                            scores[1] += 1
                    if (scores[0] > scores[1]):
                        graph[el1].add(el2)
                        reverse_graph[el2].add(el1)
                    elif (scores[0] < scores[1]):
                        graph[el2].add(el1)
                        reverse_graph[el1].add(el2)

        return graph, reverse_graph

    def copeland_score(self, cand):
        return len(self.graph[cand])

    def copeland_voting(self):
        self.graph, self.reversed_graph = self.build_dominance_graph()
        score_dict = {}
        for cand in self.alternatives:
            score_dict[cand] = self.copeland_score(cand)
        winners = self.get_winners(score_dict)
        return self.lexicografic_order(winners), score_dict


