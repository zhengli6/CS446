import numpy as np 
import csv
import readCardInfo 
# import NN
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import pandas as pd

def readS(filename):
	with open(filename,'rU') as csvfile:
		reader=csv.reader(csvfile)
		S = [[str(i) for i in r] for r in reader]
		S = np.matrix(S)
	return S

def genCardMatrix(DataM, cardInfo):
	cardNum = len(cardInfo);
	row, col = DataM.shape;
	cardMatrix = []
	matchResult = []
	for i in range(row):
		score1 = int(DataM[i,-2])
		score2 = int(DataM[i,-1])
		if score1==score2:
			continue;
		elif score1>score2:
			matchResult.append(np.array(1))
		else:
			matchResult.append(np.array(0));

		line = np.array([0 for p in range(cardNum*2)]);
		for j in range(col-2):
			cardObj = cardInfo[DataM[i,j]]
			if j <8:
				line[int(float(cardObj.ID))-1] = 1#cardObj.NAME.ljust(8)
			else:
				line[int(float(cardObj.ID))-1 + cardNum] = 1#cardObj.NAME.ljust(8)

		cardMatrix.append(line);

	return np.matrix(cardMatrix), np.matrix(matchResult).T;

def genCostMatrix(DataM, cardInfo):
	row, col = DataM.shape;
	level = 6
	costMatrix = []
	for i in range(row):
		score1 = int(DataM[i,-2])
		score2 = int(DataM[i,-1])
		if score1==score2:
			continue;
		ave1, ave2 = getAveCost(np.asarray(DataM[i,0:(col-2)]).ravel(), cardInfo);
		line = np.array([0 for p in range(level*2)]);
		idx1 = int(np.floor(ave1))-1;
		idx2 = int(np.floor(ave2))-1;
		line[idx1] = 1;
		line[idx2+level] = 1;
		costMatrix.append(line);

	return np.matrix(costMatrix);

def getAveCost(line, cardInfo):
	ave1 = 0
	ave2 = 0
	for i in range(8):
		ave1+=float(cardInfo[line[i]].COST)
		ave2+=float(cardInfo[line[i+8]].COST)

	return ave1/8.0, ave2/8.0;

def genTypeMatrix(DataM, cardInfo):
	row, col = DataM.shape;
	typeMatrix = [];
	typeList = ['Troops', 'Spells', 'Buildings'];
	for i in range(row):
		score1 = int(DataM[i,-2])
		score2 = int(DataM[i,-1])
		if score1==score2:
			continue;
		line = np.array([0 for p in range(9*len(typeList)*2)]);
		type1, type2 = getType(np.asarray(DataM[i,0:(col-2)]).ravel(), cardInfo, typeList);
		for i in range(len(type1)):
			idx1 = type1[i]+i*9;
			idx2 = type2[i]+(i+len(typeList))*9;
			line[idx1] = 1;
			line[idx2] = 1;
		typeMatrix.append(line);

	return np.matrix(typeMatrix);

def getType(line, cardInfo, typeList):
	type1 = [0 for i in range(len(typeList))];
	type2 = [0 for i in range(len(typeList))];
	for i in range(8):
		tmpT1 = cardInfo[line[i]].TYPE;
		tmpT2 = cardInfo[line[i+8]].TYPE;
		type1[typeList.index(tmpT1)]+=1;
		type2[typeList.index(tmpT2)]+=1;

	return type1, type2;

def genRarityMatrix(DataM, cardInfo):
	row, col = DataM.shape;
	RarityMatrix = [];
	rarityList = ['common', 'Rare', 'Epic', 'Legendary'];
	for i in range(row):
		score1 = int(DataM[i,-2])
		score2 = int(DataM[i,-1])
		if score1==score2:
			continue;
		line = np.array([0 for p in range(9*len(rarityList)*2)]);
		rarity1, rarity2 = getRarity(np.asarray(DataM[i,0:(col-2)]).ravel(), cardInfo, rarityList)
		for i in range(len(rarity1)):
			idx1 = rarity1[i]+i*9;
			idx2 = rarity2[i]+(i+len(rarityList))*9;
			line[idx1] = 1;
			line[idx2] = 1;
		RarityMatrix.append(line);

	return np.matrix(RarityMatrix);

def getRarity(line, cardInfo, rarityList):
	rarity1 = [0 for i in range(len(rarityList))];
	rarity2 = [0 for i in range(len(rarityList))];
	for i in range(8):
		tmpR1 = cardInfo[line[i]].RARITY;
		tmpR2 = cardInfo[line[i+8]].RARITY;
		rarity1[rarityList.index(tmpR1)]+=1;
		rarity2[rarityList.index(tmpR2)]+=1;

	return rarity1, rarity2

def genPatternMatrix(DataM, pattern):
	row, col = DataM.shape;
	numPat = len(pattern);
	patMatrix = [];
	for i in range(row):
		score1 = int(DataM[i,-2])
		score2 = int(DataM[i,-1])
		if score1==score2:
			continue;
		tmpPat = [1 for m in range(numPat*2)];
		for j in range(numPat):
			pat = pattern[j];
			player1 = np.asarray(DataM[i, 0:8]).ravel();
			player2 = np.asarray(DataM[i, 8:16]).ravel();
			if not checkSub(player1, pat):
				tmpPat[j*2] = 0;
			if not checkSub(player2, pat):
				tmpPat[j*2+1] = 0;
			
		patMatrix.append(tmpPat);

	return np.matrix(patMatrix);

def checkSub(arr, subArr):
	if set(arr)-set(subArr)!=set([]) and set(subArr)-set(arr)==set([]):
		return True;
	else:
		return False;

def readPattern(filename, threshold):
	patMatrix = []
	with open(filename) as f:
		allPattern = f.readlines()[2:];
		for pattLine in allPattern:
			line = pattLine.split(' ');
			if int(line[0]) < threshold:
				continue;
			patt = line[2:];
			patt[0] = patt[0][1:];
			patt[-1] = patt[-1][:-2]
			patMatrix.append(patt);

	return patMatrix;

def writeFeatureMatrix(featureMatrix):
	file = open('featureMatrix.csv', 'w');
	wr = csv.writer(file, dialect='excel');
	(row, col) = featureMatrix.shape;
	for i in range(row):
		tmpArr = np.asarray(featureMatrix[i,:]).ravel();
		wr.writerow(tmpArr)


xlsxName = 'cards.xlsx'
DataM = readS('data_0331.csv')
pattern = readPattern('pattern.txt', 500);
cardInfo = readCardInfo.readCARD(xlsxName)
cardMatrix, matchResult = genCardMatrix(DataM, cardInfo);
print 'cardMatrix', cardMatrix.shape
print 'matchResult', matchResult.shape;
costMatrix = genCostMatrix(DataM, cardInfo);
print 'costMatrix', costMatrix.shape;
typeMatrix = genTypeMatrix(DataM, cardInfo);
print 'typeMatrix', typeMatrix.shape
RarityMatrix = genRarityMatrix(DataM, cardInfo);
print 'RarityMatrix', RarityMatrix.shape
patMatrix = genPatternMatrix(DataM, pattern);
print 'PatMatrix', patMatrix.shape
featureMatrix = cardMatrix;
featureMatrix = np.append(featureMatrix, costMatrix, axis=1);
featureMatrix = np.append(featureMatrix, typeMatrix, axis=1);
featureMatrix = np.append(featureMatrix, RarityMatrix, axis=1);
featureMatrix = np.append(featureMatrix, patMatrix, axis=1);
# featureMatrix = np.append(featureMatrix, matchResult, axis=1);
print 'featureMatrix', featureMatrix.shape
# writeFeatureMatrix(featureMatrix)

# Implementation of nn algorithm
"""
batch_size = 10;
learning_rate = 0.1;
activation_function = 'tanh';
hidden_layer_width = 10;
domain = 'mnist';
net = NN.create_NN(col, domain, batch_size, learning_rate, activation_function, hidden_layer_width)
net.train(newTrainMatrix);
print 'accuracy ' + str(net.evaluate(newTestMatrix));

"""

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

matchResult = np.ravel(matchResult)
train_data = featureMatrix[0:35000,:]
train_label= matchResult[0:35000]
test_data = featureMatrix[35000:45000,:]
test_label= matchResult[35000:45000]
clf = RandomForestClassifier(n_estimators=25,warm_start=True,oob_score=True)
clf.fit(train_data, train_label)
pred = clf.predict(test_data)
score = metrics.accuracy_score(test_label, pred)
print 'Train accuracy is:',clf.oob_score_
print("Test accuracy is:   %0.3f" % score)
print('finished')




