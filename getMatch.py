from bs4 import BeautifulSoup
import requests
import re

def getPlayers(url):
	source_code = requests.get(url);
	plain_text = source_code.text;
	soup = BeautifulSoup(plain_text, 'html.parser');
	table = soup.find('table', attrs = {'class':"table table-condensed"})
	PlayerSet = []
	for line in table.find_all('a'):
		if 'protected' not in str(line):
			PlayerSet.append(str("".join(line.contents))[1:]);

	return PlayerSet

def getMatch(url, player):
	newURL = url+str(player);
	source_code = requests.get(newURL);
	plain_text = source_code.text;
	soup = BeautifulSoup(plain_text, 'html.parser');
	allDiv = soup.find_all('div', attrs = {'class':"panel panel-inverse"});
	cardSet = [];
	scoreSet = [];
	if len(allDiv)<2:
		return cardSet, scoreSet;
	for match in allDiv[1].find_all('div', attrs = {'class':"panel panel-inverse"}):
		scoreLine = str(match.find('span', attrs = {'style':'background-color: #000;'}).contents);
		scoreSet.append(str(re.search('.*(. - .).*', scoreLine).group(1)));
		cardsTable = match.find_all('ul', attrs = {'class':'deck'});
		if len(cardsTable)>2:
			continue;
		cards = [];
		for cardLine in cardsTable:
			for card in cardLine.find_all('li', attrs = {'class':'spell'}):
				cardURL = str(card.find('img')['src']);
				cards.append(str(re.search('/images/cards/(.*).png', cardURL).group(1)));
		cardSet.append(cards);

	# print player, cardSet[-1], scoreSet[-1]
	for i in range(0,len(cardSet)):
		print('=========================================')
		print(player)
		print("player1's deck ")
		print(cardSet[i][0:7])
		print("opponent's deck ")
		print(cardSet[i][8:15])
		print("result " + scoreSet[i])


	return scoreSet, cardSet;

PlayerSet = getPlayers('https://statsroyale.com/top/players/');
scoreTotal = {};
cardTotal = {};
for player in PlayerSet:
	if player == '22RC8G8C':
		scoreSet, cardSet = getMatch('https://statsroyale.com/profile/', player);
		scoreTotal[player] = scoreSet;
		cardTotal[player] = cardSet;

print "finished";