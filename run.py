#!/usr/bin/python
import readCardInfo 
xlsxName = 'cards.xlsx'
cardInfo = readCardInfo.readCARD(xlsxName)
for card in cardInfo:
	print(card)