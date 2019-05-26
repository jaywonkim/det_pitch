import os
import csv

nDivCont = 1000000
filePath = './data/'
fileName = 'pitches'
fileExe = '.csv'
fileFolder = 'DivFile/'

dirName = filePath + fileFolder
if not os.path.isdir(dirName):
    os.mkdir(dirName)

nLineCnt = 0
nFileIdx = 0

f = open("%s" % (filePath + fileName + fileExe), 'r')
fDivName = open("%s%06d%s" % (filePath + fileFolder + fileName, nFileIdx, fileExe),'w')

while True :
    line = f.readline()

    if not line : break

    if nLineCnt == nDivCont :
        fDivName.close()
        nFileIdx += 1
        nLineCnt = 0
        strPat = "%s%06d%s" % (filePath + fileFolder + fileName, nFileIdx, fileExe)
        fDivName = open(strPat, 'w')
        print("생성 완료 %s" % strPat)
    nLineCnt += 1
    fDivName.write(line)

fDivName.close()
f.close()