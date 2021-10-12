import math
import numpy as np
import matplotlib.pyplot as plt

class Grid:
    units = []
    SVVs = []
    SVVsSum = []

#test comment
    def convertCountToRGB(self, count):
        #sum up all of the numbers
        total = 0
        for num in count:
            total += num
        RGBArray = []
        for num in count:
            if total == 0:
                fraction = 0
            else:
                fraction = num / total
            RGBArray.append(fraction)
        return tuple(RGBArray)

    def weightVectorDistance(self, SVV, unitWV):
        #just do euclidian distance
        sum = 0.0
        for i in range(len(SVV)):
            sum += (unitWV[i] - SVV[i]) * (unitWV[i] - SVV[i])
        return math.sqrt(sum)

    def distanceFromSVVToUnitWV(self, SVV, unitWV):
        return self.weightVectorDistance(SVV, unitWV)

    def initialize(self, fileName):

        #read in the data from the file
        file = open(fileName, "r")
        theLines = file.readlines()
        for line in theLines:
            theLineData = line.split(",")
            theLineDataFloat = []
            for number in theLineData[:-1]:
                theLineDataFloat.append(float(number))
            # self.SVVsSum[0] += theLineDataFloat[0]
            # self.SVVsSum[1] += theLineDataFloat[1]
            # self.SVVsSum[2] += theLinesDataFloat[2]
            # self.SVVsSum[3] += theLinesDataFloat[3]
            theClass = theLineData[-1].strip()
            toAppend = None
            if theClass == "Iris-setosa":
                toAppend = 0
            elif theClass == "Iris-versicolor":
                toAppend = 1
            elif theClass == "Iris-virginica":
                toAppend = 2
            if toAppend == None:
                print(f"The line data is: {theLineData}")
                file.close()
                exit(1)
            theLineDataFloat.append(toAppend)
            self.SVVs.append(theLineDataFloat)
        file.close()

        self.normalizeData()

        # create units grid, values will be randomly chosen from normal distribution with mean of 0 and stdev of 1
        for i in range(20):
            currRow = []
            for j in range(20):
                theUnit = Unit()
                theUnit.x = i
                theUnit.y = j
                s = np.random.normal(0, 1, 4)
                theUnit.weightVector = [s[0],s[1],s[2],s[3],[0,0,0]]
                currRow.append(theUnit)

            self.units.append(currRow)

    def findBMUAndUpdateGrid(self, SVV):
        # you dont compare a unit to another unit, you compare a sample value vector to a unit's weight vector
        """go through all the units on the grid and find the one that has the closest
        euclidian distance, then adjust that unit accordingly"""
        smallestDistX = None
        smallestDistY = None
        smallestDist = 99999999999
        for i in range(len(self.units)):
            for j in range(len(self.units[i])):
                theDist = self.distanceFromSVVToUnitWV(SVV[:-1], self.units[i][j].weightVector[:-1])
                if theDist < smallestDist:
                    smallestDistX = self.units[i][j].x
                    smallestDistY = self.units[i][j].y
                    smallestDist = theDist
        bestMatchingUnit = self.units[smallestDistX][smallestDistY]
        theRadius = 2
        #make more similar to input
        neighbUnits = bestMatchingUnit.getNeighborhoodUnits(theRadius, self.units)

        neighbVal = 1/theRadius
        currVal = 1
        #now update the RGB values for the neighborhood
        for currR in neighbUnits:
            for unit in currR:
                unit.weightVector[-1][SVV[-1]] += currVal
            currVal -= neighbVal
        


        #now compare the weight vector to the SVV and adjust accordingly


        #loop through the neighborhood
        for n in range(len(neighbUnits)):
            #depending on the distance from the BMU, this changes
            if n == 0:
                thePercent = 0.20 #20% of the difference between the current column of the SVV and WV is what we change the current column of the WV by
            elif n == 1:
                thePercent = 0.10 #could use logarithmic decay
            neighbRadius = neighbUnits[n]
            for aUnit in neighbRadius:
                currWV = aUnit.weightVector
                for i in range(len(currWV[:-1])):
                    #if they're equal we don't need to adjust the column
                    if currWV[i] != SVV[i]:
                        #if they're not equal, then we need to adjust the WV to be more like the SVV
                        
                        #get the difference
                        diff = SVV[i] - currWV[i]
                        diffPercentVal = thePercent * diff
                        #if the diff is positive, we need to increase the WV to make it more similar to the SVV, if it's negative, we need to decrease the WV to make it more similar
                        currWV[i] += diffPercentVal
        #now the neighborhood has been adjusted

    def train(self):
        #at this point we have all of the SVVs that we want to "insert" into the map. So now we just "insert" each one into the map
        for aSVV in self.SVVs:
            self.findBMUAndUpdateGrid(aSVV)


    def normalizeData(self):
        #normalize data: calculate average and standard deviation for each column, subtract mean and divide by standard deviation

        numColumns = len(self.SVVs[0]) - 1

        for col in range(numColumns):

            sum = 0
            count = 0
            for row in self.SVVs:
                sum += row[col]
                count += 1
            avg = sum / count
            #calculate standard deviation
            sum2 = 0
            for row in self.SVVs:
                calc = (row[col] - avg) ** 2
                sum2 += calc
            calc = calc / count
            std = math.sqrt(calc)
            for i in range(len(self.SVVs)):
                self.SVVs[i][col] = (self.SVVs[i][col] - avg) / std        






    def __str__(self):
        #print how many inputs maps to each unit - reset at beggining of each epoch, stop when hasn't really changes after a couple epochs
        #print out how many of each sample mapped to each unit
        # ret = ""
        # ret += "Here is the Grid:\n"
        # #print out units grid
        # for i in range(len(self.units)):
        #     for j in range(len(self.units[i])):
        #         ret += str(self.units[i][j].weightVector)
        #     ret += "\n"

        # ret += "\n"
        # return ret

        #rectangleGrid = []
        ax = plt.gca()
        for i in range(len(self.units)):
            for j in range(len(self.units[i])):
                theRect = plt.Rectangle((20 * i, 400 - j * 20), width=20, height=20, facecolor=self.convertCountToRGB(self.units[i][j].weightVector[-1]), edgecolor="black")
                ax.add_patch(theRect)

        plt.axis("scaled")
        plt.axis("off")
        plt.show()



class Unit:
    weightVector = []
    x = None
    y = None

    def getNeighborhoodUnits(self, radius, units):
        # neighborhood is a radius up, down, left, right. NOT diagonal
        # only need to calculate once
        theUnits = []
        for theR in range(1, radius + 1):
            theR -= 1
            currRadiusUnits = []
            if (theR == 0):
                currRadiusUnits.append(units[self.x][self.y])
                theUnits.append(currRadiusUnits)
                continue
            if self.x - theR >= 0:
                currRadiusUnits.append(units[self.x - theR][self.y])
            if self.y - theR >= 0:
                currRadiusUnits.append(units[self.x][self.y - theR])
            if self.x + theR < 20:
                currRadiusUnits.append(units[self.x + theR][self.y])
            if self.y + theR < 20:
                currRadiusUnits.append(units[self.x][self.y + theR])
            theUnits.append(currRadiusUnits)
        return theUnits

    def __str__(self):
        #for now to test, we just have one number in the weight vector, so just print it out
        return str(self.weightVector)



#write a few tests to make sure it's working

theGrid = Grid()
theGrid.initialize("data/iris.txt")
theGrid.train()
print(theGrid)




# #test weightVectorDistance method
# print(f"this should be 4: {theGrid.weightVectorDistance(SVV, theGrid.units[3][4].weightVector)}")

# #test finding the best matching unit

# BMUFound = theGrid.findBMUAndUpdateGrid([13])
# print(f"The BMU is {BMUFound}")

# #test printing out grid

# print(theGrid)



# print(f"the closest unit to the unit with weight vector 12 is {theClosestUnit.weightVector[0]}")
# print()
# radiusUnits = units[3][4].getNeighborhoodUnits(3, units)
# # track how many of each class match to each unit
# currRow = 1
# for i in range(len(radiusUnits)):
#     print(f"Values at radius {currRow}: ", end="")
#     for j in range(len(radiusUnits[i])):
#         print(radiusUnits[i][j], end="")
#     print()
#     currRow += 1
        

# sample class

# epochs - number of times you run this




