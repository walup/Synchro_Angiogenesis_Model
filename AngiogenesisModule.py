import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import random
from tqdm import tqdm
import pickle

class Direction(Enum):
    '''
    Movement directions for tip cells. 
    1.Left
    2.Right
    3.Up
    4.Down
    5.Center
    '''
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4
    CENTER = 0

class TipCell:

    def __init__(self, i, j):
        '''
        To initialize a tip cell the initial location coordinates (i,j) mst be specified
        '''
        self.x = j
        self.y = i
        self.life = 0

    def move(self, direction, width, height):
        '''
        Movement of tip-cells requires a direction and dimensions of the embedding space. 

        Tip cells won't move if outside such space. 
        '''
        if(direction == Direction.UP and self.y - 1 >= 1):
            self.y = self.y - 1
        elif(direction == Direction.DOWN and self.y + 1 <= height - 2):
            self.y = self.y + 1
        elif(direction == Direction.LEFT and self.x - 1 >= 1):
            self.x = self.x - 1
        elif(direction == Direction.RIGHT and self.x + 1 <= width - 2):
            self.x = self.x + 1
        
        #Increment the life count of the tip-cell
        self.life = self.life + 1

    def getLife(self):
        return self.life

class AngiogenesisModule:

    def __init__(self, width, height):

        self.width = width
        self.height = height
        self.k = 1/(((2*10**-3)**2)/(2.9*10**-7))
        self.h = 0.005
        self.D = 0.00035
        self.alpha = 0.6
        #self.chi0 = 0.43
        self.chi0 = 0.38
        self.rho = 0.34
        self.beta = 0.05
        self.gamma = 0.1
        self.eta = 0.035
        self.tAge = 2
        self.delta = 1
        self.kn = 0.75
        self.maxBranches = 50
        self.maxTipCellLife = 100

        self.maxC = 20
        self.deltaC = 0.01
        self.Dc = 0.01
        self.tafDissipation = 0.001

        #Artist properties
        self.tipCellColor = [224/255, 190/255, 79/255]
        self.proliferatingColor = [58/255.0, 252/255.0, 84/255.0]
        self.vasculatureColor = [201/255, 74/255, 0/255]
        self.backgroundColor = [1, 1, 1]
    

    def initializeSystem(self, ecm, proliferatingLocations, nutrient):
        '''
        We will initialize concentrations of angiogenic factors.

        These initial concentrations are established using the locations of the automaton that are occupied by proliferating cancer cells. 
        '''
        self.nutrient = nutrient
        self.bloodVesselNutrientConcentration = 2
        
        #Endothelial cells
        self.nMatrix = np.zeros((self.height, self.width))
        #Angiogenic factors
        self.cMatrix = np.zeros((self.height, self.width))
        #Fibronectine
        self.fMatrix = np.zeros((self.height, self.width))
        
        #Initialize the array of tip-cells
        self.tipCells = []
        #Cells occupied by blood vessels
        self.occupiedCells = np.zeros((self.height, self.width))

        self.proliferatingVEGFInitial = 0.5
        #Set initial concentration of angiogenic factors
        #Total number of proliferating cells
        nProlif = sum(sum(proliferatingLocations))
        prolifIndexes = [index for index, x in np.ndenumerate(proliferatingLocations) if x]
        for i in range(1, self.height - 1):
            for j in range(1, self.width - 1):
                #Fibronectin is initialized as 0.4 (taken from the angiogenesis paper)
                self.fMatrix[i,j] = 0.4
                if(proliferatingLocations[i,j] == 1):
                    self.cMatrix[i,j] = self.proliferatingVEGFInitial
                for s in range(0,len(prolifIndexes)):
                    indexPair = prolifIndexes[s]
                    ind1 = indexPair[0]
                    ind2 = indexPair[1]
                    dst = np.sqrt(((ind1 - i)*self.h)**2 + ((ind2 - j)*self.h)**2)
                    if(dst < 0.1):
                        self.cMatrix[i,j] = self.cMatrix[i,j] + 2
                    else:
                        nu = (np.sqrt(5) - 0.1)/(np.sqrt(5) - 1)
                        self.cMatrix[i,j] = self.cMatrix[i,j] + (((nu - dst)**2)/(nu - 0.1))

        self.cMatrix = self.cMatrix/np.max(self.cMatrix)

        #Let's initialize tip-cells at the borders of the automaton
        tipCellPositions = np.zeros((self.height, self.width))
        for i in range(0,self.width):
            if(i%20 == 0):
                tipCellPositions[i,1] = 1
                tipCellPositions[1,i] = 1
                tipCellPositions[self.height-2,i] = 1
                tipCellPositions[i,self.width-2] = 1
        
        for i in range(0,self.height):
            for j in range(0,self.width):
                if(tipCellPositions[i,j] == 1):
                    self.tipCells.append(TipCell(i,j))
                    self.occupiedCells[i,j] = 1
                    self.nMatrix[i,j] = 1

        #That's it, this is all we need!

    def evolutionStep(self):
        '''
        One evolution step involves
        1. Updating concentrations of ECM and TAF
        2. Updating the movement of tip-cells with such conditions. 
        3. We incorporate the creation of a new neighboring tip-cell if there is a
        proliferating cell at the location.
        '''
        #Obtain previous concentration values for ECM, TAF, and endothelial cells
        n = self.nMatrix.copy()
        f = self.fMatrix.copy()
        c = self.cMatrix.copy()

        for i in range(1, self.height - 1):
            for j in range(1, self.width - 1):
                p0 = 1 - ((4*self.k*self.D/(self.h**2))) + ((self.k*self.alpha*self.chi(c[i,j]))/(4*(self.h**2)*(1 + self.alpha*c[i,j])))*((c[i, j+1] - c[i, j-1])**2 + (c[i-1,j] - c[i+1,j])**2)-((self.k*self.chi(c[i,j]))/(self.h**2))*(c[i+1,j] + c[i-1,j] + c[i,j+1] + c[i,j-1] -4*c[i,j]) - ((self.k*self.rho)/(self.h**2))*(f[i+1, j] + f[i-1, j] - 4*f[i,j] + f[i,j+1] + f[i,j-1])
                p1 = ((self.k*self.D)/(self.h**2)) - (self.k/(4*self.h**2))*(self.chi(c[i,j])*(c[i,j+1] - c[i,j-1]) + self.rho*(f[i,j+1] - f[i, j-1]))
                p2 = ((self.k*self.D)/(self.h**2)) + (self.k/(4*self.h**2))*(self.chi(c[i,j])*(c[i,j+1] - c[i,j-1]) + self.rho*(f[i,j+1] - f[i, j-1]))
                p3 = ((self.k*self.D)/(self.h**2)) + (self.k/(4*self.h**2))*(self.chi(c[i,j])*(c[i-1,j] - c[i+1,j]) + self.rho*(f[i-1,j] - f[i+1, j]))
                p4 = ((self.k*self.D)/(self.h**2)) - (self.k/(4*self.h**2))*(self.chi(c[i,j])*(c[i-1,j] - c[i+1,j]) + self.rho*(f[i-1,j] - f[i+1, j]))

                probArray = self.getProbabilities([p0, p1, p2, p3, p4])
                p0 = probArray[0]
                p1 = probArray[1]
                p2 = probArray[2]
                p3 = probArray[3]
                p4 = probArray[4]

                #Update values for ECM, TAF, and endothelial cells
                #Endothelial cells
                self.nMatrix[i,j] = p0*self.occupiedCells[i,j] + p1*self.occupiedCells[i,j-1] + p2*self.occupiedCells[i,j+1] + p3*self.occupiedCells[i-1, j] + p4*self.occupiedCells[i+1,j]
                
                #ECM (Fibronectin)
                self.fMatrix[i,j] = f[i,j]*(1 - self.k*self.gamma*n[i,j]) + self.k*self.beta*n[i,j]
                
                #TAF Concentration
                self.cMatrix[i,j] = c[i,j]*(1 - self.k*self.eta*n[i,j])
                
                if(self.tipOccupied(i,j)):
                    direction = self.getDirection(probArray)
                    tipIndex = self.getTipIndex(i,j)

                    #Move the tip
                    self.tipCells[tipIndex].move(direction, self.width, self.height)
                    #Mark the position with a blood vessel
                    self.occupiedCells[self.tipCells[tipIndex].y, self.tipCells[tipIndex].x] = 1
                    self.nutrient.putValue(self.tipCells[tipIndex].y, self.tipCells[tipIndex].x, self.bloodVesselNutrientConcentration)

                    #If after moving the tip we reach the position of another tip
                    #we will merge them (anastomosis)
                    oldTip = self.tipCells[tipIndex]
                    self.anastomosis(self.tipCells[tipIndex], tipIndex)

                    #Branching
                    branchingPositions = self.getBranchingPositions(i,j)
                    tipIndex = self.getTipIndex(i,j)
                    if(oldTip.life > self.tAge and len(branchingPositions) > 0 and n[i,j] > self.kn/c[i,j]):
                        Pn = c[i,j]/np.max(c)
                        if(random.random() < Pn and len(self.tipCells) < self.maxBranches):
                            positionIndex = random.randint(0,len(branchingPositions) - 1)
                            branchPosition = branchingPositions[positionIndex]
                            newTip = TipCell(branchPosition[0], branchPosition[1])
                            self.tipCells.append(newTip)
                            self.occupiedCells[newTip.y, newTip.x] = 1
                            
        
    def getTipCellLocations(self):
        '''
        Returns the location of tip-cells
        '''
        locations = np.zeros((self.height, self.width))
        for i in range(0,len(self.tipCells)):
            locations[self.tipCells[i].y, self.tipCells[i].x] = 1
        return locations
                    

    def getProbabilities(self, probArray):
        '''
        Computes probabilities from a set of continuous values
        '''

        probArray = np.array(probArray)
        minValue = np.min(probArray)
        maxValue = np.max(probArray)
        #First we get all values in a range from 0 to 1
        #by scaling them. This assures that we won't have any negative values
        for i in range(0,len(probArray)):
            probArray[i] = (probArray[i] - minValue)/(maxValue - minValue)

        #Now we convert our values into probabilities by dividing every value
        #by the sum of all the positive values in the array
        probArray = probArray/sum(probArray)
        return probArray

    def tipOccupied(self, i,j):
        '''
        Verifies if there is a tip-cell at location (i,j)
        '''
        for s in range(0,len(self.tipCells)):
            if(self.tipCells[s].x == j and self.tipCells[s].y == i):
                return True

        return False

    def getDirection(self, probArray):
        '''
        Returns a direction from the direction probability array. The indexes of such array correspond to the ones used in the enum

        Index 0: Center direction
        Index 1: Left direction
        Index 2: Right direction
        Index 3: Up direction
        Index 4: Down direction
        '''
        cumulative = 0
        rand = random.random()
        for i in range(0,len(probArray)):
            if(rand >= cumulative and rand <= cumulative + probArray[i]):
                return Direction(i)

            else:
                cumulative = cumulative + probArray[i]


    def getTipIndex(self, i, j):
        '''
        Returns the index of the tip-cell located at i,j (you would need to verify that such tip actually exists, otherwise the method will return a None object)
        '''
        for s in range(0,len(self.tipCells)):
            if(self.tipCells[s].x == j and self.tipCells[s].y == i):
                return s
            
        return None

    def anastomosis(self, tip, index):
        '''
        Remove one of two duplicated tip-cells if they are located in the same 
        position within the automaton
        '''
        tipsToRemove = []
        for i in range(0,len(self.tipCells)):
            if(index != i and tip.x == self.tipCells[i].x and tip.y == self.tipCells[i].y):
                tipsToRemove.append(self.tipCells[i])

        for i in range(0,len(tipsToRemove)):
            self.tipCells.remove(tipsToRemove[i])

    def getBranchingPositions(self, i, j):
        '''
        Gives you an array with the positions in the neighborhood of (i,j) that
        are available to be occupied by new tip-cells (new branch of the network)
        '''

        availablePositions = []
        for s in range(-1,2):
            for l in range(-1,2):
                if(not self.tipOccupied(i+s, j+l)):
                    availablePositions.append([i+s, j+l])

        return availablePositions

    def updateTAFConcentrations(self, proliferatingPositions):
        '''
        Updates the TAF Concentrations with the location of cancer cells, which is dynamic.

        This is one of the interactions with the tumor that is considered. 
        '''
        for i in range(1,self.height-1):
            for j in range(1,self.width-1):
                if(proliferatingPositions[i,j] == 1 and self.cMatrix[i,j] < self.maxC):
                    self.cMatrix[i,j] = self.cMatrix[i,j] + self.deltaC
                else:
                #Diffuse the concentration
                    self.cMatrix[i,j] = self.cMatrix[i,j] + self.k*self.Dc*(self.cMatrix[i+1,j] + self.cMatrix[i-1,j] + self.cMatrix[i, j+1] + self.cMatrix[i,j-1] - 4*self.cMatrix[i,j]) - self.tafDissipation

    def chi(self, x):
        '''
        Function necessary to obtain movement direction probabilities for tip-cells
        '''
        return self.chi0/(1 + self.delta*x)

    def saveInstance(self, fileName):
        '''
        Save a pickle instance of this object
        '''
        with open(fileName, 'wb') as file:
            pickle.dump(self, file)
            print("Object saved to: "+fileName)

    def openInstance(self, fileName):
        '''
        Open a pickle instance of the model located in the specified
            directory
        '''
        instance = pickle.load(open(fileName, 'rb'))
        return instance

    def removeBloodVessel(self, i, j):
        '''
        Removes a blood vessel (tip-cell also if necessary) at location (i,j) of the automaton.
        '''
        if(self.occupiedCells[i,j] == 1):
            self.occupiedCells[i,j] = 0
            if(self.tipOccupied(i,j)):
                tipIndex = self.getTipIndex(i,j)
                self.tipCells.pop(tipIndex)
            


    def getPicture(self, tipCellLocations, proliferatingLocations, vasculaturePicture):
        '''
        Returns an np.array holding a 3-channel image of the tumor and the surrounding vasculature.
        '''
        picture = np.ones((self.height, self.width, 3))
        for i in range(0,self.height):
            for j in range(0,self.width):
                if(vasculaturePicture[i,j] == 1):
                    picture[i,j,:] = self.vasculatureColor
                if(proliferatingLocations[i,j] == 1):
                    picture[i,j,:] = self.proliferatingColor
                if(tipCellLocations[i,j] == 1):
                    picture[i,j,:] = self.tipCellColor

                if(proliferatingLocations[i,j] == 0 and tipCellLocations[i,j] == 0 and vasculaturePicture[i,j] == 0):
                    picture[i,j,:] = self.backgroundColor

        return picture
            

        
        
            


        

        

        
        
        
        
    
                
        

    



        