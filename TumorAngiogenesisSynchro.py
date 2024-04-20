from enum import Enum
import random
from scipy.stats import poisson
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from AngiogenesisModule import AngiogenesisModule

class CellType(Enum):
    '''
    Biological cell types considered in our simulated tumor
    1. Proliferating
    2. Complex (Immune complexes)
    3. Dead
    4. Necrotic
    '''
    PROLIFERATING = [28.0/255, 241.0/255, 93.0/255]
    COMPLEX = [26.0/255, 69.0/255, 245.0/255]
    DEAD = [245.0/255, 72.0/255, 27.0/255]
    NECROTIC = [130.0/255, 130.0/255, 130.0/255]

class GrowthPhase(Enum):
    '''
    The model contemplates an avascular initial growth phase
    and a vascular one, where a blood vessel network develops around
    the tumor
    '''
    AVASCULAR = 0
    VASCULAR = 1

class ECM:

    def __init__(self, width, height, ec, et):
        self.width = width
        self.height = height
        self.extracellularMatrix = np.zeros((self.height, self.width))

        #Tumor ECM degradation constant
        self.ec = ec

        #Threshold of ECM necessary for proliferating cell invasion
        self.et = et

        #Initialize the matrix
        self.initializeMatrix()

    def initializeMatrix(self):
        '''
        Concentration values for ECM are set as a random number ranging from 0.8 to 1.2.
        '''
        for i in range(1,self.height - 1):
            for j in range(1, self.width -1):
                self.extracellularMatrix[i,j] = 0.8 + random.random()*(1.2 - 0.8)

    def degradeMatrix(self, nNeighbors, i, j):
        '''
        Degrades the matrix at location (i,j) based on the number of the number of proliferating cells that surround such position. 
        '''
        deltaECM = -self.ec*nNeighbors*self.extracellularMatrix[i,j]
        self.extracellularMatrix[i,j] = self.extracellularMatrix[i,j] + deltaECM

    def canInvadePosition(self, i, j):
        '''
        Returns a boolean indicating if the ECM concentration at (i,j) is sufficiently low, which means that it could be invaded by a proliferating cell. 
        '''
        if(self.extracellularMatrix[i,j] < self.et):
            return True
        return False

class Nutrient:

    def __init__(self, width, height, diffusionConstant, healthyCellConsumption, consumptionProlif, consumptionQuiescent):

        self.width = width
        self.height = height
        
        #Nutrient concentration matrix
        self.nutrientConcentration = np.zeros((height, width))

        #Nutrient consumption by proliferating cells
        self.consumptionProlif = consumptionProlif

        #Nutrient consumption by quiescent cells
        self.consumptionQuiescent = consumptionQuiescent

        #Nutrient consumption by healthy cells
        self.healthyCellConsumption = healthyCellConsumption

        #Diffusion constant for nutrient
        self.diffusionConstant = diffusionConstant

    
    def putValue(self, i, j, value):
        '''
        Method allows to set a concentration value at position (i,j) of the automaton
        '''
        self.nutrientConcentration[i,j] = value

    def updateNutrient(self, cell, x, y, growthPhase, vesselPositions):
        '''
        The method updates nutrient concentrations by applying diffusion rules in the avascular and vascular phases of tumor growth. 
        '''
        index1 = y
        index2 = x
        
        #Update for an occupied cell in the avascular phase
        if(not (cell is None) and growthPhase == GrowthPhase.AVASCULAR):
            #Update for proliferating cells
            if(cell.cellType == CellType.PROLIFERATING):
                laPlacian = self.nutrientConcentration[(index1 + 1)%self.height, index2] + self.nutrientConcentration[(index1 - 1)%self.height, index2] + self.nutrientConcentration[index1, (index2 + 1)%self.width] + self.nutrientConcentration[index1, (index2 - 1)%self.width] - 4*self.nutrientConcentration[index1, index2]
                deltaConcentration = self.diffusionConstant*laPlacian - self.consumptionProlif
                self.nutrientConcentration[index1, index2] = self.nutrientConcentration[index1, index2] + deltaConcentration

            #Update for quiescent cells:
            elif(cell.quiescent == True):
                laPlacian = self.nutrientConcentration[(index1 + 1)%self.height, index2] + self.nutrientConcentration[(index1 - 1)%self.height, index2] + self.nutrientConcentration[index1, (index2 + 1)%self.width] + self.nutrientConcentration[index1, (index2 - 1)%self.width] - 4*self.nutrientConcentration[index1, index2]
                deltaConcentration = self.diffusionConstant*laPlacian - self.consumptionQuiescent
                self.nutrientConcentration[index1, index2] = self.nutrientConcentration[index1, index2] + deltaConcentration
        
        #Update for non-occupied healthy cells in the avascular phase
        elif(cell is None and growthPhase == GrowthPhase.AVASCULAR):
            laPlacian = self.nutrientConcentration[(index1 + 1)%self.height, index2] + self.nutrientConcentration[(index1 - 1)%self.height, index2] + self.nutrientConcentration[index1, (index2 + 1)%self.width] + self.nutrientConcentration[index1, (index2 - 1)%self.width] - 4*self.nutrientConcentration[index1, index2]
            deltaConcentration = self.diffusionConstant*laPlacian - self.healthyCellConsumption
            self.nutrientConcentration[index1, index2] = self.nutrientConcentration[index1, index2] + deltaConcentration

        #Update for occupied cells in the vascular cells (we leave concentration at blood vessel locations constant)
        elif(not (cell is None) and growthPhase == GrowthPhase.VASCULAR and vesselPositions[index1, index2] == 0):
            if(cell.cellType == CellType.PROLIFERATING):
                laPlacian = self.nutrientConcentration[(index1 + 1)%self.height, index2] + self.nutrientConcentration[(index1 - 1)%self.height, index2] + self.nutrientConcentration[index1, (index2 + 1)%self.width] + self.nutrientConcentration[index1, (index2 - 1)%self.width] - 4*self.nutrientConcentration[index1, index2]
                deltaConcentration = self.diffusionConstant*laPlacian - self.consumptionProlif
                self.nutrientConcentration[index1, index2] = self.nutrientConcentration[index1, index2] + deltaConcentration

            elif(cell.quiescent == True):
                laPlacian = self.nutrientConcentration[(index1 + 1)%self.height, index2] + self.nutrientConcentration[(index1 - 1)%self.height, index2] + self.nutrientConcentration[index1, (index2 + 1)%self.width] + self.nutrientConcentration[index1, (index2 - 1)%self.width] - 4*self.nutrientConcentration[index1, index2]
                deltaConcentration = self.diffusionConstant*laPlacian - self.consumptionQuiescent
                self.nutrientConcentration[index1, index2] = self.nutrientConcentration[index1, index2] + deltaConcentration

        #Update for healthy cells in the Vascular phase
        elif(cell is None and growthPhase == GrowthPhase.VASCULAR and vesselPositions[index1, index2] == 0):
            laPlacian = self.nutrientConcentration[(index1 + 1)%self.height, index2] + self.nutrientConcentration[(index1 - 1)%self.height, index2] + self.nutrientConcentration[index1, (index2 + 1)%self.width] + self.nutrientConcentration[index1, (index2 - 1)%self.width] - 4*self.nutrientConcentration[index1, index2]
            deltaConcentration = self.diffusionConstant*laPlacian - self.healthyCellConsumption
            self.nutrientConcentration[index1, index2] = self.nutrientConcentration[index1, index2] + deltaConcentration

    def getNutrientValue(self, i, j):
        '''
        Return the nutrient concentration value at location (i,j)
        '''
        return self.nutrientConcentration[i,j]

class Cell:
    
    def __init__(self, x, y, cellType, cycleTime, treatmentAffected):
        self.x = x
        self.y = y
        self.cellType = cellType
        #self.oxygenThreshold = 0.1
        self.oxygenThreshold = 0.001
        self.quiescent = False
        #Parámetros de la célula para las terapias
        #Define si la célula será afectada por el tratamiento
        self.therapyAffected = treatmentAffected
        #Lleva el ciclo de vida de la célula, lo cual es importante en tratamientos como la radioterapia
        self.countCycle = cycleTime
    
    def __eq__(self, other):
        self.x == other.x and self.y == other.y
    
    def turnNecrotic(self):
        self.cellType = CellType.NECROTIC
        
    def breathe(self, oxygenConcentration):
        if(oxygenConcentration < self.oxygenThreshold):
            self.turnNecrotic()
    
    def setQuiescent(self, quiescent):
        self.quiescent = quiescent


class Tissue:

    def __init__(self, width, height):
        self.width = width
        self.height = height
        #Matrix to register occupied positions of the automaton
        self.occupiedPositions = np.zeros((self.height, self.width))
        
        #Matrix to register positions occupied by necrotic cells, which are no longer
        #accessible for cancer cells
        self.necroticPositions = np.zeros((self.height, self.width))
        self.quiescentCells = np.zeros((self.height, self.width))
        
        #Cell array
        self.cells = []
        
        #Artist tools
        self.colorNecrotic = [176.0/255, 176.0/255, 176.0/255]
        self.colorVessels = [201.0/255, 74.0/255, 0/255]
        
        #Parameters for ECM
        self.ec = 0.1
        self.et = 0.05
        
        #State transition probabilities
        self.rProlif = 0.8
        self.rBinding = 0.3
        self.rEscape = 0.5
        self.rLysis = 0.35
        self.rDecay = 0.35
        self.K = 1500
        
        #Nutrient parameters
        self.difussionConstant = 0.01
        self.consumptionProlif = 0.01
        self.consumptionQuiescent = 0.005
        self.consumptionHealthy = 0.0001
        
        #Initialize ECM and nutrient
        self.ecm = ECM(self.width, self.height, self.ec, self.et)
        self.nutrient = Nutrient(self.width, self.height, self.difussionConstant, self.consumptionHealthy, self.consumptionProlif, self.consumptionQuiescent)
        self.initializeNutrientAndECM()
        
        #Initial phase for the tumor is avascular
        self.growthPhase = GrowthPhase.AVASCULAR
        
        #Extra parameters used in therapies
        self.cellCycleTime = 4
        self.therapies = []

    def initializeNutrientAndECM(self):
        '''
        Method applied to initialize ECM and nutrient concentrations
        '''
        #Initialize ECM
        self.ecm.initializeMatrix()

        #Nutrient in the interior of the automaton
        for i in range(0,self.height):
            for j in range(0,self.width):
                self.nutrient.putValue(i,j,1)

        #Nutrient in the borders of the automaton
        self.nutrient.nutrientConcentration[0,:] = 2
        self.nutrient.nutrientConcentration[:,0] = 2
        self.nutrient.nutrientConcentration[self.height -1,:] = 2
        self.nutrient.nutrientConcentration[:,self.width - 1] = 2

    def setVascularPhase(self):
        '''
        Configures the vascular phase
        '''
        self.growthPhase = GrowthPhase.VASCULAR
        self.angiogenesisModule = AngiogenesisModule(self.width, self.height)
        proliferatingLocations = self.getProliferatingLocations()
        self.angiogenesisModule.initializeSystem(self.ecm, proliferatingLocations, self.nutrient)
        self.nutrient.nutrientConcentration[0,:] = 0
        self.nutrient.nutrientConcentration[:,0] = 0
        self.nutrient.nutrientConcentration[self.height - 1, :] = 0
        self.nutrient.nutrientConcentration[:,self.width - 1] = 0
        

    def countNeighbors(self, x, y):
        '''
        Counts the number of available squares around position (y,x) of the automaton.
        '''
        sumNeighbors = 0
        for i in range(-1,2):
            for j in range(-1,2):
                if(i != 0 or j != 0):
                    sumNeighbors = sumNeighbors + self.occupiedPositions[(y + j)%self.height, (x+i)%self.height]
        return sumNeighbors
        
        
    def updateNutrientAndECM(self, growthPhase):
        '''
        Updates the concentrations of nutrient and ECM in both the avascular and vascular phases
        '''
        for i in range(0,self.height):
            for j in range(0, self.width):
                nNeighbors = self.countNeighbors(j,i)
                
                #Update ECM Concentration
                self.ecm.degradeMatrix(nNeighbors, i, j)

                #Update Nutrient Concentration for squares that
                #are not occupied by cells
                if(self.occupiedPositions[i,j] == 0):
                    if(growthPhase == GrowthPhase.AVASCULAR):
                        self.nutrient.updateNutrient(None, j,i, growthPhase, None)
                    elif(growthPhase == GrowthPhase.VASCULAR):
                        self.nutrient.updateNutrient(None, j, i, growthPhase, self.angiogenesisModule.occupiedCells)

        #Update nutrient concentration of biological cells
        for i in range(0,len(self.cells)):
            if(growthPhase == GrowthPhase.AVASCULAR):
                self.nutrient.updateNutrient(self.cells[i], self.cells[i].x, self.cells[i].y, growthPhase, None)
            elif(growthPhase == GrowthPhase.VASCULAR):
                self.nutrient.updateNutrient(self.cells[i], self.cells[i].x, self.cells[i].y, growthPhase, self.angiogenesisModule.occupiedCells)

    def updateCells(self, step):
        '''
        Updates the state of cells in the automaton
        '''
        cellsToDelete = []
        #The cells are updated in a random manner
        indsList = list(range(0,len(self.cells)))
        random.shuffle(indsList)

        for i in indsList:
            cell = self.cells[i]
            r = random.random()
            #If the cell was quiescent but now there are availablle positions to infest we awake it
            if(cell.quiescent == True and len(self.getPositionsToInfest(cell.x, cell.y))>0):
                cell.setQuiescent(False)
                self.quiescentCells[cell.y, cell.x] = 0

            #Turn the cell necrotic if the nutrient (interpreted as oxygen) is low

            oxygenConcentration = self.nutrient.getNutrientValue(cell.y, cell.x)
            self.cells[i].breathe(oxygenConcentration)

            #State update for: Proliferating cells
            if(cell.cellType == CellType.PROLIFERATING):
                if(r <= self.rProlifPrime):
                    normalCells = self.getPositionsToInfest(cell.x, cell.y)
                    #Infest an available healthy cell
                    if(len(normalCells) > 0):
                        normalCell = random.choice(normalCells)
                        therapyResistance = self.getTreatmentResistance(step, cell)
                        self.addProliferatingCell(normalCell[1], normalCell[0], therapyResistance, step)
                    else:
                        cell.setQuiescent(True)
                        self.quiescentCells[cell.y, cell.x] = 1
                elif(r <= 1 - self.rBinding):
                    self.cells[i].cellType = CellType.COMPLEX

                self.updateTherapy(step, cell, oxygenConcentration)
            #State update for: Complex cells
            elif(cell.cellType == CellType.COMPLEX):
                #Escaping the immune system possibility
                if(r <= self.rEscape):
                    self.cells[i].cellType = CellType.PROLIFERATING
                #Possibility that the cell dies due to immune action
                elif(r >= 1 - self.rLysis):
                    self.cells[i].cellType = CellType.DEAD
            
            #State update for: Dead cells
            elif(cell.cellType == CellType.DEAD):
                if(r < self.rDecay):
                    cellsToDelete.append(cell)

            #Mark new necrotic positions and delete the corresponding necrotic cells so that they don't waste memory
            if(self.cells[i].cellType == CellType.NECROTIC):
                self.necroticPositions[cell.y, cell.x] = 1
                cellsToDelete.append(cell)

        for i in range(0,len(cellsToDelete)):
            if(cellsToDelete[i] in self.cells):
                self.removeCell(cellsToDelete[i])

    def removeCell(self, cell):
        '''
        Remove a cell in the automaton
        '''
        self.cells.remove(cell)
        self.occupiedPositions[cell.y, cell.x] = 0

    def getProliferatingLocations(self):
        '''
        Returns a 2D Array marked with the positions that are occupied by proliferating 
        '''

        nCells = len(self.cells)
        proliferatingLocations = np.zeros((self.height, self.width))
        for i in range(0,nCells):
            cell = self.cells[i]
            if(cell.cellType == CellType.PROLIFERATING):
                index1 = cell.y
                index2 = cell.x
                proliferatingLocations[index1, index2] = 1
        return proliferatingLocations

    def getPositionsToInfest(self, x, y):
        '''
        Gets positions that can be infested around position (y,x) in the automaton (a 3x3 neighborhood is considered).
        '''
        positions = []
        for i in range(-1,2):
            for j in range(-1,2):
                if(i != 0 or j!= 0):
                    row = (y + i)%self.height
                    col = (x + j)%self.width
                    if(self.occupiedPositions[row, col] == 0 and self.necroticPositions[row,col] == 0 and self.ecm.canInvadePosition(row, col)):
                        positions.append([row, col])
        return positions

    def getTreatmentResistance(self, step, cell):
        '''
        Gets treatment resistance for the cell, which is stochastic. 

        Probabilities for resistance are set as global properties of the class Therapy, that is the reason why we take it from the first therapy on the list of our array therapies
        '''
        if(len(self.therapies) != 0 and self.therapies[0].isStarted(step)):
            return self.therapies[0].getTreatmentAffectionInheritance(cell.therapyAffected)

        return False

    def evolveTissueInitially(self, nSteps, recordMovie, recordCounts, includeNecrotic):
        '''
        Initial evolution for the tumor.

        This is used to separate the avascular phase from the vascular one, where a blood vessel network will develop at the same time that the tumor grows. 
        '''
        if(recordMovie):
            self.evolutionMovie = np.zeros((self.height, self.width, 3, nSteps + 1))
            self.evolutionMovie[:,:,:,0] = self.getPicture(includeNecrotic)

        if(recordCounts):
            self.cellCountSeries = np.zeros((nSteps + 1, 4))
            self.cellCountSeries[0,:] = self.getCellCounts()

        for i in tqdm(range(1, nSteps + 1)):
            counts = self.getCellCounts()
            self.rProlifPrime = self.rProlif*(1 - counts[0]/self.K)
            #Update nutrient and ECM concentrations
            self.updateNutrientAndECM(self.growthPhase)
            self.updateCells(i)
            #Update treatment
            self.updateTherapyGlobally(i, self)
            
            if(self.growthPhase == GrowthPhase.VASCULAR):
                self.angiogenesisModule.evolutionStep()
                self.angiogenesisModule.updateTAFConcentrations(self.getProliferatingLocations())
            
            if(recordCounts):
                self.cellCountSeries[i,:] = self.getCellCounts()

            if(recordMovie):
                self.evolutionMovie[:,:,:,i] = self.getPicture(includeNecrotic)


    def continueTumorEvolution(self, nSteps, recordMovie, recordCounts, recordNetworkMovie, includeNecrotic):
        '''
        Continues the evolution of a previously developed tumor. This is useful to simulate growth during the angiogenesis growth phase.
        '''
        indexOffset = 0
        if(recordMovie):
            oldMovie = self.evolutionMovie.copy()
            indexOffset = np.size(oldMovie,3)
            self.evolutionMovie = np.zeros((self.height, self.width, 3, indexOffset + nSteps))
            self.evolutionMovie[:,:,:,0:indexOffset] = oldMovie

        if(recordCounts):
            oldCounts = self.cellCountSeries.copy()
            indexOffset = np.size(oldCounts,0)
            self.cellCountSeries = np.zeros((indexOffset + nSteps,4))
            self.cellCountSeries[0:indexOffset,:] = oldCounts

        networkOffset = 0
        if(recordNetworkMovie and self.growthPhase == GrowthPhase.VASCULAR):
            if(not hasattr(self, 'networkMovie')):
                self.networkMovie = np.zeros((self.height, self.width, 3, nSteps + 1))
                self.networkMovie[:,:,:,0] = self.angiogenesisModule.getPicture(self.angiogenesisModule.getTipCellLocations(), self.getProliferatingLocations(), self.angiogenesisModule.occupiedCells)
            else:
                oldMovie = self.networkMovie.copy()
                networkOffset = np.size(oldMovie,3)
                self.networkMovie = np.zeros((self.height, self.width, 3, indexOffset + nSteps))
                self.networkMovie[:,:,:,0:networkOffset] = oldMovie
                

        for i in tqdm(range(0,nSteps)):
            stepNum = indexOffset + i
            stepNumNetwork = networkOffset + i
            counts =self.getCellCounts()
            self.rProlifPrime = self.rProlif*(1 - counts[0]/self.K)
            self.updateNutrientAndECM(self.growthPhase)
            self.updateCells(stepNum)
            self.updateTherapyGlobally(stepNum, self)
            if(self.growthPhase == GrowthPhase.VASCULAR):
                self.angiogenesisModule.evolutionStep()
                self.angiogenesisModule.updateTAFConcentrations(self.getProliferatingLocations())
            if(recordMovie):
                self.evolutionMovie[:,:,:,stepNum] = self.getPicture(includeNecrotic)
            if(recordCounts):
                self.cellCountSeries[stepNum,:] = counts
            if(recordNetworkMovie and self.growthPhase == GrowthPhase.VASCULAR):
                self.networkMovie[:,:,:,stepNumNetwork] = self.angiogenesisModule.getPicture(self.angiogenesisModule.getTipCellLocations(), self.getProliferatingLocations(), self.angiogenesisModule.occupiedCells)

    def addTherapy(self, therapy):
        '''
        Adds a therapy for the tumor
        '''
        self.therapies.append(therapy)

    def updateTherapy(self, step, cell, *args):
        '''
        Updates the therapy effect at the local cell level
        '''
        if(len(self.therapies) != 0):
            for i in range(0,len(self.therapies)):
                self.therapies[i].updateTherapy(step, cell, *args)

    def updateTherapyGlobally(self, step, *args):
        '''
        Updates the therapy effect at a global level
        '''
        if(len(self.therapies) != 0):
            for i in range(0,len(self.therapies)):
                self.therapies[i].globalTherapyUpdate(step,*args)
    
    def getCellCounts(self):
        '''
        Returns a 1-dimensional array of length 4, detailing the counts for:
        1. Proliferating cells
        2. Complex cells
        3. Total cells
        4. Necrotic cells
        '''

        proliferatingCells = 0
        complexCells = 0
        necroticCells = 0
        for i in range(0,len(self.cells)):
            cell = self.cells[i]
            if(cell.cellType == CellType.PROLIFERATING):
                proliferatingCells = proliferatingCells + 1
            elif(cell.cellType == CellType.COMPLEX):
                complexCells = complexCells + 1
        necroticCells = sum(sum(self.necroticPositions))
        return np.array([proliferatingCells, complexCells, len(self.cells), necroticCells])

    def getPicture(self, includeNecrotic):
        '''
        Returns a picture of the automaton

        includeNecrotic is used for specifying whether necrotic locations are painted in the portrait. For entropy computation prposes the necrotic core needs to be omitted. 
        '''
        picture = np.zeros((self.height, self.width, 3))

        if(self.growthPhase == GrowthPhase.VASCULAR):
            bloodVesselLocations = self.angiogenesisModule.occupiedCells
            for i in range(0,self.height):
                for j in range(0,self.width):
                    if(bloodVesselLocations[i,j] == 1):
                        picture[i,j,:] = self.colorVessels
        for i in range(0,len(self.cells)):
            picture[self.cells[i].y, self.cells[i].x, :] = self.cells[i].cellType.value
            
        if(includeNecrotic):
            for i in range(0,self.height):
                for j in range(0,self.width):
                    if(self.necroticPositions[i,j] == 1):
                        picture[i,j,:] = self.colorNecrotic
        return picture
        
    def addProliferatingCell(self, x,y,treatmentAffected, step):
        '''
        Method used to add new proliferating cells
        '''
        self.cells.append(Cell(x,y,CellType.PROLIFERATING, step, treatmentAffected))
        self.occupiedPositions[y,x] = 1

    def exportTumorInstance(self, fileName):
        '''
        Exports a pickle instance of the tumor
        '''
        with open(fileName, 'wb') as file:
            pickle.dump(self, file)
            print("Tumor object saved to "+fileName)

    def importTumorInstance(self, fileName):
        '''
        Opens a specified tumor instance
        '''
        tissueInstance = pickle.load(open(fileName, 'rb'))
        return tissueInstance

class TherapyType(Enum):
    '''
    Types of therapy that the model considers

    1. Radiotherapy
    2. VDA
    '''
    RADIOTHERAPY = 0
    VDA = 1
    
class Therapy:

    def __init__(self, therapyType, *args):
        #Each therapy is initialized with a unique set of attributes, which are provided in args
        self.therapyType = therapyType
        self.inheritanceResistanceProbability = 0.6
        self.necrosisTherapyRate = 0.15

        #Radiotherapy constructor
        if(therapyType == TherapyType.RADIOTHERAPY):
            self.startDay = args[0]
            self.g0Gamma = args[1]
            self.alpha = args[2]
            self.beta = args[3]
            self.cycleTime = args[4]
            self.dose = args[5]
            self.thresholdOxygen = args[6]
            self.delayTime = args[7]
            self.initMitoticProb = args[8]
            self.finalMitoticProb = args[9]

        elif(therapyType == TherapyType.VDA):
            self.startDay = args[0]
            self.locationsToTreat = args[1]
            self.vesselKillThreshold = args[2]
            self.potentialConstant = args[3]
            self.sourceAmplification = args[4]
            self.stepDeathProbability = args[5]

            #Create a map for the vessels that will be targetted
            n = np.size(self.locationsToTreat, 0)
            m = np.size(self.locationsToTreat, 1)

            sourcePositions = []

            for i in range(0,n):
                for j in range(0,m):
                    if(self.locationsToTreat[i,j] == 1):
                        sourcePositions.append([i,j])
            vesselKillPotential = np.zeros((n,m))
            epsilon = 0.5
            for i in range(0,n):
                for j in range(0,m):
                    potential = 0
                    for s in sourcePositions:
                        dst = np.sqrt((s[0] - i)**2 + (s[1] - j)**2)
                        if(dst > epsilon):
                            potential = potential + (self.potentialConstant/dst)
                        elif(dst <= epsilon):
                            potential = potential + self.potentialConstant*self.sourceAmplification

                    vesselKillPotential[i,j] = potential

            maxPotential = np.max(np.max(vesselKillPotential))
            vesselKillPotential = vesselKillPotential/maxPotential

            self.vesselKillPotential = vesselKillPotential
            self.deathTargettedVessels = []

    def updateTherapy(self, step, cell, *args):
        '''
        Updates the local therapy effect at every cell of the automaton
        for each kind of therapy. Why has python not implemented explicit switch statements?
        '''
        if(self.therapyType == TherapyType.RADIOTHERAPY):
            cell.countCycle = cell.countCycle + 1
            #Radiotherapy when the beam is applied
            if(step == self.startDay):
                stageCellCycle = (cell.countCycle%self.cycleTime)//(self.cycleTime//4)
                gamma = self.g0Gamma*(1.5)**(stageCellCycle)
                oxygenConcentration = args[0]
                oer = 0
                if(oxygenConcentration > self.thresholdOxygen):
                    oer = 1
                else:
                    oer = 1 - (oxygenConcentration/self.thresholdOxygen)

                dOER = self.dose/oer
                probTarget = 1 - np.exp(-gamma*(self.alpha*dOER + self.beta*dOER**2))
                if(random.random()< probTarget):
                    cell.therapyAffected = True
            #Radiotherapy in the steps that follow beam application
            elif(step > self.startDay and cell.cellType == CellType.PROLIFERATING and cell.therapyAffected):
                #Acute effect
                if(step - self.startDay < self.delayTime):
                    if(random.random() < self.initMitoticProb):
                        if(random.random() < self.necrosisTherapyRate):
                            cell.turnNecrotic()
                        else:
                            cell.cellType = CellType.DEAD
                #Post-acute effect
                else:
                    if(random.random() < self.finalMitoticProb):
                        if(random.random() < self.necrosisTherapyRate):
                            cell.turnNecrotic()
                        else:
                            cell.cellType = CellType.DEAD

    def globalTherapyUpdate(self, step, tissue):
        '''
        Method used to update the effects of therapy at a global tissue level. Mainly relevant for VDA in or case, but other potential therapies could be included here.

        We actually have a version of them, but they were not relevant for our study, so i will not included them here. 
        '''
        if(self.therapyType == TherapyType.VDA):
            if(step == self.startDay and tissue.growthPhase == GrowthPhase.VASCULAR):
                n = np.size(self.vesselKillPotential,0)
                m = np.size(self.vesselKillPotential,1)

                vesselPositions = tissue.angiogenesisModule.occupiedCells

                for i in range(0,n):
                    for j in range(0,m):
                        if(self.vesselKillPotential[i,j] > self.vesselKillThreshold):
                            self.deathTargettedVessels.append([i,j])

            elif(step - self.startDay > 0):
                nVessels = len(self.deathTargettedVessels)
                if(nVessels > 0):
                    removedVessels = []
                    for i in range(0, nVessels):
                        if(random.random() < self.stepDeathProbability):
                            vesselPosition = self.deathTargettedVessels[i]
                            removedVessels.append(vesselPosition)
                            index1 = vesselPosition[0]
                            index2 = vesselPosition[1]

                            tissue.nutrient.putValue(index1, index2, 0)
                            #Here we also need to erase the blood vessel
                            tissue.angiogenesisModule.removeBloodVessel(index1, index2)
                    for i in range(0,len(removedVessels)):
                        self.deathTargettedVessels.remove(removedVessels[i])

    def getTreatmentAffectionInheritance(self, cellAffected):
        '''
        Returns a boolean indicating whether a daughter cell will inherit
        treatment resistance, in case that the mother cell called cellAffected is resistant. 
        '''
        
        if(cellAffected == False and random.random() < self.inheritanceResistanceProbability):
            return False
        else:
            return True

    def isStarted(self, step):
        '''
        Returns a boolean indicating whether the therapy has started.
        '''
        if(self.therapyType == TherapyType.RADIOTHERAPY):
            return step >= self.startDay
        elif(self.therapyType == TherapyType.VDA):
            return step >= self.startDay
    
                            
                


               



    
    

        

    


        


    
            
            
                
                
            
            
                
                
                
        

    