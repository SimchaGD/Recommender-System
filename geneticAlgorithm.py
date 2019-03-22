import numpy as np
import random

class Genome():
    def __init__(self, weights, biases):
        self.fitness = 0
        # Genes
        self.weights = weights
        self.biases = biases
        
    def mutate(self):
        mutationRate = 1.0
        for i in range(0, len(self.weights)):
            if random.random() < mutationRate:
                self.weights[i] += random.uniform(-1.0, 1.0) * 0.20
        
        for i in range(0, len(self.biases)):
            if random.random() < mutationRate:
                self.biases[i] += random.uniform(-1.0, 1.0) * 0.20        
	
    def __lt__(self, otherGenome):
        return self.fitness < otherGenome.fitness

class GeneticAlgorithm():
    def __init__(self, populationSize, numberOfUsers, numberOfMovies):
        self.populationSize = populationSize
        self.population = [None] * populationSize
        self.numberOfUsers = numberOfUsers
        self.numberOfMovies = numberOfMovies

        for i in range(populationSize):
            initialWeights = np.random.uniform(-1.0, 1.0, (numberOfUsers, numberOfMovies)).tolist()
            self.population[i] = Genome(initialWeights, initialBiases)
    
    def getGenomeByTournament(self):
        tournamentSize = 4
        combatants = random.sample(range(0, len(self.population)), tournamentSize)
        combatants.sort()
        fittestGenome = self.population[combatants[0]]      
		
        return fittestGenome
	
    def crossover(self, parent1, parent2):
        crossoverRate = 1.0
        parents = [parent1, parent2]
        if random.random() > 0.5:
            parents = [parent2, parent1]

        if random.random() < crossoverRate:
            # random.randint(a, b) returns a random integer N such that a <= N <= b
            randomWeightIndex = random.randint(0, self.numberOfWeights)
            randomBiasIndex = random.randint(0, self.numberOfBiases)
            child = Genome([None] * self.numberOfWeights, [None] * self.numberOfBiases)
			
            for i in range(randomWeightIndex):
                child.weights[i] = parents[0].weights[i]
            for i in range(randomWeightIndex, self.numberOfWeights):
                child.weights[i] = parents[1].weights[i]
            
            for i in range(randomBiasIndex):
                child.biases[i] = parents[0].biases[i]
            for i in range(randomBiasIndex, self.numberOfBiases):
                child.biases[i] = parents[1].biases[i]
            
            return child
        else:
           return parents[0]
	
    def update(self, agents):
        for i, agent in enumerate(agents):
            self.population[i].fitness = agent.fitness
    
    def upgrade(self):
        self.population.sort(reverse = True)
        
        newPopulation = [None] * self.populationSize
        newPopulation[0] = self.population[0]
        newPopulation[1] = self.population[1]
        newPopulation[2] = self.population[2]
        newPopulation[3] = self.crossover(self.population[0], self.population[1])
        newPopulation[4] = self.crossover(self.population[0], self.population[2])
        newPopulation[5] = self.crossover(self.population[1], self.population[2]) 
        
        for i in range(6, self.populationSize):
            parent1 = self.getGenomeByTournament()
            parent2 = self.getGenomeByTournament()
            newPopulation[i] = self.crossover(parent1, parent2)					
		
        for i in range(1, self.populationSize):
            newPopulation[i].mutate() 
                
        # Store all new genomes
        self.population = newPopulation
        
        # Reset fitness
        for genome in self.population:
            genome.fitness = 0