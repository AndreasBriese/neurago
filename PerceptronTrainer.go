// Package neurago is a little library providing tools to
// implement artificial neural networks
package neurago

import (
	"math/rand"
	"time"
)

type PerceptronTrainer struct{}

// Train is a method which trains perceptrons with a given set of patterns
func (t *PerceptronTrainer) Train(net IArtificialNeuralNet, patterns [][]float64) {
	var inputPattern, output []float64
	var outputDelta, newWeight, currWeight float64
	trainingPattern := patterns[0]
	//inputPatternLen := len(patterns[0])
	perceptrons := net.Perceptrons()
	errorAvg := 0.5
	errorValues := make([]float64, 0)
	lrate := 0.1

	rand.Seed(time.Now().UTC().UnixNano())
	initWeights(perceptrons)
	inputPattern = trainingPattern
	for errorAvg > 0.01 {
		net.SetInput(inputPattern)
		output = net.Output()
		for i, percepA := range perceptrons {
			outputDelta = trainingPattern[i] - output[i]
			errorValues = append(errorValues, outputDelta)
			if outputDelta != 0 {
				for j, percepB := range percepA.Connections() {
					currWeight = percepA.Weights()[j]
					newWeight = currWeight + lrate*outputDelta*percepB.Output()
					SetWeight(percepA, percepB, newWeight)
				}
			}
		}
		errorAvg = 0
		for _, value := range errorValues {
			errorAvg += value
		}
		errorAvg = errorAvg / float64(len(errorValues))
	}
}

func generateRandomPattern(size int) []float64 {
	pattern := make([]float64, size)
	flipProba := rand.Float64()

	for i := 0; i < size; i++ {
		pattern[i] = 1.0
		if rand.Float64() <= flipProba {
			pattern[i] *= -1
		}
	}
	return pattern
}

func initWeights(perceptrons []*Perceptron) {
	var randomWeight float64
	var pair [2]*Perceptron
	alreadyInitialised := make([][2]*Perceptron, 0)

	for _, perceptron := range perceptrons {
		for _, connectedPerceptron := range perceptron.Connections() {
			pair = [2]*Perceptron{perceptron, connectedPerceptron}
			if !isAlreadyInitialised(alreadyInitialised, pair) {
				randomWeight = rand.Float64() / 5
				if rand.Intn(10)%2 == 0 {
					randomWeight *= -1
				}
				SetWeight(perceptron, connectedPerceptron, randomWeight)
				alreadyInitialised = append(alreadyInitialised, pair)
			}
		}
	}
}

func isAlreadyInitialised(conns [][2]*Perceptron, elt [2]*Perceptron) bool {
	for _, connection := range conns {
		if elt[0] == connection[0] {
			if elt[1] == connection[1] {
				return true
			}
		}
		if elt[1] == connection[0] {
			if elt[0] == connection[1] {
				return true
			}
		}
	}
	return false
}

// NewPerceptronTrainer creates and returns a new initialized PerceptronTrainer
func NewPerceptronTrainer() *PerceptronTrainer {
	newTrainer := new(PerceptronTrainer)
	return newTrainer
}
