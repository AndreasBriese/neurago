// Package neurago is a little library providing tools to
// implement artificial neural networks
package neurago

import (
	"log"
)

type HebbTrainer struct{}

// Train is a method which trains perceptrons with a given set of patterns
func (t *HebbTrainer) Train(perceptrons []*Perceptron, patterns [][]float64) {
	var weights []float64
	var weight float64

	nbOfPerceptrons := len(perceptrons)
	nbOfPatterns := float64(len(patterns))
	if perceptrons == nil {
		log.Panicln("Runtime Error: called method 'Train' on an uninitialized network")
	}
	for _, pattern := range patterns {
		for i, value := range pattern {
			if i >= nbOfPerceptrons {
				log.Panicln("Runtime Error: Not enough perceptrons to represent a pattern")
			}
			perceptrons[i].SetOutput(value)
		}
		for _, perceptron := range perceptrons {
			weights = perceptron.Weights()
			for i, connectedPerceptron := range perceptron.Connections() {
				weight = 1 / nbOfPatterns * perceptron.Output() * connectedPerceptron.Output()
				weights[i] += weight
			}
			perceptron.SetWeights(weights)
		}
	}
}

// NewHebbTrainer creates and returns a new initialized HebbTrainer
func NewHebbTrainer() *HebbTrainer {
	newTrainer := new(HebbTrainer)
	return newTrainer
}
