// Package neurago is a little library providing tools to
// implement artificial neural networks
package neurago

import (
	"log"
	"math"
)

var percep_lRate = 1.0

// PerceptronTrainer trains network using the hebb learning rule
type PerceptronTrainer struct {
	errorThreshold float64
}

// ErrorThreshold returns the errorThreshold
func (t PerceptronTrainer) ErrorThreshold() float64 {
	return t.errorThreshold
}

// SetErrorThreshold sets the errorThreshold
func (t *PerceptronTrainer) SetErrorThreshold(threshold float64) {
	t.errorThreshold = threshold
}

// perceptronLearning applies the perceptron learning rule between
// neurons A and B
func perceptronLearning(net ANN, a int, b int, patterns [][]float64) float64 {
	var errA, errB float64
	var sumA, sumB float64
	var errorSum float64
	neurons := net.Neurons()
	nbOfNeurons := len(neurons)
	nA, nB := neurons[a], neurons[b]

	for _, pat := range patterns {
		for i, val := range pat {
			if i >= nbOfNeurons {
				log.Panicln("Runtime Error: Not enough neurons to represent the pattern")
			}
			neurons[i].SetValue(val)
		}
		nA.Update()
		nB.Update()
		errA = pat[a] - nA.Value()
		errB = pat[b] - nB.Value()
		sumA += errA * pat[b]
		sumB += errB * pat[a]
		errorSum += errA
	}
	oldWeight := nA.Connections()[nB]
	newWeight := oldWeight + percep_lRate*sumA + percep_lRate*sumB
	nA.SetConnection(nB, newWeight)
	nB.SetConnection(nA, newWeight)
	return errorSum
}

// See Trainer#Train
// Here PerceptronTrainer considers the first neuron of the network as the bias
func (t PerceptronTrainer) Train(net ANN, patterns [][]float64) {
	var errorSum float64
	neurons := net.Neurons()
	nbOfNeurons := len(neurons)
	currError := t.ErrorThreshold() + 1

	for math.Abs(currError) > t.ErrorThreshold() {
		errorSum = 0
		for i, _ := range neurons {
			for j, _ := range neurons {
				if i != j {
					errorSum += perceptronLearning(net, i, j, patterns)
				}
			}
		}
		currError = 1 / float64(nbOfNeurons*(nbOfNeurons-1)) * errorSum
	}
}

// NewPerceptronTrainer returns a newly intantiated PerceptronTrainer
// errThreshold is the threshold the algorithm continues to run until
// it gets the ANN output below.
func NewPerceptronTrainer(errThreshold float64) *PerceptronTrainer {
	trainer := new(PerceptronTrainer)

	trainer.SetErrorThreshold(errThreshold)
	return trainer
}
