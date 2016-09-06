// Package neurago is a little library providing tools to
// implement artificial neural networks
package neurago

import (
	"log"
	"math"
)

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
	var errorSum float64
	neurons := net.Neurons()
	nbOfNeurons := len(neurons)
	nA, nB := neurons[a], neurons[b]
	conn := nA.Connections()
	weight := conn[nB]
	learningRate := 0.01

	for _, pat := range patterns {
		for i, val := range pat {
			if i >= nbOfNeurons {
				log.Panicln("Runtime Error: Not enough neurons to represent the pattern")
			}
			neurons[i].SetValue(val)
		}
		nA.Update()
		errorSum += (pat[a] - nA.Value()) * pat[b]
	}
	nA.SetConnection(nB, weight+learningRate*errorSum)
	return errorSum
}

// See Trainer#Train
// Here PerceptronTrainer considers the first neuron of the network as the bias
func (t PerceptronTrainer) Train(net ANN, patterns [][]float64) {
	var errorSum float64
	neurons := net.Neurons()
	nbOfPatterns := len(patterns)
	currError := t.ErrorThreshold() + 1

	for math.Abs(currError) > t.ErrorThreshold() {
		for i, _ := range neurons {
			for j, _ := range neurons {
				if i != j {
					errorSum = perceptronLearning(net, i, j, patterns)
					currError = 1 / float64(nbOfPatterns) * errorSum
				}
			}
		}
	}
}

// for currError = errorThreshold + 1; currError > errorThreshold; {
// currError = 1 / float64(nbOfPatterns) * errorSum

// NewPerceptronTrainer returns a newly intantiated PerceptronTrainer
// errThreshold is the threshold the algorithm continues to run until
// it gets the ANN output below.
func NewPerceptronTrainer(errThreshold float64) *PerceptronTrainer {
	trainer := new(PerceptronTrainer)

	trainer.SetErrorThreshold(errThreshold)
	return trainer
}
