// Package neurago is a little library providing tools to
// implement artificial neural networks
package neurago

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"os"
	"strconv"

	"github.com/gonum/matrix/mat64"
)

var percep_lRate = 0.01

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
// neurons A and B and returns the calculated weight
func perceptronLearning(net ANN, a int, b int, patterns [][]float64) float64 {
	var errA, errB float64
	var sumA, sumB float64
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
	}
	oldWeight := nA.Connections()[nB]
	newWeight := oldWeight + percep_lRate*sumA + percep_lRate*sumB
	return newWeight
}

func networkCopy(neurons []Neuron) []Neuron {
	var connections map[Neuron]float64
	nLen := len(neurons)
	cpy := make([]Neuron, nLen)
	for i, neuron := range neurons {
		cpy[i] = NewMCPNeuron(neuron.Val(), 0)
	}
	for i, nA := range neurons {
		connections = nA.Connections()
		for j, nB := range neurons {
			if i != j {
				cpy[i].SetConnection(cpy[j], connections[nB])
			}
		}
	}
	return cpy
}

// computeError returns the mean squared error of a network
func computeError(neurons []Neuron, patterns [][]float64) float64 {
	var sse float64
	nbOfNeurons := len(neurons)
	nbOfPatterns := len(patterns)

	for a, nA := range neurons {
		for _, pat := range patterns {
			for i, val := range pat {
				neurons[i].SetValue(val)
			}
			nA.Update()
			sse += math.Pow(pat[a]-nA.Value(), 2)
		}
	}
	return 1 / float64(nbOfNeurons*nbOfPatterns) * sse
}

// See Trainer#Train
func (t PerceptronTrainer) Train(net ANN, patterns [][]float64) {
	var newWeight, prevError float64
	var sameErrorCount int
	neurons := net.Neurons()
	nbOfNeurons := len(neurons)
	currError := t.ErrorThreshold() + 1
	weights := mat64.NewDense(nbOfNeurons, nbOfNeurons, make([]float64, nbOfNeurons*nbOfNeurons))

	file, _ := os.OpenFile("percep_result.csv", os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0666)
	defer file.Close()
	csvWriter := csv.NewWriter(file)
	csvWriter.Write([]string{
		"update",
		"error",
	})
	defer csvWriter.Flush()
	update := 1
	var error float64

	for math.Abs(currError) > t.ErrorThreshold() {
		for i, _ := range neurons {
			for j, _ := range neurons {
				if i != j {
					newWeight = perceptronLearning(net, i, j, patterns)
					weights.Set(i, j, newWeight)
					weights.Set(j, i, newWeight)
					//--
					error = computeError(networkCopy(neurons), patterns)
					csvWriter.Write([]string{
						strconv.Itoa(update),
						strconv.FormatFloat(error, 'f', -1, 64),
					})
					update++
				}
			}
		}
		for i, nA := range neurons {
			for j, nB := range neurons {
				if i != j {
					nA.SetConnection(nB, weights.At(i, j))
				}
			}
		}
		prevError = currError
		currError = computeError(neurons, patterns)
		if prevError == currError {
			sameErrorCount++
			if sameErrorCount > 10 {
				fmt.Println("Error is not minimized anymore, last value: ", currError)
				sameErrorCount = 0
			} else {
				sameErrorCount = 0
			}
		}
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
