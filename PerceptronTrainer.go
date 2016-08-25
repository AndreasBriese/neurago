// Package neurago is a little library providing tools to
// implement artificial neural networks.
package neurago

import (
	"fmt"
	"math"
)

// PerceptronTrainer trains network using the hebb learning rule
type PerceptronTrainer struct {
	errorThreshold float64
}

// ErrorThreshold returns the errorThreshold.
func (t PerceptronTrainer) ErrorThreshold() float64 {
	return t.errorThreshold
}

// SetErrorThreshold sets the errorThreshold.
func (t *PerceptronTrainer) SetErrorThreshold(threshold float64) {
	t.errorThreshold = threshold
}

func printWeightMatrix(neurons []Neuron) {
	for _, neuron := range neurons {
		for _, weight := range neuron.Connections() {
			fmt.Printf("%f ", weight)
		}
		fmt.Println()
	}
}

// See Trainer#Train
// Here PerceptronTrainer considers the first neuron of the network as the bias
func (t PerceptronTrainer) Train(net ANN, patterns [][]float64) {
	var newWeight float64
	var inputs map[Neuron]float64
	var errorSum float64
	neurons := net.Neurons()
	errorThreshold := t.ErrorThreshold()
	currError := errorThreshold + 1
	nbOfPatterns := len(patterns)

	for i, neuron := range neurons {
		inputs = neuron.Connections()
		fmt.Println("Neuron ", i)

		for currError = errorThreshold + 1; currError > errorThreshold; {
			errorSum = 0

			for _, pattern := range patterns {
				fmt.Println("-------------")
				fmt.Println("pattern = ", pattern)
				net.SetInput(pattern)
				neuron.Update()

				fmt.Println("neuron value = ", neuron.Value(), "// expected = ", pattern[i])
				if neuron.Value() != pattern[i] {
					for inputUnit, weight := range inputs {
						fmt.Println("******")
						fmt.Println("******")
						fmt.Println("input value: ", inputUnit.Value(), "input weight: ", weight)
						newWeight = weight +
							(pattern[i]-neuron.Value())*inputUnit.Value()
						fmt.Println("new weight: ", newWeight)
						neuron.SetConnection(inputUnit, newWeight)
						fmt.Println("******")
						fmt.Println("******")
					}
					errorSum += math.Abs(pattern[i] - neuron.Value())
					// printWeightMatrix(net.Neurons())
					// fmt.Println("pattern[", i, "] = ", pattern[i],
					// 	"// neuron = ", neuron.Value())
					// fmt.Println("abs(delta) = ", math.Abs(pattern[i]-neuron.Value()))
					// fmt.Println("-------------")
				}

			} // for _, pattern := range patterns
			currError = 1 / float64(nbOfPatterns) * errorSum
		} // for currError > errorThreshold
	} // for i, neuron := range neurons
}

// NewPerceptronTrainer returns a newly intantiated PerceptronTrainer
// errThreshold is the threshold the algorithm continues to run until
// it gets the ANN output below.
func NewPerceptronTrainer(errThreshold float64) *PerceptronTrainer {
	trainer := new(PerceptronTrainer)

	trainer.SetErrorThreshold(errThreshold)
	return trainer
}
