// Package neurago is a little library providing tools to
// implement artificial neural networks.
package neurago

import (
	"fmt"
	"log"
	"math/rand"
	"time"
)

// HopfieldNetwork is the implementation of Hopfield networks
type HopfieldNetwork struct {
	perceptrons []*Perceptron
}

// Perceptrons returns the perceptrons of the network "n"
func (n *HopfieldNetwork) Perceptrons() []*Perceptron {
	return n.perceptrons
}

// SetPerceptrons changes the perceptrons of the network "n" for the perceptrons "p"
func (n *HopfieldNetwork) SetPerceptrons(p []*Perceptron) {
	n.perceptrons = p
}

// Train makes the network "n" learn the patterns in "data"
func (n *HopfieldNetwork) Train(data [][]float64) {
	var weights []float64
	var weight float64
	nbOfPerceptrons := len(n.perceptrons)
	nbOfPatterns := float64(len(data))
	fmt.Printf("nb of patterns: %f\n", nbOfPatterns)
	if n.perceptrons == nil {
		log.Panicln("Runtime Error: called method 'Train' on an uninitialized network")
	}
	for _, pattern := range data {
		for i, value := range pattern {
			if i >= nbOfPerceptrons {
				log.Panicln("Runtime Error: Not enough perceptrons to represent a pattern")
			}
			n.perceptrons[i].SetOutput(value)
		}
		for _, perceptron := range n.perceptrons {
			weights = perceptron.Weights()
			for i, connectedPerceptron := range perceptron.Connections() {
				weight = 1 / nbOfPatterns * perceptron.Output() * connectedPerceptron.Output()
				weights[i] += weight
			}
			perceptron.SetWeights(weights)
		}
	}
}

// SetInput sets the given pattern as input of the Hopfield network "n"
func (n *HopfieldNetwork) SetInput(pattern []float64) {
	if len(n.Perceptrons()) != len(pattern) {
		log.Panicln("Runtime Error: Pattern too small or too large to be represented")
	}

	for i, perceptron := range n.Perceptrons() {
		perceptron.SetOutput(pattern[i])
	}
}

// Output returns the value of the perceptrons after the network has been stabilized
func (n *HopfieldNetwork) Output() []float64 {
	output := make([]float64, len(n.Perceptrons()))
	var updatedPerceptron *Perceptron
	var weightedSum float64
	var stabilized bool
	var rIndex int
	nbOfPerceptrons := len(n.Perceptrons())
	rand.Seed(time.Now().UTC().UnixNano())

	for !stabilized {
		rIndex = rand.Intn(nbOfPerceptrons)
		updatedPerceptron = n.Perceptrons()[rIndex]
		weightedSum = 0
		for i, connectedPerceptron := range updatedPerceptron.Connections() {
			weightedSum += connectedPerceptron.Output() * updatedPerceptron.Weights()[i]
		}
		if weightedSum >= 0 {
			updatedPerceptron.SetOutput(1)
		} else {
			updatedPerceptron.SetOutput(-1)
		}
		fmt.Println(n.Perceptrons())
		stabilized = isStable(n)
	}
	for i, perceptron := range n.Perceptrons() {
		output[i] = perceptron.Output()
	}
	return output
}

// isStable returns whether the network "n" is stable or not
func isStable(n *HopfieldNetwork) bool {
	var localField float64
	perceptrons := n.Perceptrons()
	nbOfPerceptrons := len(perceptrons)
	count := 0

	for _, perceptron := range perceptrons {
		localField = computeLocalField(perceptron.Weights(), perceptron.Connections())
		if localField >= 0 && perceptron.Output() == 1 ||
			localField < 0 && perceptron.Output() == -1 {

			count++
		}
	}

	if count == nbOfPerceptrons {
		return true
	}
	return false
}

// computeLocalField returns the local field for a perceptron with
// weighted inputs "connections"
func computeLocalField(weights []float64, connections []*Perceptron) float64 {
	var localField float64

	for i, connectedPerceptron := range connections {
		localField += weights[i] * connectedPerceptron.Output()
	}
	return localField
}

// computeEnergy returns the energy of the hopfield network "n"
func computeEnergy(n *HopfieldNetwork) float64 {
	var netEnergy, percepEnergy, localField float64
	perceptrons := n.Perceptrons()

	for _, perceptron := range perceptrons {
		localField = computeLocalField(perceptron.Weights(), perceptron.Connections())
		percepEnergy = -0.5 * localField * perceptron.Output()
		netEnergy += percepEnergy
	}
	return netEnergy
}

// NewHopfieldNetwork creates and returns a new initialized Hopfield network
func NewHopfieldNetwork(perceps []*Perceptron) *HopfieldNetwork {
	if len(perceps) < 2 {
		log.Panicln("Runtime Error: Hopfield networks should have at least two perceptrons")
	}
	n := new(HopfieldNetwork)
	n.SetPerceptrons(perceps)
	return n
}
