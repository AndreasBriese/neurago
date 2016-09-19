// Package neurago_test is the package which holds tests for
// github.com/lemourA/neurago
package neurago_test

import (
	"fmt"
	"math/rand"
	"reflect"
	"testing"

	"github.com/lemourA/neurago"
)

var hebbTests = []struct {
	netSize             int
	nbOfLearnedPatterns int
}{
	{10, 1},
	{10, 3},
	{100, 10},
	{100, 30},
	//{500, 50},
}

// TestHebbTrain tests that HebbTrainer correctly trains ANNs by checking
// that it correctly recalls learned patterns.
func TestHebbTrain(t *testing.T) {
	var netSize, nbOfLearnedPatterns int
	var patterns [][]float64
	var neurons []neurago.Neuron
	var net neurago.ANN
	trainer := neurago.NewHebbTrainer()

	fmt.Println("--------------------")
	for _, test := range hebbTests {
		netSize = test.netSize
		neurons = make([]neurago.Neuron, netSize)
		for i := 0; i < netSize; i++ {
			neurons[i] = neurago.NewTestNeuron(0, 0)
		}
		net = neurago.NewTestANN(neurons)
		nbOfLearnedPatterns = test.nbOfLearnedPatterns
		patterns = generatePatterns(netSize, nbOfLearnedPatterns)
		trainer.Train(net, patterns)

		for _, neuronA := range neurons {
			for neuronB, weight := range neuronA.Connections() {
				if weight != neuronB.Connections()[neuronA] {
					t.Error("HebbTrainer#Train failed (The weights are not symmetric")
				}
			}
		}

		for i, pat := range patterns {
			net.SetInput(pat)
			fmt.Printf("[TEST]HebbTrainer//Network size: %d. pattern %d on %d -> ", netSize, i+1, nbOfLearnedPatterns)
			if !reflect.DeepEqual(net.Output(), pat) {
				fmt.Println("Recall Failed")
			} else {
				fmt.Println("Successfully Recalled")
			}
		}
		fmt.Printf("\n")
	}
	fmt.Println("--------------------")
}

func generatePattern(size int) []float64 {
	pattern := make([]float64, size)
	probaFlip := rand.Float64()
	for i := 0; i < size; i++ {
		pattern[i] = 1
		if rand.Float64() < probaFlip {
			pattern[i] = -1
		}
	}
	return pattern
}

func generatePatterns(size int, nbOfLearnedPatterns int) [][]float64 {
	patterns := make([][]float64, nbOfLearnedPatterns)
	for i := 0; i < nbOfLearnedPatterns; i++ {
		patterns[i] = generatePattern(size)
	}
	return patterns
}

func mutatePattern(pattern []float64, mutationLevel float64) []float64 {
	if mutationLevel > 1.0 || mutationLevel < 0.0 {
		panic("Mutation level must range between 0 and 1")
	}
	var randomIndex int
	patternLen := len(pattern)
	mutated := make([]float64, patternLen)
	remainingIndexes := make([]int, 0)
	nbOfMutations := int(float64(patternLen) * mutationLevel)

	copy(mutated, pattern)
	if mutationLevel == 0 {
		return mutated
	}

	for i := 0; i < patternLen; i++ {
		remainingIndexes = append(remainingIndexes, i)
	}
	for i := 0; i < nbOfMutations; i++ {
		randomIndex = rand.Intn(len(remainingIndexes))
		if mutated[randomIndex] == 1 {
			mutated[randomIndex] = -1
		} else {
			mutated[randomIndex] = 1
		}
		remainingIndexes = append(
			remainingIndexes[:randomIndex],
			remainingIndexes[randomIndex+1:]...,
		)
	}
	return mutated
}
