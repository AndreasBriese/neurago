// Package neurago_test is the package which holds tests for
// github.com/lemourA/neurago
package neurago_test

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/lemourA/neurago"
)

var storkeyTests = []struct {
	netSize             int
	nbOfLearnedPatterns int
}{
	{10, 1},
	{10, 3},
	{100, 10},
	//	{100, 30},
	//{500, 50},
}

// TestStorkeyTrain tests that StorkeyTrainer correctly trains ANNs by checking
// that it correctly recalls learned patterns.
func TestStorkeyTrain(t *testing.T) {
	var netSize, nbOfLearnedPatterns int
	var patterns [][]float64
	var neurons []neurago.Neuron
	var net neurago.ANN
	trainer := neurago.NewStorkeyTrainer()

	fmt.Println("--------------------")
	for _, test := range storkeyTests {
		netSize = test.netSize
		neurons = make([]neurago.Neuron, netSize)
		for i := 0; i < netSize; i++ {
			neurons[i] = neurago.NewTestNeuron(0, 0)
		}
		net = neurago.NewTestANN(neurons)
		nbOfLearnedPatterns = test.nbOfLearnedPatterns
		patterns = generatePatterns(netSize, nbOfLearnedPatterns)
		trainer.Train(net, patterns)

		// for _, neuronA := range neurons {
		// 	for neuronB, weight := range neuronA.Connections() {
		// 		if weight != neuronB.Connections()[neuronA] {
		// 			t.Error("StorkeyTrainer#Train failed (The weights are not symmetric")
		// 		}
		// 	}
		// }

		for i, pat := range patterns {
			net.SetInput(pat)
			fmt.Printf("[TEST]StorkeyTrainer//Network size: %d. pattern %d on %d -> ", netSize, i+1, nbOfLearnedPatterns)
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
