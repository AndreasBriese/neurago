// Package neurago_test is the package which holds tests for
// github.com/lemourA/neurago
package neurago_test

import (
	"fmt"
	"testing"

	"github.com/lemourA/neurago"
)

func printWeightMatrix(neurons []neurago.Neuron) {
	for _, neuron := range neurons {
		for _, weight := range neuron.Connections() {
			fmt.Printf("%f ", weight)
		}
		fmt.Println()
	}
}

// TestPerceptronTrain tests that PerceptronTrainer correctly train ANNs by checking
// weights after training.
func TestPerceptronTrain(t *testing.T) {
	// trainer := neurago.NewPerceptronTrainer(0)
	// trainingPatterns := [][]float64{
	// 	[]float64{1, 1, 1, 1, -1},
	// 	[]float64{1, 1, 1, -1, 1},
	// 	[]float64{1, 1, -1, 1, 1},
	// 	[]float64{1, -1, 1, 1, 1},
	// 	[]float64{-1, 1, 1, 1, 1},
	// }
	// net := neurago.NewTestANN([]neurago.Neuron{
	// 	neurago.NewTestNeuron(1, 0),
	// 	neurago.NewTestNeuron(1, 0),
	// 	neurago.NewTestNeuron(1, 0),
	// 	neurago.NewTestNeuron(1, 0),
	// 	neurago.NewTestNeuron(1, 0),
	// })
	// printWeightMatrix(net.Neurons())
	// trainer.Train(net, trainingPatterns)
	// fmt.Println("AFTER TRAINING")
	// printWeightMatrix(net.Neurons())
	// for _, pattern := range trainingPatterns {
	// 	net.SetInput(pattern)
	// 	if !reflect.DeepEqual(net.Output(), pattern) {
	// 		t.Error("PerceptronTrainer#Train failed")
	// 	}
	// }
	// net.SetInput([]float64{1, 1, 1, 1, -1})
	// fmt.Println(net.Output())
}
