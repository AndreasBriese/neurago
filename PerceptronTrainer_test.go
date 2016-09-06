// Package neurago_test is the package which holds tests for
// github.com/lemourA/neurago
package neurago_test

import (
	"reflect"
	"testing"

	"github.com/lemourA/neurago"
)

// TestPerceptronTrain tests that PerceptronTrainer correctly train ANNs by checking
// weights after training.
func TestPerceptronTrain(t *testing.T) {
	trainer := neurago.NewPerceptronTrainer(0)
	trainingPatterns := [][]float64{
		[]float64{1, 1, -1},
		[]float64{1, -1, 1},
		[]float64{-1, 1, 1},
	}
	net := neurago.NewTestANN([]neurago.Neuron{
		neurago.NewTestNeuron(1, 0),
		neurago.NewTestNeuron(1, 0),
		neurago.NewTestNeuron(1, 0),
	})
	trainer.Train(net, trainingPatterns)
	for _, pat := range trainingPatterns {
		net.SetInput(pat)
		if !reflect.DeepEqual(net.Output(), pat) {
			t.Error("HebbTrainer#Train failed")
		}
	}
}
