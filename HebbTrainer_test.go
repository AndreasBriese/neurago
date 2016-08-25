// Package neurago_test is the package which holds tests for
// github.com/lemourA/neurago
package neurago_test

import (
	"reflect"
	"testing"

	"github.com/lemourA/neurago"
)

// TestHebbTrain tests that HebbTrainer correctly train ANNs by checking
// weights after training.
func TestHebbTrain(t *testing.T) {
	trainer := neurago.NewHebbTrainer()
	trainingPattern := []float64{1, 1, -1}
	net := neurago.NewTestANN([]neurago.Neuron{
		neurago.NewTestNeuron(1, 0),
		neurago.NewTestNeuron(1, 0),
		neurago.NewTestNeuron(1, 0),
	})
	trainer.Train(net, [][]float64{
		trainingPattern,
	})
	net.SetInput([]float64{1, 1, 1})
	if !reflect.DeepEqual(net.Output(), trainingPattern) {
		t.Error("HebbTrainer#Train failed")
	}
}
