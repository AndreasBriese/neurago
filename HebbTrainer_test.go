// Package neurago_test is the package which holds tests for
// github.com/lemourA/neurago
package neurago_test

import (
	"reflect"
	"testing"

	"github.com/lemourA/neurago"
)

// TestHebbTrain tests that HebbTrainer correctly trains ANNs by checking
// that it correctly recalls learned patterns.
func TestHebbTrain(t *testing.T) {
	trainer := neurago.NewHebbTrainer()
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
