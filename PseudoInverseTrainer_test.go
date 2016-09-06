// Package neurago_test is the package which holds tests for
// github.com/lemourA/neurago
package neurago_test

import (
	"reflect"
	"testing"

	"github.com/lemourA/neurago"
)

// TestPseudoInverseTrain tests that PseudoInverseTrainer correctly trains ANNs by checking
// that it correctly recalls learned patterns.
func TestPseudoInverseTrain(t *testing.T) {
	trainer := neurago.NewPseudoInverseTrainer()
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
	neurons := net.Neurons()
	for _, neuronA := range neurons {
		for neuronB, weight := range neuronA.Connections() {
			if weight != neuronB.Connections()[neuronA] {
				t.Error("PseudoInverseTrainer#Train failed (The weights are not symmetric")
			}
		}
	}
	for _, pat := range trainingPatterns {
		net.SetInput(pat)
		if !reflect.DeepEqual(net.Output(), pat) {
			t.Error("PseudoInverseTrainer#Train failed")
		}
	}
}
