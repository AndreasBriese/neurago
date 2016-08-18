// Package neurago_test is the package which holds tests for
// github.com/lemourA/neurago
package neurago_test

import (
	"testing"

	"github.com/lemourA/neurago"
)

// IntegrateTest tests the Integrate method by checking that it correctly returns the
// weighted sum of the given inputs.
func IntegrateTest(t *testing.T) {
	fct := neurago.NewWeightedSumFunction()
	connections := map[neurago.Neuron]float64{
		neurago.NewMCPNeuron(-1, 0): 1,
		neurago.NewMCPNeuron(-1, 0): -1,
		neurago.NewMCPNeuron(-1, 0): -1,
	}

	if fct.Integrate(connections) != 1 {
		t.Error("WeightedSumFunction#Integrate failed")
	}
}
