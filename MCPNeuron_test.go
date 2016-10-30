// Package neurago_test is the package which holds tests for
// github.com/lemourA/neurago
package neurago_test

import (
	"fmt"
	"testing"

	"github.com/lemourA/neurago"
)

// InputNeurons represents the input neurons for each test case.
type InputNeurons [3]struct {
	neuron *neurago.MCPNeuron
	weight float64
}

// TestUpdate test cases.
var refreshTests = []struct {
	connections InputNeurons
	output      float64
}{
	// Test 1
	{
		InputNeurons{ // Test 1: input neurons
			{neurago.NewMCPNeuron(1, 0), -1.0},
			{neurago.NewMCPNeuron(1, 0), -1.0},
			{neurago.NewMCPNeuron(1, 0), -1.0},
		},
		-1, // Test 1: expected output
	},
	// Test 2
	{
		InputNeurons{ // Test 2: input neurons
			{neurago.NewMCPNeuron(-1, 0), -1},
			{neurago.NewMCPNeuron(-1, 0), -1},
			{neurago.NewMCPNeuron(1, 0), -1},
		},
		1, // Test 2: expected output
	},
	// Test 3
	{
		InputNeurons{ // Test 3: input neurons
			{neurago.NewMCPNeuron(1, 0), -1},
			{neurago.NewMCPNeuron(1, 0), 1},
			{neurago.NewMCPNeuron(1, 0), 0},
		},
		-1, // Test 3: expected output
	},
}

// TestUpdate tests that MCP neurons are correctly refreshing (and calculating)
// their value from their inputs.
func TestUpdate(t *testing.T) {
	neuron := neurago.NewMCPNeuron(0, 0)

	for n, testCase := range refreshTests {
		neuron.SetConnections(make(map[neurago.Neuron]float64)) // resets neuron

		// Adds connections
		for _, newConnection := range testCase.connections {
			neuron.SetConnection(newConnection.neuron, newConnection.weight)
		}

		neuron.Update() // calculates neuron new output value

		// checks that the computed neuron output is correct
		if testCase.output != neuron.Value() {
			t.Error(fmt.Sprintf("MCPNeuron#Update failed at test case #%d", n+1))
		}
	}
}
