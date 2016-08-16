// Package neurago is a little library providing tools to
// implement artificial neural networks.
package neurago

import (
	"reflect"
	"testing"
)

// TestValue tests the method "Value" of McCullochPitts Neurons by
// checking that it correctly returns the current neuron value.
func TestValue(t *testing.T) {
	neuron := NewMCPNeuron(0)
	exampleValue := 4.2

	neuron.value = exampleValue
	if neuron.Value() != exampleValue {
		t.Error("MCPNeuron#Value failed")
	}
}

// TestSetValue tests the method "SetValue" of McCullochPitts Neurons by
// checking that it correctly sets the current neuron value.
func TestSetValue(t *testing.T) {
	neuron := NewMCPNeuron(0)
	exampleValue := 4.2

	neuron.SetValue(exampleValue)
	if neuron.value != exampleValue {
		t.Error("MCPNeuron#SetValue failed")
	}
}

// TestConnections tests the method "Connections" of McCullochPitts Neurons by
// checking that it correctly returns the current neuron connections.
func TestConnections(t *testing.T) {
	neuronA := NewMCPNeuron(0)
	neuronB := NewMCPNeuron(0)
	neuronC := NewMCPNeuron(0)
	connections := make(map[Neuron]float64)

	connections[neuronB] = -0.2
	connections[neuronC] = 1.0
	neuronA.connections = connections
	if !reflect.DeepEqual(connections, neuronA.Connections()) {
		t.Error("MCPNeuron#Connections failed")
	}
}

// TestSetConnections tests the method "SetConnections" of McCullochPitts Neurons by
// checking that it correctly sets the current connections.
func TestSetConnections(t *testing.T) {
	neuronA := NewMCPNeuron(0)
	neuronB := NewMCPNeuron(0)
	neuronC := NewMCPNeuron(0)
	connections := make(map[Neuron]float64)

	connections[neuronB] = -0.2
	connections[neuronC] = 1.0
	neuronA.SetConnections(connections)
	if !reflect.DeepEqual(connections, neuronA.connections) {
		t.Error("MCPNeuron#SetConnections failed")
	}
}

// TestSetConnection tests the method "SetConnection" of McCullochPitts Neurons by
// checking that it correctly adds the given neuron to the connections and that it assigns
// the given weight.
func TestSetConnection(t *testing.T) {
	neuronA := NewMCPNeuron(0)
	neuronB := NewMCPNeuron(0)
	exampleWeight := -1.0

	neuronA.SetConnection(neuronB, exampleWeight)
	if weight, keyExists := neuronA.connections[neuronB]; keyExists {
		if weight != exampleWeight {
			t.Error("MCPNeuron#SetConnection failed: connection doesn't hold the given weight")
		}
	} else {
		t.Error("MCPNeuron#SetConnection failed: connected neuron not found within connections")
	}
}
