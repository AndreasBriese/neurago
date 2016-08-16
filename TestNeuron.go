// Package neurago is a little library providing tools to
// implement artificial neural networks.
package neurago

// TestNeuron is a McCulloc-Pitts neuron used for testing purposes.
type TestNeuron struct {
	value       float64
	connections map[Neuron]float64
}

// see Neuron#Value
func (n TestNeuron) Value() float64 {
	return n.value
}

// see Neuron#SetValue
func (n *TestNeuron) SetValue(value float64) {
	n.value = value
}

// see Neuron#Connections
func (n TestNeuron) Connections() map[Neuron]float64 {
	return n.connections
}

// see Neuron#SetConnections
func (n *TestNeuron) SetConnections(connections map[Neuron]float64) {
	n.connections = connections
}

// see Neuron#SetConnection
func (n *TestNeuron) SetConnection(neuron Neuron, weight float64) {
	n.connections[neuron] = weight
}

// see Neuron#Update
func (n *TestNeuron) Update() {
	weightedSum := 0.0

	for neuron, weight := range n.Connections() {
		weightedSum += (neuron.Value() * weight)
	}
	if weightedSum >= 0 {
		n.SetValue(1.0)
	} else {
		n.SetValue(-1.0)
	}
}

// NewTestNeuron returns a new, initialised, TestNeuron
func NewTestNeuron(value float64) *TestNeuron {
	neuron := &TestNeuron{value, make(map[Neuron]float64)}
	return neuron
}
