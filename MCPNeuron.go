// Package neurago is a little library providing tools to
// implement artificial neural networks.
package neurago

// MCPNeuron is the type of McCulloch-Pitts neurons
type MCPNeuron struct {
	value       float64
	connections map[Neuron]float64
}

// see Neuron#Value
func (n MCPNeuron) Value() float64 {
	return n.value
}

// see Neuron#SetValue
func (n *MCPNeuron) SetValue(value float64) {
	n.value = value
}

// see Neuron#Connections
func (n MCPNeuron) Connections() map[Neuron]float64 {
	return n.connections
}

// see Neuron#SetConnections
func (n *MCPNeuron) SetConnections(connections map[Neuron]float64) {
	n.connections = connections
}

// see Neuron#SetConnection
func (n *MCPNeuron) SetConnection(neuron Neuron, weight float64) {
	n.connections[neuron] = weight
}

// see Neuron#Update
func (n *MCPNeuron) Update() {
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

// NewMCPNeuron returns a new, initialised, MCPNeuron
func NewMCPNeuron(value float64) *MCPNeuron {
	neuron := &MCPNeuron{value, make(map[Neuron]float64)}

	return neuron
}
