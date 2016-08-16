// Package neurago is a little library providing tools to
// implement artificial neural networks.
package neurago

// Neuron is the interface each neuron types must implement.
type Neuron interface {

	// Value returns the current value (output) of the neuron.
	Value() float64
	// SetValue sets the given value as the new current neuron value.
	SetValue(float64)
	// Connections returns all the input connections of the calling neuron.
	Connections() map[Neuron]float64
	// SetConnections sets the given parameter as the new set of connections.
	SetConnections(map[Neuron]float64)
	// SetConnection adds a new connection or modify an existing one..
	SetConnection(Neuron, float64)
	// Update computes the calling neuron output value and store it, it can then
	// be accessed with the method "Value".
	Update()
}
