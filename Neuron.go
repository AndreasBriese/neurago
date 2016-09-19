// Package neurago is a little library providing tools to
// implement artificial neural networks.
package neurago

// Neuron is the interface each neuron types must implement.
type Neuron interface {
	// Value returns the current value (output) of the neuron.
	Value() float64
	Val() float64
	// SetValue sets the given value as the new current neuron value.
	SetValue(value float64)

	// OutputFunction returns the current output function of the neuron.
	OutputFunction() OutputFunction

	// SetOutputFunction sets the given OutputFunction as the neuron current output function.
	SetOutputFunction(outFn OutputFunction)

	// InputFunction returns the current input function of the neuron.
	InputFunction() InputFunction

	// SetInputFunction sets the given InputFunction as the neuron current input function.
	SetInputFunction(inFn InputFunction)

	// Connections returns all the input connections of the calling neuron.
	Connections() map[Neuron]float64

	// SetConnections sets the given parameter as the new set of connections.
	SetConnections(connections map[Neuron]float64)

	// SetConnection adds a new connection or modify an existing one..
	SetConnection(neuron Neuron, weight float64)

	// Update computes the calling neuron output value and store it, it can then
	// be accessed with the method "Value". If Update results in a change of
	// the neuron value it returns true and false otherwise
	Update() bool
}
