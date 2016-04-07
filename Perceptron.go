// Package neurago is a little library providing tools to
// implement artificial neural networks.
package neurago

// Perceptron is the type used to represent perceptrons
type Perceptron struct {
	outputValue  float64
	inputWeights []float64
	connections  []Perceptron
}

// Output returns the output value of the perceptron "p"
func (p *Perceptron) Output() float64 {
	return p.outputValue
}

// SetOutput updates the output value of the perceptron "p" with the value "v"
func (p *Perceptron) SetOutput(v float64) {
	p.outputValue = v
}

// Weights returns the input connections weights of the perceptron "p"
func (p *Perceptron) Weights() []float64 {
	return p.inputWeights
}

// SetWeights updates the input connections weights of the perceptron "p" with the weights "w"
func (p *Perceptron) SetWeights(w []float64) {
	p.inputWeights = w
}

// Connections returns the other perceptrons connected to the perceptron "p"
func (p *Perceptron) Connections() []Perceptron {
	return p.connections
}

// SetConnections connects the perceptrons "perceps" to the the perceptron "p"
func (p *Perceptron) SetConnections(perceps []Perceptron) {
	p.connections = perceps
}

// NewPerceptron returns a pointer on a new initialized Perceptron
func NewPerceptron(v, w, c) *Perceptron {
	p := new(Perceptron)
	p.SetOutput(v)
	p.SetWeights(w)
	p.SetConnections(c)
	return p
}
