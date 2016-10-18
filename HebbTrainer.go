// Package neurago is a little library providing tools to
// implement artificial neural networks.
package neurago

import "log"

// HebbTrainer trains network using the hebb learning rule
type HebbTrainer struct{}

// See Trainer#Train
func (t HebbTrainer) Train(net ANN, patterns [][]float64) {
	var connections map[Neuron]float64
	neurons := net.Neurons()
	nbOfNeurons := len(neurons)
	nbOfPatterns := len(patterns)

	// file, _ := os.OpenFile("hebb_result.csv", os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0666)
	// defer file.Close()
	// csvWriter := csv.NewWriter(file)
	// csvWriter.Write([]string{
	// 	"update",
	// 	"error",
	// })
	// defer csvWriter.Flush()
	// update := 1

	if neurons == nil {
		log.Panicln("Runtime Error: called method 'Train' on an uninitialized network")
	}
	for _, pattern := range patterns {
		for i, value := range pattern {
			if i >= nbOfNeurons {
				log.Panicln("Runtime Error: Not enough neurons to represent the pattern")
			}
			neurons[i].SetValue(value)
		}
		for _, neuron := range neurons {
			connections = neuron.Connections()
			for inputNeuron, weight := range connections {
				weight = 1 / float64(nbOfPatterns) * neuron.Value() * inputNeuron.Value()
				neuron.SetConnection(inputNeuron, connections[inputNeuron]+weight)
				//error := computeError(neurons, patterns)
				for i, value := range pattern {
					if i >= nbOfNeurons {
						log.Panicln("Runtime Error: Not enough neurons to represent the pattern")
					}
					neurons[i].SetValue(value)
				}
				// csvWriter.Write([]string{
				// 	strconv.Itoa(update),
				// 	strconv.FormatFloat(error, 'f', -1, 64),
				// })
				// update++
			}
		}
	}
}

// NewHebbTrainer returns a newly intantiated HebbTrainer
func NewHebbTrainer() *HebbTrainer {
	return new(HebbTrainer)
}
