// Package neurago is a little library providing tools to
// implement artificial neural networks.
package neurago

import (
	"encoding/csv"
	"local/utils/math"
	"log"
	"os"
	"strconv"

	"github.com/gonum/matrix/mat64"
)

//var pinv_lRate = 0.1

// Pseudoinverse trainer which trains networks using the hebb learning rule
type PseudoInverseTrainer struct{}

// computeWeight computes the weights between neurons A and neurons B.
func computeWeight(net ANN,
	nA, nB Neuron,
	i, j int,
	patterns [][]float64,
	correlInv *mat64.Dense) {

	var weight float64
	nbOfPatterns := len(patterns)
	nbOfNeurons := float64(len(net.Neurons()))
	for v := 0; v < nbOfPatterns; v++ {
		for u := 0; u < nbOfPatterns; u++ {
			weight += patterns[v][i] * correlInv.At(v, u) * patterns[u][j]
		}
	}
	weight *= (1.0 / nbOfNeurons)
	nA.SetConnection(nB, math.ToFixed(weight, 11))
}

// see Trainer#train
func (t PseudoInverseTrainer) Train(net ANN, patterns [][]float64) {
	var sum float64
	nbOfPatterns := len(patterns)
	nbOfNeurons := len(net.Neurons())
	correl := mat64.NewDense(nbOfPatterns, nbOfPatterns, make([]float64, nbOfPatterns*nbOfPatterns))
	correlInv := mat64.NewDense(nbOfPatterns, nbOfPatterns, make([]float64, nbOfPatterns*nbOfPatterns))

	file, _ := os.OpenFile("pinv_result.csv", os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0666)
	defer file.Close()
	csvWriter := csv.NewWriter(file)
	csvWriter.Write([]string{
		"update",
		"error",
	})
	defer csvWriter.Flush()
	update := 1

	for v := 0; v < nbOfPatterns; v++ {
		for u := 0; u < nbOfPatterns; u++ {
			sum = 0
			for k := 0; k < nbOfNeurons; k++ {
				sum += patterns[v][k] * patterns[u][k]
			}
			correl.Set(v, u, sum/float64(nbOfNeurons))
		}
	}
	if err := correlInv.Inverse(correl); err != nil {
		log.Panicln("PseudoInverseTrainer#Train, Mat64.Dense#Inverse error: ", err)
	}
	for i, nA := range net.Neurons() {
		for j, nB := range net.Neurons() {
			if i != j {
				computeWeight(net, nA, nB, i, j, patterns, correlInv)
				//--
				error := computeError(net.Neurons(), patterns)
				csvWriter.Write([]string{
					strconv.Itoa(update),
					strconv.FormatFloat(error, 'f', -1, 64),
				})
				update++
			}
		}
	}
}

// NewPseudoInverseTrainer returns a newly intantiated PseudoInverseTrainer
func NewPseudoInverseTrainer() *PseudoInverseTrainer {
	return new(PseudoInverseTrainer)
}
