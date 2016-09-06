package neurago_example

import (
	"fmt"
	"log"

	"github.com/lemourA/neurago"
)

var inputPatterns = [8][]float64{
	[]float64{
		-1, -1, -1,
	},
	[]float64{
		-1, -1, 1,
	},
	[]float64{
		-1, 1, -1,
	},
	[]float64{
		-1, 1, 1,
	},
	[]float64{
		1, -1, -1,
	},
	[]float64{
		1, -1, 1,
	},
	[]float64{
		1, 1, -1,
	},
	[]float64{
		1, 1, 1,
	},
}

func ExampleHopfieldNetwork() {
	net, err := neurago.NewHopfieldNetwork([]neurago.Neuron{
		neurago.NewMCPNeuron(0, 0),
		neurago.NewMCPNeuron(0, 0),
		neurago.NewMCPNeuron(0, 0),
	})
	if err != nil {
		log.Panicln("Error: ", err)
	} else {
		trainingPatterns := [][]float64{
			[]float64{1, 1, -1},
			[]float64{-1, 1, 1},
		}
		trainer := neurago.NewHebbTrainer()

		fmt.Println(net.Output())

		trainer.Train(net, trainingPatterns)

		for _, pattern := range inputPatterns {
			net.SetInput(pattern)
			fmt.Println("Inputed: ", pattern)
			fmt.Println("Recalled: ", net.Output())
			fmt.Println("-------------------")
		}
	}
}
