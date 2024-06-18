package mnist

import (
	"bufio"
	"bytes"
	"compress/gzip"
	_ "embed"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
)

//go:embed data/train-images-idx3-ubyte.gz
var TrainData []byte

//go:embed data/t10k-images-idx3-ubyte.gz
var TestData []byte

//go:embed data/train-labels-idx1-ubyte.gz
var TrainLabels []byte

//go:embed data/t10k-labels-idx1-ubyte.gz
var TestLabels []byte

// A Sample is one instance of a handwritten digit.
type Sample struct {
	// Intensities is a bitmap of white-and-black
	// values, where 1 is black and 0 is white.
	Intensities []float64

	// Label is a number between 0 and 9 (inclusive)
	// indicating what digit this is.
	Label int
}

// A DataSet is a collection of samples.
type DataSet struct {
	Samples []Sample

	// These fields indicate the dimensions of
	// the sample bitmaps.
	Width  int
	Height int
}

func LoadTrainingDataSet() DataSet {
	return loadDataSet("train")
}

func LoadTestingDataSet() DataSet {
	return loadDataSet("t10k")
}

func loadDataSet(prefix string) DataSet {
	labelFilename := prefix + "-labels-idx1-ubyte.gz"
	imageFilename := prefix + "-images-idx3-ubyte.gz"
	intensities, w, h, err := readIntensities(assetReader(imageFilename))
	if err != nil {
		panic("failed to read images: " + err.Error())
	}
	labels, err := readLabels(assetReader(labelFilename), len(intensities))
	if err != nil {
		panic("failed to read labels: " + err.Error())
	}
	var dataSet DataSet
	dataSet.Width = w
	dataSet.Height = h
	dataSet.Samples = make([]Sample, len(intensities))
	for i := range dataSet.Samples {
		floats := make([]float64, len(intensities[i]))
		for i, x := range intensities[i] {
			floats[i] = float64(x) / 255.0
		}
		dataSet.Samples[i].Intensities = floats
		dataSet.Samples[i].Label = labels[i]
	}
	return dataSet
}

// IntensityVectors returns a slice of intensity
// vectors, one per sample.
func (d DataSet) IntensityVectors() [][]float64 {
	res := make([][]float64, len(d.Samples))
	for i, sample := range d.Samples {
		res[i] = sample.Intensities
	}
	return res
}

// LabelVectors returns a slice of output vectors,
// where the first value of an output vector is 1
// for samples labeled 0, the second value is
// 1 for samples labeled 1, etc.
//
// This is useful for classifiers such as neural
// networks where the output of the network is a
// vector of probabilities.
func (d DataSet) LabelVectors() [][]float64 {
	res := make([][]float64, len(d.Samples))
	for i, sample := range d.Samples {
		res[i] = make([]float64, 10)
		res[i][sample.Label] = 1
	}
	return res
}

func assetReader(name string) io.Reader {
	var r *bytes.Reader
	if name == "t10k-images-idx3-ubyte.gz" {
		r = bytes.NewReader(TestData)
	} else if name == "train-images-idx3-ubyte.gz" {
		r = bytes.NewReader(TrainData)
	} else if name == "t10k-labels-idx1-ubyte.gz" {
		r = bytes.NewReader(TestLabels)
	} else if name == "train-labels-idx1-ubyte.gz" {
		r = bytes.NewReader(TrainLabels)
	} else {
		panic("file not found: " + name)
	}
	fmt.Println(r.Len())
	reader, err := gzip.NewReader(r)
	if err != nil {
		panic(fmt.Sprintf("could not decompress %s: %s", name, err.Error()))
	}
	return reader
}

func readIntensities(reader io.Reader) (results [][]uint8, width, height int, err error) {
	r := bufio.NewReader(reader)
	if _, err := r.Discard(4); err != nil {
		return nil, 0, 0, err
	}

	var params [3]uint32

	for i := 0; i < 3; i++ {
		if err := binary.Read(r, binary.BigEndian, &params[i]); err != nil {
			return nil, 0, 0, err
		}
	}

	count := int(params[0])
	width = int(params[1])
	height = int(params[2])

	results = make([][]uint8, count)
	for j := range results {
		var buffer bytes.Buffer
		limited := io.LimitedReader{R: r, N: int64(width * height)}
		if n, err := io.Copy(&buffer, &limited); err != nil {
			return nil, 0, 0, err
		} else if n < int64(width*height) {
			return nil, 0, 0, errors.New("not enough data for image")
		}

		vec := make([]uint8, width*height)
		for i, b := range buffer.Bytes() {
			vec[i] = uint8(b)
		}
		results[j] = vec
	}

	return
}

func readLabels(reader io.Reader, count int) ([]int, error) {
	r := bufio.NewReader(reader)

	if _, err := r.Discard(8); err != nil {
		return nil, err
	}

	res := make([]int, count)
	for i := range res {
		label, err := r.ReadByte()
		if err != nil {
			return nil, err
		}
		res[i] = int(label)
	}

	return res, nil
}
