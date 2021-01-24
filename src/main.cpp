#include <charconv>
#include <iostream>
#include <fstream>

#include <CustomLibrary/NeuralNet.h>
#include <CustomLibrary/RandomGenerator.h>

#include <CustomLibrary/Streamer.h>

using namespace ctl;

rnd::Random<rnd::Mersenne> g_rand;

using MNIST_Entry = std::pair<mth::Matrix<double>, mth::Matrix<double>>;

auto load_mnist(std::string_view name)
{
	std::ifstream			 mnist_file(name.data(), std::ios::in);
	std::vector<MNIST_Entry> entries;

	assert(mnist_file && "File couldn't open.");

	for (char c; mnist_file.get(c);)
	{
		auto &entry = entries.emplace_back();

		entry.first			 = mth::Matrix<double>(10, 1, 0.);
		entry.first[c - '0'] = 1.;

		entry.second = mth::Matrix<double>(28 * 28, 1);

		while (mnist_file.get() != '\n')
		{
			std::string b;
			while (mnist_file.peek() != ',' && mnist_file.peek() != '\n') b.push_back(mnist_file.get());

			unsigned char c;
			std::from_chars(b.data(), b.data() + b.size(), c);

			entry.second.emplace_back(c / 255.);
		}
	}

	return entries;
}

auto highest(const mth::Matrix<double> &e) { return std::distance(e.begin(), std::max_element(e.begin(), e.end())); }

auto operator<<(std::ostream &o, const MNIST_Entry &e) -> std::ostream &
{
	for (size_t y = 0; y < 28; ++y)
	{
		for (size_t x = 0; x < 28; ++x) o.put(e.second[y * 28 + x] > 0 ? '#' : ' ');
		o.put('\n');
	}

	o << "Correct Value: " << highest(e.first);
	return o;
}

int main(int argc, char **argv)
{
	if (argc > 2)
	{
		std::cerr << "Usage: MNIST_Recog train"
					 "train: Specify if you want the network trained or use the pretrained parameters.\n";
	}

	const auto				init = [] { return g_rand.rand_number(-1., 1.); };
	mcl::BasicNeuralNetwork nn({ 28 * 28, 200, 10 }, init);

	std::vector<std::pair<mth::Matrix<double>, mth::Matrix<double>>> set;

	if (argc > 1) // true ~> train the network; false ~> import trained network
	{
		set = load_mnist("mnist_train.csv");

		// Train the network
		std::clog << "Started training...\n";
		for (size_t epoch = 0; epoch < 5; ++epoch)
		{
			for (const auto &e : set) mcl::fit(nn, e.second, e.first, .1);
			std::clog << "Finished epoch " << epoch + 1 << '\n';
		}

		std::ofstream file_out("data.nn", std::ios::binary);
		file_out << nn;
		file_out.close();
	}
	else
	{
		std::ifstream file_in("data.nn", std::ios::binary);
		file_in >> nn;
		file_in.close();
	}

	set = load_mnist("mnist_test.csv");

	// Evaluate network
	for (const auto &e : set)
	{
		const auto res = nn.query(e.second);
		std::cout << e << "\nPredicted value: " << highest(res) << "\n\n Next?";

		::getchar();
	}

	return 0;
}