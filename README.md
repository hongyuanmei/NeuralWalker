# SelectiveGeneration
Code base for [Mei et al. AAAI 2016 paper](https://arxiv.org/abs/1506.04089)

## Dependencies
* [Anaconda](https://www.continuum.io/) - Anaconda includes all the Python-related dependencies
* [Theano](http://deeplearning.net/software/theano/) - Computational graphs are built on Theano
* [ArgParse](https://docs.python.org/2/howto/argparse.html) - Command line parsing in Python

## Instructions
Here are the instructions to use the code base.

### Train Models
To train the model with options, use the command line 
```
python train_model.py --options
```
For the details of options, please check
```
python train_model.py --help
```

### Test Models
Choose a model to evaluate on test map, with the command line:
```
python test_model.py --options
python test_ensemble.py --options
```
For the details of options, please check
```
python test_models.py --help
python test_ensemble.py --options
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

