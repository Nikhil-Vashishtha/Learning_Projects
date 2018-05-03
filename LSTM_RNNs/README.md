# LSTM_RNNs

work source: 

https://github.com/spiglerg/RNN_Text_Generation_Tensorflow

# Text generation using a RNN (LSTM) using Tensorflow.

## Changes made in the form of:-
	#myAddition
	<code blocks here>
	#

In train mode:

hyperparametersEffects.csv saves the model variables and time taken in training.

If different file name needed, can be reinitiated and commented out again in lines 300-1, and used in 307 and 311

In talk mode:
createdText.txt saves the generated text with timestamp whenever called. Keeps a log.

savedBackup folder shows example models saved.

##

Any file, shakespeare, Republic of Plato, Story of Plato can be trained and text can be generated. Get and try on new texts.

Test the coherence and flow of text created as per tweaking of hyperparameters. This gives better understanding of the model being used and studied.

## Usage

To train the model you can set the textfile you want to use to train the network by using command line options:

Run the network in train mode:

  $ python textGenerator.py --input_file=data/shakespeare.txt --ckpt_file="saved/model.ckpt" --mode=train

Run the network to generate text:

  $ python textGenerator.py --input_file=data/shakespeare.txt --ckpt_file="saved/model.ckpt" --test_prefix="The " --mode=talk
