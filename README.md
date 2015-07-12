# RNN-LSTM documents soft-clustering

Many unsupervised learning methods exist to generate fixed length feature representations of documents. The large majority of them (LSI, LDA, deep belief networks ...) use the bag of words model. For those methods, if it is possible to set up some workarounds and use n-grams, it is always at the expense of the computational time, flexibility, and often overall result quality. Text data is sequential by nature, two words might have a different meaning depending on the context. To my knowledge we are still lacking today a good method to generate feature vectors that fully exploit the information contained in those sequences. 

Recently, recurrent neural networks are under the spotlights. The long-short term memory (LSTM) neurons improvement answers the vanishing gradient problem and allows to find correlation in long sequences. Many articles have been published in the past year, successfully using this technology for optical character recognition, machine translation, word representation, language models, reinforcement learning ... In my case I got familiar with LSTM networks working on connected handwritting recognition. Some papers can be found here and there using RNN for document classification, yet I couldn't find a proper unsupervised setup.

The initial idea that I had was to use a recurrent autoencoder inspired from "[Unsupervised Learning of Video Representations using LSTMs](http://arxiv.org/pdf/1502.04681.pdf)". The word-tokens sequences would be given to the LSTM-RNN which would generate a feature vector representing the entire sequence. From this representation the network would then try to reconstruct the input sequence. This way we would force the network to build representation containing the semantic information of the entire sequence. Unfortunately, if this approach is possible with images, asking a network to reconstruct an entire sequence of word-tokens from a low dimensional vector seems too optimistic and prone to overfitting. On another hand I heard about several successful applications using characters sequences as inputs instead of word tokens. This approach seems interesting since it greatly reduces the input and output size of our networks (less characters than words).

###Setup

In the end I decided to try this character sequence approach and hijack the [char-rnn](https://github.com/karpathy/char-rnn) implementation. Char-rnn aims to create character level language models. In order to predict correctly which character might come next, the network has to know the context in which he has to do his prediction. Hence we can expect the RNN to learn how to encode the context of sequences of characters, and using this context to predict the next character. Let's call this character based context the local context, accumulating the local contexts for all the characters of a document can give us a global context vector. 

For my first setup, I trained a language model on 140 Mo of wikipedia articles (one article is one sequence). For few documents of my test set, I generate a feature vector accumulating all the local context vectors made of all the hidden activations of the last hidden layer. Due to hardware limitations I was only able to set up a rather small network of two LSTM layers of size 300.  

###Results

A lot of work still has to be done (see future guidelines), the actual results serve as proof of concept showing some semantic clustering is happening. Multiples context vectors are generated for several random wikipedia articles, the cosine distance between the vectors is used to rank those articles among themselves.

*Table showing the cosine distance rankings for several wikipedia documents:*

| austin | american_revolution | anarcho_capitalism
|:-----:|:-----:|:-----:
| alaska | american_civil_war | anarchism
| asia | alaska | american_civil_war
| american_civil_war | asia | american_revolution
| american_revolution | austin | autism
| anarcho_capitalism | anarcho_capitalism | alaska
| achilles | achilles | asia
| anarchism | anarchism | albedo
| albedo | albedo | austin
| autism | autism | achilles

Of course better experiments are required prior to draw any conclusion. This is just a small preliminary work. My point here is that documents about locations are close to other documents about locations, similarly american history documents are close together, etc ... 

###Future guidelines

Among the ideas I would like to try: 

* Always inspired by "[Unsupervised Learning of Video Representations using LSTMs](http://arxiv.org/pdf/1502.04681.pdf)" the first step would be to change the training procedure of the RNN. Char-rnn tries to guess the next character only, however the context representation needed to guess this t+1 character can be very simple. To force the network to create better context representation we might ask him to guess the next k characters ( k>1).  

* Another improvement would be to use a bidirectional network. With the current procedure the sequence is handled from left to right only, we are tryig to predict the future from the past. However in our case the sequences are given and we already know the future. With a bidirectional network, local context vectors would represent the accumulation of the information from left to right and right to left. 

* An important limitation of the current setup is also the network size. A network with 3 or more hidden layers might generate more elaborate representations. 

* Finally the character-based and word-based approach should be compared. A comparison with LDA would also be interesting.

* Try [gated feedback rnn](http://arxiv.org/pdf/1502.02367v4.pdf)

* Find a way to modify the loss function to generate sparse vectors for representations. 

Right now I am working on implementing the following net:

![](https://github.com/mpagli/RNN-doc-soft-clustering/blob/master/rnn_proto_1.jpg)

This net represents a language model where each prediction is done from a context vector and the current character. Learning this way will hopefully make it possible to dissociate the semantic (context) from the character representation.

