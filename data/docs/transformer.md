# The Transformer Architecture

The Transformer is a neural network architecture for sequence modeling that
relies entirely on attention mechanisms, dispensing with the recurrence and
convolutions used by earlier models. Because it processes all positions in a
sequence in parallel rather than step by step, it trains far more efficiently on
modern hardware than recurrent networks do.

## Self-attention

The core idea is self-attention: for each position in the input, the model
computes a weighted sum of the representations of all positions, where the
weights reflect how relevant each other position is. Each input is projected
into three vectors — a query, a key, and a value. The attention weight between
two positions is computed by taking the dot product of one position's query with
another's key, scaling it down by the square root of the key dimension, and
passing the result through a softmax so the weights sum to one. The output for a
position is the weighted sum of all value vectors. This lets every position draw
information directly from every other position in a single step.

## Multi-head attention

Rather than computing attention once, the Transformer runs several attention
operations in parallel, each called a head. Each head uses its own learned query,
key, and value projections, so different heads can specialize in different kinds
of relationships — one head might track syntactic dependencies while another
tracks longer-range topical links. The outputs of all heads are concatenated and
projected back to the model dimension. This multi-head design gives the model
several independent "views" of the sequence at once.

## Positional encoding

Because self-attention has no inherent notion of order, the Transformer adds
positional information to the input embeddings. The original design uses fixed
sinusoidal functions of different frequencies, which lets the model attend to
relative positions and generalize to sequence lengths not seen during training.
Later variants learn positional embeddings instead.

## Encoder and decoder

The full architecture has an encoder that maps an input sequence to a set of
continuous representations, and a decoder that generates an output sequence one
token at a time. Both are stacks of identical layers; each layer contains a
multi-head attention block and a position-wise feed-forward network, with
residual connections and layer normalization around each sub-block. The decoder
additionally attends over the encoder's output, which is how, for example, a
translation model conditions each output word on the source sentence.

## Why it mattered

By removing recurrence, the Transformer made it practical to train very large
models on very large datasets, since the computation parallelizes across
sequence positions. This scalability is the foundation of the large language
models that followed, which are predominantly stacks of Transformer layers
trained on enormous text corpora.
