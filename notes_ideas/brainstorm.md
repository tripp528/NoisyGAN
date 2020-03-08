#### Where to put dataset?

options:

1) Just pass it to the DDSP_TRAINER constructor.  

2) Pass it individually to train & predict functions.

pros:
  - this is how keras models usually are. You don't pass in a dataset when you
    compile the  model, you pass  it into the fit function.

  - passing in different datasets to train and test.

cons:
  - the buildModel() needs to know n_samples, which you get from DDSP_DATASET..
    so you need to pass it in somewhere into initialization.

3) Make an argument tfrecord_filepattern=None and an argument n_samples=DEFAULT_N_SAMPLES

- I don't really understand what n_samples is doing....

- So it gets an audio sample from the dataset, then gets the first argument of the shape.

- It doesn't actually care how many are in the dataset, it cares how many individual
  waves there are in a given sample.

- Since the dataset has samples that are all uniform length (4s.. where is that defined??),
  and the sample rate has a default value, we can theoretically make a default n_samples...


Yes! Using option 3!

- But now it can't build the model, because there's no dataset.
  (it uses forward pass to build)

- maybe don't build until train / predict?
