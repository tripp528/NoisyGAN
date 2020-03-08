just running with ddsp ideas:


- right now

We feed in a sample

we encode it into  f0, amp, and z

using this encoding, call it FAZ, we decode it into audio file

We compare constructed sample to real sample, and backpropogate


- what if


we feed in noise vector FAZ

using FAZ, decode it audio file

feed this into discriminator, say this ain’t a real sample

	backpropogate

feed in a real sample into the discriminator, say this is real

	backpropogate


The network will learn to make “realistic” sounding samples. Midi, f0, and z!!

No pretraining of a bunch of auto encoders.

Just music…..FULLY constructed by GANNNNNNNN BITCH




expansion …..

so, how do we generate the ZZZ from a noise vector??????

- idea: compositional pattern-producing networks
    - (CPNN)
    - expained in HYPERNEAT paper
    -



Dataset is NSynth, available [4] in the paper
