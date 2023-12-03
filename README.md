## Inspiration
The brain is the most complex electrical circuit known to mankind. Our team is truly motivated to try to understand and reverse-engineer this complex machine so essential to us all.

## What it does
Our project simulates the brain activity when presented with an auditory stimulus.

## How we built it
We modelised the cochlea using Continuous Wavelet Transforms to separate a complicated audio input stream into ranges of frequencies. Then, with an equation describing the distribution of frequencies in the human cochlea, we were abled to determine which neurons would be stimulated by said sounds.

These 'neurons' form the base of our neural network, which we trained to reproduce the brain regions' reaction to the stimulus. We used real fMRI data to train our model!

## Challenges we ran into and what we learned from them
We realized that we could not use the data taken from the research team without doing a lot of preprocessing on our part beforehand. We also realized that producing an artificial cochlea which is the most anatomically accurate as possible is pretty difficult as it requires a lot of heavy math.

However, we managed to overcome these difficulties and we learned a lot about data science, machine learning along with imitating biological structures.

## What's next for Simulated Auditory Evoked Hemodynamics (SAEH)
This approach to analysing biological systems such as the brain using artificial components has limitless applications. Not only could we go further by applying this method to other senses (smell, vision, physical touch) but it could also be combined with modern medecine in a larger scale : scientists discovered that mice can 'mute' some of the sounds that are supposed to enter their brain. If we can understand the human brain better, notably using neural networks, we could be able to understand how diseases such as chronic pain work and find a cure for them.
