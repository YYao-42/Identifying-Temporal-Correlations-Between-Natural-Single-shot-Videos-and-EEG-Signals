Stimulus Response Correlation Toolbox

Provided here is a set of Matlab functions that allow one to implement the “stimulus-response correlation” (SRC) technique of Dmochowski et al (2018).  The technique is applicable when neural activity is recorded to a stimulus that lacks the properties that are required to perform a conventional Event-Related analysis.  For example, the stimulus may be long, continuous and naturalistic, and there may only be a single repetition per subject (in contrast to the event-related design where a brief stimulus is repeated many times over).


The following are the requirements for deploying the code: (i) a time series of stimulus features (see below) that were presented to the subject(s), (ii) a multi-sensor neural response, such as a matrix of EEG or MEG responses that was recorded during the stimulus presentation, and (iii) time alignment between the stimulus and the neural response.  


Notes about alignment: Our experience is that precise time alignment is unfortunately necessary to achieve relatively high correlation values.  A good tool for collecting temporally synchronized data is Lab Streaming Layer (LSL).   In our recordings, we embed small bright squares that flash periodically in the top-right corner of the screen.  These flashes are captured with a photodiode and then recorded with an auxiliary channel of an EEG recording system.  We have also included a function that we have successfully used to align frames of a video game stimulus with the recorded EEG samples: alignStim2EEG.m performs a non-linear procedure that enforces strict alignment between the “flash” events, while linearly interpolating between co-registered events.  We have been using Open Broadcaster Software for screen capture.  


The core function is myCanonCorr.m -- this is our regularized version of Mathworks’ canoncorr.m.  In the case of neural responses, regularization is often required to prevent overfitting when learning projections of the data.  The function supports missing values (must be coded as NaNs).  The script demo.m walks through a standard SRC analysis, here shown on sample data collected from a participant viewing a film clip.  


The basic idea behind SRC is that the stimulus-driven neural response is correlated with some feature of the stimulus.   Here, the feature may be a time series of visual contrast or optic flow, or it could be the sound envelope waveform.  In all cases, the algorithm learns a temporal filter to apply to the stimulus, and a spatial filter to apply to the neural response, such that the filter outputs are maximally correlated.  The output is a set of components, where each component pair reflects an independent set of stimulus/response mappings.  The components are ranked in order of descending correlation, and often the bulk of the correlation is captured by just a few components.  This means that you can perform your analysis of experimental effects just confining yourself to these few dimensions.  


Reference:
Dmochowski, J. P., Ki, J. J., DeGuzman, P., Sajda, P., & Parra, L. C. (2018). Extracting multidimensional stimulus-response correlations using hybrid encoding-decoding of neural activity. NeuroImage, 180, 134-146.

Acknowledgments: This research was funded by the Army Research Office (ARO) Partnered Research Initiative, administered by the Army Research Laboratory (ARL) under the The Cognition and Neuroergonomics Collaborative Technology Alliance (Can-CTA)         W911-NF-10-2-0022.