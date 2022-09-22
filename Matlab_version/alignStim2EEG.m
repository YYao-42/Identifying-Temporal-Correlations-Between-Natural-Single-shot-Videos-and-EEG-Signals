function [eegResamp,stimResamp] = alignStim2EEG(stim,indicator,fps,eegSeries,eegStamps,triggerStamps,eegFs,targetFs)
% [EEGRESAMP,STIMRESAMP] = ALIGNSTIM2EEG(STIM,INDICATOR,FPS,EEGSERIES,EEGSTAMPS,TRIGGERSTAMPS,EEGFS,TARGETFS)
%
% align a stimulus feature time series with EEG
%
% NB: To use this function, you need at least two screen events that are
% captured in a trigger channel that is on the same clock as the EEG.  In
% our case, we had white squares flashing periodically in the top
% right of the screen ("flash frames").   These flashes were captured by a
% photodiode and the timing of captures are represented by variable
% triggerStamps (see below).
%
% stim: a vector of stimulus feature values
% indicator: a binary vector of the same size as stim, where 1s indicate
%   the presence of screen events that are registered with a trigger
% fps: video frame rate
% eegSeries: a matrix of EEG data (rows are electrodes, columns are
%   samples) 
% eegStamps: a vector of time stamps that correspond to the EEG data (as
%   produced by LSL)
% triggerStamps: time stamps of trigger events matching the events coded by
%   indicator
% eegFs: EEG sampling rate
% targetFs: the sampling rate of the output EEG and stimulus data
%
% eegResamp: the EEG matrix after alignment
% stimResamp: the stimulus feature after alignment
%
% (c) Jacek P. Dmochowski, 2019-

%% trigger
triggerEventsTime=triggerStamps; 
diffTriggerEventsTime=[0 diff(triggerEventsTime)];

%% stimulus
if numel(stim(:))~=numel(indicator(:))
    error('stim and indicator must have the same number of elements');
end
indicator=indicator(:).'; % row
nFrames=numel(indicator);
indicatorTime=(0:nFrames-1)/fps;
indicatorOnsetsIndices=find(indicator);
if isempty(indicatorOnsetsIndices)
    error('No indicator events found');
end
indicatorOnsetsTime=indicatorTime(indicatorOnsetsIndices);

%% assuming that first indicator matches first trigger event...
if numel(triggerStamps)~=numel(indicatorOnsetsTime)
    error('Indicator and trigger lengths do not match');
end

nEvents=numel(triggerStamps);

%% alignment  
% basic algorithm
% first flash frame time is given by the time of the first photodiode spike
% subsequent flash frame times are given by the previous time + photodiode
% ISI
flashFrameTime=zeros(nEvents,1);
flashFrameTime(1)=triggerEventsTime(1);  
for f=2:nEvents
    flashFrameTime(f)=flashFrameTime(f-1)+diffTriggerEventsTime(f);
end
flashFrameValue=stim(indicatorOnsetsIndices);
flashFrameIndices=indicatorOnsetsIndices;

% to fill in features in between flash frames
% linearly interpolate
allV=[]; allT=[];
for f=1:nEvents-1
    nIntervening= flashFrameIndices(f+1) - flashFrameIndices(f) - 1;
    vals=stim(flashFrameIndices(f)+1:flashFrameIndices(f+1)-1);
    dt= ( flashFrameTime(f+1)-flashFrameTime(f) ) / ( flashFrameIndices(f+1) - flashFrameIndices(f) );
    tmes=(1:nIntervening)'*dt+flashFrameTime(f);
    allV=cat(1,allV,[flashFrameValue(f);vals]);
    allT=cat(1,allT,[flashFrameTime(f);tmes]);
end

[~,eeg_stamp_start_index]=min(abs(eegStamps-triggerEventsTime(1)));
[~,eeg_stamp_stop_index]=min(abs(eegStamps-triggerEventsTime(end)));
    
tInterp=eegStamps(eeg_stamp_start_index):1/eegFs:eegStamps(eeg_stamp_stop_index);
stimInterp=interp1(allT,allV,tInterp,'linear','extrap');

%%
% stimInterp and eeg are now on the same (eeg) clock
% resample them both to a more reasonable rate
stimResamp=resample(stimInterp,targetFs,eegFs);
nStimResampFrames=numel(stimResamp);

eegSamplesKept=eeg_stamp_start_index:eeg_stamp_stop_index;
eeg_series_cut=eegSeries(:,eegSamplesKept);
eegResamp=(resample(eeg_series_cut.',targetFs,eegFs)).';
nEEGresamplesKept=size(eegResamp,2);

%% logic bit to make the stim and eeg same length
if nEEGresamplesKept>nStimResampFrames
    nSamplesToRemove=nEEGresamplesKept-nStimResampFrames;
    eegResamp=eegResamp(:,1:end-nSamplesToRemove);
elseif nEEGresamplesKept<nStimResampFrames
    nSamplesToRemove=nStimResampFrames-nEEGresamplesKept;
    stimResamp=stimResamp(1:end-nSamplesToRemove);
else
    % lengths match
end
end

