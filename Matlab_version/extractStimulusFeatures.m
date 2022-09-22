function features = extractStimulusFeatures(stimFilename,outFilename)
%FEATURES = EXTRACTSTIMULUTSFEATURES(STIMFILENAME,OUTFILENAME)
% read in a movie and return visual and auditory features
%
% stimFilename: string movie file (e.g 'avi' file)
% outFilename: string mat file where features will be stored (optional)
%
% features: struct whose fields hold the time series of features

if nargin<2
    outFilename=[stimFilename(1:end-4) '-features-' date '.mat'];
end

% read in stimulus file
vidObj=VideoReader(stimFilename);
vidFrames=read(vidObj);
nFrames=size(vidFrames,4);
fsVideo=vidObj.FrameRate;
fsVideo=round(fsVideo);

% set up optical flow computation
opticalFlow = vision.OpticalFlow('ReferenceFrameSource','Input port');

% pre-allocate features
muFlow=zeros(nFrames,1);
muSqFlow=zeros(nFrames,1);
muLuminance=zeros(nFrames,1);
muSqLuminance=zeros(nFrames,1);
muLocalContrast=zeros(nFrames,1);
muSqLocalContrast=zeros(nFrames,1);
stdLocalContrast=zeros(nFrames,1);
muTemporalContrast=zeros(nFrames,1);
muSqTemporalContrast=zeros(nFrames,1);
kern=ones(30);  % 2-D kernel function for local contrast
diode=zeros(nFrames,1);
for f=1:nFrames
    f/nFrames*100;
    img=squeeze(vidFrames(:,:,:,f));
    
    grayImg=rgb2gray(img);
    floatGrayImg=double(grayImg);
    
    if f>1
        imgRef=squeeze(vidFrames(:,:,:,f-1));
        floatGrayImgRef=double(rgb2gray(imgRef));
    else
        floatGrayImgRef=zeros(size(vidFrames,1),size(vidFrames,2));
    end
    
    flow=step(opticalFlow,floatGrayImg,floatGrayImgRef);
    muFlow(f)=mean2(flow);
    muSqFlow(f)=mean2(flow.^2);
       
    % if you have a flickering square at the top right of the screen,
    % the next few lines of code will represent that to facilitate time 
    % alignment between stimulus and eeg
    topCorner=floatGrayImg(1:3,end-2:end); % 3 x 3 square, for now
    topCorner=topCorner(:);
    diode(f)=mean(topCorner);
    
    muTemporalContrast(f)=mean2(floatGrayImg-floatGrayImgRef);
    muSqTemporalContrast(f)=mean2((floatGrayImg-floatGrayImgRef).^2);
    
    % compute luminance
    muLuminance(f)=mean2(floatGrayImg);
    muSqLuminance(f)=mean2((floatGrayImg).^2);
    
    % local contrast
    bg = conv2(floatGrayImg,kern,'same');
    bg(isnan(bg)) = 0;
    lc = abs((floatGrayImg - bg) ./ bg);
    lc(isnan(lc)) = 0;
    muLocalContrast(f)=mean2(lc);
    muSqLocalContrast(f)=mean2((lc).^2);
    stdLocalContrast(f)=std2(lc);
end

% read in soundtrack
try
    [y,fsAudio]=audioread(stimFilename);
catch
    error('failed to read audio file');
end
yh=hilbert(y(:,1));  % soundtrack is often mono
soundEnvelope=sqrt(real(yh).^2+imag(yh).^2);

fsAudio=round(fsAudio)
soundEnvelopeDown=resample(soundEnvelope,fsVideo,fsAudio); % downsample to video frame rate

% legacy
fs=fsVideo;

% output features
features.fs=fs;
features.fsVideo=fsVideo;
features.fsAudio=fsAudio;
features.muFlow=muFlow;
features.muSqFlow=muSqFlow;
features.muTemporalContrast=muTemporalContrast;
features.muSqTemporalContrast=muSqTemporalContrast;
features.muLuminance=muLuminance;
features.muSqLuminance=muSqLuminance;
features.muLocalContrast=muLocalContrast;
features.muSqLocalContrast=muSqLocalContrast;
features.stdLocalContrast=stdLocalContrast;
features.soundEnvelope=soundEnvelope;
features.soundEnvelopeDown=soundEnvelopeDown;
features.diode=diode;

% output to file
save(outFilename,'features','fs','fsVideo','fsAudio',...
    'muFlow','muSqFlow',...
    'muTemporalContrast','muSqTemporalContrast',...
    'muLuminance','muSqLuminance',...
    'muLocalContrast','muSqLocalContrast','stdLocalContrast',...
    'soundEnvelope','soundEnvelopeDown','diode');
end