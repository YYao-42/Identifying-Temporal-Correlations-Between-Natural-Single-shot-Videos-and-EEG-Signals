function xClean = pcaDenoise( X , nPcsToKeep )
%PCADENOISE denoise data matrix by removing pca noise subspace
if ~ismatrix(X), error('JD: X must have two dimensions'); end;
if size(X,1)>size(X,2), X=X.'; warning('JD: transposing EEG'); end;
X=X-repmat(mean(X,2),1,size(X,2)); % row-centered
nChannels=size(X,1);

R=nancov(X','pairwise');
[V,D]=eig(R);
[~,sortind]=sort(diag(D),'descend');
nPcsToRemove=rank(R)-nPcsToKeep;
if nPcsToRemove<0 || nPcsToRemove > nChannels,
   error('JD: invalid value for nPcsToKeep');
end
indremove=sortind(1:nPcsToRemove);
Vn=V(:,indremove);
xClean=X-Vn*Vn'*X; 

%X(isnan(X))=0; 
%[U,S,V]=svd(X,0); % economy-size svd
%Strunc=S; % singular values
%Strunc(K+1:end,K+1:end)=0;  % truncate spectrum
%xClean=U*Strunc*V'; % reconstruct clean data

return  % end of function

eegPathname='/Users/jacek/Documents/MATLAB/NMC/data/sb2012/';
eegFilenames=dir([eegPathname '*.mat']); 
load(fullfile(eegPathname,eegFilenames(1).name));
X=Y1{1};
xClean=svdDenoise(X,10);
%figure(1);
%subplot(221);


