function SqrtInvR = regSqrtInv( R,K )
%invR = regInv( R,K )
%   PCA regularized square root of inverse of square symmetric positive 
%   definite matrix R = V D^(-1/2) V' using only K principal components V.

% Jacek Dmochowski, Lucas Parra, Jason Ki
if nargin<2, K=size(R,1); end;
if ~ismatrix(R), error('JD: R must have two dimensions'); end;
if size(R,1)~=size(R,2), error('JD: R must be a square matrix'); end;


[V,D]=eig(R); 
[d,sortIndx]=sort(diag(D),'ascend');
V=V(:,sortIndx);  % in case eigenvectors/eigenvalues not sorted
d=d(end-K+1:end);

% regularized sqrt inverse
SqrtInvR=V(:,end-K+1:end)*diag(1./sqrt(d))*V(:,end-K+1:end)';  

end

