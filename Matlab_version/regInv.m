function invR = regInv( R,K )
%invR = regInv( R,K )
%   PCA regularized inverse of square symmetric positive definite matrix R
if nargin<2, K=size(R,1); end;
if ~ismatrix(R), error('JD: R must have two dimensions'); end;
if size(R,1)~=size(R,2), error('JD: R must be a square matrix'); end;


[V,D]=eig(R); 
[d,sortIndx]=sort(diag(D),'ascend');
V=V(:,sortIndx);  % in case eigenvectors/eigenvalues not sorted
d=d(end-K+1:end);
invR=V(:,end-K+1:end)*diag(1./d)*V(:,end-K+1:end)';  % regularized inverse

end

