function [A,B,rhos,pvals,U,V,Rxx,Ryy] = myCanonCorr(X,Y,Kx,Ky)
%[A,B,rhos,pvals,U,V] = myCanonCorr(X,Y,Kx,Ky) 
%   regularized canonical correlation
if nargin<2, error('JD: at least two argument required'); end
if ~ismatrix(X), error('JD: X must have two dimensions'); end;
if ~ismatrix(Y), error('JD: Y must have two dimensions'); end;
if size(X,1)>size(X,2), X=X.'; warning('JD: transposing X'); end;
if size(Y,1)>size(Y,2), Y=Y.'; warning('JD: transposing Y'); end;


[Rxy,Rxx,Ryy,Ryx] = nanRXY(X,Y);
if nargin<4 || isempty(Ky), Ky=rank(Ryy); end;
if nargin<3 || isempty(Kx), Kx=rank(Rxx); end;

% compute A
Rxxnsq=regSqrtInv(Rxx,Kx); % regularized Rxx^(-1/2)
M = Rxxnsq*Rxy*regInv(Ryy,Ky)*Ryx*Rxxnsq; % textbook
M = (M+M')/2; % fix nummerical precision asymmetric
[C,Dc]=eig(M);  % textbook
[~,indx]=sort(diag(Dc),'descend'); 
C=C(:,indx(1:min(Kx,Ky))); % dump zero eigenvalue dimensions 
A=Rxxnsq*C; % invert coordinate transformation

% compute B
Ryynsq=regSqrtInv(Ryy,Ky); % regularized Ryy^(-1/2)
D=Ryynsq*Ryx*Rxxnsq*C;
B=Ryynsq*D;


[A,Da]=eig(regInv(Rxx,Kx)*Rxy*regInv(Ryy,Ky)*Ryx);
[B,Db]=eig(regInv(Ryy,Ky)*Ryx*regInv(Rxx,Kx)*Rxy);

if sum(~isreal(A(:))) | sum(~isreal(B(:)))
    warning('Imaginary components. Something is wrong!'); 
    A=real(A); B=real(B);
end

U=A.'*X;
V=B.'*Y;

nVars=min(size(U,1),size(V,1));
rhos=zeros(nVars,1);
pvals=zeros(nVars,1);
for n=1:nVars
    n
    [r,p]=corrcoef(U(n,:),V(n,:));
    rhos(n)=r(1,2);
    pvals(n)=p(1,2);
end

return

X=randn(5,1000);
Y=randn(5,1000);
[A,B,rhos,pvals,U,V] = myCanonCorr(X,Y,5,5);
rhos
pvals










