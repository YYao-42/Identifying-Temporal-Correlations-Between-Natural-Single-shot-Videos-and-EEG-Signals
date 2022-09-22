function [Rxy,Rxx,Ryy,Ryx] = nanRXY(X,Y)
D=size(X,1);
RXY=nancov([X.' Y.'],'pairwise');
Rxx=RXY(1:D,1:D);
Ryy=RXY(D+1:end,D+1:end);
Rxy=RXY(1:D,D+1:end);
if nargout==4, Ryx=RXY(D+1:end,1:D); end

% D=size(X,1);
% RXY=nancov([X.' Y.'],'pairwise');
% Rxx1=RXY(1:D,1:D);
% Rxx2=RXY(2*D+1:3*D,2*D+1:3*D);
% 
% Ryy=RXY(D+1:2*D,D+1:);
% Rxy=RXY(1:D,D+1:end);