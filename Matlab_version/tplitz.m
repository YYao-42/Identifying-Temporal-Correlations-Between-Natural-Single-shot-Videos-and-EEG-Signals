function X=tplitz(x,K)

% construct block toeplitz matrix
X=[];

% to understand this function, take the most common case where 'x' is a
% single column
for i=1:size(x,2)
    Xtmp=toeplitz(x(:,i));
    uind=find(triu(Xtmp,1));
    Xtmp(uind)=NaN; 
    %Xtmp=Xtmp(:,1:K+1);
    Xtmp=Xtmp(:,1:K);
    
    % aggregate
    X=cat(2,X,Xtmp);
end

for row=1:size(X,1)
    nanind=find(isnan(X(row,:)));
    if ~isempty(nanind)
        %X(row,nanind)=X(row,nanind(1)-1);
        X(row,nanind)=0;
    end
end

% add zero-degree coefficient
X=cat(2,X,ones(size(X,1),1));

return