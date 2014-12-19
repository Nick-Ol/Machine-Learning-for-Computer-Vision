clc, clear; %Hell yeah Celio Leonard Collado
N = 5;

%%  Potential functions

phi = cell(5, 5)
phi{1,2} = [.1,.9; .5,.2; .1,.1];
phi{2,1} = phi{1,2}'; %useless
phi{2,3} = [.1,.9,.2,.3; .8,.2,.3,.6];
phi{3,2} = phi{2,3}'; %useless
phi{2,4} = [.1,.9;.8,.2];
phi{4,2} = phi{2,4}'; 
phi{1,5} = [.1,.2;.8,.1;.3,.7];
phi{5,1} = phi{1,5}';

%% Sum-Product :
% forward messages:
messages{3,2} = sum(phi{2,3},2);
messages{4,2} = sum(phi{2,4},2);
messages{2,1} = phi{1,2} * (messages{3,2} .* messages{4,2});
messages{5,1} = sum(phi{1,5},2);

% backward :
messages{1,2} = phi{1,2}' * messages{5,1};
messages{2,3} = phi{2,3}' * (messages{1,2} .* messages{4,2});
messages{2,4} = phi{2,4}' * (messages{1,2} .* messages{3,2});
messages{1,5} = phi{1,5}' * messages{2,1};

%% Marginal distribs :
MargDist = cell(5, 1);
% P(X1) :
B1 = messages{2,1} .* messages{5,1}
Z1 = sum(B1) % partition function
MargDist{1} = B1/Z1

% P(X2) :
B2 = messages{3,2} .* messages{4,2} .* messages{1,2}
MargDist{2} = B2 / Z1

% P(X3) :
P3 = messages{2,3};
MargDist{3} = P3 / sum(P3);

% P(X4) :
P4 = messages{2,4};
MargDist{4} = P4 / sum(P4);

% P(X5) :
P5 = messages{1,5};
MargDist{5} = P5 / sum(P5);

proba([1;2;4;2;1], MargDist)
proba([3;1;2;1;2], MargDist)
proba([2;2;1;2;1], MargDist)

%% Max-Product :
% forward :
messages_max{3,2} = max(phi{2,3},[],2); % maximum on each row
messages_max{4,2} = max(phi{2,4},[],2);
messages_max{2,1} = zeros(3,1);
for i =1:3
    messages_max{2,1}(i) = max(phi{1,2}(i,:)' .* (messages{3,2} .* messages{4,2}));
end
messages_max{5,1} = max(phi{1,5},[],2);

% backward :
messages_max{1,2} = zeros(2,1);
for i = 1:2
    messages_max{1,2}(i) = max(phi{1,2}(:,i) .* messages{5,1});
end
messages_max{2,3} = zeros(4,1);
for i = 1:4
    messages_max{2,3}(i) = max(phi{2,3}(:,i) .* (messages{1,2} .* messages{4,2}));
end
messages_max{2,4} = zeros(2,1);
for i = 1:2
    messages_max{2,4}(i) = max(phi{2,4}(:,i) .* (messages{1,2} .* messages{3,2}));
end
messages_max{1,5} = zeros(2,1);
for i = 1:2
    messages_max{1,5}(i) = max(phi{1,5}(:,i) .* messages{2,1});
end

% B(X1) :
Bmax1 = messages_max{2,1} .* messages_max{5,1};

% B(X2) :
Bmax2 = messages_max{3,2} .* messages_max{4,2} .* messages_max{1,2};

% B(X3) :
Bmax3 = messages_max{2,3};

% B(X4) :
Bmax4 = messages_max{2,4};

% B(X5) :
Bmax5 = messages_max{1,5};

%% X5=1, X4=2, X1=3
phi_bis{1,2} = [.1,.1];
phi_bis{2,3} = phi{2,3};
phi_bis{2,4} = [.9;.2];
phi_bis{1,5} = .3;

% sum-product :
% forward :
messages_bis{3,2} = sum(phi_bis{2,3},2);
messages_bis{4,2} = sum(phi_bis{2,4},2);
messages_bis{2,1} = phi_bis{1,2} * (messages_bis{3,2} .* messages_bis{4,2});
messages_bis{5,1} = sum(phi_bis{1,5},2);

% backward :
messages_bis{1,2} = phi_bis{1,2}' * messages_bis{5,1};
messages_bis{2,3} = phi_bis{2,3}' * (messages_bis{1,2} .* messages_bis{4,2});
messages_bis{2,4} = phi_bis{2,4}' * (messages_bis{1,2} .* messages_bis{3,2});
messages_bis{1,5} = phi_bis{1,5}' * messages_bis{2,1};

% Pbis(X2) :
P2_bis = messages_bis{3,2} .* messages_bis{4,2} .* messages_bis{1,2};
P2_bis = P2_bis / sum(P2_bis)

% Pbis(X3) :
P3_bis = messages_bis{2,3};
P3_bis = P3_bis / sum(P3_bis)

% Partition function bis :
Z_bis = sum(messages_bis{2,3})



