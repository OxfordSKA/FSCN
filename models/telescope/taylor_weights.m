function Wch = taylor_weights(x,y,SLL)
% Wch = taylor_weights(x,y,SLL)
% x,y = positions of antenna in array
% SLL = sidelobe level desired
% e.g. SLL = -28;

x = x-(min(x)+max(x))/2;
y = y-(min(y)+max(y))/2;

NBAR = ceil(2*(acosh(10^(-SLL/20))/pi)^2+0.5);

W_taylor = taylorwin(10000, NBAR, SLL);
W_taylor(1:5000) = [];

Rvec = sqrt(x.^2 + y.^2);
rr = linspace(0.0001,max(Rvec)+0.001,5000);

ab = ones(1,length(x));
for ii = 1:length(x)
    rdif = abs(Rvec(ii)-rr);
    bb = find(rdif == min(rdif));
    ab(ii) = bb(1);
end

Wch = W_taylor(ab);
Wch = Wch./max(Wch);